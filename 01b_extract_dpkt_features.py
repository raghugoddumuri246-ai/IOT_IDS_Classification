"""
==========================================================
SCRIPT 01b : DPKT FLOW-LEVEL FEATURE EXTRACTION
==========================================================

PIPELINE POSITION:
  01_extract_flows.py (NFStream)  --\
                                      >--  [THIS SCRIPT merges with NFStream in 04_*.py]
  01b_extract_dpkt_features.py    --/

INPUT  : Same raw PCAP files as 01_extract_flows.py
         CIC_IOT_PCAP/<attack_name>/*.pcap
OUTPUT : DPKT_CSV/<attack_name>/<pcap_name>.csv
         One row per flow, with a "flow_key" column used to
         JOIN against the corresponding NFStream CSV in
         Script 04.

USAGE:
   python 01b_extract_dpkt_features.py DDoS-ACK_Fragmentation
   python 01b_extract_dpkt_features.py DDoS-SlowLoris


WHY A SEPARATE SCRIPT INSTEAD OF MODIFYING NFStream'S OUTPUT?
NFStream does not expose low-level IP/TCP header fields such
as per-packet TTL, IP fragmentation flags, or TCP window size
sequences. DPKT gives direct access to every header field at
the byte level, which is needed for:

  - DDoS-ACK/ICMP/UDP_Fragmentation : IP fragmentation flags
    (MF, DF, fragment offset) are the textbook signature of
    fragmentation attacks. NFStream does not expose these.
  - Mirai-greeth vs Mirai-greip vs Mirai-udpplain : TTL stats
    can reveal encapsulation-layer differences not fully
    captured by NFStream's existing packet-size features.
  - Recon-OSScan : TCP window size patterns vary between
    OS-fingerprinting tools (nmap, etc.) and normal OS stacks.
  - DDoS-SlowLoris : header-byte entropy / structure of the
    handful of packets sent over a long-lived connection.


CRITICAL DESIGN REQUIREMENT — FLOW BOUNDARIES MUST MATCH NFSTREAM
==================================================================
NFStream groups packets into "flows" using the standard 5-tuple
(src_ip, dst_ip, src_port, dst_port, protocol), AND ends a flow
once IDLE_TIMEOUT seconds pass with no new packet on that tuple
(NFStream's default idle_timeout = 120 seconds). If the SAME
5-tuple appears again after that gap, NFStream treats it as a
BRAND NEW flow, not a continuation.

This script replicates that exact behaviour:
  1. The 5-tuple is normalised so that A->B and B->A packets
     (same flow, opposite direction) map to the SAME base key
     (NFStream flows are bidirectional).
  2. If no packet has been seen for that base key in the last
     IDLE_TIMEOUT_S seconds, a NEW sub-flow counter is started
     for that key, exactly mirroring NFStream's flow-splitting.

If this timeout logic is removed or set differently than
NFStream's, flow_key values will NOT correspond 1:1 between
the NFStream and DPKT outputs, and the merge in Script 04
will be silently wrong (rows joined to the wrong flow, or
flows that should be separate getting merged together).

The merge step in Script 04 reports a "match rate" between
NFStream flow_keys and DPKT flow_keys precisely because a
PERFECT 100% match cannot be guaranteed (DPKT and NFStream
parse certain edge-case packets, like malformed headers or
non-IP traffic, slightly differently) — this is expected and
documented as a known limitation, not a bug to chase to zero.
==========================================================
"""

import os
import sys
import socket
import struct
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import dpkt

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

INPUT_ROOT  = Path("CIC_IOT_PCAP")
OUTPUT_ROOT = Path("DPKT_CSV")

# MUST match NFStream's idle_timeout setting used in
# 01_extract_flows.py. NFStream's default is 120 seconds.
# If 01_extract_flows.py is ever changed to pass a custom
# idle_timeout to NFStreamer(), update this value to match.
IDLE_TIMEOUT_S = 120.0

# How many of the first N packets in a flow to compute
# window-size / TTL "first-N" statistics over, mirroring
# NFStream's splt_analysis=10 window for comparability.
FIRST_N_PACKETS = 10


# ==========================================================
# HELPERS
# ==========================================================

def ip_to_str(addr_bytes):
    return socket.inet_ntoa(addr_bytes)


def normalize_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """
    Build a direction-independent flow key so that A->B and
    B->A packets of the same flow map to the same key.
    This MUST use the same field order/format that Script 04
    will reconstruct from the NFStream CSV's src_ip/dst_ip/
    src_port/dst_port/protocol columns, before those columns
    are dropped.
    """
    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)
    if endpoint_a <= endpoint_b:
        return f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_{proto}"
    else:
        return f"{dst_ip}_{dst_port}_{src_ip}_{src_port}_{proto}"


def safe_stats(values):
    """min/mean/std/max for a list of numbers, 0 if empty."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.min()), float(arr.mean()), float(arr.std()), float(arr.max())


def new_flow_record():
    return {
        "packet_count": 0,
        "ttl_values": [],
        "tcp_window_values": [],
        "ip_total_len_values": [],
        "header_byte_samples": [],   # first bytes of IP header per packet, for entropy
        "frag_mf_count": 0,          # "more fragments" flag set
        "frag_df_count": 0,          # "don't fragment" flag set
        "frag_offset_nonzero_count": 0,  # mid-fragment packets
        "ip_options_present_count": 0,   # IP header > 20 bytes (options present)

        # ---- GRE encapsulation detection (separates Mirai-greeth vs
        # Mirai-greip; this is the SINGLE STRONGEST new feature added
        # in this round — see docstring section "GRE INNER PROTOCOL") ----
        "gre_packet_count": 0,
        "gre_inner_proto_ip_count": 0,    # inner protocol = IP (0x0800)   -> greip signature
        "gre_inner_proto_ether_count": 0, # inner protocol = Ethernet (0x6558) -> greeth signature

        # ---- ICMP type/code tracking (separates Recon-HostDiscovery
        # ping sweeps from generic ICMP floods) ----
        "icmp_echo_request_count": 0,   # type 8 (ping)
        "icmp_echo_reply_count": 0,     # type 0 (pong)
        "icmp_other_count": 0,          # any other ICMP type (dest unreachable, etc.)

        # ---- TCP scan-pattern flag combinations (separates
        # Recon-OSScan / Recon-PortScan style probes from normal
        # connections, which always start with a plain SYN) ----
        "tcp_null_scan_count": 0,   # flags == 0 (no flags set at all)
        "tcp_fin_scan_count": 0,    # only FIN set
        "tcp_xmas_scan_count": 0,   # FIN + PUSH + URG set together
    }


def byte_entropy(byte_list):
    """Shannon entropy (bits) over a flat list of byte values, 0 if empty."""
    if not byte_list:
        return 0.0
    arr = np.array(byte_list, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


# ==========================================================
# CORE EXTRACTION — ONE PCAP FILE
# ==========================================================

def extract_dpkt_features(pcap_path):
    """
    Reads one PCAP file packet-by-packet and groups packets into
    flows using the same 5-tuple + idle-timeout logic NFStream
    uses, then computes header-level statistics per flow.

    FRAGMENTATION HANDLING:
    Only the FIRST fragment of a fragmented IP datagram carries
    the transport-layer header (TCP/UDP ports). Subsequent
    fragments carry only raw payload bytes, with port=0 in this
    script's parsing. If left unhandled, fragment 2+ would be
    keyed as a DIFFERENT flow (wrong ports) than fragment 1.

    The IP header's "id" (identification) field is the same
    across ALL fragments of one original datagram (this is how
    every real flow tracker, including NFStream, associates
    fragments — it is mandated by RFC 791, not an NFStream-
    specific choice). This script tracks, for each (src_ip,
    dst_ip, ip_id) triple, which flow_key the FIRST fragment
    was assigned to, and routes all subsequent fragments of
    that same datagram to the same flow_key.

    Returns a pandas DataFrame, one row per flow, with a
    'flow_key' column for joining against the NFStream CSV.
    """
    last_seen_ts   = {}                  # base_key -> last packet timestamp
    sub_flow_index = defaultdict(int)    # base_key -> current sub-flow counter
    flows          = defaultdict(new_flow_record)

    # Maps (src_ip, dst_ip, ip_identification) -> the full_key that
    # the FIRST fragment of that datagram was assigned to. Lets
    # later fragments (which lack L4 ports) join the same flow.
    fragment_to_flow_key = {}

    n_packets_total = 0
    n_packets_parsed = 0

    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)

        for ts, buf in reader:
            n_packets_total += 1

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue

            ip_pkt = eth.data
            if not isinstance(ip_pkt, dpkt.ip.IP):
                # Skip non-IP packets (ARP, etc.) — DDoS/DoS/Mirai/Recon
                # classes targeted here are all IP-based attacks.
                continue

            try:
                src_ip = ip_to_str(ip_pkt.src)
                dst_ip = ip_to_str(ip_pkt.dst)
            except Exception:
                continue

            proto = ip_pkt.p
            ip_id = getattr(ip_pkt, "id", 0)
            frag_offset = getattr(ip_pkt, "offset", 0)
            is_fragment_continuation = frag_offset > 0

            src_port = dst_port = 0
            tcp_window = None
            tcp_flags = None
            icmp_type = None
            is_gre = False
            gre_inner_proto = None

            if not is_fragment_continuation:
                # Only the first fragment (offset=0) — or a normal,
                # unfragmented packet — carries the L4 header.
                transport = ip_pkt.data
                if isinstance(transport, dpkt.tcp.TCP):
                    src_port = transport.sport
                    dst_port = transport.dport
                    tcp_window = transport.win
                    tcp_flags = transport.flags
                elif isinstance(transport, dpkt.udp.UDP):
                    src_port = transport.sport
                    dst_port = transport.dport
                elif isinstance(transport, dpkt.icmp.ICMP):
                    icmp_type = transport.type
                elif isinstance(transport, dpkt.gre.GRE):
                    # GRE encapsulation (Mirai-greeth / Mirai-greip).
                    # transport.p is the GRE "protocol type" field,
                    # which identifies what is encapsulated INSIDE:
                    #   0x0800 (ETH_TYPE_IP)      -> a raw IP packet inside
                    #                                 -> Mirai-greip signature
                    #   0x6558 (Transparent Eth.
                    #           Bridging)           -> a raw Ethernet frame inside
                    #                                 -> Mirai-greeth signature
                    is_gre = True
                    gre_inner_proto = getattr(transport, "p", None)
                # ICMP and other protocols: ports stay 0, which is fine —
                # the 5-tuple still uniquely identifies ICMP flows by
                # (src_ip, dst_ip, proto) since port is constant (0) for them.

            datagram_key = (src_ip, dst_ip, ip_id)

            if is_fragment_continuation and datagram_key in fragment_to_flow_key:
                # Route this continuation fragment to the SAME flow
                # the first fragment of this datagram was assigned to.
                full_key = fragment_to_flow_key[datagram_key]
                base_key = full_key.rsplit("#", 1)[0]
                last_seen_ts[base_key] = ts
            else:
                base_key = normalize_flow_key(src_ip, dst_ip, src_port, dst_port, proto)

                # ---- Idle-timeout flow splitting (mirrors NFStream) ----
                if base_key in last_seen_ts:
                    gap = ts - last_seen_ts[base_key]
                    if gap > IDLE_TIMEOUT_S:
                        sub_flow_index[base_key] += 1
                last_seen_ts[base_key] = ts

                full_key = f"{base_key}#{sub_flow_index[base_key]}"

                if not is_fragment_continuation:
                    # Remember this assignment in case later fragments
                    # of the SAME datagram (offset>0) arrive afterward.
                    fragment_to_flow_key[datagram_key] = full_key

            record = flows[full_key]

            # ---- Per-packet feature accumulation ----
            record["packet_count"] += 1
            record["ttl_values"].append(ip_pkt.ttl)
            record["ip_total_len_values"].append(ip_pkt.len)

            if tcp_window is not None:
                record["tcp_window_values"].append(tcp_window)

            # IP fragmentation fields — direct signature for the
            # *_Fragmentation attack classes.
            if getattr(ip_pkt, "mf", 0):
                record["frag_mf_count"] += 1
            if getattr(ip_pkt, "df", 0):
                record["frag_df_count"] += 1
            if frag_offset > 0:
                record["frag_offset_nonzero_count"] += 1

            # IP header length > 20 bytes means IP options are present
            # (ip_pkt.hl is header length in 32-bit words; 5 = 20 bytes)
            if getattr(ip_pkt, "hl", 5) > 5:
                record["ip_options_present_count"] += 1

            # ---- GRE inner-protocol detection ----
            if is_gre:
                record["gre_packet_count"] += 1
                if gre_inner_proto == 0x0800:
                    record["gre_inner_proto_ip_count"] += 1
                elif gre_inner_proto == 0x6558:
                    record["gre_inner_proto_ether_count"] += 1

            # ---- ICMP type tracking ----
            if icmp_type is not None:
                if icmp_type == 8:
                    record["icmp_echo_request_count"] += 1
                elif icmp_type == 0:
                    record["icmp_echo_reply_count"] += 1
                else:
                    record["icmp_other_count"] += 1

            # ---- TCP scan-pattern flag combinations ----
            # These flag combinations almost never occur in normal
            # traffic and are textbook signatures of port/OS scanning
            # tools (e.g. nmap -sN, -sF, -sX scan modes).
            if tcp_flags is not None:
                if tcp_flags == 0:
                    record["tcp_null_scan_count"] += 1
                elif tcp_flags == dpkt.tcp.TH_FIN:
                    record["tcp_fin_scan_count"] += 1
                elif (tcp_flags & dpkt.tcp.TH_FIN) and \
                     (tcp_flags & dpkt.tcp.TH_PUSH) and \
                     (tcp_flags & dpkt.tcp.TH_URG):
                    record["tcp_xmas_scan_count"] += 1

            # Byte-entropy sample: first up to 8 raw bytes of the IP
            # payload, only for the first FIRST_N_PACKETS packets of
            # the flow (keeps computation cheap on huge flood flows).
            if record["packet_count"] <= FIRST_N_PACKETS:
                try:
                    payload = bytes(ip_pkt.data.data) if hasattr(ip_pkt.data, "data") else bytes(ip_pkt.data) if isinstance(ip_pkt.data, (bytes, bytearray)) else b""
                    record["header_byte_samples"].extend(list(payload[:8]))
                except Exception:
                    pass

            n_packets_parsed += 1

    # ---- Build one output row per flow ----
    rows = []
    for full_key, rec in flows.items():
        base_key = full_key.rsplit("#", 1)[0]

        ttl_min, ttl_mean, ttl_std, ttl_max = safe_stats(rec["ttl_values"])
        win_min, win_mean, win_std, win_max = safe_stats(rec["tcp_window_values"])
        len_min, len_mean, len_std, len_max = safe_stats(rec["ip_total_len_values"])

        rows.append({
            "flow_key": full_key,
            "dpkt_packet_count": rec["packet_count"],

            "dpkt_ttl_min": ttl_min,
            "dpkt_ttl_mean": ttl_mean,
            "dpkt_ttl_std": ttl_std,
            "dpkt_ttl_max": ttl_max,

            "dpkt_tcp_window_min": win_min,
            "dpkt_tcp_window_mean": win_mean,
            "dpkt_tcp_window_std": win_std,
            "dpkt_tcp_window_max": win_max,

            "dpkt_ip_total_len_min": len_min,
            "dpkt_ip_total_len_mean": len_mean,
            "dpkt_ip_total_len_std": len_std,
            "dpkt_ip_total_len_max": len_max,

            "dpkt_frag_mf_count": rec["frag_mf_count"],
            "dpkt_frag_df_count": rec["frag_df_count"],
            "dpkt_frag_offset_nonzero_count": rec["frag_offset_nonzero_count"],
            "dpkt_ip_options_present_count": rec["ip_options_present_count"],

            "dpkt_header_byte_entropy": byte_entropy(rec["header_byte_samples"]),

            # ---- GRE encapsulation (Mirai-greeth vs Mirai-greip) ----
            "dpkt_gre_packet_count": rec["gre_packet_count"],
            "dpkt_gre_inner_proto_ip_count": rec["gre_inner_proto_ip_count"],
            "dpkt_gre_inner_proto_ether_count": rec["gre_inner_proto_ether_count"],

            # ---- GRE RATIO FEATURES (NEW) ----
            #
            # WHY: Raw GRE counts are diluted by background non-GRE flows
            # in Mirai PCAPs. A flow where EVERY packet is GRE-encapsulated
            # (gre_ratio=1.0) is unambiguously a Mirai-GRE attack flow.
            # A background flow with gre_count=0 gets gre_ratio=0.0.
            #
            # dpkt_gre_ratio: overall fraction of this flow's packets that
            #   were GRE. Confirmed 0.0 for all Mirai-udpplain flows
            #   (verified from real data). Will be 1.0 for pure GRE flows
            #   in greeth/greip captures.
            #
            # dpkt_gre_inner_ip_ratio: within GRE packets, what fraction
            #   carried an IP inner protocol (greip signature)?
            #   greip -> ratio ~1.0, greeth -> ratio ~0.0
            #
            # dpkt_gre_inner_ether_ratio: within GRE packets, what fraction
            #   carried an Ethernet inner protocol (greeth signature)?
            #   greeth -> ratio ~1.0, greip -> ratio ~0.0
            #
            # Together these three ratios give the model a clean
            # 3-way separation:
            #   udpplain:  gre_ratio=0
            #   greip:     gre_ratio=1, inner_ip_ratio=1, inner_ether_ratio=0
            #   greeth:    gre_ratio=1, inner_ip_ratio=0, inner_ether_ratio=1
            "dpkt_gre_ratio": (
                rec["gre_packet_count"] / rec["packet_count"]
                if rec["packet_count"] > 0 else 0.0
            ),
            "dpkt_gre_inner_ip_ratio": (
                rec["gre_inner_proto_ip_count"] / rec["gre_packet_count"]
                if rec["gre_packet_count"] > 0 else 0.0
            ),
            "dpkt_gre_inner_ether_ratio": (
                rec["gre_inner_proto_ether_count"] / rec["gre_packet_count"]
                if rec["gre_packet_count"] > 0 else 0.0
            ),

            # ---- FRAGMENTATION RATIO (NEW) ----
            #
            # WHY: Raw frag_mf_count is absolute. A 10,000-packet flood
            # with 100 fragmented packets looks less fragmented than a
            # 10-packet flow where all 10 are fragmented.
            # The RATIO captures what fraction of this flow's packets
            # carried the "more fragments" flag.
            #
            # Verified from data:
            #   Mirai-udpplain: always 0.0 (no fragmentation at all)
            #   DDoS-ICMP/UDP_Fragmentation: > 0 on actual fragmented flows
            #   DDoS-ICMP/UDP_Flood: 0.0 (pure flood, no fragmentation)
            #
            # Note: does NOT fix unfragmented flows WITHIN a fragmentation
            # attack PCAP — those still look like flood flows. The
            # file_frag_rate feature in Script 04 addresses that gap.
            "dpkt_frag_ratio": (
                rec["frag_mf_count"] / rec["packet_count"]
                if rec["packet_count"] > 0 else 0.0
            ),

            # ---- ICMP type breakdown ----
            "dpkt_icmp_echo_request_count": rec["icmp_echo_request_count"],
            "dpkt_icmp_echo_reply_count": rec["icmp_echo_reply_count"],
            "dpkt_icmp_other_count": rec["icmp_other_count"],

            # ---- TCP scan-pattern flag combinations ----
            "dpkt_tcp_null_scan_count": rec["tcp_null_scan_count"],
            "dpkt_tcp_fin_scan_count": rec["tcp_fin_scan_count"],
            "dpkt_tcp_xmas_scan_count": rec["tcp_xmas_scan_count"],
        })

    stats = {
        "n_packets_total": n_packets_total,
        "n_packets_parsed": n_packets_parsed,
        "n_flows": len(flows),
    }
    return pd.DataFrame(rows), stats


# ==========================================================
# MAIN — mirrors 01_extract_flows.py usage pattern
# ==========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python 01b_extract_dpkt_features.py <attack_name>")
        sys.exit(1)

    attack_name = sys.argv[1]

    input_folder  = INPUT_ROOT / attack_name
    output_folder = OUTPUT_ROOT / attack_name
    output_folder.mkdir(parents=True, exist_ok=True)

    pcaps = sorted(input_folder.glob("*.pcap"))

    print(f"\nAttack : {attack_name}")
    print(f"PCAPs  : {len(pcaps)}")

    for i, pcap in enumerate(pcaps, start=1):
        output_csv = output_folder / f"{pcap.stem}.csv"

        if output_csv.exists():
            print(f"[{i}/{len(pcaps)}] Skip : {pcap.name}")
            continue

        print(f"[{i}/{len(pcaps)}] Processing : {pcap.name}")

        df, stats = extract_dpkt_features(pcap)

        df.to_csv(output_csv, index=False)

        print(f"  Packets total/parsed : {stats['n_packets_total']:,} / {stats['n_packets_parsed']:,}")
        print(f"  Flows extracted      : {stats['n_flows']:,}")
        print(f"  Saved : {output_csv.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()