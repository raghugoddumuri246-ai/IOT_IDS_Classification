"""
==========================================================
INFERENCE PIPELINE — Shared feature extraction for 11 and 12
==========================================================

This module replicates EXACTLY the steps that
04_create_balanced_attack_files.py performed on training data,
so that 11_predict_pcap.py and 12_predict_pcap_xai.py
produce the same 130-feature matrix the model was trained on.

PIPELINE STEPS:
  PCAP
   -> NFStream (statistical_analysis, splt_analysis=10, n_dissections=20)
   -> DPKT extraction on same PCAP                [NEW vs old scripts]
   -> Drop 24 identifier/IP/MAC/port/timestamp/payload columns
   -> SPLT expansion (3 string cols -> 30 numeric)
   -> DPKT merge on flow_key                      [NEW vs old scripts]
   -> Cardinality features (fan_in/fan_out)        [NEW vs old scripts]
   -> file_frag_rate from DPKT flows               [NEW vs old scripts]
   -> application_name/category encoding
   -> inf/NaN -> 0
   -> Reorder to feature_order.pkl (130 features)
   -> RobustScaler.transform(scale_cols)
   -> model.predict / predict_proba

FEATURE COUNT:
  92 NFStream-derived (62 flow stats + 30 SPLT)
  31 DPKT header/packet features
   6 cardinality (fan_in/fan_out derived)
   1 file_frag_rate (capture-level context)
  ─────
  130 total features
==========================================================
"""

import ast
import json
import socket
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import dpkt
from nfstream import NFStreamer


# ==========================================================
# CONSTANTS — MUST match 04_create_balanced_attack_files.py
# ==========================================================

SPLT_N            = 10
IDLE_TIMEOUT_S    = 120.0       # seconds — matches NFStream default
IDLE_TIMEOUT_MS   = 120 * 1000  # milliseconds — for NFStream CSV join
FIRST_N_PACKETS   = 10
CARDINALITY_WINDOW_MS = 5000   # 5-second time buckets

DROP_COLUMNS = [
    "id", "expiration_id",
    "src_ip", "dst_ip",
    "src_mac", "dst_mac",
    "src_oui", "dst_oui",
    "vlan_id", "tunnel_id",
    "src_port", "dst_port",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms",       "src2dst_last_seen_ms",
    "dst2src_first_seen_ms",       "dst2src_last_seen_ms",
    "application_is_guessed",
    "requested_server_name",
    "client_fingerprint",
    "server_fingerprint",
    "user_agent",
    "content_type",
]

DPKT_FEATURE_COLUMNS = [
    "dpkt_packet_count",
    "dpkt_ttl_min", "dpkt_ttl_mean", "dpkt_ttl_std", "dpkt_ttl_max",
    "dpkt_tcp_window_min", "dpkt_tcp_window_mean", "dpkt_tcp_window_std", "dpkt_tcp_window_max",
    "dpkt_ip_total_len_min", "dpkt_ip_total_len_mean", "dpkt_ip_total_len_std", "dpkt_ip_total_len_max",
    "dpkt_frag_mf_count", "dpkt_frag_df_count", "dpkt_frag_offset_nonzero_count",
    "dpkt_ip_options_present_count", "dpkt_header_byte_entropy",
    "dpkt_gre_packet_count", "dpkt_gre_inner_proto_ip_count", "dpkt_gre_inner_proto_ether_count",
    "dpkt_gre_ratio", "dpkt_gre_inner_ip_ratio", "dpkt_gre_inner_ether_ratio",
    "dpkt_frag_ratio",
    "dpkt_icmp_echo_request_count", "dpkt_icmp_echo_reply_count", "dpkt_icmp_other_count",
    "dpkt_tcp_null_scan_count", "dpkt_tcp_fin_scan_count", "dpkt_tcp_xmas_scan_count",
]


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def ip_to_str(addr_bytes):
    return socket.inet_ntoa(addr_bytes)

def safe_stats(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.min()), float(arr.mean()), float(arr.std()), float(arr.max())

def byte_entropy(byte_list):
    if not byte_list:
        return 0.0
    arr = np.array(byte_list, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def normalize_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Bidirectional flow key — same logic as 01b_extract_dpkt_features.py."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_{proto}"
    else:
        return f"{dst_ip}_{dst_port}_{src_ip}_{src_port}_{proto}"


# ==========================================================
# STEP 1 — SPLT EXPANSION (same as 04_*.py)
# ==========================================================

def expand_splt_column(series, prefix, n=SPLT_N):
    def safe_parse(val):
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return list(parsed[:n]) + [0] * (n - len(parsed))
        except Exception:
            pass
        return [0] * n
    parsed = series.apply(safe_parse)
    return pd.DataFrame(
        parsed.tolist(),
        columns=[f"{prefix}_{i}" for i in range(n)],
        index=series.index
    )

def expand_all_splt(df):
    for col, prefix in [("splt_direction","splt_direction"),
                         ("splt_ps","splt_ps"),
                         ("splt_piat_ms","splt_piat_ms")]:
        if col in df.columns:
            expanded = expand_splt_column(df[col], prefix, SPLT_N)
            df = df.drop(columns=[col])
            df = pd.concat([df, expanded], axis=1)
    return df


# ==========================================================
# STEP 2 — DPKT FEATURE EXTRACTION FROM PCAP
# ==========================================================

def new_flow_record():
    return {
        "packet_count": 0, "ttl_values": [], "tcp_window_values": [],
        "ip_total_len_values": [], "header_byte_samples": [],
        "frag_mf_count": 0, "frag_df_count": 0,
        "frag_offset_nonzero_count": 0, "ip_options_present_count": 0,
        "gre_packet_count": 0, "gre_inner_proto_ip_count": 0,
        "gre_inner_proto_ether_count": 0,
        "icmp_echo_request_count": 0, "icmp_echo_reply_count": 0,
        "icmp_other_count": 0,
        "tcp_null_scan_count": 0, "tcp_fin_scan_count": 0,
        "tcp_xmas_scan_count": 0,
    }

def extract_dpkt_features(pcap_path):
    """
    Runs DPKT over a PCAP file and returns a DataFrame of per-flow
    header/packet features, with a flow_key column for joining against
    the NFStream output.

    Uses the same idle-timeout (120s) and flow-key logic as
    01b_extract_dpkt_features.py so that flow_key values match
    what would have been produced during training.
    """
    last_seen_ts   = {}
    sub_flow_index = defaultdict(int)
    flows          = defaultdict(new_flow_record)
    fragment_to_flow_key = {}

    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            ip_pkt = eth.data
            if not isinstance(ip_pkt, dpkt.ip.IP):
                continue
            try:
                src_ip = ip_to_str(ip_pkt.src)
                dst_ip = ip_to_str(ip_pkt.dst)
            except Exception:
                continue

            proto      = ip_pkt.p
            ip_id      = getattr(ip_pkt, "id", 0)
            frag_offset = getattr(ip_pkt, "offset", 0)
            is_frag_continuation = frag_offset > 0

            src_port = dst_port = 0
            tcp_window = tcp_flags = None
            icmp_type = None
            is_gre = False
            gre_inner_proto = None

            if not is_frag_continuation:
                transport = ip_pkt.data
                if isinstance(transport, dpkt.tcp.TCP):
                    src_port, dst_port = transport.sport, transport.dport
                    tcp_window, tcp_flags = transport.win, transport.flags
                elif isinstance(transport, dpkt.udp.UDP):
                    src_port, dst_port = transport.sport, transport.dport
                elif isinstance(transport, dpkt.icmp.ICMP):
                    icmp_type = transport.type
                elif isinstance(transport, dpkt.gre.GRE):
                    is_gre = True
                    gre_inner_proto = getattr(transport, "p", None)

            datagram_key = (src_ip, dst_ip, ip_id)

            if is_frag_continuation and datagram_key in fragment_to_flow_key:
                full_key = fragment_to_flow_key[datagram_key]
                base_key = full_key.rsplit("#", 1)[0]
                last_seen_ts[base_key] = ts
            else:
                base_key = normalize_flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                if base_key in last_seen_ts and (ts - last_seen_ts[base_key]) > IDLE_TIMEOUT_S:
                    sub_flow_index[base_key] += 1
                last_seen_ts[base_key] = ts
                full_key = f"{base_key}#{sub_flow_index[base_key]}"
                if not is_frag_continuation:
                    fragment_to_flow_key[datagram_key] = full_key

            rec = flows[full_key]
            rec["packet_count"] += 1
            rec["ttl_values"].append(ip_pkt.ttl)
            rec["ip_total_len_values"].append(ip_pkt.len)

            if tcp_window is not None:
                rec["tcp_window_values"].append(tcp_window)
            if getattr(ip_pkt, "mf", 0):
                rec["frag_mf_count"] += 1
            if getattr(ip_pkt, "df", 0):
                rec["frag_df_count"] += 1
            if frag_offset > 0:
                rec["frag_offset_nonzero_count"] += 1
            if getattr(ip_pkt, "hl", 5) > 5:
                rec["ip_options_present_count"] += 1

            if is_gre:
                rec["gre_packet_count"] += 1
                if gre_inner_proto == 0x0800:
                    rec["gre_inner_proto_ip_count"] += 1
                elif gre_inner_proto == 0x6558:
                    rec["gre_inner_proto_ether_count"] += 1

            if icmp_type is not None:
                if icmp_type == 8:
                    rec["icmp_echo_request_count"] += 1
                elif icmp_type == 0:
                    rec["icmp_echo_reply_count"] += 1
                else:
                    rec["icmp_other_count"] += 1

            if tcp_flags is not None:
                if tcp_flags == 0:
                    rec["tcp_null_scan_count"] += 1
                elif tcp_flags == dpkt.tcp.TH_FIN:
                    rec["tcp_fin_scan_count"] += 1
                elif (tcp_flags & dpkt.tcp.TH_FIN) and \
                     (tcp_flags & dpkt.tcp.TH_PUSH) and \
                     (tcp_flags & dpkt.tcp.TH_URG):
                    rec["tcp_xmas_scan_count"] += 1

            if rec["packet_count"] <= FIRST_N_PACKETS:
                try:
                    payload = bytes(ip_pkt.data.data) if hasattr(ip_pkt.data, "data") \
                              else bytes(ip_pkt.data) if isinstance(ip_pkt.data, (bytes, bytearray)) else b""
                    rec["header_byte_samples"].extend(list(payload[:8]))
                except Exception:
                    pass

    rows = []
    for full_key, rec in flows.items():
        n_pkt = rec["packet_count"]
        ttl_min, ttl_mean, ttl_std, ttl_max = safe_stats(rec["ttl_values"])
        win_min, win_mean, win_std, win_max = safe_stats(rec["tcp_window_values"])
        len_min, len_mean, len_std, len_max = safe_stats(rec["ip_total_len_values"])
        gre_ct  = rec["gre_packet_count"]
        frag_ct = rec["frag_mf_count"]

        rows.append({
            "flow_key": full_key,
            "dpkt_packet_count": n_pkt,
            "dpkt_ttl_min": ttl_min, "dpkt_ttl_mean": ttl_mean,
            "dpkt_ttl_std": ttl_std, "dpkt_ttl_max": ttl_max,
            "dpkt_tcp_window_min": win_min, "dpkt_tcp_window_mean": win_mean,
            "dpkt_tcp_window_std": win_std, "dpkt_tcp_window_max": win_max,
            "dpkt_ip_total_len_min": len_min, "dpkt_ip_total_len_mean": len_mean,
            "dpkt_ip_total_len_std": len_std, "dpkt_ip_total_len_max": len_max,
            "dpkt_frag_mf_count": frag_ct,
            "dpkt_frag_df_count": rec["frag_df_count"],
            "dpkt_frag_offset_nonzero_count": rec["frag_offset_nonzero_count"],
            "dpkt_ip_options_present_count": rec["ip_options_present_count"],
            "dpkt_header_byte_entropy": byte_entropy(rec["header_byte_samples"]),
            "dpkt_gre_packet_count": gre_ct,
            "dpkt_gre_inner_proto_ip_count": rec["gre_inner_proto_ip_count"],
            "dpkt_gre_inner_proto_ether_count": rec["gre_inner_proto_ether_count"],
            "dpkt_gre_ratio": gre_ct / n_pkt if n_pkt > 0 else 0.0,
            "dpkt_gre_inner_ip_ratio": rec["gre_inner_proto_ip_count"] / gre_ct if gre_ct > 0 else 0.0,
            "dpkt_gre_inner_ether_ratio": rec["gre_inner_proto_ether_count"] / gre_ct if gre_ct > 0 else 0.0,
            "dpkt_frag_ratio": frag_ct / n_pkt if n_pkt > 0 else 0.0,
            "dpkt_icmp_echo_request_count": rec["icmp_echo_request_count"],
            "dpkt_icmp_echo_reply_count": rec["icmp_echo_reply_count"],
            "dpkt_icmp_other_count": rec["icmp_other_count"],
            "dpkt_tcp_null_scan_count": rec["tcp_null_scan_count"],
            "dpkt_tcp_fin_scan_count": rec["tcp_fin_scan_count"],
            "dpkt_tcp_xmas_scan_count": rec["tcp_xmas_scan_count"],
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["flow_key"] + DPKT_FEATURE_COLUMNS)


# ==========================================================
# STEP 3 — FLOW KEY RECONSTRUCTION FROM NFSTREAM CSV
# ==========================================================

def add_flow_keys(df):
    """
    Reconstructs the flow_key for each NFStream row using the same
    normalize_flow_key() + idle-timeout logic as 04_create_balanced_attack_files.py,
    so the keys match what DPKT produced for the same PCAP.
    """
    required = ["src_ip","dst_ip","src_port","dst_port","protocol","bidirectional_first_seen_ms"]
    if any(c not in df.columns for c in required):
        df["flow_key"] = None
        return df

    base_keys = df.apply(
        lambda r: normalize_flow_key(str(r["src_ip"]), str(r["dst_ip"]),
                                      r["src_port"], r["dst_port"], r["protocol"]),
        axis=1
    )

    order = df.assign(_bk=base_keys, _orig_idx=df.index) \
               .sort_values(["_bk", "bidirectional_first_seen_ms"])

    last_ms, sub_idx, fk_list = {}, {}, []
    for bk, ms in zip(order["_bk"], order["bidirectional_first_seen_ms"]):
        if bk in last_ms and (ms - last_ms[bk]) > IDLE_TIMEOUT_MS:
            sub_idx[bk] = sub_idx.get(bk, 0) + 1
        elif bk not in sub_idx:
            sub_idx[bk] = 0
        last_ms[bk] = ms
        fk_list.append(f"{bk}#{sub_idx[bk]}")

    fk_series = pd.Series(fk_list, index=order["_orig_idx"].values)
    df = df.copy()
    df["flow_key"] = df.index.map(fk_series)
    return df


# ==========================================================
# STEP 4 — CARDINALITY FEATURES (same as 04_*.py)
# ==========================================================

def add_cardinality_features(df):
    """Fan-in and fan-out cardinality features. Must run BEFORE dropping IPs/ports."""
    required = ["src_ip","dst_ip","dst_port","protocol","bidirectional_first_seen_ms"]
    if any(c not in df.columns for c in required):
        for col in ["fan_in_src_count","fan_out_port_count","fan_out_ip_count",
                    "fan_out_proto_count","fan_out_scope","fan_app_diversity"]:
            df[col] = 0
        return df

    df = df.copy()
    df["_tb"] = (df["bidirectional_first_seen_ms"] // CARDINALITY_WINDOW_MS).astype(int)

    df["fan_in_src_count"]   = df.groupby(["dst_ip","_tb"])["src_ip"].transform("nunique")
    df["fan_out_port_count"] = df.groupby(["src_ip","_tb"])["dst_port"].transform("nunique")
    df["fan_out_ip_count"]   = df.groupby(["src_ip","_tb"])["dst_ip"].transform("nunique")
    df["fan_out_proto_count"]= df.groupby(["src_ip","_tb"])["protocol"].transform("nunique")
    df["fan_out_scope"]      = df["fan_out_port_count"] * df["fan_out_ip_count"]

    if "application_name" in df.columns:
        df["fan_app_diversity"] = df.groupby(["src_ip","_tb"])["application_name"].transform("nunique")
    else:
        df["fan_app_diversity"] = 0

    df.drop(columns=["_tb"], inplace=True)
    return df


# ==========================================================
# MAIN FUNCTION — full pipeline from PCAP to feature matrix
# ==========================================================

def build_feature_matrix(pcap_path, app_name_vocab, app_category_vocab,
                          feature_order, scale_cols, scaler, verbose=True):
    """
    Full inference feature pipeline.
    Returns (X, dpkt_df) where X is the scaled 130-feature matrix ready
    for model.predict(), and dpkt_df is the raw DPKT output (for file_frag_rate
    reporting and debugging).
    """
    pcap_path = str(pcap_path)

    # ---- NFStream extraction ----
    if verbose:
        print(f"\nExtracting flows with NFStream...")
    nf_df = NFStreamer(
        source=pcap_path,
        statistical_analysis=True,
        splt_analysis=10,
        n_dissections=20
    ).to_pandas()
    if verbose:
        print(f"  NFStream flows extracted: {len(nf_df)}")

    if len(nf_df) == 0:
        return None, None

    # ---- DPKT extraction ----
    if verbose:
        print("Extracting packet-level features with DPKT...")
    dpkt_df = extract_dpkt_features(pcap_path)
    if verbose:
        print(f"  DPKT flows extracted: {len(dpkt_df)}")

    # ---- file_frag_rate: fraction of flows with any fragmented packets ----
    # Computed from DPKT output of THIS PCAP — exactly as training did it
    # per PCAP file. Gives the model capture-level context (is this a
    # fragmentation attack environment or a plain flood environment?).
    if len(dpkt_df) > 0 and "dpkt_frag_mf_count" in dpkt_df.columns:
        file_frag_rate = float((dpkt_df["dpkt_frag_mf_count"] > 0).sum()) / len(dpkt_df)
    else:
        file_frag_rate = 0.0
    if verbose:
        print(f"  file_frag_rate: {file_frag_rate:.4f}")

    # ---- Build flow_key on NFStream rows (before dropping IPs/ports) ----
    nf_df = add_flow_keys(nf_df)

    # ---- Merge DPKT features ----
    if "flow_key" in nf_df.columns and not nf_df["flow_key"].isnull().all():
        merged = nf_df.merge(dpkt_df, on="flow_key", how="left", suffixes=("","_dpkt"))
        for col in DPKT_FEATURE_COLUMNS:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0.0)
        matched = merged[DPKT_FEATURE_COLUMNS[0]].notna().sum()
        if verbose:
            pct = 100 * matched / len(merged) if len(merged) > 0 else 0
            print(f"  DPKT merge: {matched}/{len(merged)} flows matched ({pct:.1f}%)")
        merged.drop(columns=["flow_key"], errors="ignore", inplace=True)
    else:
        merged = nf_df.copy()
        for col in DPKT_FEATURE_COLUMNS:
            merged[col] = 0.0
        merged.drop(columns=["flow_key"], errors="ignore", inplace=True)

    # ---- Cardinality features (before dropping IPs/ports) ----
    merged = add_cardinality_features(merged)

    # ---- file_frag_rate column ----
    merged["file_frag_rate"] = file_frag_rate

    # ---- Drop identifier columns ----
    merged.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

    # ---- SPLT expansion ----
    merged = expand_all_splt(merged)

    # ---- Encode application columns ----
    if "application_name" in merged.columns:
        merged["application_name"] = merged["application_name"].map(app_name_vocab).fillna(0).astype(int)
    if "application_category_name" in merged.columns:
        merged["application_category_name"] = merged["application_category_name"].map(app_category_vocab).fillna(0).astype(int)

    # ---- inf/NaN -> 0 ----
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.fillna(0, inplace=True)

    # ---- Build feature matrix in exact training column order ----
    missing = [c for c in feature_order if c not in merged.columns]
    if missing and verbose:
        print(f"  WARNING: {len(missing)} features missing from PCAP — filled with 0")

    X = pd.DataFrame(0.0, index=merged.index, columns=feature_order)
    for col in feature_order:
        if col in merged.columns:
            X[col] = merged[col].values

    # ---- Scale ----
    X[scale_cols] = scaler.transform(X[scale_cols])

    if verbose:
        print(f"  Feature matrix: {X.shape}")

    return X, dpkt_df