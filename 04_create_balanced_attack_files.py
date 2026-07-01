"""
==========================================================
SCRIPT 04 : CREATE BALANCED PER-ATTACK CSV FILES
            (NFStream + DPKT FUSION VERSION)
==========================================================

PIPELINE POSITION:
  01_extract_flows.py (NFStream)        --\
                                            >-- [THIS SCRIPT] --> 05_master_dataset_creation.py
  01b_extract_dpkt_features.py (DPKT)   --/

INPUT  : NFSTREAM_CSV/<attack_name>/*.csv   (raw NFStream output, 89 columns)
         DPKT_CSV/<attack_name>/*.csv       (raw DPKT output, 18 columns incl. flow_key)
OUTPUT : BALANCED_ATTACKS/<attack_name>.csv (TARGET_ROWS rows x ~110 columns each)
         BALANCED_ATTACKS/app_name_vocab.json
         BALANCED_ATTACKS/app_category_vocab.json

WHAT CHANGED FROM THE ORIGINAL (NFStream-only) VERSION:

  1. ATTACKS list expanded from 14 to 24 classes (added 10 new
     DDoS/DoS flood variants + VulnerabilityScan; PingSweep
     excluded due to insufficient raw flow count, see Section
     "EXCLUDED CLASSES" below).

  2. A new merge step (STEP 2c.0) joins each NFStream chunk with
     the corresponding DPKT-extracted CSV on a reconstructed
     "flow_key", BEFORE src_ip/dst_ip/ports are dropped. This
     adds ~17 new "dpkt_*" columns (TTL stats, TCP window stats,
     IP length stats, fragmentation flags, header byte entropy).

  3. TARGET_ROWS is now adaptive per-attack: if an attack has
     fewer raw flows than TARGET_ROWS, ALL available rows are
     used instead of crashing on an impossible sample() call.
     This is reported clearly in the console output so it is
     never a silent surprise.

WHY THE MERGE MUST HAPPEN HERE, BEFORE COLUMN DROPPING:
  The DPKT-extracted features are computed per-flow using a
  flow_key built from (src_ip, dst_ip, src_port, dst_port,
  protocol) plus an idle-timeout-based sub-flow index. NFStream's
  CSV must still have src_ip/dst_ip/ports/protocol AND
  bidirectional_first_seen_ms available to reconstruct the
  IDENTICAL flow_key on its side and join correctly. Once those
  columns are dropped (as in the original script), the join is
  no longer possible. So in this version, the join happens
  FIRST, and only then are src_ip/dst_ip/ports dropped as before.

EXCLUDED CLASSES — Recon-PingSweep:
  Recon-PingSweep had only 1,686 raw flows available (vs. a
  60,000 target), 35x too few to reach the target even with the
  adaptive TARGET_ROWS logic producing a meaningfully-sized
  class. It has been EXCLUDED from this version of the pipeline.
  It can be reintroduced later if more PingSweep PCAPs are
  sourced. This is a documented limitation, not an oversight.
==========================================================
"""

import os
import ast
import math
import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

NFSTREAM_ROOT = "NFSTREAM_CSV"    # input: raw NFStream CSVs per attack
DPKT_ROOT     = "DPKT_CSV"        # input: raw DPKT CSVs per attack (from 01b_*.py)
OUTPUT_DIR    = "BALANCED_ATTACKS"  # output: one balanced CSV per attack

TARGET_ROWS  = 60000    # DESIRED final row count for every attack class
                         # (used as-is when available; reduced automatically
                         # for any attack with fewer raw flows than this)
CHUNK_SIZE   = 100000   # read large CSVs in chunks to limit RAM use
RANDOM_STATE = 42       # fixed seed -> reproducible sampling

# MUST match the IDLE_TIMEOUT_S used in 01b_extract_dpkt_features.py,
# which in turn matches NFStream's idle_timeout default (120s, confirmed
# from NFStream 6.5.3 official documentation).
IDLE_TIMEOUT_MS = 120 * 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# PART A — SPLT COLUMN EXPANSION  (unchanged from original)
# ==========================================================
#
# NFStream's splt_analysis=10 produces THREE columns where
# each cell is a STRING representation of a 10-element list:
#
#   splt_direction  ->  "[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]"
#   splt_ps         ->  "[60, 60, 60, 60, 60, 60, 60, 60, 60, 60]"
#   splt_piat_ms    ->  "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
#
# XGBoost cannot use string columns. Each of these three
# columns must be parsed and expanded into 10 separate
# numeric columns (30 columns total). If a flow has fewer
# than 10 packets, missing values are padded with 0.
#
# SPLT FEATURES ARE KEPT IN THIS VERSION. They solve a
# DIFFERENT problem than the new DPKT features (packet
# SEQUENCE shape vs packet HEADER content) -- additive,
# not substitutive. See project report Section 5.3.
# ==========================================================

SPLT_N = 10


def expand_splt_column(series, prefix, n=SPLT_N):
    """Parse a column of string-lists -> n separate numeric columns."""

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
    """Expand all three SPLT columns (if present) and drop the originals."""
    for col, prefix in [("splt_direction", "splt_direction"),
                         ("splt_ps", "splt_ps"),
                         ("splt_piat_ms", "splt_piat_ms")]:
        if col in df.columns:
            expanded = expand_splt_column(df[col], prefix, SPLT_N)
            df = df.drop(columns=[col])
            df = pd.concat([df, expanded], axis=1)
    return df


# ==========================================================
# PART A2 — NFSTREAM <-> DPKT FLOW KEY RECONSTRUCTION (NEW)
# ==========================================================
#
# Must be IDENTICAL in logic to normalize_flow_key() and the
# idle-timeout sub-flow splitting in 01b_extract_dpkt_features.py.
# If this drifts out of sync with that script, the merge below
# will silently join rows to the WRONG flow, or fail to match
# rows that should have matched.
# ==========================================================

def normalize_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Direction-independent 5-tuple key — same logic as the DPKT script."""
    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)
    if endpoint_a <= endpoint_b:
        return f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_{proto}"
    else:
        return f"{dst_ip}_{dst_port}_{src_ip}_{src_port}_{proto}"


def add_flow_keys(chunk):
    """
    Reconstructs the same 'flow_key' values 01b_extract_dpkt_features.py
    produced, using NFStream's src_ip/dst_ip/ports/protocol and
    bidirectional_first_seen_ms (for idle-timeout sub-flow splitting).

    Returns the chunk with a new 'flow_key' column added. Does NOT
    drop src_ip/dst_ip/ports here -- that still happens later in the
    normal DROP_COLUMNS step, AFTER this key has been used for the
    DPKT merge.
    """
    required = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                "bidirectional_first_seen_ms"]
    missing = [c for c in required if c not in chunk.columns]
    if missing:
        # Can't build flow_key without these — DPKT merge will be skipped
        # for this chunk (handled by the caller).
        chunk["flow_key"] = None
        return chunk

    base_keys = chunk.apply(
        lambda r: normalize_flow_key(
            str(r["src_ip"]), str(r["dst_ip"]),
            r["src_port"], r["dst_port"], r["protocol"]
        ),
        axis=1
    )

    # Sort by (base_key, start time) to process flows in chronological
    # order per 5-tuple -- required for correct idle-timeout splitting.
    order = chunk.assign(_base_key=base_keys, _orig_idx=chunk.index) \
                 .sort_values(["_base_key", "bidirectional_first_seen_ms"])

    last_seen_ms = {}
    sub_idx = {}
    flow_keys_sorted = []

    for bk, start_ms in zip(order["_base_key"], order["bidirectional_first_seen_ms"]):
        if bk in last_seen_ms and (start_ms - last_seen_ms[bk]) > IDLE_TIMEOUT_MS:
            sub_idx[bk] = sub_idx.get(bk, 0) + 1
        elif bk not in sub_idx:
            sub_idx[bk] = 0
        last_seen_ms[bk] = start_ms
        flow_keys_sorted.append(f"{bk}#{sub_idx[bk]}")

    # Map back to original row order
    flow_key_series = pd.Series(flow_keys_sorted, index=order["_orig_idx"].values)
    chunk = chunk.copy()
    chunk["flow_key"] = chunk.index.map(flow_key_series)
    return chunk


# ==========================================================
# PART A3 — DERIVED CARDINALITY FEATURES (NEW)
# ==========================================================
#
# These features answer questions that NEITHER NFStream NOR DPKT
# can answer at the single-packet or single-flow level, because
# they require observing PATTERNS ACROSS MANY FLOWS simultaneously.
#
# The three problems they solve:
#
# 1. DoS vs DDoS (same flood type, different attacker count)
#    - DoS-UDP_Flood:  1 source IP floods the target
#    - DDoS-UDP_Flood: hundreds of source IPs flood the target
#    FAN-IN (distinct src_ip per dst_ip per time window) is the
#    direct, exact signal for this distinction.
#    DoS -> fan_in = 1 (always)
#    DDoS -> fan_in = tens to hundreds
#
# 2. Recon-PortScan vs Recon-HostDiscovery (cross-target pattern)
#    - PortScan:       1 scanner -> many ports on 1 target
#    - HostDiscovery:  1 scanner -> 1 port on many targets
#    FAN-OUT-PORTS (distinct dst_port per src_ip per window) is
#    high for PortScan, low for HostDiscovery.
#    FAN-OUT-IPS (distinct dst_ip per src_ip per window) is
#    low for PortScan, high for HostDiscovery.
#
# 3. VulnerabilityScan (broad sweep: many ports AND many IPs)
#    Both fan_out_port_count AND fan_out_ip_count will be high,
#    distinguishing it from focused scans.
#
# IMPLEMENTATION NOTE — window size choice:
#   WINDOW_MS = 5000 (5 seconds) is a reasonable default for flood
#   and scan attacks, which operate at high rates. A smaller window
#   (1s) would split slow attacks across too many buckets, producing
#   fan_in=1 even for real DDoS. A larger window (30s) would merge
#   separate burst events together. 5s balances sensitivity vs noise.
#   This is computed PER FILE (per PCAP), which is correct — DoS and
#   DDoS flows are never mixed in the same source PCAP file, so
#   computing cardinality within a file always reflects true single-
#   attack behaviour, not a cross-attack mixture.
#
# IMPORTANT: src_ip, dst_ip, dst_port must still be present in the
#   chunk when this function is called. This runs BEFORE DROP_COLUMNS
#   removes them, immediately after the DPKT merge step.
# ==========================================================

CARDINALITY_WINDOW_MS = 5000   # 5-second time bucket for cardinality counts

CARDINALITY_COLUMNS = [
    "fan_in_src_count",        # distinct src_ip -> this dst_ip in window (DoS vs DDoS)
    "fan_out_port_count",      # distinct dst_port this src_ip contacted in window (PortScan)
    "fan_out_ip_count",        # distinct dst_ip this src_ip contacted in window (HostDiscovery)
    "fan_out_proto_count",     # distinct protocols this src_ip used in window (OSScan)
]


def add_cardinality_features(chunk):
    """
    Adds fan-in and fan-out cardinality columns to the chunk.
    Must be called BEFORE DROP_COLUMNS removes src_ip/dst_ip/dst_port.
    Returns chunk with CARDINALITY_COLUMNS added.

    If any required column is missing (defensive handling),
    all cardinality columns are filled with 0.
    """
    required = ["src_ip", "dst_ip", "dst_port", "protocol",
                "bidirectional_first_seen_ms"]
    missing = [c for c in required if c not in chunk.columns]
    if missing:
        for col in CARDINALITY_COLUMNS:
            chunk[col] = 0
        return chunk

    chunk = chunk.copy()
    chunk["_time_bucket"] = (
        chunk["bidirectional_first_seen_ms"] // CARDINALITY_WINDOW_MS
    ).astype(int)

    # Fan-in: how many distinct src_ip hit this dst_ip in this time bucket?
    # -> HIGH for DDoS (many attackers), LOW (=1) for DoS (one attacker)
    chunk["fan_in_src_count"] = chunk.groupby(
        ["dst_ip", "_time_bucket"]
    )["src_ip"].transform("nunique")

    # Fan-out ports: how many distinct dst_port did this src_ip contact?
    # -> HIGH for PortScan, LOW for HostDiscovery (one port, many IPs)
    chunk["fan_out_port_count"] = chunk.groupby(
        ["src_ip", "_time_bucket"]
    )["dst_port"].transform("nunique")

    # Fan-out IPs: how many distinct dst_ip did this src_ip contact?
    # -> HIGH for HostDiscovery and VulnerabilityScan, LOW for PortScan
    chunk["fan_out_ip_count"] = chunk.groupby(
        ["src_ip", "_time_bucket"]
    )["dst_ip"].transform("nunique")

    # Fan-out protocols: how many distinct protocols did this src_ip use?
    # -> HIGH for OSScan (nmap mixes TCP/ICMP/UDP probes), LOW for pure floods
    chunk["fan_out_proto_count"] = chunk.groupby(
        ["src_ip", "_time_bucket"]
    )["protocol"].transform("nunique")

    # fan_out_scope: product of distinct ports × distinct IPs contacted
    # WHY: VulnerabilityScan hits MANY ports AND MANY IPs -> high product
    #      PortScan: many ports, 1 IP -> low product (ports × 1)
    #      OSScan: few ports, 1 IP -> very low product
    #      Data-verified: this interaction is hard for XGBoost to learn
    #      from two separate columns but trivial from one explicit product.
    chunk["fan_out_scope"] = (
        chunk["fan_out_port_count"] * chunk["fan_out_ip_count"]
    )

    # fan_app_diversity: distinct application_name values used by this
    # src_ip in this time window (before vocab encoding, it's a string;
    # after encoding, nunique on integers still gives the right count).
    # WHY: Benign IoT traffic uses DNS + NTP + HTTP + MQTT + mDNS etc.
    #      Attack traffic repeats one application type obsessively.
    #      High diversity -> strong benign signal.
    if "application_name" in chunk.columns:
        chunk["fan_app_diversity"] = chunk.groupby(
            ["src_ip", "_time_bucket"]
        )["application_name"].transform("nunique")
    else:
        chunk["fan_app_diversity"] = 0

    chunk.drop(columns=["_time_bucket"], inplace=True)

    return chunk
DPKT_FEATURE_COLUMNS = [
    "dpkt_packet_count",
    "dpkt_ttl_min", "dpkt_ttl_mean", "dpkt_ttl_std", "dpkt_ttl_max",
    "dpkt_tcp_window_min", "dpkt_tcp_window_mean", "dpkt_tcp_window_std", "dpkt_tcp_window_max",
    "dpkt_ip_total_len_min", "dpkt_ip_total_len_mean", "dpkt_ip_total_len_std", "dpkt_ip_total_len_max",
    "dpkt_frag_mf_count", "dpkt_frag_df_count", "dpkt_frag_offset_nonzero_count",
    "dpkt_ip_options_present_count", "dpkt_header_byte_entropy",
    # GRE encapsulation (Mirai-greeth vs Mirai-greip)
    "dpkt_gre_packet_count", "dpkt_gre_inner_proto_ip_count", "dpkt_gre_inner_proto_ether_count",
    # GRE ratio features (NEW — data-verified to separate Mirai subtypes)
    "dpkt_gre_ratio",            # 0.0 for udpplain, ~1.0 for GRE attack flows
    "dpkt_gre_inner_ip_ratio",   # ~1.0 for greip, ~0.0 for greeth
    "dpkt_gre_inner_ether_ratio",# ~1.0 for greeth, ~0.0 for greip
    # Fragmentation ratio (NEW — normalizes count by flow size)
    "dpkt_frag_ratio",           # 0.0 for pure floods, >0 for fragmented flows
    # ICMP type breakdown (Recon-HostDiscovery vs DDoS-ICMP_Flood)
    "dpkt_icmp_echo_request_count", "dpkt_icmp_echo_reply_count", "dpkt_icmp_other_count",
    # TCP scan-pattern flag combinations (Recon-OSScan / Recon-PortScan)
    "dpkt_tcp_null_scan_count", "dpkt_tcp_fin_scan_count", "dpkt_tcp_xmas_scan_count",
]


def load_dpkt_features_for_file(attack, pcap_stem):
    """
    Loads the DPKT CSV corresponding to one NFStream CSV file
    (same attack, same pcap filename stem). Returns an empty
    DataFrame with the right columns if the DPKT file is missing,
    so the merge degrades gracefully (DPKT columns filled with 0)
    rather than crashing the whole pipeline.
    """
    dpkt_path = os.path.join(DPKT_ROOT, attack, pcap_stem + ".csv")
    if not os.path.exists(dpkt_path):
        print(f"    WARNING: no DPKT file found at {dpkt_path} — "
              f"DPKT columns will be 0 for this file's rows")
        return pd.DataFrame(columns=["flow_key"] + DPKT_FEATURE_COLUMNS)
    return pd.read_csv(dpkt_path)


def merge_dpkt_into_chunk(chunk, dpkt_df, match_stats):
    """
    Left-joins DPKT features onto an NFStream chunk using flow_key.
    Unmatched NFStream rows (flow present in NFStream but not found
    in the DPKT extraction, or flow_key could not be built) get 0
    for all DPKT columns rather than NaN, consistent with how the
    rest of this script treats missing values.

    match_stats is a dict this function updates in-place with
    'total' and 'matched' counts, so the caller can print an
    overall match-rate summary per attack.
    """
    if "flow_key" not in chunk.columns or chunk["flow_key"].isnull().all():
        for col in DPKT_FEATURE_COLUMNS:
            chunk[col] = 0.0
        match_stats["total"] += len(chunk)
        return chunk.drop(columns=["flow_key"], errors="ignore")

    merged = chunk.merge(dpkt_df, on="flow_key", how="left", suffixes=("", "_dpkt"))

    matched_mask = merged[DPKT_FEATURE_COLUMNS[0]].notna() if DPKT_FEATURE_COLUMNS[0] in merged.columns else pd.Series([False]*len(merged))
    match_stats["total"]   += len(merged)
    match_stats["matched"] += int(matched_mask.sum())

    for col in DPKT_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    return merged.drop(columns=["flow_key"], errors="ignore")


# ==========================================================
# PART B — COLUMNS TO DROP  (unchanged from original)
# ==========================================================

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


# ==========================================================
# PART C — THE 24 ATTACK CLASSES (EXPANDED FROM 14)
# ==========================================================
# Recon-PingSweep intentionally excluded (see docstring above).
# Each folder name here MUST match a subfolder under both
# NFSTREAM_CSV/ (from 01_extract_flows.py) AND DPKT_CSV/
# (from 01b_extract_dpkt_features.py).
# ==========================================================

ATTACKS = [
    # Original 14
    "Benign_Final",
    "DDoS-ICMP_Flood",
    "DDoS-SYN_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "Recon-HostDiscovery",
    "Recon-PortScan",
    "Recon-OSScan",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    # New (10)
    "DDoS-ACK_Fragmentation",
    "DDoS-ICMP_Fragmentation",
    "DDoS-UDP_Fragmentation",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-HTTP_Flood",
    "DoS-HTTP_Flood",
    "VulnerabilityScan",
]


# ==========================================================
# STEP 1 : BUILD GLOBAL CATEGORICAL VOCABULARIES  (unchanged)
# ==========================================================

print("=" * 70)
print("STEP 1 : Building global categorical vocabularies")
print("=" * 70)

app_name_vals     = set()
app_category_vals = set()

for attack in ATTACKS:
    attack_path = os.path.join(NFSTREAM_ROOT, attack)
    csv_files   = sorted([f for f in os.listdir(attack_path) if f.endswith(".csv")])

    for file in csv_files:
        path = os.path.join(attack_path, file)
        try:
            sample = pd.read_csv(
                path,
                usecols=["application_name", "application_category_name"],
                low_memory=False
            )
            app_name_vals.update(sample["application_name"].dropna().unique())
            app_category_vals.update(sample["application_category_name"].dropna().unique())
        except (ValueError, KeyError):
            pass

app_name_vocab     = {v: i + 1 for i, v in enumerate(sorted(app_name_vals))}
app_category_vocab = {v: i + 1 for i, v in enumerate(sorted(app_category_vals))}

print(f"  application_name unique values     : {len(app_name_vocab)}")
print(f"  application_category unique values : {len(app_category_vocab)}")

with open(os.path.join(OUTPUT_DIR, "app_name_vocab.json"), "w") as f:
    json.dump(app_name_vocab, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "app_category_vocab.json"), "w") as f:
    json.dump(app_category_vocab, f, indent=2)

print(f"  Vocabularies saved to {OUTPUT_DIR}/\n")


def count_rows(csv_path):
    """Fast row count (excluding header) without loading the full file."""
    return sum(1 for _ in open(csv_path, encoding="utf-8", errors="ignore")) - 1


# ==========================================================
# STEP 2 : PROCESS EACH ATTACK CATEGORY
# ==========================================================

first_attack_done = False
summary_rows = []  # for the final per-attack summary table

for attack in ATTACKS:

    print("\n" + "=" * 70)
    print(f"ATTACK : {attack}")
    print("=" * 70)

    attack_path = os.path.join(NFSTREAM_ROOT, attack)
    csv_files   = sorted([f for f in os.listdir(attack_path) if f.endswith(".csv")])

    # ---- 2a. Count rows in every file ----
    file_rows         = {}
    attack_total_rows = 0

    for file in csv_files:
        rows = count_rows(os.path.join(attack_path, file))
        file_rows[file]    = rows
        attack_total_rows += rows

    print(f"  Total rows available : {attack_total_rows:,}")

    # ---- ADAPTIVE TARGET_ROWS ----
    # If this attack has fewer raw flows than TARGET_ROWS, use ALL
    # available rows instead of crashing on an impossible sample().
    # This is reported clearly so it's never a silent surprise.
    if attack_total_rows < TARGET_ROWS:
        effective_target = attack_total_rows
        print(f"  WARNING: only {attack_total_rows:,} rows available "
              f"(< target {TARGET_ROWS:,}). Using ALL available rows "
              f"for this class instead of the usual target.")
    else:
        effective_target = TARGET_ROWS

    # ---- 2b. Proportional allocation per file ----
    allocations     = {}
    allocated_total = 0

    for file in csv_files:
        alloc = round((file_rows[file] / attack_total_rows) * effective_target)
        allocations[file]  = alloc
        allocated_total   += alloc

    allocations[csv_files[0]] += (effective_target - allocated_total)

    attack_parts = []
    dpkt_match_stats = {"total": 0, "matched": 0}

    # ---- 2c. Process each file in chunks ----
    for file in csv_files:

        path   = os.path.join(attack_path, file)
        target = allocations[file]
        pcap_stem = file[:-4]  # strip ".csv" to get the original pcap filename stem

        print(f"  {file:<40}  rows={file_rows[file]:>10,}  alloc={target:>8,}")

        # Load this file's corresponding DPKT-extracted features ONCE
        # (not per-chunk) — DPKT CSVs are one-row-per-flow and much
        # smaller than the raw packet-level data, so this is cheap.
        dpkt_df = load_dpkt_features_for_file(attack, pcap_stem)

        # ---- Compute file_frag_rate BEFORE chunk processing ----
        #
        # WHY THIS FEATURE: DDoS-ICMP_Flood vs DDoS-ICMP_Fragmentation
        # is the hardest class pair. Both have ICMP traffic. The only
        # difference is that in a Fragmentation attack PCAP, a HIGH
        # FRACTION of all flows carry IP fragmentation flags.
        #
        # A per-flow dpkt_frag_ratio tells us about THIS flow, but an
        # unfragmented ICMP flow within a fragmentation attack PCAP
        # still has frag_ratio=0, identical to a plain flood flow.
        #
        # file_frag_rate is the fraction of ALL flows in THIS PCAP file
        # that have at least one fragmented packet (dpkt_frag_mf_count > 0).
        # This is a FILE-LEVEL context feature assigned identically to
        # every row from this file — it tells the model "what kind of
        # attack ENVIRONMENT did this flow come from?"
        #
        # DDoS-ICMP_Fragmentation PCAP: file_frag_rate ~ 0.8-1.0
        # DDoS-ICMP_Flood PCAP:         file_frag_rate ~ 0.0
        # DDoS-UDP_Fragmentation PCAP:  file_frag_rate ~ 0.8-1.0
        # DDoS-UDP_Flood PCAP:          file_frag_rate ~ 0.0
        #
        # At inference time (new PCAP), file_frag_rate is computed from
        # the incoming PCAP's own DPKT-extracted flows before prediction.
        if "dpkt_frag_mf_count" in dpkt_df.columns and len(dpkt_df) > 0:
            frag_flows = (dpkt_df["dpkt_frag_mf_count"] > 0).sum()
            file_frag_rate = float(frag_flows) / len(dpkt_df)
        else:
            file_frag_rate = 0.0

        sampled_chunks = []

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):

            # 0. NEW: build flow_key and merge DPKT features in,
            #    BEFORE src_ip/dst_ip/ports are dropped below.
            chunk = add_flow_keys(chunk)
            chunk = merge_dpkt_into_chunk(chunk, dpkt_df, dpkt_match_stats)

            # 0b. NEW: compute fan-in / fan-out cardinality features
            #     BEFORE dropping src_ip/dst_ip/dst_port below.
            chunk = add_cardinality_features(chunk)

            # 0c. NEW: assign file-level fragmentation rate
            #     Same value for every row in this file — it reflects
            #     what fraction of this PCAP's flows are fragmented,
            #     giving context that per-flow features cannot provide.
            chunk["file_frag_rate"] = file_frag_rate

            # 1. Drop identifier/environment/payload columns
            chunk.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

            # 2. Expand splt_* string columns into 30 numeric columns
            chunk = expand_all_splt(chunk)

            # 3. Encode application_name / application_category_name
            if "application_name" in chunk.columns:
                chunk["application_name"] = (
                    chunk["application_name"].map(app_name_vocab).fillna(0).astype(int)
                )
            if "application_category_name" in chunk.columns:
                chunk["application_category_name"] = (
                    chunk["application_category_name"].map(app_category_vocab).fillna(0).astype(int)
                )

            # 4. Replace inf -> NaN -> 0
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.fillna(0, inplace=True)

            # 5. Randomly sample this chunk's share of the target
            frac     = target / file_rows[file]
            sample_n = min(math.ceil(len(chunk) * frac), len(chunk))
            sampled_chunks.append(chunk.sample(n=sample_n, random_state=RANDOM_STATE))

        file_df = pd.concat(sampled_chunks, ignore_index=True)

        if len(file_df) > target:
            file_df = file_df.sample(n=target, random_state=RANDOM_STATE)

        attack_parts.append(file_df)

    # ---- Report DPKT match rate for this attack ----
    if dpkt_match_stats["total"] > 0:
        match_pct = 100.0 * dpkt_match_stats["matched"] / dpkt_match_stats["total"]
        print(f"\n  DPKT match rate : {dpkt_match_stats['matched']:,} / "
              f"{dpkt_match_stats['total']:,} flows matched ({match_pct:.1f}%)")
    else:
        match_pct = 0.0

    # ---- 2d. Combine, trim to exactly effective_target, label, save ----
    attack_df = pd.concat(attack_parts, ignore_index=True)

    if len(attack_df) > effective_target:
        attack_df = attack_df.sample(n=effective_target, random_state=RANDOM_STATE)

    nan_count = attack_df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN remain — filling with 0")
        attack_df.fillna(0, inplace=True)

    attack_df["label"] = attack

    save_path = os.path.join(OUTPUT_DIR, attack + ".csv")
    attack_df.to_csv(save_path, index=False)

    print(f"\n  Final shape : {attack_df.shape}")
    print(f"  Saved       : {save_path}")

    summary_rows.append({
        "attack": attack,
        "raw_rows": attack_total_rows,
        "final_rows": len(attack_df),
        "dpkt_match_pct": round(match_pct, 1),
    })

    if not first_attack_done:
        print(f"\n  Final columns ({len(attack_df.columns)}):")
        for col in attack_df.columns:
            print(f"    {col}")
        first_attack_done = True

# ==========================================================
# FINAL SUMMARY — sanity-check this before moving to Script 05
# ==========================================================

print("\n\n" + "=" * 70)
print("SUMMARY — per-attack row counts and DPKT match rates")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

low_match = summary_df[summary_df["dpkt_match_pct"] < 80]
if len(low_match) > 0:
    print("\n  WARNING: the following attacks had a DPKT match rate below 80%.")
    print("  This usually means the DPKT extraction (01b_*.py) was not yet")
    print("  run for that attack, or the idle_timeout/flow_key logic has")
    print("  drifted out of sync between the two scripts. DPKT columns for")
    print("  unmatched rows are 0, which is safe but reduces their value.")
    print(low_match.to_string(index=False))

print(f"\nDONE — all {len(ATTACKS)} balanced attack files saved to {OUTPUT_DIR}/")
