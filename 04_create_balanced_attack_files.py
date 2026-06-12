"""
==========================================================
SCRIPT 04 : CREATE BALANCED PER-ATTACK CSV FILES
==========================================================

PIPELINE POSITION:
  01_extract_flows.py  ->  [THIS SCRIPT]  ->  05_master_dataset_creation.py

INPUT  : NFSTREAM_CSV/<attack_name>/*.csv   (raw NFStream output, 89 columns)
OUTPUT : BALANCED_ATTACKS/<attack_name>.csv (60,000 rows x 93 columns each)
         BALANCED_ATTACKS/app_name_vocab.json
         BALANCED_ATTACKS/app_category_vocab.json

WHAT THIS SCRIPT DOES (in order):

  STEP 1 — Build global vocabularies
    Scan every CSV across every attack ONCE to collect all
    distinct values of application_name and
    application_category_name. Build a consistent
    string -> integer mapping. Saved as JSON so the SAME
    mapping can be reused at inference time.

  STEP 2 — For each attack category:
    a) Count total rows across all CSV files for that attack
    b) Decide how many rows to sample from EACH file so the
       attack ends up with exactly 60,000 rows total
       (proportional sampling — files with more rows
       contribute more)
    c) For each file, process in CHUNKS of 100,000 rows:
         - Drop identifier/IP/MAC/port/timestamp/payload columns
         - Expand splt_direction / splt_ps / splt_piat_ms
           (string-lists) into 30 numeric columns
         - Encode application_name / application_category_name
           to integers using the global vocab from Step 1
         - Replace inf -> NaN -> 0
         - Sample the required number of rows from this chunk
    d) Concatenate all sampled chunks, trim/pad to exactly
       60,000 rows, add a "label" column, save to CSV

WHY 60,000 ROWS PER CLASS?
  The 14 attack classes have WILDLY different amounts of raw
  data (from 64,389 flows for Mirai-udpplain to 17,365,430
  for DDoS-TCP_Flood). Training on the raw imbalanced data
  would bias the model toward the huge classes. Balancing to
  60,000 rows each gives every class equal representation.
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

ROOT_DIR   = "NFSTREAM_CSV"      # input: raw NFStream CSVs per attack
OUTPUT_DIR = "BALANCED_ATTACKS"  # output: one balanced CSV per attack

TARGET_ROWS  = 60000    # final row count for every attack class
CHUNK_SIZE   = 100000   # read large CSVs in chunks to limit RAM use
RANDOM_STATE = 42       # fixed seed -> reproducible sampling

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# PART A — SPLT COLUMN EXPANSION
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
# numeric columns (30 columns total):
#
#   splt_direction_0 .. splt_direction_9   (0 = src->dst, 1 = dst->src)
#   splt_ps_0        .. splt_ps_9          (packet size of Nth packet)
#   splt_piat_ms_0   .. splt_piat_ms_9     (gap before Nth packet, ms)
#
# If a flow has fewer than 10 packets, missing values are
# padded with 0.
# ==========================================================

SPLT_N = 10


def expand_splt_column(series, prefix, n=SPLT_N):
    """Parse a column of string-lists -> n separate numeric columns."""

    def safe_parse(val):
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                # Pad with zeros if the flow had fewer than n packets
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
# PART B — COLUMNS TO DROP
# ==========================================================
#
# KEPT (behavioural, useful for classification):
#   application_name           -> encoded to int (e.g. GRE/ICMP/UDP
#                                  signal helps separate Mirai variants)
#   application_category_name  -> encoded to int
#   application_confidence      -> values 0/1/2/6. ~97% of flows have
#                                  a "guessed" DPI label, so this value
#                                  tells the model how reliable the
#                                  application_name is — real signal.
#
# DROPPED (identifiers / environment-specific / payload-level):
#   id, expiration_id              -> per-row identifiers, no meaning
#   src_ip, dst_ip                 -> changes in every deployment
#   src_mac, dst_mac, src_oui, dst_oui -> hardware-specific
#   vlan_id, tunnel_id              -> always constant in this dataset
#   src_port, dst_port              -> environment dependent, overfits
#   *_first_seen_ms, *_last_seen_ms -> absolute timestamps, meaningless
#   application_is_guessed          -> redundant with application_confidence
#   requested_server_name           -> payload-level SNI/hostname
#   client_fingerprint, server_fingerprint -> TLS fingerprints, env-specific
#   user_agent, content_type        -> ~99.9% NULL in this dataset
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
# PART C — THE 14 ATTACK CLASSES
# ==========================================================
# Each folder name here MUST match a subfolder under
# NFSTREAM_CSV/ (created by 01_extract_flows.py) and the
# output filename under BALANCED_ATTACKS/.
# ==========================================================

ATTACKS = [
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
]


# ==========================================================
# STEP 1 : BUILD GLOBAL CATEGORICAL VOCABULARIES
# ==========================================================
#
# WHY GLOBAL (not per-file or per-attack)?
# application_name values like "ICMP" or "GRE" must map to
# the SAME integer everywhere — in every attack's CSV and
# later during live PCAP inference. If each file built its
# own mapping, "GRE" could be 3 in one file and 7 in another,
# and the model would learn meaningless patterns.
#
# This step scans every CSV ONCE (reading only the two
# application columns, to save memory) and builds one
# shared {string: integer} dictionary per column.
#
# 0 is reserved for "unknown" — any value NOT seen during
# this scan (e.g. a brand-new protocol in a future PCAP)
# will map to 0 at inference time via .fillna(0).
# ==========================================================

print("=" * 70)
print("STEP 1 : Building global categorical vocabularies")
print("=" * 70)

app_name_vals     = set()
app_category_vals = set()

for attack in ATTACKS:
    attack_path = os.path.join(ROOT_DIR, attack)
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
            # Column not present in this file — skip
            pass

# Sort for determinism: re-running this script always
# produces the same {string: int} mapping.
app_name_vocab     = {v: i + 1 for i, v in enumerate(sorted(app_name_vals))}
app_category_vocab = {v: i + 1 for i, v in enumerate(sorted(app_category_vals))}

print(f"  application_name unique values     : {len(app_name_vocab)}")
print(f"  application_category unique values : {len(app_category_vocab)}")

# Save vocabularies — 12_predict_pcap_xai.py and
# 11_predict_pcap.py load these to encode application
# columns consistently for new PCAP files.
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
#
# For each attack:
#   2a. Count total available rows across all its CSV files
#   2b. Allocate a target row count to EACH file,
#       proportional to that file's share of the total
#       (so a file with 30% of the data contributes ~30%
#       of the final 60,000 rows)
#   2c. Read each file in chunks, clean + transform each
#       chunk, then randomly sample the allocated number
#       of rows from it
#   2d. Concatenate all sampled rows, trim/pad to exactly
#       60,000, label, and save
# ==========================================================

first_attack_done = False

for attack in ATTACKS:

    print("\n" + "=" * 70)
    print(f"ATTACK : {attack}")
    print("=" * 70)

    attack_path = os.path.join(ROOT_DIR, attack)
    csv_files   = sorted([f for f in os.listdir(attack_path) if f.endswith(".csv")])

    # ---- 2a. Count rows in every file ----
    file_rows         = {}
    attack_total_rows = 0

    for file in csv_files:
        rows = count_rows(os.path.join(attack_path, file))
        file_rows[file]    = rows
        attack_total_rows += rows

    print(f"  Total rows available : {attack_total_rows:,}")

    # ---- 2b. Proportional allocation per file ----
    allocations     = {}
    allocated_total = 0

    for file in csv_files:
        alloc = round((file_rows[file] / attack_total_rows) * TARGET_ROWS)
        allocations[file]  = alloc
        allocated_total   += alloc

    # Rounding can leave us a few rows short/over TARGET_ROWS.
    # Fix this by adjusting the FIRST file's allocation.
    allocations[csv_files[0]] += (TARGET_ROWS - allocated_total)

    attack_parts = []

    # ---- 2c. Process each file in chunks ----
    for file in csv_files:

        path   = os.path.join(attack_path, file)
        target = allocations[file]

        print(f"  {file:<40}  rows={file_rows[file]:>10,}  alloc={target:>8,}")

        sampled_chunks = []

        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):

            # 1. Drop identifier/environment/payload columns
            chunk.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

            # 2. Expand splt_* string columns into 30 numeric columns
            chunk = expand_all_splt(chunk)

            # 3. Encode application_name / application_category_name
            #    to integers using the global vocab from Step 1
            if "application_name" in chunk.columns:
                chunk["application_name"] = (
                    chunk["application_name"].map(app_name_vocab).fillna(0).astype(int)
                )
            if "application_category_name" in chunk.columns:
                chunk["application_category_name"] = (
                    chunk["application_category_name"].map(app_category_vocab).fillna(0).astype(int)
                )

            # 4. Replace inf -> NaN -> 0
            #    (DDoS flows are often one-directional, so
            #    dst2src_* columns can be NaN/inf for them)
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.fillna(0, inplace=True)

            # 5. Randomly sample this chunk's share of the target
            frac     = target / file_rows[file]
            sample_n = min(math.ceil(len(chunk) * frac), len(chunk))
            sampled_chunks.append(chunk.sample(n=sample_n, random_state=RANDOM_STATE))

        file_df = pd.concat(sampled_chunks, ignore_index=True)

        # Trim if rounding gave us slightly too many rows
        if len(file_df) > target:
            file_df = file_df.sample(n=target, random_state=RANDOM_STATE)

        attack_parts.append(file_df)

    # ---- 2d. Combine, trim to exactly 60,000, label, save ----
    attack_df = pd.concat(attack_parts, ignore_index=True)

    if len(attack_df) > TARGET_ROWS:
        attack_df = attack_df.sample(n=TARGET_ROWS, random_state=RANDOM_STATE)

    # Final safety check — should already be 0 after step 2c.4
    nan_count = attack_df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN remain — filling with 0")
        attack_df.fillna(0, inplace=True)

    attack_df["label"] = attack

    save_path = os.path.join(OUTPUT_DIR, attack + ".csv")
    attack_df.to_csv(save_path, index=False)

    print(f"\n  Final shape : {attack_df.shape}")
    print(f"  Saved       : {save_path}")

    # Print the full column list once, for verification
    if not first_attack_done:
        print(f"\n  Final columns ({len(attack_df.columns)}):")
        for col in attack_df.columns:
            print(f"    {col}")
        first_attack_done = True

print("\n\nDONE")
print(f"All {len(ATTACKS)} balanced attack files saved to {OUTPUT_DIR}/")