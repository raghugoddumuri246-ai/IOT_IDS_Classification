"""
==========================================================
SCRIPT 11 : PREDICT ON A NEW PCAP (NO EXPLAINABILITY)
==========================================================

PURPOSE:
This is the LIGHTWEIGHT version of 12_predict_pcap_xai.py —
it does everything 12_*.py does EXCEPT the SHAP explanation
step at the end.

WHEN TO USE THIS INSTEAD OF 12_*.py:
  - You just want the attack classification quickly,
    without waiting for SHAP to compute (SHAP can take
    significant time for large PCAPs).
  - You're running this on a machine without enough
    memory/CPU for SHAP's tree traversal.
  - You're batch-testing many PCAP files and only care
    about the final distribution, not per-file explanations.

PIPELINE (must mirror 04_create_balanced_attack_files.py
EXACTLY, otherwise the model receives features in a
different format than it was trained on):

  PCAP
   -> NFStreamer (same settings: statistical_analysis,
                   splt_analysis=10, n_dissections=20)
   -> drop DROP_COLUMNS (same list as 04_*.py)
   -> expand_all_splt() (splt_direction/ps/piat_ms
                          string-lists -> 30 numeric columns)
   -> encode application_name / application_category_name
      using the SAME vocab JSON files saved by 04_*.py
   -> reorder columns to feature_order.pkl (saved by
      06_splitting_master_dataset.py)
   -> apply robust_scaler.pkl to scale_cols
   -> model.predict()

USAGE:
   python 11_predict_pcap.py path/to/traffic.pcap
==========================================================
"""

import sys
import ast
import json
import numpy as np
import pandas as pd
import joblib

from nfstream import NFStreamer

# ----------------------------------------------------------
# PCAP PATH
# ----------------------------------------------------------

if len(sys.argv) < 2:
    pcap_file = "Test_Pcap/DoS-UDP_Flood16.pcap"
    print(f"No PCAP argument given — using default: {pcap_file}")
else:
    pcap_file = sys.argv[1]

SPLT_N = 10

# ----------------------------------------------------------
# SPLT EXPANSION
#
# NFStream stores splt_direction / splt_ps / splt_piat_ms
# as STRING representations of 10-element lists, e.g.
#   "[60, 60, 60, 60, 60, 60, 60, 60, 60, 60]"
#
# This must be expanded into 10 separate numeric columns
# per field (30 columns total) — exactly as done in
# 04_create_balanced_attack_files.py — otherwise the
# model will receive strings instead of numbers.
# ----------------------------------------------------------

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
    for col, prefix in [("splt_direction", "splt_direction"),
                         ("splt_ps", "splt_ps"),
                         ("splt_piat_ms", "splt_piat_ms")]:
        if col in df.columns:
            expanded = expand_splt_column(df[col], prefix, SPLT_N)
            df = df.drop(columns=[col])
            df = pd.concat([df, expanded], axis=1)
    return df


# ----------------------------------------------------------
# LOAD ALL TRAINING ARTIFACTS
#
# These files are produced by:
#   - app_*_vocab.json     <- 04_create_balanced_attack_files.py
#   - label_encoder.pkl    <- 06_splitting_master_dataset.py
#   - robust_scaler.pkl    <- 06_splitting_master_dataset.py
#   - scale_cols.pkl       <- 06_splitting_master_dataset.py
#   - feature_order.pkl    <- 06_splitting_master_dataset.py
#   - xgb_model.pkl        <- 08_xgboost_model_creation.py
#
# If ANY of these is missing, re-run the corresponding
# script before using this prediction script.
# ----------------------------------------------------------

print("\nLoading model artifacts...")

model         = joblib.load("xgb_model.pkl")
le            = joblib.load("TRAINING_DATA/label_encoder.pkl")
scaler        = joblib.load("TRAINING_DATA/robust_scaler.pkl")
scale_cols    = joblib.load("TRAINING_DATA/scale_cols.pkl")
feature_order = joblib.load("TRAINING_DATA/feature_order.pkl")

# Force CPU — avoids "mismatched devices" warning and
# possible CUDA OOM seen when running on this machine's GPU.
model.get_booster().set_param({"device": "cpu"})

with open("BALANCED_ATTACKS/app_name_vocab.json") as f:
    app_name_vocab = json.load(f)
with open("BALANCED_ATTACKS/app_category_vocab.json") as f:
    app_category_vocab = json.load(f)

class_names = list(le.classes_)
num_classes = len(class_names)

print(f"  Classes ({num_classes}): {class_names}")


# ----------------------------------------------------------
# COARSE LABEL GROUPING
#
# Groups the 3 Mirai subtypes into "Mirai" and the 3 Recon
# subtypes into "Recon" for the high-level summary —
# matches the grouping used in 09_xgboost_evaluation_10.py
# ----------------------------------------------------------

def get_coarse_label(fine_label):
    if fine_label.startswith("Mirai"):
        return "Mirai"
    if fine_label.startswith("Recon"):
        return "Recon"
    return fine_label


# ----------------------------------------------------------
# RUN NFSTREAM ON THE PCAP
#
# MUST use the exact same settings as 01_extract_flows.py:
#   statistical_analysis=True  -> min/mean/stddev/max for
#                                  packet sizes and inter-
#                                  arrival times
#   splt_analysis=10           -> first 10 packet
#                                  sizes/directions/timings
#   n_dissections=20           -> DPI for application_name /
#                                  application_category_name
# ----------------------------------------------------------

print(f"\nLoading PCAP : {pcap_file}")
print("Extracting flows with NFStream...")

df = NFStreamer(
    source=pcap_file,
    statistical_analysis=True,
    splt_analysis=10,
    n_dissections=20
).to_pandas()

print(f"  Flows extracted : {len(df)}")

if len(df) == 0:
    print("No flows found in this PCAP. Exiting.")
    sys.exit(1)

# ----------------------------------------------------------
# FEATURE PREPARATION
#
# Must mirror 04_create_balanced_attack_files.py exactly:
#   1. Drop identifier / IP / MAC / port / timestamp /
#      payload-metadata columns
#   2. Expand SPLT string-list columns into numeric columns
#   3. Encode application_name / application_category_name
#      to integers using the SAME vocabulary built during
#      training (unseen values map to 0 = "unknown")
#   4. Replace inf -> NaN -> 0
# ----------------------------------------------------------

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

df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

df = expand_all_splt(df)

if "application_name" in df.columns:
    df["application_name"] = (
        df["application_name"].map(app_name_vocab).fillna(0).astype(int)
    )

if "application_category_name" in df.columns:
    df["application_category_name"] = (
        df["application_category_name"].map(app_category_vocab).fillna(0).astype(int)
    )

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ----------------------------------------------------------
# BUILD FEATURE MATRIX IN THE EXACT TRAINING COLUMN ORDER
#
# XGBoost validates feature order strictly — even if the
# SET of columns matches, a different ORDER raises a
# "feature_names mismatch" error. feature_order.pkl was
# saved during 06_splitting_master_dataset.py and captures
# the exact order used during training.
# ----------------------------------------------------------

missing = [c for c in feature_order if c not in df.columns]
if missing:
    print(f"\n  WARNING: {len(missing)} features missing from this PCAP — filled with 0")
    print(f"  Missing: {missing}")

X = pd.DataFrame(0.0, index=df.index, columns=feature_order)
for col in feature_order:
    if col in df.columns:
        X[col] = df[col].values

# Apply the SAME RobustScaler fitted during training,
# only to the columns it was originally fitted on.
X[scale_cols] = scaler.transform(X[scale_cols])

print(f"  Feature matrix : {X.shape}")

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

print("\nPredicting...")

preds      = model.predict(X)
pred_proba = model.predict_proba(X)

fine_labels   = [class_names[int(p)] for p in preds]
coarse_labels = [get_coarse_label(l) for l in fine_labels]
confidences   = [pred_proba[i, int(p)] for i, p in enumerate(preds)]

fine_dist   = pd.Series(fine_labels).value_counts()
coarse_dist = pd.Series(coarse_labels).value_counts()
total       = len(preds)

# ----------------------------------------------------------
# REPORT
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("PCAP ANALYSIS REPORT")
print("=" * 60)
print(f"\n  File  : {pcap_file}")
print(f"  Flows : {total}")

print("\nCoarse distribution (Mirai/Recon subtypes merged):")
for label, count in coarse_dist.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<22}  {count:>6,}  ({pct:5.1f}%)  {bar}")

print("\nFine-grained distribution (all 14 classes):")
for label, count in fine_dist.items():
    pct = (count / total) * 100
    print(f"  {label:<30}  {count:>6,}  ({pct:5.1f}%)")

# ----------------------------------------------------------
# FINAL DECISION
#
# The dominant COARSE class (by flow count) is reported
# as the overall verdict for this PCAP. If it's Mirai or
# Recon, also report the most likely fine-grained subtype.
# ----------------------------------------------------------

top_coarse       = coarse_dist.index[0]
top_coarse_count = coarse_dist.iloc[0]
top_confidence   = (top_coarse_count / total) * 100

mean_flow_conf = np.mean([
    c for l, c in zip(coarse_labels, confidences) if l == top_coarse
]) * 100

print("\n" + "=" * 60)
print("FINAL DECISION")
print("=" * 60)
print(f"\n  Attack class        : {top_coarse}")
print(f"  Flow coverage       : {top_confidence:.1f}%")
print(f"  Mean flow confidence: {mean_flow_conf:.1f}%")

if top_coarse in ("Mirai", "Recon"):
    sub_counts = {k: v for k, v in fine_dist.items() if k.startswith(top_coarse)}
    if sub_counts:
        top_sub = max(sub_counts, key=sub_counts.get)
        sub_pct = (sub_counts[top_sub] / sum(sub_counts.values())) * 100
        print(f"\n  Most likely subtype : {top_sub}")
        print(f"  Subtype confidence  : {sub_pct:.1f}%")

print("=" * 60)
print("\nDone. (Run 12_predict_pcap_xai.py for SHAP-based explanations.)")