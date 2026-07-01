"""
==========================================================
SCRIPT 11 : PREDICT ON A NEW PCAP (NO EXPLAINABILITY)
==========================================================

PIPELINE: NFStream + DPKT + Cardinality + file_frag_rate -> 130 features
USAGE:    python 11_predict_pcap.py path/to/traffic.pcap
==========================================================
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib

from inference_pipeline import build_feature_matrix

def get_coarse_label(fine_label):
    if fine_label.startswith("Mirai"):  return "Mirai"
    if fine_label.startswith("Recon"):  return "Recon"
    return fine_label

CONFIDENCE_THRESHOLD = 35.0

if len(sys.argv) < 2:
    pcap_file = "Test_Pcap/DDoS-SYN_Flood15.pcap"
    print(f"No PCAP argument given — using default: {pcap_file}")
else:
    pcap_file = sys.argv[1]

# ----------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------
print("\nLoading model artifacts...")

model         = joblib.load("xgb_model.pkl")
le            = joblib.load("TRAINING_DATA/label_encoder.pkl")
scaler        = joblib.load("TRAINING_DATA/robust_scaler.pkl")
scale_cols    = joblib.load("TRAINING_DATA/scale_cols.pkl")
feature_order = joblib.load("TRAINING_DATA/feature_order.pkl")
model.get_booster().set_param({"device": "cpu"})

with open("BALANCED_ATTACKS/app_name_vocab.json") as f:
    app_name_vocab = json.load(f)
with open("BALANCED_ATTACKS/app_category_vocab.json") as f:
    app_category_vocab = json.load(f)

class_names = list(le.classes_)
print(f"  Classes ({len(class_names)}): {class_names}")

# ----------------------------------------------------------
# FULL 130-FEATURE PIPELINE
# ----------------------------------------------------------
print(f"\nLoading PCAP : {pcap_file}")

X, dpkt_df = build_feature_matrix(
    pcap_path=pcap_file,
    app_name_vocab=app_name_vocab,
    app_category_vocab=app_category_vocab,
    feature_order=feature_order,
    scale_cols=scale_cols,
    scaler=scaler,
    verbose=True,
)

if X is None:
    print("No flows found in this PCAP. Exiting.")
    sys.exit(1)

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
frag_rate_used = (dpkt_df["dpkt_frag_mf_count"] > 0).sum() / len(dpkt_df) \
                 if dpkt_df is not None and len(dpkt_df) > 0 else 0.0

print("\n" + "=" * 60)
print("PCAP ANALYSIS REPORT")
print("=" * 60)
print(f"\n  File             : {pcap_file}")
print(f"  Flows            : {total}")
print(f"  file_frag_rate   : {frag_rate_used:.4f}  (>0.1 = fragmentation attack context)")

print("\nCoarse distribution:")
for label, count in coarse_dist.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<28}  {count:>6,}  ({pct:5.1f}%)  {bar}")

print("\nFine-grained distribution (all 24 classes):")
for label, count in fine_dist.items():
    pct = (count / total) * 100
    print(f"  {label:<30}  {count:>6,}  ({pct:5.1f}%)")

top_coarse       = coarse_dist.index[0]
top_coarse_count = coarse_dist.iloc[0]
top_confidence   = (top_coarse_count / total) * 100
mean_flow_conf   = np.mean([
    c for l, c in zip(coarse_labels, confidences) if l == top_coarse
]) * 100

print("\n" + "=" * 60)
print("FINAL DECISION")
print("=" * 60)

if mean_flow_conf < CONFIDENCE_THRESHOLD:
    print(f"\n  Attack class        : UNCERTAIN")
    print(f"  Closest match       : {top_coarse}")
    print(f"  Flow coverage       : {top_confidence:.1f}%")
    print(f"  Mean flow confidence: {mean_flow_conf:.1f}%  (below {CONFIDENCE_THRESHOLD}% threshold)")
    print(f"  NOTE: Traffic does not closely match any trained class.")
else:
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
