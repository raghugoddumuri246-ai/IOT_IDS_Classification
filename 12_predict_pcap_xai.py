"""
==========================================================
SCRIPT 12 : PREDICT ON A NEW PCAP + EXPLAIN WITH SHAP (XAI)
==========================================================

PIPELINE POSITION:
  08_xgboost_model_creation.py  ->  [THIS SCRIPT]   (end of pipeline)

INPUT  : Any .pcap file (e.g. Test_Pcap/BenignTraffic3.pcap)
         + all artifacts saved by 04_*.py, 06_*.py, 08_*.py:
           xgb_model.pkl
           TRAINING_DATA/label_encoder.pkl
           TRAINING_DATA/robust_scaler.pkl
           TRAINING_DATA/scale_cols.pkl
           TRAINING_DATA/feature_order.pkl
           BALANCED_ATTACKS/app_name_vocab.json
           BALANCED_ATTACKS/app_category_vocab.json

OUTPUT : Console report (class distribution + final decision)
         pcap_explanation.csv   <- ranked SHAP feature importances
         pcap_explanation.png   <- bar chart of top 15 features

USAGE:
   python 12_predict_pcap_xai.py path/to/traffic.pcap


WHAT THIS SCRIPT DOES (in order):

  PART 1 — LOAD ARTIFACTS
    Load the trained model + every preprocessing artifact
    saved during training, so the new PCAP can be processed
    in EXACTLY the same way as the training data was.

  PART 2 — EXTRACT FLOWS FROM THE PCAP
    Run NFStream with the SAME settings as 01_extract_flows.py
    (statistical_analysis, splt_analysis=10, n_dissections=20).

  PART 3 — REPLICATE THE 04_*.py FEATURE PIPELINE
    Drop the same columns, expand splt_* string columns,
    encode application_name/category using the SAVED
    vocabularies (not re-built — must match training!),
    fill inf/NaN with 0.

  PART 4 — BUILD FEATURE MATRIX + SCALE
    Reorder columns to feature_order.pkl, apply the SAVED
    RobustScaler to scale_cols.

  PART 5 — PREDICT
    model.predict() for class labels,
    model.predict_proba() for per-flow confidence.
    Mirai/Recon subtypes are grouped into coarse "Mirai" /
    "Recon" categories for the headline decision.

  PART 6 — REPORT
    Print coarse + fine-grained distributions and the
    final overall decision for this PCAP.

  PART 7 — SHAP EXPLAINABILITY (XAI)
    For the flows predicted as the DOMINANT class, compute
    SHAP values to show WHICH features drove that decision.
    Saves a ranked CSV and a bar chart PNG.
==========================================================
"""

import sys
import ast
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nfstream import NFStreamer

# ----------------------------------------------------------
# PCAP PATH
# ----------------------------------------------------------

if len(sys.argv) < 2:
    pcap_file = "Test_Pcap/BenignTraffic3.pcap"
    print(f"No PCAP argument given — using default: {pcap_file}")
else:
    pcap_file = sys.argv[1]

SPLT_N = 10


# ==========================================================
# PART 1a — SPLT EXPANSION HELPER
# ==========================================================
#
# MUST be IDENTICAL to the version in
# 04_create_balanced_attack_files.py — this is what turns
# the string-encoded splt_direction / splt_ps / splt_piat_ms
# columns into 30 numeric columns the model can use.
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
    for col, prefix in [("splt_direction", "splt_direction"),
                         ("splt_ps", "splt_ps"),
                         ("splt_piat_ms", "splt_piat_ms")]:
        if col in df.columns:
            expanded = expand_splt_column(df[col], prefix, SPLT_N)
            df = df.drop(columns=[col])
            df = pd.concat([df, expanded], axis=1)
    return df


# ==========================================================
# PART 1b — LOAD ALL TRAINING ARTIFACTS
# ==========================================================
#
# Every one of these files was produced by an earlier
# pipeline step and is REQUIRED for inference to match
# training exactly. See the artifact reference doc for
# full details on each file.
#
#   xgb_model.pkl              <- 08_xgboost_model_creation.py
#   label_encoder.pkl          <- 06_splitting_master_dataset.py
#   robust_scaler.pkl          <- 06_splitting_master_dataset.py
#   scale_cols.pkl             <- 06_splitting_master_dataset.py
#   feature_order.pkl          <- 06_splitting_master_dataset.py
#   app_name_vocab.json        <- 04_create_balanced_attack_files.py
#   app_category_vocab.json    <- 04_create_balanced_attack_files.py
# ==========================================================

print("\nLoading model artifacts...")

model         = joblib.load("xgb_model.pkl")
le            = joblib.load("TRAINING_DATA/label_encoder.pkl")
scaler        = joblib.load("TRAINING_DATA/robust_scaler.pkl")
scale_cols    = joblib.load("TRAINING_DATA/scale_cols.pkl")
feature_order = joblib.load("TRAINING_DATA/feature_order.pkl")

# ----------------------------------------------------------
# Force CPU for inference.
#
# The model was trained with device="cuda", but on this
# machine the GPU may have very little free memory (we
# observed ~27MB free during SHAP). Inference on a single
# PCAP (tens of thousands of flows) is fast enough on CPU
# and avoids both the "mismatched devices" warning and
# potential CUDA out-of-memory crashes.
# ----------------------------------------------------------

model.get_booster().set_param({"device": "cpu"})

with open("BALANCED_ATTACKS/app_name_vocab.json") as f:
    app_name_vocab = json.load(f)
with open("BALANCED_ATTACKS/app_category_vocab.json") as f:
    app_category_vocab = json.load(f)

class_names = list(le.classes_)
num_classes = len(class_names)

print(f"  Classes ({num_classes}): {class_names}")


def get_coarse_label(fine_label):
    """Group Mirai/Recon subtypes into one coarse label each."""
    if fine_label.startswith("Mirai"):
        return "Mirai"
    if fine_label.startswith("Recon"):
        return "Recon"
    return fine_label


# ==========================================================
# PART 2 — RUN NFSTREAM ON THE PCAP
# ==========================================================
#
# MUST use the exact same NFStreamer settings as
# 01_extract_flows.py — otherwise the raw columns produced
# (and therefore the features available) will differ from
# what the model was trained on.
# ==========================================================

print(f"\nLoading PCAP : {pcap_file}")
print("Extracting flows...")

df = NFStreamer(
    source=pcap_file,
    statistical_analysis=True,
    splt_analysis=10,
    n_dissections=20
).to_pandas()

print(f"  Flows extracted : {len(df)}")

if len(df) == 0:
    print("No flows found. Exiting.")
    sys.exit(1)


# ==========================================================
# PART 3 — REPLICATE THE 04_*.py FEATURE PIPELINE
# ==========================================================
#
# This block must produce the SAME columns, in the SAME
# transformed form, as 04_create_balanced_attack_files.py
# did for the training data.
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

# 3a. Drop identifier / environment / payload columns
df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

# 3b. Expand splt_* string-list columns into 30 numeric columns
df = expand_all_splt(df)

# 3c. Encode application_name / application_category_name using
#     the SAVED vocab from training (NOT rebuilt here — any
#     value not seen during training maps to 0 = "unknown")
if "application_name" in df.columns:
    df["application_name"] = (
        df["application_name"].map(app_name_vocab).fillna(0).astype(int)
    )
if "application_category_name" in df.columns:
    df["application_category_name"] = (
        df["application_category_name"].map(app_category_vocab).fillna(0).astype(int)
    )

# 3d. Replace inf -> NaN -> 0 (same as training)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# ==========================================================
# PART 4 — BUILD FEATURE MATRIX IN TRAINING COLUMN ORDER + SCALE
# ==========================================================
#
# feature_order.pkl guarantees the columns are in the EXACT
# same order XGBoost was trained on (order matters, not just
# the set of names — see earlier debugging notes).
#
# If this PCAP is missing a feature the training data had
# (e.g. NFStream produced fewer splt values for very short
# flows), that column is filled with 0.
# ==========================================================

missing = [c for c in feature_order if c not in df.columns]

if missing:
    print(f"\n  WARNING: {len(missing)} features missing — filled with 0")
    print(f"  Missing: {missing}")

X = pd.DataFrame(0.0, index=df.index, columns=feature_order)
for col in feature_order:
    if col in df.columns:
        X[col] = df[col].values

# Apply the SAME RobustScaler fitted on training data
X[scale_cols] = scaler.transform(X[scale_cols])

print(f"  Feature matrix : {X.shape}")


# ==========================================================
# PART 5 — PREDICTION
# ==========================================================

print("\nPredicting...")

preds      = model.predict(X)           # class index per flow (0-13)
pred_proba = model.predict_proba(X)      # per-class probability per flow

fine_labels   = [class_names[int(p)] for p in preds]
coarse_labels = [get_coarse_label(l) for l in fine_labels]
confidences   = [pred_proba[i, int(p)] for i, p in enumerate(preds)]

fine_dist   = pd.Series(fine_labels).value_counts()
coarse_dist = pd.Series(coarse_labels).value_counts()
total       = len(preds)


# ==========================================================
# PART 6 — REPORT
# ==========================================================

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
# The dominant coarse class (most flows) is reported as the
# overall verdict for the whole PCAP. "Flow coverage" is
# what fraction of flows fall into this class. "Mean flow
# confidence" is the average predict_proba value for those
# flows — a rough measure of how sure the model was.
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


# ==========================================================
# PART 7 — SHAP EXPLAINABILITY (XAI)
# ==========================================================
#
# GOAL: for the flows predicted as the dominant class, show
# WHICH features pushed the model toward that decision.
#
# Multiclass XGBoost technical note:
#   total boosted trees = n_estimators * num_class
#   (500 * 14 = 7000), but model.best_iteration is stored
#   in "rounds" (e.g. 499), not "total trees". SHAP's
#   TreeExplainer reads best_iteration to determine which
#   trees to use — if this value is stale/inconsistent it
#   can raise a "Check failed: end <= BoostedRounds()" error.
#   We explicitly recompute and re-set it here to avoid that.
# ==========================================================

print("\n\nRunning SHAP analysis...")

# Model is already on CPU (set in Part 1b). TreeExplainer
# walks the tree structure directly and works fine on CPU.

booster = model.get_booster()

total_trees = int(booster.num_boosted_rounds())
num_class_  = num_classes

booster.best_iteration = (total_trees // num_class_) - 1
try:
    booster.set_attr(best_iteration=str(booster.best_iteration))
    booster.set_attr(best_ntree_limit=str(total_trees))
except Exception:
    pass

print(f"  Total boosted trees : {total_trees}")
print(f"  Trees per class     : {total_trees // num_class_}")
print(f"  Adjusted best_iteration -> {booster.best_iteration}")

explainer = shap.TreeExplainer(model)

# Select only the flows predicted as the dominant class
dominant_mask = [l == top_coarse for l in coarse_labels]
dominant_X    = X[dominant_mask].copy()

# Cap at 1500 flows for speed — SHAP on tree models scales
# with the number of samples x number of trees.
MAX_SAMPLE = 1500
if len(dominant_X) > MAX_SAMPLE:
    dominant_X = dominant_X.sample(n=MAX_SAMPLE, random_state=42)

print(f"  Computing SHAP on {len(dominant_X)} flows...")

try:
    shap_values = explainer.shap_values(dominant_X)
except Exception as e:
    # The additivity check can fail by a small margin on some
    # XGBoost/SHAP version combinations — this is usually safe
    # to ignore for explanation purposes.
    print(f"  Primary SHAP call failed ({type(e).__name__}: {e})")
    print("  Retrying with check_additivity=False ...")
    shap_values = explainer.shap_values(dominant_X, check_additivity=False)

# For multiclass SHAP, shap_values is a list with one array
# per class. If the dominant class is a SINGLE class (e.g.
# "Benign_Final"), use that class's SHAP values directly.
# If it's a GROUPED class (Mirai or Recon, made of 3 fine-
# grained classes), average the SHAP values across all 3.
candidate_ids = [i for i, n in enumerate(class_names) if get_coarse_label(n) == top_coarse]

if len(candidate_ids) == 1:
    mean_shap = np.mean(np.abs(shap_values[candidate_ids[0]]), axis=0)
else:
    stacked   = np.stack([np.abs(shap_values[cid]) for cid in candidate_ids])
    mean_shap = np.mean(stacked, axis=(0, 1))

importance_df = pd.DataFrame({
    "Feature": dominant_X.columns,
    "SHAP_Importance": mean_shap
}).sort_values("SHAP_Importance", ascending=False)

print("\n" + "=" * 60)
print(f"TOP FEATURES — Why classified as {top_coarse}")
print("=" * 60)
for rank, row in enumerate(importance_df.head(15).itertuples(), start=1):
    bar = "█" * int(row.SHAP_Importance * 500)
    print(f"  {rank:>2}. {row.Feature:<35}  {row.SHAP_Importance:.6f}  {bar}")

importance_df.to_csv("pcap_explanation.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 7))
top15 = importance_df.head(15)
ax.barh(top15["Feature"][::-1], top15["SHAP_Importance"][::-1])
ax.set_xlabel("Average |SHAP| value")
ax.set_title(f"Feature contributions — {top_coarse}")
plt.tight_layout()
plt.savefig("pcap_explanation.png", dpi=150)
plt.close()

print("\nSaved -> pcap_explanation.csv")
print("Saved -> pcap_explanation.png")
print("\nAnalysis complete.")