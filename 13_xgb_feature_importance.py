"""
==========================================================
SCRIPT 13 : XGBOOST FEATURE IMPORTANCE (STANDALONE)
==========================================================

PURPOSE:
Loads the saved XGBoost model and produces a ranked list +
bar chart of feature importances, WITHOUT re-running
training or evaluation.

DIFFERENCE FROM 08_xgboost_model_creation.py:
08_*.py already saves xgb_feature_importance.csv right
after training. This script is for when you want to
re-generate the chart (e.g. with different styling) or
re-check importances later without retraining.

WHAT "IMPORTANCE" MEANS HERE:
model.feature_importances_ for XGBoost defaults to "gain"
based importance — it reflects how much each feature
contributed to reducing the loss function across all
splits in all trees, averaged and normalised to sum to 1.

A feature with high importance is one the model relies on
heavily to separate classes. For this IDS, features like
src2dst_syn_packets and bidirectional_syn_packets ranking
highly makes sense — SYN packet counts are a direct
behavioural signature of SYN-flood style attacks.

HOW TO READ FEATURE NAMES IN THE CHART:
  - splt_ps_N       : packet size of the Nth packet in the
                       flow (N = 0..9)
  - splt_piat_ms_N  : inter-arrival time before the Nth
                       packet (N = 0..9)
  - splt_direction_N: direction of the Nth packet
                       (0 = src->dst, 1 = dst->src)
  - application_name / application_category_name :
                       NFStream's DPI-detected protocol,
                       encoded as integers via
                       BALANCED_ATTACKS/app_name_vocab.json
==========================================================
"""

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

print("Loading XGBoost model...")

model = joblib.load("xgb_model.pkl")

# ----------------------------------------------------------
# LOAD FEATURE NAMES IN TRAINING ORDER
#
# feature_order.pkl (saved by 06_splitting_master_dataset.py)
# is the authoritative source for column order — using it
# here (rather than re-reading train.csv) guarantees the
# importance values line up with the correct feature names
# even if train.csv's column order ever changes.
# ----------------------------------------------------------

feature_names = joblib.load("TRAINING_DATA/feature_order.pkl")

print(f"Loaded {len(feature_names)} feature names")

# ----------------------------------------------------------
# BUILD IMPORTANCE TABLE
# ----------------------------------------------------------

importance = model.feature_importances_

if len(importance) != len(feature_names):
    print(
        f"\n  WARNING: model has {len(importance)} importances "
        f"but feature_order.pkl has {len(feature_names)} names. "
        "These should match — check that the model was trained "
        "on the current feature set."
    )

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

# ----------------------------------------------------------
# PRINT TOP 30
# ----------------------------------------------------------

print("\nTOP 30 FEATURES (by XGBoost gain importance)\n")
print(feature_df.head(30).to_string(index=False))

# ----------------------------------------------------------
# SAVE CSV
# ----------------------------------------------------------

feature_df.to_csv("xgb_feature_importance.csv", index=False)
print("\nSaved -> xgb_feature_importance.csv")

# ----------------------------------------------------------
# TOP 20 BAR CHART
# ----------------------------------------------------------

top20 = feature_df.head(20)

plt.figure(figsize=(12, 8))
plt.barh(top20["Feature"], top20["Importance"])
plt.xlabel("Importance Score (XGBoost gain, normalised)")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Feature Importances")
plt.gca().invert_yaxis()   # highest importance at the top
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
plt.close()

print("Saved -> xgb_feature_importance.png")
print("\nDone.")