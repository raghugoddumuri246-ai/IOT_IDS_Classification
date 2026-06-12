"""
==========================================================
SCRIPT 07 : RANDOM FOREST BASELINE + FEATURE IMPORTANCE
==========================================================

PURPOSE:
Random Forest is used here NOT as the final model, but as
a fast, simple baseline to:

  1. Get a quick sense of achievable accuracy before
     spending time tuning XGBoost.

  2. Get an INDEPENDENT feature importance ranking.
     XGBoost and Random Forest compute importance
     differently (Random Forest uses Gini impurity
     decrease, XGBoost uses gain). If a feature ranks
     high in BOTH, that is strong evidence it is genuinely
     useful — not just an artifact of one algorithm's
     splitting strategy.

WHY ONLY 100,000 SAMPLES:
Random Forest with many deep trees on the full 588,000-row
training set would be slow. Since this is a BASELINE/
SANITY-CHECK step (not the final model), a 100k random
sample trains in a couple of minutes and gives a feature
importance ranking that is stable enough for comparison.

OUTPUT:
  rf_feature_importance.csv  -> ranked feature importances
  rf_baseline_report.txt     -> accuracy/precision/recall/F1
                                 on the validation set
==========================================================
"""

import pandas as pd
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ----------------------------------------------------------
# LOAD TRAINING DATA
# ----------------------------------------------------------

print("Loading training dataset...")

train_df = pd.read_csv("TRAINING_DATA/train.csv")
val_df   = pd.read_csv("TRAINING_DATA/val.csv")

X_train_full = train_df.drop("label", axis=1)
y_train_full = train_df["label"]

X_val = val_df.drop("label", axis=1)
y_val = val_df["label"]

print(f"Full training set : {X_train_full.shape}")

# Load class names so the report is readable
le = joblib.load("TRAINING_DATA/label_encoder.pkl")
class_names = list(le.classes_)

# ----------------------------------------------------------
# SAMPLE 100,000 ROWS FOR THE BASELINE
#
# Sampling is done on the already-combined train_df so that
# X and y rows stay aligned (avoids the index-mismatch bug
# that occurs if you sample X and y separately).
# ----------------------------------------------------------

SAMPLE_SIZE = 100000

sample_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)

X_sample = sample_df.drop("label", axis=1)
y_sample = sample_df["label"]

print(f"Sampled training set : {X_sample.shape}")

# ----------------------------------------------------------
# TRAIN RANDOM FOREST
#
# n_estimators=100 : 100 trees is enough for a stable
#                     importance ranking without taking
#                     too long.
# n_jobs=-1        : use all CPU cores (RF trains on CPU
#                     only, unlike XGBoost which can use GPU)
# ----------------------------------------------------------

print("\nTraining Random Forest (100 trees, 100k samples)...")

start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_sample, y_sample)

elapsed = round(time.time() - start, 2)
print(f"Training time : {elapsed} seconds")

# ----------------------------------------------------------
# VALIDATION ACCURACY
#
# This number is a BASELINE — your XGBoost model (08_*.py)
# should outperform this. If XGBoost performs WORSE than
# this baseline, something is wrong with the XGBoost
# configuration or training process.
# ----------------------------------------------------------

print("\nEvaluating on validation set...")

y_val_pred = rf.predict(X_val)

val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nRandom Forest Validation Accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")

report = classification_report(
    y_val, y_val_pred,
    target_names=class_names,
    digits=4
)
print("\nClassification Report:\n")
print(report)

with open("rf_baseline_report.txt", "w") as f:
    f.write(f"Random Forest Baseline Accuracy: {val_acc:.4f}\n")
    f.write(f"Trained on {SAMPLE_SIZE:,} samples (100 trees)\n\n")
    f.write(report)

print("Saved -> rf_baseline_report.txt")

# ----------------------------------------------------------
# FEATURE IMPORTANCE
#
# Random Forest importance = average decrease in Gini
# impurity contributed by each feature across all trees.
#
# Compare this ranking with xgb_feature_importance.csv
# (from 08_xgboost_model_creation.py). Features that rank
# highly in BOTH are the most trustworthy discriminators.
# ----------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 20 features (Random Forest):\n")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv("rf_feature_importance.csv", index=False)
print("\nSaved -> rf_feature_importance.csv")

print("\nDone.")