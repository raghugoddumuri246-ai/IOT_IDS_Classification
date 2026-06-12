"""
==========================================================
SCRIPT 08 : TRAIN THE XGBOOST IDS MODEL
==========================================================

PIPELINE POSITION:
  06_splitting_master_dataset.py  ->  [THIS SCRIPT]  ->  12_predict_pcap_xai.py

INPUT  : TRAINING_DATA/train.csv, val.csv, test.csv
         TRAINING_DATA/label_encoder.pkl
OUTPUT : xgb_model.pkl                 <- the trained model (DEPLOY THIS)
         xgb_confusion_matrix.csv      <- 14x14 confusion matrix, labeled
         xgb_feature_importance.csv    <- ranked feature importances
         xgb_val_report.txt            <- validation classification report
         xgb_test_report.txt           <- test classification report

WHAT THIS SCRIPT DOES:

  1. Load train/val/test splits (already scaled and label-
     encoded by 06_splitting_master_dataset.py)
  2. Configure and train an XGBoost multiclass classifier
     (14 classes, see hyperparameter notes below)
  3. Evaluate on BOTH validation and test sets
  4. Print a per-class recall bar chart to quickly spot
     which classes are weak
  5. Save the confusion matrix (with readable class-name
     labels), feature importances, model, and text reports
==========================================================
"""

import pandas as pd
import numpy as np
import joblib
import time

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score
)

# ==========================================================
# STEP 1 : LOAD DATA
# ==========================================================

print("Loading datasets...")

train_df = pd.read_csv("TRAINING_DATA/train.csv")
val_df   = pd.read_csv("TRAINING_DATA/val.csv")
test_df  = pd.read_csv("TRAINING_DATA/test.csv")

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_val   = val_df.drop("label", axis=1)
y_val   = val_df["label"]

X_test  = test_df.drop("label", axis=1)
y_test  = test_df["label"]

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

# Load class names (e.g. "DDoS-SYN_Flood") for readable reports
le = joblib.load("TRAINING_DATA/label_encoder.pkl")
class_names = list(le.classes_)

print(f"\nClasses ({len(class_names)}):")
for i, name in enumerate(class_names):
    print(f"  {i:>2}  {name}")

num_classes = len(class_names)


# ==========================================================
# STEP 2 : MODEL CONFIGURATION
# ==========================================================
#
# objective="multi:softprob", num_class=14
#   Multiclass classification with probability outputs
#   (needed for predict_proba in 12_predict_pcap_xai.py
#   to report per-flow confidence).
#
# tree_method="hist", device="cuda"
#   GPU-accelerated histogram-based training. Remove
#   device="cuda" (or set device="cpu") on machines
#   without a usable GPU.
#
# n_estimators=500, max_depth=10, learning_rate=0.05
#   500 boosting rounds, fairly deep trees (10 levels) to
#   capture interactions between the 92 features —
#   particularly important now that application_name,
#   application_category_name, and the 30 splt_* columns
#   have been added on top of the original flow statistics.
#
# min_child_weight=5, gamma=0.1
#   Regularisation: requires at least 5 samples per leaf
#   and a minimum loss reduction to split — reduces
#   overfitting to noisy individual flows.
#
# subsample=0.8, colsample_bytree=0.8
#   Each tree sees 80% of rows and 80% of columns —
#   standard regularisation, also speeds up training.
#
# reg_alpha=0.1 (L1), reg_lambda=1.0 (L2)
#   Additional regularisation terms penalising large
#   leaf weights.
#
# early_stopping_rounds=30
#   Stop training if validation mlogloss does not improve
#   for 30 consecutive rounds — prevents wasting time/
#   overfitting once the model has converged.
# ==========================================================

model = XGBClassifier(
    objective       = "multi:softprob",
    num_class       = num_classes,
    n_estimators    = 500,
    max_depth       = 10,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 5,
    gamma           = 0.1,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    tree_method     = "hist",
    device          = "cuda",    # change to "cpu" if no GPU
    eval_metric     = "mlogloss",
    early_stopping_rounds = 30,
    random_state    = 42,
    n_jobs          = -1,
    verbosity       = 1
)


# ==========================================================
# STEP 3 : TRAINING
# ==========================================================

print("\nTraining...")
start = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=10   # print mlogloss every 10 rounds
)

elapsed = round(time.time() - start, 2)
print(f"\nTraining time : {elapsed} seconds")
print(f"Best iteration: {model.best_iteration}")


# ==========================================================
# STEP 4 : VALIDATION EVALUATION
# ==========================================================

print("\n" + "=" * 60)
print("VALIDATION SET RESULTS")
print("=" * 60)

y_val_pred = model.predict(X_val)

val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")

val_report = classification_report(
    y_val, y_val_pred,
    target_names=class_names,
    digits=4
)
print("\nClassification Report (Validation):\n")
print(val_report)


# ==========================================================
# STEP 5 : TEST SET EVALUATION
# ==========================================================
#
# Why evaluate on BOTH val and test?
# Validation was used (indirectly) during training via
# early_stopping_rounds — the model "saw" validation
# performance and stopped based on it. The TEST set was
# never used in any way during training, so test accuracy
# is the more trustworthy/unbiased estimate of real-world
# performance.
# ==========================================================

print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

y_test_pred = model.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

test_report = classification_report(
    y_test, y_test_pred,
    target_names=class_names,
    digits=4
)
print("\nClassification Report (Test):\n")
print(test_report)


# ==========================================================
# STEP 6 : PER-CLASS RECALL SUMMARY
# ==========================================================
#
# Sorted from WORST to BEST recall. This is the fastest way
# to spot which attack classes the model struggles with
# (e.g. Mirai/Recon subtypes typically show up at the bottom).
# ==========================================================

print("\nPer-class recall summary (sorted, worst first):")

recalls = recall_score(y_test, y_test_pred, average=None)

for name, rec in sorted(zip(class_names, recalls), key=lambda x: x[1]):
    bar = "█" * int(rec * 30)
    print(f"  {name:<30}  {rec:.3f}  {bar}")


# ==========================================================
# STEP 7 : CONFUSION MATRIX (labeled)
# ==========================================================
#
# Rows = true class, Columns = predicted class.
# Saved WITH class names as row/column headers so it can
# be opened directly in Excel/Sheets for inspection.
# ==========================================================

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

cm_df.to_csv("xgb_confusion_matrix.csv")
print("\nSaved -> xgb_confusion_matrix.csv")


# ==========================================================
# STEP 8 : FEATURE IMPORTANCE
# ==========================================================
#
# model.feature_importances_ uses XGBoost's default
# "gain"-based importance — how much each feature
# contributed to reducing the loss across all splits,
# normalised to sum to 1.
# ==========================================================

importance_df = pd.DataFrame({
    "Feature"   : X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance_df.to_csv("xgb_feature_importance.csv", index=False)

print("Saved -> xgb_feature_importance.csv")
print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))


# ==========================================================
# STEP 9 : SAVE MODEL + TEXT REPORTS
# ==========================================================
#
# xgb_model.pkl is the ONE artifact required for inference
# (along with the TRAINING_DATA/*.pkl artifacts saved by
# 06_splitting_master_dataset.py — see the full artifact
# explanation document for details).
# ==========================================================

joblib.dump(model, "xgb_model.pkl")
print("\nSaved -> xgb_model.pkl")

with open("xgb_val_report.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
    f.write(val_report)

with open("xgb_test_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(test_report)

print("Saved -> xgb_val_report.txt")
print("Saved -> xgb_test_report.txt")
print("\nDone.")