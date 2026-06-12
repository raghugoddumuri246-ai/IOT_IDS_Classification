"""
==========================================================
SCRIPT 10 : QUICK METRICS SUMMARY (FINE-GRAINED, 14-CLASS)
==========================================================

PURPOSE:
A short script that loads the trained model and prints
the four headline numbers — Accuracy, Precision, Recall,
F1 — plus the full classification report and raw confusion
matrix, on the VALIDATION set.

This is useful as a quick "is the model still working"
sanity check after re-running training (08_*.py), without
needing to re-read the full training log.

DIFFERENCE FROM 08_xgboost_model_creation.py:
08_*.py prints these metrics ONCE, right after training,
and also evaluates on the TEST set. This script is a
standalone re-evaluation tool you can run anytime against
the saved xgb_model.pkl without retraining.

DIFFERENCE FROM 09_xgboost_evaluation_10.py:
09_*.py collapses Mirai/Recon subtypes into coarse classes
(10-class view). This script shows the full 14-class
("fine-grained") view.
==========================================================
"""

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------------------------------
# LOAD VALIDATION DATA
# ----------------------------------------------------------

print("Loading validation dataset...")

df = pd.read_csv("TRAINING_DATA/val.csv")

X      = df.drop("label", axis=1)
y_true = df["label"]

# ----------------------------------------------------------
# LOAD MODEL + CLASS NAMES
# ----------------------------------------------------------

print("Loading XGBoost model...")

model = joblib.load("xgb_model.pkl")
le    = joblib.load("TRAINING_DATA/label_encoder.pkl")

# Force CPU — consistent with 09_*.py and 12_*.py, avoids
# GPU memory issues on machines with limited free VRAM.
model.get_booster().set_param({"device": "cpu"})

class_names = list(le.classes_)

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

print("Predicting...")

y_pred = model.predict(X)

# ----------------------------------------------------------
# HEADLINE METRICS
#
# average="weighted" : each class's score is weighted by
# how many samples it has. Since all 14 classes have
# exactly 9,000 validation samples (balanced dataset),
# "weighted" and "macro" averages will be nearly identical
# here — but "weighted" is used for consistency with
# sklearn's default reporting style.
# ----------------------------------------------------------

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall    = recall_score(y_true, y_pred, average="weighted")
f1        = f1_score(y_true, y_pred, average="weighted")

print("\n" + "=" * 60)
print("XGBOOST MODEL METRICS (14-CLASS, FINE-GRAINED)")
print("=" * 60)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# ----------------------------------------------------------
# FULL CLASSIFICATION REPORT
#
# target_names=class_names makes the report show actual
# attack names (e.g. "DDoS-ICMP_Flood") instead of just
# numeric indices (0-13) — much easier to read and explain.
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4
))

# ----------------------------------------------------------
# RAW CONFUSION MATRIX (14x14, numeric)
#
# Rows = true class, Columns = predicted class.
# Diagonal values = correct predictions.
# Off-diagonal values = confusions between classes.
#
# For a labeled version, see xgb_confusion_matrix.csv
# (saved by 08_xgboost_model_creation.py with class
# names as row/column headers).
# ----------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX (numeric, 0-13 = class index)")
print("=" * 60)
print(cm)

print("\nShape:", cm.shape)
print("\nClass index reference:")
for i, name in enumerate(class_names):
    print(f"  {i:>2} = {name}")