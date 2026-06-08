# show_xgb_metrics.py

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

# =====================================================
# LOAD VALIDATION DATA
# =====================================================

print("Loading Validation Dataset...")

df = pd.read_csv(
    "TRAINING_DATA/val.csv"
)

X = df.drop(
    "label",
    axis=1
)

y_true = df["label"]

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading XGBoost Model...")

model = joblib.load(
    "xgb_model.pkl"
)

# =====================================================
# PREDICT
# =====================================================

print("Predicting...")

y_pred = model.predict(X)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

# =====================================================
# REPORT
# =====================================================

print("\n" + "=" * 60)
print("XGBOOST MODEL METRICS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nShape:", cm.shape)