# evaluate_final_ids_v4.py

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# LOAD VALIDATION DATA
# =====================================================

print("Loading Validation Dataset...")

val_df = pd.read_csv(
    "TRAINING_DATA/val.csv"
)

X_val = val_df.drop(
    "label",
    axis=1
)

y_true = val_df["label"]

# =====================================================
# LOAD XGBOOST V4 MODEL
# =====================================================

print("Loading Model...")

model = joblib.load(
    "xgb_model.pkl"
)

# =====================================================
# PREDICTIONS
# =====================================================

print("Predicting...")

y_pred = model.predict(X_val)

# =====================================================
# LABEL MAP
# =====================================================

label_map = {
    0: "Benign",

    1: "DDoS-ICMP",
    2: "DDoS-SYN",
    3: "DDoS-TCP",
    4: "DDoS-UDP",

    5: "DoS-SYN",
    6: "DoS-TCP",
    7: "DoS-UDP",

    8: "Mirai-greeth",
    9: "Mirai-greip",
    10: "Mirai-udpplain",

    11: "Recon-HostDiscovery",
    12: "Recon-OSScan",
    13: "Recon-PortScan"
}

# =====================================================
# COLLAPSE LABELS
# =====================================================

def collapse(label_id):

    name = label_map[int(label_id)]

    if name.startswith("Mirai"):
        return "Mirai"

    if name.startswith("Recon"):
        return "Recon"

    return name

# =====================================================
# CONVERT TRUE LABELS
# =====================================================

y_true_final = [
    collapse(x)
    for x in y_true
]

# =====================================================
# CONVERT PREDICTIONS
# =====================================================

y_pred_final = [
    collapse(x)
    for x in y_pred
]

# =====================================================
# ACCURACY
# =====================================================

acc = accuracy_score(
    y_true_final,
    y_pred_final
)

print("\n")
print("="*60)
print("FINAL IDS ACCURACY")
print("="*60)

print(acc)

# =====================================================
# REPORT
# =====================================================

report = classification_report(
    y_true_final,
    y_pred_final,
    digits=4
)

print("\n")
print("="*60)
print("FINAL IDS REPORT")
print("="*60)

print(report)


# =====================================================
# CONFUSION MATRIX
# =====================================================

labels = [
    "Benign",
    "DDoS-ICMP",
    "DDoS-SYN",
    "DDoS-TCP",
    "DDoS-UDP",
    "DoS-SYN",
    "DoS-TCP",
    "DoS-UDP",
    "Mirai",
    "Recon"
]

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

cm = confusion_matrix(
    y_true_final,
    y_pred_final,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm_df)


