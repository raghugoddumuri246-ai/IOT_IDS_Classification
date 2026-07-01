import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# =====================================================
# CONFIG
# =====================================================

DATASET    = "MASTER_FINE_GRAINED.csv"
OUTPUT_DIR = "TRAINING_DATA"

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD
# =====================================================

print("Loading dataset...")
df = pd.read_csv(DATASET)
print(f"Shape : {df.shape}")

X = df.drop("label", axis=1)
y = df["label"]

# =====================================================
# LABEL ENCODING
# =====================================================

print("\nEncoding labels...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nLabel mapping:")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    print(f"  {idx:>2}  {cls}")

joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\nSaved label encoder -> {OUTPUT_DIR}/label_encoder.pkl")

# =====================================================
# TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# Stratified to preserve class balance
# =====================================================

print("\nSplitting...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

X_train = X_train.reset_index(drop=True)
X_val   = X_val.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)

y_train = pd.Series(y_train).reset_index(drop=True)
y_val   = pd.Series(y_val).reset_index(drop=True)
y_test  = pd.Series(y_test).reset_index(drop=True)

# =====================================================
# ROBUST SCALER
#
# Why RobustScaler and not StandardScaler?
#
# Features like bidirectional_mean_piat_ms have values
# in the range of tens of millions with extreme outliers
# from DDoS flows. StandardScaler uses mean and std, so
# outliers dominate and the majority of values end up
# compressed near zero.
#
# RobustScaler uses median and IQR, which are not
# affected by outliers. This gives XGBoost better split
# candidates across the full value range.
#
# IMPORTANT: Scaler is fit ONLY on training data.
#            Val and Test are transformed using the
#            training scaler — never fit on them.
#
# The scaler is saved to disk so that inference on
# new PCAP files applies the exact same transformation.
# =====================================================

print("\nFitting RobustScaler on training data...")

# Identify which columns to scale:
# - Exclude columns that are already integer-encoded categoricals
#   (application_name, application_category_name, protocol,
#    ip_version, splt_direction_*)
# - These should NOT be scaled because they are label-encoded
#   integers and scaling would give them fractional values

CATEGORICAL_COLS = [
    "protocol",
    "ip_version",
    "application_name",
    "application_category_name",
]

# Also exclude SPLT direction columns (0 or 1 values)
splt_dir_cols = [c for c in X_train.columns if c.startswith("splt_direction")]

COLS_TO_SKIP = set(CATEGORICAL_COLS + splt_dir_cols)

scale_cols = [
    c for c in X_train.columns
    if c not in COLS_TO_SKIP
]

print(f"  Columns to scale        : {len(scale_cols)}")
print(f"  Columns kept as-is      : {len(COLS_TO_SKIP & set(X_train.columns))}")

scaler = RobustScaler()

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

# Save scaler and the list of columns it was applied to
joblib.dump(scaler, f"{OUTPUT_DIR}/robust_scaler.pkl")
joblib.dump(scale_cols, f"{OUTPUT_DIR}/scale_cols.pkl")

# =====================================================
# SAVE EXACT TRAINING COLUMN ORDER
#
# This is CRITICAL for inference. XGBoost validates
# feature_names by exact order, not just by set membership.
# X_train.columns at this point is the definitive order
# the model will be trained on. Save it so 12_predict_pcap_xai.py
# can build its feature matrix in the identical order.
# =====================================================

feature_order = list(X_train.columns)
joblib.dump(feature_order, f"{OUTPUT_DIR}/feature_order.pkl")

print(f"Saved scaler -> {OUTPUT_DIR}/robust_scaler.pkl")
print(f"Saved scale_cols -> {OUTPUT_DIR}/scale_cols.pkl")
print(f"Saved feature_order -> {OUTPUT_DIR}/feature_order.pkl  ({len(feature_order)} features)")

# =====================================================
# SAVE SPLIT DATASETS
# =====================================================

train_df = X_train.copy()
train_df["label"] = y_train

val_df = X_val.copy()
val_df["label"] = y_val

test_df = X_test.copy()
test_df["label"] = y_test

print("\nSaving split datasets...")

train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(  f"{OUTPUT_DIR}/val.csv",   index=False)
test_df.to_csv( f"{OUTPUT_DIR}/test.csv",  index=False)

# =====================================================
# VERIFICATION
# =====================================================

print("\n" + "=" * 50)
print("SPLIT COMPLETE")
print("=" * 50)

print(f"\nTrain : {train_df.shape}  NaN={train_df.isnull().sum().sum()}")
print(f"Val   : {val_df.shape}    NaN={val_df.isnull().sum().sum()}")
print(f"Test  : {test_df.shape}   NaN={test_df.isnull().sum().sum()}")

print("\nClass distribution in train (expect 42,000 per class x 24 classes = 1,008,000 total):")
print(train_df["label"].value_counts().sort_index())

print(f"\nAll files saved to {OUTPUT_DIR}/")
inference_pipeline.py