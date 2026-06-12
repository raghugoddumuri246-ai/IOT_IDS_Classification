"""
==========================================================
SCRIPT 06 : ENCODE LABELS, SCALE FEATURES, SPLIT DATASET
==========================================================

PIPELINE POSITION:
  05_master_dataset_creation.py  ->  [THIS SCRIPT]  ->  08_xgboost_model_creation.py

INPUT  : MASTER_FINE_GRAINED.csv     (840,000 rows x 93 columns)
OUTPUT : TRAINING_DATA/train.csv     (588,000 rows, 70%)
         TRAINING_DATA/val.csv       (126,000 rows, 15%)
         TRAINING_DATA/test.csv      (126,000 rows, 15%)
         TRAINING_DATA/label_encoder.pkl
         TRAINING_DATA/robust_scaler.pkl
         TRAINING_DATA/scale_cols.pkl
         TRAINING_DATA/feature_order.pkl

WHAT THIS SCRIPT DOES (in order):

  1. LABEL ENCODING
     Convert the 14 string labels (e.g. "DDoS-SYN_Flood")
     into integers 0-13. XGBoost requires integer class
     labels. The mapping is saved so predictions (integers)
     can be converted back to attack names later.

  2. TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
     Stratified — every split contains the same proportion
     of each of the 14 classes (since the dataset is already
     perfectly balanced at 60,000/class, each split gets
     42,000 / 9,000 / 9,000 per class).

  3. FEATURE SCALING (RobustScaler)
     Fitted ONLY on the training set, then applied to
     train/val/test. See "WHY ROBUSTSCALER" below.

  4. SAVE ALL ARTIFACTS NEEDED FOR INFERENCE
     label_encoder, scaler, scale_cols, feature_order —
     see the explanation at the end of this file for what
     each one is and why it's needed.

WHY ROBUSTSCALER (NOT STANDARDSCALER)?
  Timing features like bidirectional_mean_piat_ms range
  from 0 to tens of millions of milliseconds, with extreme
  outliers from long-lived DDoS flows. StandardScaler uses
  MEAN and STD DEV — a few huge outliers would dominate
  these statistics and compress the majority of "normal"
  values near zero, destroying their usefulness as split
  candidates for XGBoost.

  RobustScaler uses MEDIAN and IQR (interquartile range),
  which are not affected by extreme outliers. This keeps
  the bulk of the data spread out across a usable range.

WHY FIT THE SCALER ONLY ON TRAINING DATA?
  If the scaler were fit on the full dataset (including
  val/test), information about the val/test distribution
  would "leak" into the scaling parameters — an optimistic
  bias called "data leakage". Fitting only on train and
  then TRANSFORMING val/test with those same parameters
  mimics the real deployment scenario: at inference time,
  you only know the training distribution, not the new
  PCAP's distribution.
==========================================================
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

DATASET    = "MASTER_FINE_GRAINED.csv"
OUTPUT_DIR = "TRAINING_DATA"

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# STEP 1 : LOAD MASTER DATASET
# ==========================================================

print("Loading dataset...")
df = pd.read_csv(DATASET)
print(f"Shape : {df.shape}")

X = df.drop("label", axis=1)
y = df["label"]


# ==========================================================
# STEP 2 : LABEL ENCODING
# ==========================================================
#
# LabelEncoder sorts class names ALPHABETICALLY and assigns
# integers 0, 1, 2, ... in that order. The printed mapping
# below shows exactly which integer corresponds to which
# attack — this mapping is saved to label_encoder.pkl and
# used everywhere downstream (confusion matrix labels,
# classification reports, inference output).
# ==========================================================

print("\nEncoding labels...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nLabel mapping:")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    print(f"  {idx:>2}  {cls}")

joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"\nSaved label encoder -> {OUTPUT_DIR}/label_encoder.pkl")


# ==========================================================
# STEP 3 : TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# ==========================================================
#
# Two-step split:
#   First split off 30% as "temp" (val+test combined)
#   Then split "temp" 50/50 into val and test
#   Result: 70% train, 15% val, 15% test
#
# stratify=y_encoded ensures each split has the SAME class
# proportions as the full dataset (here: each class is
# exactly 1/14 of every split).
# ==========================================================

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


# ==========================================================
# STEP 4 : ROBUST SCALER
# ==========================================================
#
# WHICH COLUMNS GET SCALED?
# Categorical / encoded-integer columns must NOT be scaled
# — scaling would turn meaningful integers (e.g. protocol=6
# for TCP) into fractional values that lose their categorical
# meaning. These columns are excluded:
#
#   protocol, ip_version               -> protocol numbers
#   application_name,
#   application_category_name          -> vocab-encoded integers
#   splt_direction_0 .. splt_direction_9 -> binary (0/1) direction flags
#
# Everything else (packet counts, byte counts, timing
# statistics, flag counts, splt_ps_*, splt_piat_ms_*) gets
# scaled with RobustScaler.
# ==========================================================

print("\nFitting RobustScaler on training data...")

CATEGORICAL_COLS = [
    "protocol",
    "ip_version",
    "application_name",
    "application_category_name",
]

splt_dir_cols = [c for c in X_train.columns if c.startswith("splt_direction")]

COLS_TO_SKIP = set(CATEGORICAL_COLS + splt_dir_cols)

scale_cols = [c for c in X_train.columns if c not in COLS_TO_SKIP]

print(f"  Columns to scale   : {len(scale_cols)}")
print(f"  Columns kept as-is : {len(COLS_TO_SKIP & set(X_train.columns))}")

scaler = RobustScaler()

# Fit ONLY on training data (see docstring above for why)
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

joblib.dump(scaler, f"{OUTPUT_DIR}/robust_scaler.pkl")
joblib.dump(scale_cols, f"{OUTPUT_DIR}/scale_cols.pkl")


# ==========================================================
# STEP 5 : SAVE EXACT TRAINING COLUMN ORDER
# ==========================================================
#
# CRITICAL FOR INFERENCE.
# XGBoost validates feature_names by EXACT ORDER, not just
# by set membership — even if a new PCAP's feature matrix
# contains the same 92 column names, a DIFFERENT order
# raises a "feature_names mismatch" error.
#
# X_train.columns at this exact point (after scaling,
# before training) is the definitive order the model will
# be trained on. Saving it here guarantees
# 11_predict_pcap.py and 12_predict_pcap_xai.py can
# reproduce the identical order for any new PCAP.
# ==========================================================

feature_order = list(X_train.columns)
joblib.dump(feature_order, f"{OUTPUT_DIR}/feature_order.pkl")

print(f"Saved scaler        -> {OUTPUT_DIR}/robust_scaler.pkl")
print(f"Saved scale_cols    -> {OUTPUT_DIR}/scale_cols.pkl")
print(f"Saved feature_order -> {OUTPUT_DIR}/feature_order.pkl  ({len(feature_order)} features)")


# ==========================================================
# STEP 6 : SAVE TRAIN / VAL / TEST CSVs
# ==========================================================

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


# ==========================================================
# STEP 7 : VERIFICATION
# ==========================================================

print("\n" + "=" * 50)
print("SPLIT COMPLETE")
print("=" * 50)

print(f"\nTrain : {train_df.shape}  NaN={train_df.isnull().sum().sum()}")
print(f"Val   : {val_df.shape}    NaN={val_df.isnull().sum().sum()}")
print(f"Test  : {test_df.shape}   NaN={test_df.isnull().sum().sum()}")

print("\nClass distribution in train (expect 42,000 per class):")
print(train_df["label"].value_counts().sort_index())

print(f"\nAll files saved to {OUTPUT_DIR}/")