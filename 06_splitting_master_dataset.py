import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# =====================================================
# CONFIG
# =====================================================

DATASET = "MASTER_FINE_GRAINED.csv"
OUTPUT_DIR = "TRAINING_DATA"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

df = pd.read_csv(DATASET)

print("Shape:", df.shape)

# =====================================================
# FEATURES + LABEL
# =====================================================

X = df.drop("label", axis=1)
y = df["label"]

print("\nFeatures Shape:", X.shape)
print("Labels Shape:", y.shape)

# =====================================================
# LABEL ENCODING
# =====================================================

print("\nEncoding labels...")

le = LabelEncoder()

y_encoded = le.fit_transform(y)

print("\nLabel Mapping:")

for cls, idx in zip(le.classes_, range(len(le.classes_))):
    print(f"{cls:25s} -> {idx}")

joblib.dump(
    le,
    f"{OUTPUT_DIR}/label_encoder.pkl"
)

# =====================================================
# TRAIN / VAL / TEST SPLIT
# =====================================================

print("\nCreating Train Split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=42
)

print("Creating Validation/Test Split...")

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# =====================================================
# RESET INDEXES
# =====================================================

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = pd.Series(y_train).reset_index(drop=True)
y_val = pd.Series(y_val).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# =====================================================
# MERGE FEATURES + LABELS
# =====================================================

train_df = X_train.copy()
train_df["label"] = y_train

val_df = X_val.copy()
val_df["label"] = y_val

test_df = X_test.copy()
test_df["label"] = y_test

# =====================================================
# SAVE FILES
# =====================================================

print("\nSaving datasets...")

train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

val_df.to_csv(
    f"{OUTPUT_DIR}/val.csv",
    index=False
)

test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)

# =====================================================
# VERIFY NO NAN
# =====================================================

print("\nVerification")

print("Train NaN :", train_df.isnull().sum().sum())
print("Val NaN   :", val_df.isnull().sum().sum())
print("Test NaN  :", test_df.isnull().sum().sum())

# =====================================================
# REPORT
# =====================================================

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)

print("\nTrain Labels")
print(train_df["label"].value_counts().sort_index())

print("\nVal Labels")
print(val_df["label"].value_counts().sort_index())

print("\nTest Labels")
print(test_df["label"].value_counts().sort_index())

print("\nSaved To:", OUTPUT_DIR)