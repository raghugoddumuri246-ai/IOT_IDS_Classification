# =====================================================
# 12_shap_test.py
#
# PURPOSE:
# Verify SHAP works correctly with:
# XGBoost 2.1.4
# SHAP 0.44.1
#
# This script:
# 1. Loads trained model
# 2. Loads validation data
# 3. Predicts few samples
# 4. Computes SHAP values
# 5. Prints shape information
#
# We need this output before building
# PCAP-level explainability.
# =====================================================

import pandas as pd
import joblib
import shap

# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = joblib.load(
    "xgb_model.pkl"
)

print("Model Loaded")

# =====================================================
# LOAD VALIDATION DATA
# =====================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

df = pd.read_csv(
    "TRAINING_DATA/val.csv"
)

X = df.drop(
    "label",
    axis=1
)

y = df["label"]

print("Dataset Shape :", X.shape)

# =====================================================
# TAKE SMALL SAMPLE
#
# We only need few rows for testing.
# =====================================================

sample = X.iloc[:5]

print("\nSample Shape :", sample.shape)

# =====================================================
# PREDICT
# =====================================================

print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

preds = model.predict(sample)

print("Predictions:")
print(preds)

# =====================================================
# CREATE SHAP EXPLAINER
# =====================================================

print("\n" + "=" * 60)
print("CREATING SHAP EXPLAINER")
print("=" * 60)

explainer = shap.TreeExplainer(model)

print("Explainer Created")

# =====================================================
# COMPUTE SHAP VALUES
# =====================================================

print("\n" + "=" * 60)
print("COMPUTING SHAP VALUES")
print("=" * 60)

shap_values = explainer.shap_values(
    sample
)

print("SHAP Computation Complete")

# =====================================================
# TYPE
# =====================================================

print("\n" + "=" * 60)
print("SHAP TYPE")
print("=" * 60)

print(type(shap_values))

# =====================================================
# SHAPE
# =====================================================

print("\n" + "=" * 60)
print("SHAP SHAPE")
print("=" * 60)

try:

    print(shap_values.shape)

except Exception as e:

    print("No direct shape available")
    print("Error:", e)

# =====================================================
# EXTRA INSPECTION
# =====================================================

print("\n" + "=" * 60)
print("DETAILED INSPECTION")
print("=" * 60)

if isinstance(shap_values, list):

    print("Returned as LIST")

    print(
        "Number of elements:",
        len(shap_values)
    )

    for i, arr in enumerate(shap_values):

        try:

            print(
                f"Class {i} Shape:",
                arr.shape
            )

        except:

            print(
                f"Class {i} Shape Unknown"
            )

else:

    print(
        "Returned as NUMPY ARRAY"
    )

    try:

        print(
            "Array Shape:",
            shap_values.shape
        )

    except:

        print(
            "Cannot determine shape"
        )

# =====================================================
# EXPECTED VALUE
# =====================================================

print("\n" + "=" * 60)
print("EXPECTED VALUE")
print("=" * 60)

try:

    print(type(explainer.expected_value))

    print(explainer.expected_value)

except Exception as e:

    print("Error:", e)

# =====================================================
# FEATURE COUNT
# =====================================================

print("\n" + "=" * 60)
print("FEATURE COUNT")
print("=" * 60)

print(len(sample.columns))

print("\nDONE")