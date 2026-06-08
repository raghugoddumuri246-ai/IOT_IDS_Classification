    # xgb_feature_importance.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading XGBoost Model...")

model = joblib.load(
    "xgb_model.pkl"
)

# =====================================================
# LOAD FEATURE NAMES
# =====================================================

train_df = pd.read_csv(
    "TRAINING_DATA/train.csv",
    nrows=5
)

feature_names = list(
    train_df.drop(
        "label",
        axis=1
    ).columns
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

# =====================================================
# PRINT TOP 30
# =====================================================

print("\nTOP 30 FEATURES\n")

print(
    feature_df.head(30)
    .to_string(index=False)
)

# =====================================================
# SAVE CSV
# =====================================================

feature_df.to_csv(
    "xgb_feature_importance.csv",
    index=False
)

print(
    "\nSaved -> xgb_feature_importance.csv"
)

# =====================================================
# TOP 20 PLOT
# =====================================================

top20 = feature_df.head(20)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.xlabel(
    "Importance Score"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 20 XGBoost Features"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "xgb_feature_importance.png",
    dpi=300
)

plt.show()

print(
    "\nSaved -> xgb_feature_importance.png"
)