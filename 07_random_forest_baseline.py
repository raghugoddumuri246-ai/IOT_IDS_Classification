import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Loading Train Dataset...")

df = pd.read_csv("TRAINING_DATA/train.csv")

X = df.drop("label", axis=1)
y = df["label"]

print("Original Shape:", X.shape)

sample_size = 100000

sample = df.sample(
    n=sample_size,
    random_state=42
)

X_sample = sample.drop("label", axis=1)
y_sample = sample["label"]

print("Sample Shape:", X_sample.shape)

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_sample, y_sample)

print("Done.")

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 30 FEATURES\n")
print(importance_df.head(30))

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nSaved -> feature_importance.csv")