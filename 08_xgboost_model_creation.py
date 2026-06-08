# xgboost_v4.py

import pandas as pd
import joblib
import time

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading datasets...")

train_df = pd.read_csv(
    "TRAINING_DATA/train.csv"
)

val_df = pd.read_csv(
    "TRAINING_DATA/val.csv"
)

X_train = train_df.drop(
    "label",
    axis=1
)

y_train = train_df["label"]

X_val = val_df.drop(
    "label",
    axis=1
)

y_val = val_df["label"]

print("Train:", X_train.shape)
print("Val  :", X_val.shape)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=14,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")

start = time.time()

model.fit(
    X_train,
    y_train
)

end = time.time()

print(
    "\nTraining Time:",
    round(end-start,2),
    "seconds"
)

print("\nPredicting...")

y_pred = model.predict(X_val)

acc = accuracy_score(
    y_val,
    y_pred
)

print("\nAccuracy:")
print(acc)

report = classification_report(
    y_val,
    y_pred,
    digits=4
)

print("\nClassification Report\n")
print(report)

joblib.dump(
    model,
    "xgb_model.pkl"
)

with open(
    "xgb_report.txt",
    "w"
) as f:
    f.write(report)

pd.DataFrame(
    confusion_matrix(
        y_val,
        y_pred
    )
).to_csv(
    "xgb_confusion_matrix.csv",
    index=False
)

print("\nSaved")