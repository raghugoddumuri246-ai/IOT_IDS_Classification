import pandas as pd

df = pd.read_csv("MASTER_FINE_GRAINED.csv")

cols = [
    "bidirectional_packets",
    "bidirectional_duration_ms",
    "bidirectional_bytes"
]

print(
    df.groupby("label")[cols].mean()
)