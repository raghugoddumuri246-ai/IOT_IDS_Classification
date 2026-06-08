import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "NFSTREAM_CSV/DoS-TCP_Flood/DoS-TCP_Flood.csv"     
SAMPLE_ROWS = 100000

print("="*80)
print("LOADING DATA")
print("="*80)

df = pd.read_csv(
    CSV_FILE,
    low_memory=False,
    nrows=SAMPLE_ROWS
)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for col in df.columns:
    print(col)

print("\n")
print("="*80)
print("DATA TYPES")
print("="*80)

print(df.dtypes)

print("\n")
print("="*80)
print("NULL VALUES")
print("="*80)

nulls = df.isnull().sum().sort_values(ascending=False)

print(nulls)

print("\n")
print("="*80)
print("UNIQUE VALUES")
print("="*80)

for col in df.columns:
    print(col, ":", df[col].nunique())

print("\n")
print("="*80)
print("NUMERIC SUMMARY")
print("="*80)

print(df.describe().T)

print("\n")
print("="*80)
print("TOP APPLICATIONS")
print("="*80)

if "application_name" in df.columns:
    print(df["application_name"].value_counts().head(20))

print("\n")
print("="*80)
print("TOP PROTOCOLS")
print("="*80)

if "protocol" in df.columns:
    print(df["protocol"].value_counts())

print("\n")
print("="*80)
print("TOP DESTINATION PORTS")
print("="*80)

if "dst_port" in df.columns:
    print(df["dst_port"].value_counts().head(20))


# ------------------------------------------------
# Histograms
# ------------------------------------------------

numeric_cols = df.select_dtypes(include="number").columns

"""for col in numeric_cols[:15]:

    plt.figure(figsize=(8,4))

    df[col].hist(
        bins=50
    )

    plt.title(col)

    plt.tight_layout()

    plt.savefig(
        f"hist_{col}.png"
    )

    plt.close() """


# ------------------------------------------------
# Correlation
# ------------------------------------------------

print("\nGenerating Correlation Matrix...")

corr = df[numeric_cols].corr()

plt.figure(
    figsize=(20,15)
)

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.tight_layout()

plt.savefig(
    "correlation_matrix.png"
)

plt.close()

print("\nEDA COMPLETE")
