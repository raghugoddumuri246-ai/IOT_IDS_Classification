from pathlib import Path
import pandas as pd

INPUT_DIR = Path("BALANCED_ATTACKS")
OUTPUT_FILE = "MASTER_FINE_GRAINED.csv"

RANDOM_STATE = 42

# --------------------------------------------------
# Load all balanced files
# --------------------------------------------------

all_dfs = []

csv_files = sorted(INPUT_DIR.glob("*.csv"))

print(f"\nFound {len(csv_files)} files\n")

for csv_file in csv_files:

    df = pd.read_csv(csv_file)

    print(
        f"{csv_file.name:35} "
        f"{df.shape}"
    )

    all_dfs.append(df)

# --------------------------------------------------
# Merge
# --------------------------------------------------

print("\nMerging files...")

master_df = pd.concat(
    all_dfs,
    axis=0,
    ignore_index=True
)

print("\nShape Before Shuffle:")
print(master_df.shape)

# --------------------------------------------------
# Global Shuffle
# --------------------------------------------------

print("\nShuffling dataset...")

master_df = master_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# --------------------------------------------------
# Verification
# --------------------------------------------------

print("\nFinal Shape:")
print(master_df.shape)

print("\nLabel Distribution:")
print(master_df["label"].value_counts().sort_index())

# --------------------------------------------------
# Duplicate Check
# --------------------------------------------------

duplicates = master_df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicates)

# --------------------------------------------------
# Save
# --------------------------------------------------

print("\nSaving CSV...")

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved -> {OUTPUT_FILE}")