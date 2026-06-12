"""
==========================================================
SCRIPT 05 : MERGE BALANCED ATTACK FILES INTO ONE MASTER DATASET
==========================================================

PIPELINE POSITION:
  04_create_balanced_attack_files.py  ->  [THIS SCRIPT]  ->  06_splitting_master_dataset.py

INPUT  : BALANCED_ATTACKS/<attack_name>.csv   (14 files, 60,000 rows x 93 cols each)
OUTPUT : MASTER_FINE_GRAINED.csv               (840,000 rows x 93 cols, shuffled)

WHAT THIS SCRIPT DOES:

  1. Load all 14 balanced attack CSVs
  2. CHECK that every file has the EXACT same columns
     (catches mistakes like running 04_*.py with different
     NFStreamer settings for different attacks, which would
     produce mismatched splt_* or application_* columns)
  3. Concatenate all 14 files into one DataFrame
     (14 x 60,000 = 840,000 rows)
  4. SHUFFLE the combined dataset
     (so that train/val/test splits in 06_*.py contain a
     random mix of all classes, not blocks of one class
     followed by blocks of another)
  5. Verify: print label distribution, NaN count, duplicate
     row count
  6. Save as MASTER_FINE_GRAINED.csv

WHY CHECK COLUMN ALIGNMENT?
If 01_extract_flows.py was run with different NFStreamer
settings at different times (e.g. splt_analysis enabled
for some attacks but not others), the resulting balanced
CSVs would have different column sets. Concatenating
mismatched columns silently introduces NaN columns for
whichever rows don't have that column — this check catches
that BEFORE it becomes a confusing problem in training.
==========================================================
"""

from pathlib import Path
import pandas as pd

INPUT_DIR   = Path("BALANCED_ATTACKS")
OUTPUT_FILE = "MASTER_FINE_GRAINED.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------
# STEP 1 : LOAD ALL BALANCED ATTACK FILES
# ----------------------------------------------------------

all_dfs   = []
csv_files = sorted(INPUT_DIR.glob("*.csv"))

# BALANCED_ATTACKS/ also contains app_name_vocab.json and
# app_category_vocab.json (saved by 04_*.py) — only load
# the .csv attack files here.
csv_files = [f for f in csv_files if f.suffix == ".csv"]

print(f"\nFound {len(csv_files)} attack CSV files\n")

column_sets = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    print(f"  {csv_file.name:45}  shape={df.shape}")
    column_sets.append(set(df.columns))
    all_dfs.append(df)

# ----------------------------------------------------------
# STEP 2 : COLUMN ALIGNMENT CHECK
# ----------------------------------------------------------
#
# All 14 files must have IDENTICAL column sets. If they
# don't, it usually means 01_extract_flows.py was run with
# different NFStreamer settings for different attacks.
#
# If a mismatch IS found, this script falls back to using
# only the columns common to ALL files (intersection), so
# the pipeline can still proceed — but you should
# investigate WHY the mismatch happened.
# ----------------------------------------------------------

print("\nChecking column alignment...")

reference_cols = column_sets[0]
all_same = True

for i, cols in enumerate(column_sets[1:], start=1):
    if cols != reference_cols:
        extra   = cols - reference_cols
        missing = reference_cols - cols
        print(f"  MISMATCH in file index {i}:")
        if extra:
            print(f"    Extra columns   : {extra}")
        if missing:
            print(f"    Missing columns : {missing}")
        all_same = False

if all_same:
    print("  All columns aligned correctly.")
else:
    print("\n  WARNING: Column mismatch detected.")
    print("  Using intersection of all columns to align.")

    common_cols = reference_cols.copy()
    for cols in column_sets[1:]:
        common_cols &= cols

    all_dfs = [df[list(common_cols)] for df in all_dfs]
    print(f"  Using {len(common_cols)} common columns.\n")

# ----------------------------------------------------------
# STEP 3 : MERGE ALL 14 FILES
# ----------------------------------------------------------

print("\nMerging...")

master_df = pd.concat(all_dfs, axis=0, ignore_index=True)

print(f"Shape before shuffle : {master_df.shape}")

# ----------------------------------------------------------
# STEP 4 : GLOBAL SHUFFLE
# ----------------------------------------------------------
#
# Without this, the dataset is 14 BLOCKS of 60,000 rows,
# one block per attack, in the order of csv_files. If
# 06_splitting_master_dataset.py's train_test_split used
# stratify=y (which it does), shuffling isn't strictly
# REQUIRED for correctness — but it's good practice and
# makes any manual head()/sample() inspection meaningful.
# ----------------------------------------------------------

print("Shuffling...")

master_df = master_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ----------------------------------------------------------
# STEP 5 : VERIFICATION
# ----------------------------------------------------------

print(f"\nFinal shape : {master_df.shape}")

print("\nLabel distribution:")
print(master_df["label"].value_counts().sort_index())
# Expect: all 14 classes show exactly 60,000

print("\nNaN check:", master_df.isnull().sum().sum(), "NaN values")
# Expect: 0 (all NaN were filled with 0 in 04_*.py)

print("\nDuplicate rows:", master_df.duplicated().sum())
# Some duplicates are EXPECTED and normal here — e.g. many
# DDoS flood flows genuinely have identical statistics
# (same packet size, same flag pattern). This is not a bug.

# ----------------------------------------------------------
# STEP 6 : SAVE
# ----------------------------------------------------------

print("\nSaving...")
master_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved -> {OUTPUT_FILE}")