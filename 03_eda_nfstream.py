"""
==========================================================
SCRIPT 03 : EXPLORATORY DATA ANALYSIS (EDA) ON RAW NFSTREAM CSV
==========================================================

PURPOSE:
Before any feature engineering, this script inspects ONE
raw NFStream CSV file (before the column drop / SPLT
expansion / encoding steps in 04_create_balanced_attack_files.py)
to understand what NFStream actually produced.

This is useful for:
  - Verifying the expected 89 raw columns are present
  - Checking which columns are mostly NULL
    (helps justify why user_agent/content_type were dropped)
  - Checking value ranges and outliers
    (justifies why RobustScaler is used instead of StandardScaler)
  - Checking application_name / application_category_name
    distributions (justifies keeping these as features)
  - Generating a correlation heatmap of numeric features
    (helps explain to your guide which features move together)

NOTE: This script reads the RAW CSV (before 04_*.py runs).
The splt_direction / splt_ps / splt_piat_ms columns will
still be STRING-encoded lists at this stage — this is
expected and is exactly the issue that 04_*.py fixes via
expand_all_splt().
==========================================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no GUI needed, just save PNG files
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# CONFIG — change this to inspect a different attack's CSV
# ----------------------------------------------------------

CSV_FILE    = "NFSTREAM_CSV/DoS-TCP_Flood/DoS-TCP_Flood.csv"
SAMPLE_ROWS = 100000   # load only first N rows for speed

# ----------------------------------------------------------
# LOAD
# ----------------------------------------------------------

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print(f"\nFile : {CSV_FILE}")
print(f"Rows requested : {SAMPLE_ROWS:,}\n")

df = pd.read_csv(CSV_FILE, low_memory=False, nrows=SAMPLE_ROWS)

print("Shape:", df.shape)

# ----------------------------------------------------------
# COLUMN LIST
# ----------------------------------------------------------

print("\n" + "=" * 80)
print(f"COLUMNS ({len(df.columns)} total)")
print("=" * 80)
for col in df.columns:
    print(" ", col)

# ----------------------------------------------------------
# DATA TYPES
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)
print(df.dtypes)

# ----------------------------------------------------------
# NULL VALUES
#
# Look for columns that are mostly NULL — these are
# candidates for dropping (e.g. user_agent, content_type
# were ~99.9% null and were dropped in 04_*.py)
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("NULL VALUES (sorted, highest first)")
print("=" * 80)

nulls = df.isnull().sum().sort_values(ascending=False)
print(nulls[nulls > 0])

if (nulls > 0).sum() == 0:
    print("(no null values found)")

# ----------------------------------------------------------
# UNIQUE VALUE COUNTS
#
# Helps spot:
#   - Identifier columns (id, src_ip etc. -> nunique very high)
#   - Constant columns (expiration_id, vlan_id -> nunique == 1)
#   - SPLT columns (high nunique because each row has a
#     different "[a,b,c,...]" string — expected at this stage)
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("UNIQUE VALUES PER COLUMN")
print("=" * 80)

for col in df.columns:
    print(f"  {col:<30} : {df[col].nunique()}")

# ----------------------------------------------------------
# NUMERIC SUMMARY
#
# Look at min/max/std for timing features
# (bidirectional_mean_piat_ms etc.) — large ranges and
# heavy-tailed distributions here are WHY RobustScaler
# (median + IQR based) was chosen over StandardScaler
# (mean + std based) in 06_splitting_master_dataset.py
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("NUMERIC SUMMARY (describe)")
print("=" * 80)

print(df.describe().T)

# ----------------------------------------------------------
# TOP APPLICATIONS (NFStream DPI output)
#
# This is the application_name column kept as a feature
# in 04_*.py. Useful to see how many distinct protocols
# NFStream detected for this attack type.
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("TOP APPLICATIONS (application_name)")
print("=" * 80)

if "application_name" in df.columns:
    print(df["application_name"].value_counts().head(20))
else:
    print("(application_name column not present)")

# ----------------------------------------------------------
# TOP PROTOCOLS (transport layer protocol number)
#   6  = TCP
#   17 = UDP
#   1  = ICMP
#   2  = IGMP
#   58 = ICMPv6
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("PROTOCOL DISTRIBUTION")
print("=" * 80)

if "protocol" in df.columns:
    print(df["protocol"].value_counts())
else:
    print("(protocol column not present)")

# ----------------------------------------------------------
# TOP DESTINATION PORTS
#
# Useful for understanding WHAT the attack is targeting
# (e.g. port 80 = HTTP, port 22 = SSH). Note: dst_port
# itself is DROPPED in 04_*.py because it is environment
# dependent — this is just for EDA understanding.
# ----------------------------------------------------------

print("\n" + "=" * 80)
print("TOP DESTINATION PORTS")
print("=" * 80)

if "dst_port" in df.columns:
    print(df["dst_port"].value_counts().head(20))
else:
    print("(dst_port column not present — already removed)")

# ----------------------------------------------------------
# CORRELATION HEATMAP
#
# Shows which numeric features are highly correlated with
# each other. Highly correlated feature pairs are candidates
# for removal in future feature-selection experiments
# (not currently removed — XGBoost handles correlated
# features reasonably well, but this is useful for discussion
# with your guide).
# ----------------------------------------------------------

print("\nGenerating correlation heatmap...")

numeric_cols = df.select_dtypes(include="number").columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(20, 15))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title(f"Feature Correlation Heatmap — {CSV_FILE}")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.close()

print("Saved -> correlation_matrix.png")
print("\nEDA COMPLETE")