"""
==========================================================
SCRIPT 02 : FLOW COUNT SUMMARY
==========================================================

PURPOSE:
After running 01_extract_flows.py for every attack, each
attack folder inside NFSTREAM_CSV/ contains one or more
CSV files (one per source PCAP).

This script counts the TOTAL number of flow rows available
for each attack category. This number is important because
it tells us:

  1. Which attacks have PLENTY of data (millions of flows,
     e.g. DDoS-SYN_Flood) vs which attacks have LIMITED data
     (tens of thousands, e.g. Mirai-udpplain).

  2. It is the input to 04_create_balanced_attack_files.py,
     which uses these totals to decide how many rows to
     sample from each individual CSV file so that every
     attack contributes exactly 60,000 rows in the end.

HOW TO READ THE OUTPUT:
The printed numbers are simply:
    (total lines in all CSVs for that attack) - (header rows)

Example output:
    DDoS-SYN_Flood   : 17,043,461   <- huge, will be heavily downsampled
    Mirai-udpplain   :     64,389   <- small, almost all rows will be used
==========================================================
"""

import pandas as pd
from pathlib import Path

ROOT = Path("NFSTREAM_CSV")

print("=" * 60)
print("FLOW COUNTS PER ATTACK CATEGORY")
print("=" * 60)
print()

results = []

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir():
        continue

    total_rows = 0
    file_count = 0

    for csv_file in folder.glob("*.csv"):

        # Count lines minus 1 for the header row.
        # This is fast because it doesn't load the
        # full CSV into memory — just counts newlines.
        rows = sum(1 for _ in open(csv_file, encoding="utf-8", errors="ignore")) - 1

        total_rows += rows
        file_count += 1

    results.append((folder.name, file_count, total_rows))

    print(f"{folder.name:<25}  files={file_count:>3}  rows={total_rows:>12,}")

print("\n" + "=" * 60)
print(f"Total attack categories found : {len(results)}")
print("=" * 60)