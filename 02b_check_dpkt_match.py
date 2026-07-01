"""
==========================================================
SCRIPT : CHECK DPKT vs NFSTREAM FLOW COUNT MATCH RATES
==========================================================

PURPOSE:
Before running the full 24-class 04_create_balanced_attack_files.py,
this script does a quick sanity check:

  For each attack that has DPKT CSVs extracted:
    - Count total flows in all DPKT_CSV/<attack>/*.csv files
    - Count total flows in all NFSTREAM_CSV/<attack>/*.csv files
    - Report the match percentage

A match rate >= 95% means the idle-timeout flow-key logic is
working correctly and the merge in Script 04 will be reliable.

A match rate well below 95% for a specific attack means either:
  a) 01b_extract_dpkt_features.py was not run on all PCAPs for
     that attack yet (some DPKT files missing)
  b) That attack contains unusual traffic (e.g. many non-IP
     packets, non-Ethernet link layer) that DPKT skips but
     NFStream counted differently

USAGE:
  python check_dpkt_match.py
==========================================================
"""

import os
from pathlib import Path

NFSTREAM_ROOT = Path("NFSTREAM_CSV")
DPKT_ROOT     = Path("DPKT_CSV")


def count_rows_fast(csv_path):
    """Count data rows (excluding header) without loading into pandas."""
    try:
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f) - 1  # minus 1 for header
    except Exception:
        return 0


def count_rows_in_folder(folder):
    """Sum rows across all CSV files in a folder."""
    total = 0
    files = sorted(Path(folder).glob("*.csv"))
    for f in files:
        total += count_rows_fast(f)
    return total, len(files)


print("=" * 75)
print("DPKT vs NFSTREAM FLOW COUNT COMPARISON")
print("=" * 75)
print(f"{'Attack':<30} {'NFS flows':>12} {'DPKT flows':>12} {'Match %':>9} {'Status'}")
print("-" * 75)

results = []

# Find all attack folders that exist in DPKT_CSV/
if not DPKT_ROOT.exists():
    print(f"\nERROR: {DPKT_ROOT} does not exist.")
    print("Run 01b_extract_dpkt_features.py for each attack first.")
    raise SystemExit(1)

dpkt_attacks = sorted([d.name for d in DPKT_ROOT.iterdir() if d.is_dir()])

for attack in dpkt_attacks:
    nfs_folder  = NFSTREAM_ROOT / attack
    dpkt_folder = DPKT_ROOT / attack

    if not nfs_folder.exists():
        print(f"  {attack:<30} {'???':>12} {'???':>12} {'N/A':>9}  "
              f"WARNING: no NFSTREAM_CSV folder found")
        continue

    nfs_rows,  nfs_files  = count_rows_in_folder(nfs_folder)
    dpkt_rows, dpkt_files = count_rows_in_folder(dpkt_folder)

    if nfs_rows == 0:
        print(f"  {attack:<30} {'0':>12} {dpkt_rows:>12,} {'N/A':>9}  "
              f"WARNING: NFStream folder empty")
        continue

    match_pct = 100.0 * dpkt_rows / nfs_rows

    if dpkt_files < nfs_files:
        status = f"INCOMPLETE — only {dpkt_files}/{nfs_files} DPKT files extracted"
    elif match_pct >= 95:
        status = "OK"
    elif match_pct >= 85:
        status = "ACCEPTABLE (minor edge-case gap)"
    else:
        status = "LOW — investigate before merging"

    print(f"  {attack:<30} {nfs_rows:>12,} {dpkt_rows:>12,} {match_pct:>8.1f}%  {status}")

    results.append({
        "attack": attack,
        "nfs_rows": nfs_rows,
        "dpkt_rows": dpkt_rows,
        "match_pct": match_pct,
        "nfs_files": nfs_files,
        "dpkt_files": dpkt_files,
    })

print("-" * 75)

if results:
    total_nfs  = sum(r["nfs_rows"]  for r in results)
    total_dpkt = sum(r["dpkt_rows"] for r in results)
    overall    = 100.0 * total_dpkt / total_nfs if total_nfs > 0 else 0
    print(f"  {'OVERALL':<30} {total_nfs:>12,} {total_dpkt:>12,} {overall:>8.1f}%")

print("=" * 75)

# Summary of attacks with issues
incomplete = [r for r in results if r["dpkt_files"] < r["nfs_files"]]
low_match  = [r for r in results if r["match_pct"] < 85 and r["dpkt_files"] >= r["nfs_files"]]

if incomplete:
    print(f"\nINCOMPLETE extractions ({len(incomplete)} attacks):")
    print("Run 01b_extract_dpkt_features.py <attack_name> for each:")
    for r in incomplete:
        missing = r["nfs_files"] - r["dpkt_files"]
        print(f"  {r['attack']:<30} missing {missing} DPKT file(s)")

if low_match:
    print(f"\nLOW MATCH RATE ({len(low_match)} attacks) — investigate:")
    for r in low_match:
        print(f"  {r['attack']:<30} {r['match_pct']:.1f}%")

# Also show which NFSTREAM attacks are NOT yet in DPKT at all
all_nfs_attacks = sorted([d.name for d in NFSTREAM_ROOT.iterdir() if d.is_dir()])
not_started = [a for a in all_nfs_attacks if a not in dpkt_attacks]
if not_started:
    print(f"\nNOT YET EXTRACTED with DPKT ({len(not_started)} attacks):")
    print("Run 01b_extract_dpkt_features.py <attack_name> for each:")
    for a in not_started:
        print(f"  {a}")

if not incomplete and not low_match and not not_started:
    print("\nAll attacks extracted and match rates look good.")
    print("Safe to run 04_create_balanced_attack_files.py on all 24 classes.")