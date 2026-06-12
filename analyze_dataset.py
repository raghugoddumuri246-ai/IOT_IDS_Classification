"""
==========================================================
SCRIPT : ANALYZE RAW PCAP DATASET FOLDER
==========================================================

PURPOSE:
This is the FIRST script you run, before any NFStream
processing. It scans CIC_IOT_PCAP/ (the folder where you
downloaded raw CICIoT2023 PCAP files, organised into one
subfolder per attack category) and reports:

  - How many attack category folders exist
  - How many .pcap files are in each folder
  - The filenames of (up to) the first 10 files per folder,
    so you can sanity-check naming conventions before
    running 01_extract_flows.py

EXPECTED FOLDER STRUCTURE:
  CIC_IOT_PCAP/
  ├── Benign_Final/
  │     BenignTraffic.pcap, BenignTraffic1.pcap, ...
  ├── DDoS-SYN_Flood/
  │     DDoS-SYN_Flood1.pcap, DDoS-SYN_Flood2.pcap, ...
  ├── Mirai-greip_flood/
  │     ...
  └── ...

The folder NAMES here must exactly match the ATTACKS list
in 04_create_balanced_attack_files.py and the attack_name
argument passed to 01_extract_flows.py.
==========================================================
"""

from pathlib import Path

ROOT = Path("./CIC_IOT_PCAP")

print("=" * 80)
print("RAW PCAP DATASET ANALYSIS")
print("=" * 80)

if not ROOT.exists():
    print(f"\nERROR: {ROOT} does not exist.")
    print("Make sure raw PCAP files are organised under CIC_IOT_PCAP/<attack_name>/")
    raise SystemExit(1)

total_pcaps   = 0
folder_counts = []

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir():
        continue

    pcap_files = list(folder.glob("*.pcap"))

    print(f"\nFolder : {folder.name}")
    print(f"PCAPs  : {len(pcap_files)}")

    total_pcaps += len(pcap_files)
    folder_counts.append((folder.name, len(pcap_files)))

    # Show up to 10 example filenames so you can verify
    # naming conventions look as expected
    for pcap in sorted(pcap_files)[:10]:
        print("   ", pcap.name)

    if len(pcap_files) > 10:
        print(f"    ... and {len(pcap_files) - 10} more")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for name, count in folder_counts:
    print(f"  {name:<25} : {count:>4} pcap files")

print(f"\n  TOTAL FOLDERS : {len(folder_counts)}")
print(f"  TOTAL PCAPS   : {total_pcaps}")
print("=" * 80)