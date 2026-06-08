# analyze_dataset.py

from pathlib import Path

ROOT = Path("./CIC_IOT_PCAP")

print("="*80)
print("DATASET ANALYSIS")
print("="*80)

total_pcaps = 0

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir():
        continue

    pcap_files = list(folder.glob("*.pcap"))

    print(f"\nFolder : {folder.name}")
    print(f"PCAPs  : {len(pcap_files)}")

    total_pcaps += len(pcap_files)

    for pcap in pcap_files[:10]:
        print("   ", pcap.name)

print("\n")
print("="*80)
print("TOTAL PCAPS :", total_pcaps)
print("="*80)
