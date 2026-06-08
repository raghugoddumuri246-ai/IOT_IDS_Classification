from pathlib import Path
from nfstream import NFStreamer
import sys

attack_name = sys.argv[1]

INPUT_ROOT = Path("CIC_IOT_PCAP")
OUTPUT_ROOT = Path("NFSTREAM_CSV")

input_folder = INPUT_ROOT / attack_name
output_folder = OUTPUT_ROOT / attack_name

output_folder.mkdir(parents=True, exist_ok=True)

pcaps = list(input_folder.glob("*.pcap"))

print(f"\nAttack: {attack_name}")
print(f"Total PCAPs: {len(pcaps)}")

for i, pcap in enumerate(pcaps, start=1):

    output_csv = output_folder / f"{pcap.stem}.csv"

    if output_csv.exists():
        print(f"[{i}/{len(pcaps)}] Skip: {pcap.name}")
        continue

    print(f"[{i}/{len(pcaps)}] Processing: {pcap.name}")

    NFStreamer(
        source=str(pcap),
        statistical_analysis=True
    ).to_csv(
        path=str(output_csv)
    )

    print(f"Saved: {output_csv.name}")

print("\nDone.")


#to run : 

# python 01_extract_flows.py Bemign_Final
# python 01_extract_flows.py DDoS_SYN_Flood
# python 01_extract_flows.py DoS_SYN_Flood



