from pathlib import Path
from nfstream import NFStreamer
import sys

# ==========================================================
# USAGE:
#   python 01_extract_flows.py Benign_Final
#   python 01_extract_flows.py DDoS-SYN_Flood
#   python 01_extract_flows.py Mirai-greip_flood
# ==========================================================

attack_name = sys.argv[1]

INPUT_ROOT  = Path("CIC_IOT_PCAP")
OUTPUT_ROOT = Path("NFSTREAM_CSV")

input_folder  = INPUT_ROOT  / attack_name
output_folder = OUTPUT_ROOT / attack_name

output_folder.mkdir(parents=True, exist_ok=True)

pcaps = list(input_folder.glob("*.pcap"))

print(f"\nAttack : {attack_name}")
print(f"PCAPs  : {len(pcaps)}")

for i, pcap in enumerate(pcaps, start=1):

    output_csv = output_folder / f"{pcap.stem}.csv"

    if output_csv.exists():
        print(f"[{i}/{len(pcaps)}] Skip : {pcap.name}")
        continue

    print(f"[{i}/{len(pcaps)}] Processing : {pcap.name}")

    # ----------------------------------------------------------
    # NFStreamer configuration
    #
    # statistical_analysis=True
    #   Enables per-direction packet size stats and inter-arrival
    #   time stats (min/mean/stddev/max for piat and ps).
    #   These are the most discriminating features.
    #
    # splt_analysis=10
    #   Captures the first 10 packet sizes and directions per flow.
    #   Helps distinguish Mirai GRE variants from UDP Plain because
    #   GRE encapsulation produces a fixed outer packet size pattern.
    #   Also helps separate Recon subtypes (ICMP vs TCP probes have
    #   very different initial packet sizes).
    #
    # n_dissections=20
    #   Runs DPI for up to 20 packets to identify the application.
    #   Produces application_name (e.g. "GRE", "ICMP", "DNS") and
    #   application_category_name. These are behavioural because
    #   they reflect the protocol used by the attack, not the IP.
    #   Keeping n_dissections at default (non-zero) is important
    #   for Mirai vs Recon separation.
    #
    # udps.entropy=True is NOT enabled because it adds significant
    # processing time and the entropy values are not consistent
    # across different PCAP sources (cross-dataset problem).
    # ----------------------------------------------------------

    NFStreamer(
        source=str(pcap),
        statistical_analysis=True,
        splt_analysis=10,
        n_dissections=20
    ).to_csv(
        path=str(output_csv)
    )

    print(f"  Saved : {output_csv.name}")

print("\nDone.")