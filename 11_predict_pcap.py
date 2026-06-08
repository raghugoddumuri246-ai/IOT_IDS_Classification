import sys
import pandas as pd
import joblib

from nfstream import NFStreamer

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "xgb_model.pkl"

# =====================================================
# FINAL V4 FEATURES (59)
# =====================================================

FEATURES = [
    'protocol',
    'ip_version',
    'bidirectional_duration_ms',
    'bidirectional_packets',
    'bidirectional_bytes',
    'src2dst_duration_ms',
    'src2dst_packets',
    'src2dst_bytes',
    'dst2src_duration_ms',
    'dst2src_packets',
    'dst2src_bytes',
    'bidirectional_min_ps',
    'bidirectional_mean_ps',
    'bidirectional_stddev_ps',
    'bidirectional_max_ps',
    'src2dst_min_ps',
    'src2dst_mean_ps',
    'src2dst_stddev_ps',
    'src2dst_max_ps',
    'dst2src_min_ps',
    'dst2src_mean_ps',
    'dst2src_stddev_ps',
    'dst2src_max_ps',
    'bidirectional_min_piat_ms',
    'bidirectional_mean_piat_ms',
    'bidirectional_stddev_piat_ms',
    'bidirectional_max_piat_ms',
    'src2dst_min_piat_ms',
    'src2dst_mean_piat_ms',
    'src2dst_stddev_piat_ms',
    'src2dst_max_piat_ms',
    'dst2src_min_piat_ms',
    'dst2src_mean_piat_ms',
    'dst2src_stddev_piat_ms',
    'dst2src_max_piat_ms',
    'bidirectional_syn_packets',
    'bidirectional_cwr_packets',
    'bidirectional_ece_packets',
    'bidirectional_urg_packets',
    'bidirectional_ack_packets',
    'bidirectional_psh_packets',
    'bidirectional_rst_packets',
    'bidirectional_fin_packets',
    'src2dst_syn_packets',
    'src2dst_cwr_packets',
    'src2dst_ece_packets',
    'src2dst_urg_packets',
    'src2dst_ack_packets',
    'src2dst_psh_packets',
    'src2dst_rst_packets',
    'src2dst_fin_packets',
    'dst2src_syn_packets',
    'dst2src_cwr_packets',
    'dst2src_ece_packets',
    'dst2src_urg_packets',
    'dst2src_ack_packets',
    'dst2src_psh_packets',
    'dst2src_rst_packets',
    'dst2src_fin_packets'
]

# =====================================================
# IDS LABEL MAP (10 CLASSES)
# =====================================================

label_map = {
    0: "Benign",

    1: "DDoS-ICMP",
    2: "DDoS-SYN",
    3: "DDoS-TCP",
    4: "DDoS-UDP",

    5: "DoS-SYN",
    6: "DoS-TCP",
    7: "DoS-UDP",

    8: "Mirai",
    9: "Mirai",
    10: "Mirai",

    11: "Recon",
    12: "Recon",
    13: "Recon"
}

# =====================================================
# ORIGINAL 14-CLASS LABEL MAP
# =====================================================

fine_label_map = {
    0: "Benign",

    1: "DDoS-ICMP_Flood",
    2: "DDoS-SYN_Flood",
    3: "DDoS-TCP_Flood",
    4: "DDoS-UDP_Flood",

    5: "DoS-SYN_Flood",
    6: "DoS-TCP_Flood",
    7: "DoS-UDP_Flood",

    8: "Mirai-greeth_flood",
    9: "Mirai-greip_flood",
    10: "Mirai-udpplain",

    11: "Recon-HostDiscovery",
    12: "Recon-OSScan",
    13: "Recon-PortScan"
}

# =====================================================
# INPUT CHECK
# =====================================================

pcap_file = "CIC_IOT_PCAP/DoS-UDP_Flood/DoS-UDP_Flood1.pcap"

print("\nLoading PCAP...")
print(pcap_file)

# =====================================================
# NFSTREAM EXTRACTION
# =====================================================

print("\nExtracting Flows...")

streamer = NFStreamer(
    source=pcap_file,
    statistical_analysis=True
)

df = streamer.to_pandas()

print("Flows:", len(df))

# =====================================================
# FEATURE CHECK
# =====================================================

missing = [
    col
    for col in FEATURES
    if col not in df.columns
]

if len(missing) > 0:
    print("\nMissing Features Found:")
    print(missing)
    sys.exit()

# =====================================================
# PREPARE MODEL INPUT
# =====================================================

X = df[FEATURES].copy()
X = X.fillna(0)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Model...")

model = joblib.load(MODEL_PATH)

# =====================================================
# PREDICT
# =====================================================

print("\nPredicting...")

preds = model.predict(X)

# =====================================================
# CONVERT TO IDS CLASSES
# =====================================================

pred_labels = [
    label_map[int(x)]
    for x in preds
]

# =====================================================
# CONVERT TO ORIGINAL SUBTYPES
# =====================================================

fine_labels = [
    fine_label_map[int(x)]
    for x in preds
]

# =====================================================
# DISTRIBUTION
# =====================================================

dist = pd.Series(pred_labels).value_counts()
fine_dist = pd.Series(fine_labels).value_counts()

total = len(pred_labels)

# =====================================================
# REPORT
# =====================================================

print("\n" + "=" * 60)
print("PCAP ANALYSIS REPORT")
print("=" * 60)

print("\nTotal Flows:", total)

print("\nPrediction Distribution:\n")

for label, count in dist.items():
    percent = (count / total) * 100
    print(f"{label:15s}{count:10,d} ({percent:.2f}%)")

# =====================================================
# FINAL DECISION
# =====================================================

top_class = dist.index[0]
top_count = dist.iloc[0]

confidence = (top_count / total) * 100

# =====================================================
# SUBTYPE DETECTION
# =====================================================

subtype_name = None
subtype_conf = None

if top_class == "Mirai":

    mirai_subtypes = [
        "Mirai-greeth_flood",
        "Mirai-greip_flood",
        "Mirai-udpplain"
    ]

    mirai_counts = {
        k: v
        for k, v in fine_dist.items()
        if k in mirai_subtypes
    }

    subtype_name = max(mirai_counts, key=mirai_counts.get)

    subtype_conf = (
        mirai_counts[subtype_name] /
        sum(mirai_counts.values())
    ) * 100

elif top_class == "Recon":

    recon_subtypes = [
        "Recon-HostDiscovery",
        "Recon-OSScan",
        "Recon-PortScan"
    ]

    recon_counts = {
        k: v
        for k, v in fine_dist.items()
        if k in recon_subtypes
    }

    subtype_name = max(recon_counts, key=recon_counts.get)

    subtype_conf = (
        recon_counts[subtype_name] /
        sum(recon_counts.values())
    ) * 100

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n" + "=" * 60)

print("FINAL DECISION :", top_class)
print("CONFIDENCE     :", round(confidence, 2), "%")

if subtype_name is not None:
    print()
    print("MOST LIKELY SUBTYPE :", subtype_name)
    print("SUBTYPE CONFIDENCE  :", round(subtype_conf, 2), "%")

print("=" * 60)
