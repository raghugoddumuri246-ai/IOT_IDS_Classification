"""
Standalone evaluation script.
Loads saved MLP and ADST models and runs full evaluation
on the complete val and test sets (216,000 rows each).

Usage:
  python evaluate_models.py
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

TRAINING_DIR = "TRAINING_DATA"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# SHARED PREPROCESSING
# ==========================================================

def load_and_transform(csv_path, feature_order):
    print(f"  Loading {csv_path}...")
    df  = pd.read_csv(csv_path)
    y   = torch.tensor(df["label"].values, dtype=torch.long)
    X   = torch.tensor(df[feature_order].values.astype("float32"))
    X   = torch.asinh(X)
    X   = torch.clamp(X, -15.0, 15.0)
    return X, y


# ==========================================================
# MLP ARCHITECTURE  (must match 08b_mlp_flat_baseline.py)
# ==========================================================

class FlatMLP(nn.Module):
    def __init__(self, n_features, hidden_dims, dropouts, n_classes):
        super().__init__()
        layers = []
        in_dim = n_features
        for h, d in zip(hidden_dims, dropouts):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(d)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(in_dim, n_classes)
        # Must match training script exactly — residual_proj was saved
        self.residual_proj = nn.Linear(n_features, hidden_dims[1]) \
            if len(hidden_dims) > 1 and hidden_dims[0] == hidden_dims[1] \
            else None

    def forward(self, x):
        return self.head(self.backbone(x))


# ==========================================================
# ADST ARCHITECTURE  (must match 08c_adst_transformer.py)
# ==========================================================

SEMANTIC_GROUPS = {
    "flow_volume": [
        "protocol","ip_version",
        "bidirectional_duration_ms","bidirectional_packets","bidirectional_bytes",
        "src2dst_duration_ms","src2dst_packets","src2dst_bytes",
        "dst2src_duration_ms","dst2src_packets","dst2src_bytes",
        "bidirectional_min_ps","bidirectional_mean_ps","bidirectional_stddev_ps","bidirectional_max_ps",
        "src2dst_min_ps","src2dst_mean_ps","src2dst_stddev_ps","src2dst_max_ps",
        "dst2src_min_ps","dst2src_mean_ps","dst2src_stddev_ps","dst2src_max_ps",
        "bidirectional_min_piat_ms","bidirectional_mean_piat_ms","bidirectional_stddev_piat_ms","bidirectional_max_piat_ms",
        "src2dst_min_piat_ms","src2dst_mean_piat_ms","src2dst_stddev_piat_ms","src2dst_max_piat_ms",
        "dst2src_min_piat_ms","dst2src_mean_piat_ms","dst2src_stddev_piat_ms","dst2src_max_piat_ms",
        "application_name","application_category_name","application_confidence",
        "dpkt_packet_count",
        "dpkt_ip_total_len_min","dpkt_ip_total_len_mean","dpkt_ip_total_len_std","dpkt_ip_total_len_max",
    ],
    "tcp_flags": [
        "bidirectional_syn_packets","bidirectional_cwr_packets","bidirectional_ece_packets",
        "bidirectional_urg_packets","bidirectional_ack_packets","bidirectional_psh_packets",
        "bidirectional_rst_packets","bidirectional_fin_packets",
        "src2dst_syn_packets","src2dst_cwr_packets","src2dst_ece_packets",
        "src2dst_urg_packets","src2dst_ack_packets","src2dst_psh_packets",
        "src2dst_rst_packets","src2dst_fin_packets",
        "dst2src_syn_packets","dst2src_cwr_packets","dst2src_ece_packets",
        "dst2src_urg_packets","dst2src_ack_packets","dst2src_psh_packets",
        "dst2src_rst_packets","dst2src_fin_packets",
    ],
    "fragmentation": [
        "dpkt_frag_mf_count","dpkt_frag_df_count","dpkt_frag_offset_nonzero_count",
        "dpkt_frag_ratio","dpkt_ip_options_present_count",
    ],
    "gre_header": [
        "dpkt_gre_packet_count","dpkt_gre_inner_proto_ip_count","dpkt_gre_inner_proto_ether_count",
        "dpkt_gre_ratio","dpkt_gre_inner_ip_ratio","dpkt_gre_inner_ether_ratio",
        "dpkt_ttl_min","dpkt_ttl_mean","dpkt_ttl_std","dpkt_ttl_max",
        "dpkt_tcp_window_min","dpkt_tcp_window_mean","dpkt_tcp_window_std","dpkt_tcp_window_max",
        "dpkt_header_byte_entropy",
    ],
    "recon_cardinality": [
        "dpkt_tcp_null_scan_count","dpkt_tcp_fin_scan_count","dpkt_tcp_xmas_scan_count",
        "dpkt_icmp_echo_request_count","dpkt_icmp_echo_reply_count","dpkt_icmp_other_count",
        "fan_in_src_count","fan_out_port_count","fan_out_ip_count",
        "fan_out_proto_count","fan_out_scope","fan_app_diversity",
    ],
    "temporal_splt": [
        *[f"splt_direction_{i}" for i in range(10)],
        *[f"splt_ps_{i}" for i in range(10)],
        *[f"splt_piat_ms_{i}" for i in range(10)],
    ],
}
GLOBAL_CONTEXT_FEATURE = "file_frag_rate"
GROUP_NAMES = list(SEMANTIC_GROUPS.keys()) + ["global_context"]


def make_mlp_encoder(in_dim, hidden_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
        nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class SPLTConvEncoder(nn.Module):
    def __init__(self, d_token, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.proj  = nn.Linear(64, d_token)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.shape[0], 3, 10)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.proj(self.drop(x))


class ADSTTransformer(nn.Module):
    def __init__(self, group_sizes, d_token, n_heads, n_layers,
                 d_ff, n_classes, dropout):
        super().__init__()
        def hidden(n): return max(16, min(128, n * 2))
        self.encoders = nn.ModuleDict({
            "flow_volume":       make_mlp_encoder(group_sizes["flow_volume"],       hidden(group_sizes["flow_volume"]),       d_token, dropout),
            "tcp_flags":         make_mlp_encoder(group_sizes["tcp_flags"],         hidden(group_sizes["tcp_flags"]),         d_token, dropout),
            "fragmentation":     make_mlp_encoder(group_sizes["fragmentation"],     hidden(group_sizes["fragmentation"]),     d_token, dropout),
            "gre_header":        make_mlp_encoder(group_sizes["gre_header"],        hidden(group_sizes["gre_header"]),        d_token, dropout),
            "recon_cardinality": make_mlp_encoder(group_sizes["recon_cardinality"], hidden(group_sizes["recon_cardinality"]), d_token, dropout),
        })
        self.splt_encoder     = SPLTConvEncoder(d_token, dropout)
        self.context_encoder  = nn.Sequential(nn.Linear(1, d_token), nn.GELU())
        self.group_id_embed   = nn.Embedding(7, d_token)
        self.cls_token        = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_token, n_classes))

    def forward(self, groups, context):
        tokens = [self.encoders[g](groups[g])
                  for g in ["flow_volume","tcp_flags","fragmentation",
                             "gre_header","recon_cardinality"]]
        tokens.append(self.splt_encoder(groups["temporal_splt"]))
        tokens.append(self.context_encoder(context.unsqueeze(-1)))
        x = torch.stack(tokens, dim=1)
        ids = torch.arange(7, device=x.device)
        x   = x + self.group_id_embed(ids).unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.norm(self.transformer(x))
        return self.head(x[:, 0, :])

    @torch.no_grad()
    def get_group_attention_per_class(self, groups, context):
        tokens = [self.encoders[g](groups[g])
                  for g in ["flow_volume","tcp_flags","fragmentation",
                             "gre_header","recon_cardinality"]]
        tokens.append(self.splt_encoder(groups["temporal_splt"]))
        tokens.append(self.context_encoder(context.unsqueeze(-1)))
        x = torch.stack(tokens, dim=1)
        ids = torch.arange(7, device=x.device)
        x   = x + self.group_id_embed(ids).unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        layer = self.transformer.layers[0]
        xn = layer.norm1(x)
        _, attn = layer.self_attn(xn, xn, xn,
                                  need_weights=True,
                                  average_attn_weights=True)
        # CLS row, columns 1-7 (the group tokens)
        return attn[:, 0, 1:]   # (batch, 7)


# ==========================================================
# EVALUATION HELPER
# ==========================================================

@torch.no_grad()
def eval_model(model, loader, device, class_names, model_type="mlp",
               feat_idx=None, semantic_groups=None, global_ctx_idx=None):
    model.eval()
    all_preds, all_labels = [], []
    # For ADST: collect per-class attention
    attn_per_class = {i: [] for i in range(len(class_names))} if model_type == "adst" else None

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if model_type == "mlp":
            logits = model(X)
        else:
            # Split into groups
            groups  = {g: X[:, [feat_idx[f] for f in feats]]
                       for g, feats in semantic_groups.items()}
            context = X[:, global_ctx_idx]
            logits  = model(groups, context)
            # Collect attention
            attn = model.get_group_attention_per_class(groups, context)
            preds_batch = logits.argmax(1)
            for i, (pred, label) in enumerate(zip(preds_batch.cpu(), y.cpu())):
                attn_per_class[label.item()].append(attn[i].cpu().numpy())

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=4, zero_division=0)
    recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)

    return acc, report, recalls, attn_per_class


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 65)
    print("FINAL EVALUATION — MLP and ADST on full test sets")
    print("=" * 65)

    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)
    feat_idx      = {f: i for i, f in enumerate(feature_order)}
    use_amp       = DEVICE.type == "cuda"

    print(f"\nDevice: {DEVICE},  Classes: {n_classes},  Features: {len(feature_order)}")

    # Load data once, share between both models
    print("\nLoading and transforming datasets...")
    X_val,  y_val  = load_and_transform(f"{TRAINING_DIR}/val.csv",  feature_order)
    X_test, y_test = load_and_transform(f"{TRAINING_DIR}/test.csv", feature_order)

    val_loader  = DataLoader(TensorDataset(X_val,  y_val),
                             batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=1024, shuffle=False, num_workers=0)

    # ----------------------------------------------------------
    # MLP EVALUATION
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("FLAT MLP BASELINE — FULL EVALUATION")
    print("=" * 65)

    mlp_ckpt = torch.load("mlp_flat_model.pt", map_location=DEVICE,
                          weights_only=False)
    mlp = FlatMLP(len(feature_order), [512,512,256,128], [0.3,0.3,0.2,0.1],
                  n_classes).to(DEVICE)
    mlp.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp.eval()

    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    val_acc_mlp, _, _, _ = eval_model(mlp, val_loader, DEVICE, class_names, "mlp")
    print(f"MLP Validation Accuracy (full 216k): {val_acc_mlp*100:.2f}%")

    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    test_acc_mlp, test_report_mlp, recalls_mlp, _ = eval_model(
        mlp, test_loader, DEVICE, class_names, "mlp")
    print(f"MLP Test Accuracy (full 216k):       {test_acc_mlp*100:.2f}%")
    print()
    print(test_report_mlp)

    # ----------------------------------------------------------
    # ADST EVALUATION
    # ----------------------------------------------------------
    print("=" * 65)
    print("ADST TRANSFORMER — FULL EVALUATION")
    print("=" * 65)

    adst_ckpt   = torch.load("adst_model.pt", map_location=DEVICE,
                             weights_only=False)
    group_sizes = {g: len(feats) for g, feats in SEMANTIC_GROUPS.items()}
    adst = ADSTTransformer(group_sizes, d_token=64, n_heads=4, n_layers=2,
                           d_ff=256, n_classes=n_classes, dropout=0.1).to(DEVICE)
    adst.load_state_dict(adst_ckpt["model_state_dict"])
    adst.eval()

    global_ctx_idx = feat_idx[GLOBAL_CONTEXT_FEATURE]

    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    val_acc_adst, _, _, _ = eval_model(
        adst, val_loader, DEVICE, class_names, "adst",
        feat_idx, SEMANTIC_GROUPS, global_ctx_idx)
    print(f"ADST Validation Accuracy (full 216k): {val_acc_adst*100:.2f}%")

    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    test_acc_adst, test_report_adst, recalls_adst, attn_per_class = eval_model(
        adst, test_loader, DEVICE, class_names, "adst",
        feat_idx, SEMANTIC_GROUPS, global_ctx_idx)
    print(f"ADST Test Accuracy (full 216k):       {test_acc_adst*100:.2f}%")
    print()
    print(test_report_adst)

    # ----------------------------------------------------------
    # THREE-WAY COMPARISON TABLE
    # ----------------------------------------------------------
    xgb_recalls = {
        "Benign_Final":0.903,"DDoS-ACK_Fragmentation":1.000,"DDoS-HTTP_Flood":1.000,
        "DDoS-ICMP_Flood":0.703,"DDoS-ICMP_Fragmentation":1.000,"DDoS-PSHACK_Flood":0.996,
        "DDoS-RSTFINFlood":0.995,"DDoS-SYN_Flood":0.993,"DDoS-SlowLoris":1.000,
        "DDoS-SynonymousIP_Flood":0.974,"DDoS-TCP_Flood":0.984,"DDoS-UDP_Flood":0.797,
        "DDoS-UDP_Fragmentation":1.000,"DoS-HTTP_Flood":0.901,"DoS-SYN_Flood":0.953,
        "DoS-TCP_Flood":0.971,"DoS-UDP_Flood":0.908,"Mirai-greeth_flood":0.632,
        "Mirai-greip_flood":0.681,"Mirai-udpplain":0.636,"Recon-HostDiscovery":0.934,
        "Recon-OSScan":0.811,"Recon-PortScan":0.843,"VulnerabilityScan":1.000,
    }

    print("=" * 80)
    print("THREE-WAY RECALL COMPARISON")
    print("=" * 80)
    print(f"{'Class':<28} {'XGB':>7} {'MLP':>7} {'ADST':>7}  {'A>M':>6}  {'A>X':>6}")
    print("-" * 65)

    adst_beats_mlp = adst_beats_xgb = 0
    for i, cls in enumerate(class_names):
        x = xgb_recalls.get(cls, 0)
        m = recalls_mlp[i]
        a = recalls_adst[i]
        am = f"+{(a-m)*100:.1f}%" if a > m else f"{(a-m)*100:.1f}%"
        ax = f"+{(a-x)*100:.1f}%" if a > x else f"{(a-x)*100:.1f}%"
        if a > m: adst_beats_mlp += 1
        if a > x: adst_beats_xgb += 1
        print(f"  {cls:<26} {x:>7.3f} {m:>7.3f} {a:>7.3f}  {am:>6}  {ax:>6}")

    print()
    print(f"  XGBoost accuracy:  90.06%")
    print(f"  MLP accuracy:      {test_acc_mlp*100:.2f}%")
    print(f"  ADST accuracy:     {test_acc_adst*100:.2f}%")
    print(f"  ADST beats MLP:    {adst_beats_mlp}/24 classes")
    print(f"  ADST beats XGBoost:{adst_beats_xgb}/24 classes")

    # ----------------------------------------------------------
    # GROUP ATTENTION PER ATTACK CLASS (key paper visualization)
    # ----------------------------------------------------------
    print()
    print("=" * 65)
    print("GROUP ATTENTION PER ATTACK CLASS")
    print("=" * 65)
    print(f"  {'Class':<28}", end="")
    for g in GROUP_NAMES:
        short = g[:6]
        print(f" {short:>7}", end="")
    print()
    print("  " + "-" * 85)

    attn_matrix = []
    for i, cls in enumerate(class_names):
        attns = attn_per_class.get(i, [])
        if attns:
            mean_attn = np.stack(attns).mean(axis=0)
            mean_attn = mean_attn / mean_attn.sum()
        else:
            mean_attn = np.ones(7) / 7
        attn_matrix.append(mean_attn)
        print(f"  {cls:<28}", end="")
        for v in mean_attn:
            print(f" {v:>7.4f}", end="")
        print()

    # Save attention matrix
    attn_df = pd.DataFrame(
        attn_matrix, index=class_names, columns=GROUP_NAMES)
    attn_df.to_csv("adst_per_class_attention.csv")
    print(f"\nSaved per-class attention -> adst_per_class_attention.csv")

    # Highlight top group per class
    print()
    print("DOMINANT GROUP PER ATTACK CLASS:")
    for cls, attn in zip(class_names, attn_matrix):
        top_idx = np.argmax(attn)
        print(f"  {cls:<30} -> {GROUP_NAMES[top_idx]:<20} ({attn[top_idx]:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()