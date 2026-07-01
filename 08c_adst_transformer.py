"""
==========================================================
SCRIPT 08c : ADST — ATTACK-DOMAIN SEMANTIC TOKENIZATION
             TRANSFORMER
==========================================================

PIPELINE POSITION:
  06_splitting_master_dataset.py  ->  [THIS SCRIPT]
  (same train/val/test splits as XGBoost and MLP baseline)

PAPER CONTRIBUTION:
  This is the proposed method. The core idea:
  Instead of feeding all 130 features as a flat vector
  (MLP baseline) or as 130 individual tokens (flat
  FT-Transformer), features are first GROUPED by attack
  domain semantics, then each group is encoded into a
  learned embedding (token), and these 7 tokens are
  processed by a Transformer that learns inter-group
  interactions.

  The hypothesis: a Transformer can learn attack behavior
  more effectively from 7 semantically-organized tokens
  than from 130 independent flat features, because:
  (1) Intra-group encoders learn co-activation patterns
      within each attack behavior domain
  (2) Inter-group Transformer attention learns which
      domains are jointly active for each attack class
  (3) The global context token (file_frag_rate) enables
      conditional reasoning that flat models cannot do

ARCHITECTURE — 7 TOKENS:
  Token 1 (G1 — Flow Volume, 43 features):
    MLP encoder: 43 -> 128 -> d_token
    Covers: flow duration, packet counts, byte counts,
    packet size stats, inter-arrival time stats,
    application type, IP packet length stats
    Target classes: SlowLoris (slow flow) vs SYN_Flood
    (high packet rate) vs HTTP_Flood (application=HTTP)

  Token 2 (G2 — TCP Flag Behavior, 24 features):
    MLP encoder: 24 -> 64 -> d_token
    Covers: bidirectional/src2dst/dst2src SYN/ACK/FIN/
    RST/PSH/URG/CWR/ECE packet counts
    Target classes: SYN_Flood (all SYN), PSHACK (PSH+ACK),
    RSTFINFlood (RST+FIN), normal TCP (balanced flags)

  Token 3 (G3 — Fragmentation, 5 features):
    MLP encoder: 5 -> 16 -> d_token
    Covers: frag_mf_count, frag_df_count,
    frag_offset_nonzero_count, frag_ratio, ip_options
    Target classes: DDoS-ACK/ICMP/UDP_Fragmentation
    Note: tiny group but XGBoost rank #1, #3, #9

  Token 4 (G4 — GRE + Header, 15 features):
    MLP encoder: 15 -> 48 -> d_token
    Covers: GRE encapsulation features (packet count,
    inner protocol type ratios), TTL stats, TCP window
    stats, header byte entropy
    Target classes: Mirai-greeth (GRE-Ethernet inner)
    vs Mirai-greip (GRE-IP inner) vs Mirai-udpplain (no GRE)

  Token 5 (G5 — Recon + Cardinality, 12 features):
    MLP encoder: 12 -> 32 -> d_token
    Covers: TCP scan patterns (NULL/FIN/XMAS), ICMP types
    (echo request/reply/other), fan-in/fan-out cardinality
    Target classes: PortScan (high fan_out_port, null_scan)
    vs HostDiscovery (high fan_out_ip, icmp_echo)
    vs VulnScan (high fan_out_scope) vs DDoS (high fan_in)

  Token 6 (G6 — Temporal SPLT, 30 features):
    1D-Conv encoder: reshape to [10 timesteps x 3 channels]
    then Conv1D -> d_token
    The 30 SPLT features ARE a temporal sequence:
    10 timesteps x (direction, packet_size, inter-arrival)
    A 1D-Conv preserves this temporal structure, which
    MLP would flatten away.
    Target classes: SlowLoris (slow piat, regular direction)
    vs SYN_Flood (fast piat, one-directional)

  Token 7 (Global Context, 1 feature — file_frag_rate):
    Linear: 1 -> d_token
    file_frag_rate = fraction of ALL flows in this PCAP
    file with IP fragmentation flags set. This captures
    capture-level context that no per-flow feature can.
    As a standalone token, the Transformer can learn to
    ATTEND or IGNORE this context depending on other tokens:
    - If G4 (GRE) is active: ignore file_frag_rate
    - If G3 (Frag) is active AND file_frag_rate > 0.5:
      strongly confirm fragmentation attack
    This conditional logic is impossible in XGBoost/MLP.

OUTPUT:
  adst_model.pt
  adst_val_report.txt
  adst_test_report.txt
  adst_group_attention.csv  <- which semantic groups matter most
  adst_history.csv
==========================================================
"""

import os
import math
import time
import joblib
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ==========================================================
# SECTION 1: CONFIGURATION
# ==========================================================

TRAINING_DIR  = "TRAINING_DATA"
OUTPUT_PREFIX = "adst"

# Token dimension — each group encoder outputs a d_token-dim vector
# 7 tokens x d_token -> Transformer
D_TOKEN   = 64     # each semantic group token size
N_HEADS   = 4      # attention heads (D_TOKEN must be divisible)
N_LAYERS  = 2      # Transformer encoder blocks
D_FF      = 256    # FFN hidden dim inside Transformer block
DROPOUT   = 0.1

BATCH_SIZE    = 2048
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
PATIENCE      = 7

MAX_STEPS_PER_EPOCH = 500
MAX_EVAL_STEPS      = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================================
# SECTION 2: ATTACK-DOMAIN SEMANTIC GROUPS
# ==========================================================
# Each entry maps group_name -> ordered list of feature names.
# The ORDER within each group matches how they appear in the
# feature_order.pkl file — this is verified at dataset load time.
# file_frag_rate is handled separately as the global context token.

SEMANTIC_GROUPS = {
    "flow_volume": [
        # NFStream flow statistics + application type + IP length
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
        # Bidirectional + directional TCP flag counters
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
        # IP fragmentation flags and ratios (XGBoost ranks 1, 3, 9)
        "dpkt_frag_mf_count","dpkt_frag_df_count","dpkt_frag_offset_nonzero_count",
        "dpkt_frag_ratio","dpkt_ip_options_present_count",
    ],
    "gre_header": [
        # GRE encapsulation + TTL + TCP window (Mirai subtype signals)
        "dpkt_gre_packet_count","dpkt_gre_inner_proto_ip_count","dpkt_gre_inner_proto_ether_count",
        "dpkt_gre_ratio","dpkt_gre_inner_ip_ratio","dpkt_gre_inner_ether_ratio",
        "dpkt_ttl_min","dpkt_ttl_mean","dpkt_ttl_std","dpkt_ttl_max",
        "dpkt_tcp_window_min","dpkt_tcp_window_mean","dpkt_tcp_window_std","dpkt_tcp_window_max",
        "dpkt_header_byte_entropy",
    ],
    "recon_cardinality": [
        # TCP scan patterns + ICMP types + fan-in/fan-out (Recon signals)
        "dpkt_tcp_null_scan_count","dpkt_tcp_fin_scan_count","dpkt_tcp_xmas_scan_count",
        "dpkt_icmp_echo_request_count","dpkt_icmp_echo_reply_count","dpkt_icmp_other_count",
        "fan_in_src_count","fan_out_port_count","fan_out_ip_count",
        "fan_out_proto_count","fan_out_scope","fan_app_diversity",
    ],
    "temporal_splt": [
        # First-10-packet sequence: direction, size, inter-arrival
        # Treated as [10 timesteps x 3 channels] for 1D-Conv encoder
        *[f"splt_direction_{i}" for i in range(10)],
        *[f"splt_ps_{i}" for i in range(10)],
        *[f"splt_piat_ms_{i}" for i in range(10)],
    ],
}

GLOBAL_CONTEXT_FEATURE = "file_frag_rate"  # 7th standalone token

GROUP_NAMES = list(SEMANTIC_GROUPS.keys()) + ["global_context"]


# ==========================================================
# SECTION 3: DATASET
# ==========================================================

class ADSTDataset(Dataset):
    """
    Loads train/val/test CSV and organizes features into
    semantic groups + global context.

    Returns a tuple per sample:
      group_tensors: dict {group_name: tensor of group features}
      context:       scalar tensor (file_frag_rate)
      label:         integer class label
    """

    def __init__(self, csv_path, feature_order, semantic_groups,
                 global_context_feature):
        print(f"  Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Build index lookup: feature_name -> column position in feature_order
        feat_idx = {f: i for i, f in enumerate(feature_order)}

        # Load ALL features as one float tensor first, then slice by group
        X_np = df[feature_order].values.astype(np.float32)
        X    = torch.tensor(X_np, dtype=torch.float32)

        # CRITICAL FIX: replace hard clipping [-10,10] with arcsinh transform.
        #
        # WHY HARD CLIPPING BROKE SPARSE FEATURES:
        # Features like dpkt_frag_mf_count are 0 for ~87.5% of rows (21/24
        # classes never fragment) and large/varied (hundreds to 90,000+)
        # for the remaining ~12.5% (the 3 Fragmentation attack classes).
        # RobustScaler computes IQR from the WHOLE column — since the
        # median and Q1/Q3 are all 0 (most rows are 0), the IQR is ~0,
        # so RobustScaler's scaling factor explodes. ANY non-zero raw
        # value gets scaled to a huge number. After clipping to [-10,10],
        # 99.78% of the non-zero (informative!) values all collapse to
        # the SAME ceiling of 10.0 — destroying the ability to
        # distinguish "lightly fragmented" from "heavily fragmented"
        # flows. This is why the fragmentation token carried almost no
        # information and got the LOWEST attention weight despite being
        # XGBoost's #1 most important feature.
        #
        # ARCSINH FIX:
        # arcsinh(x) = ln(x + sqrt(x^2 + 1))
        # Behaves like the identity function near 0 (preserves fine-
        # grained distinctions for small/typical values) but compresses
        # large values LOGARITHMICALLY instead of clipping them to a
        # hard wall. A value of 100 and a value of 90,000 remain
        # DIFFERENT after arcsinh (unlike hard clipping where both
        # become 10.0). This preserves relative ordering information
        # that the sparse fragmentation/GRE/scan count features need.
        X = torch.asinh(X)

        # Light clipping AFTER arcsinh as a final safety net only —
        # arcsinh already compresses extreme values into a small range
        # (asinh(90000) ≈ 12.1), so this clip rarely triggers and never
        # collapses distinct values to the same number the way the
        # pre-transform hard clip did.
        X = torch.clamp(X, min=-15.0, max=15.0)

        clip_pct = ((X < -15) | (X > 15)).float().mean().item() * 100
        print(f"  arcsinh transform applied. Post-transform clip [-15,15]: "
              f"{clip_pct:.4f}% of values clipped (should be ~0%)")

        # Slice each semantic group
        self.group_data = {}
        for gname, feats in semantic_groups.items():
            idxs = [feat_idx[f] for f in feats]
            self.group_data[gname] = X[:, idxs]

        # Global context token (single feature)
        ctx_idx = feat_idx[global_context_feature]
        self.context = X[:, ctx_idx]   # (N,)

        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        groups  = {g: t[idx] for g, t in self.group_data.items()}
        context = self.context[idx]
        label   = self.labels[idx]
        return groups, context, label


def collate_adst(batch):
    groups_list, contexts, labels = zip(*batch)
    groups_batch = {
        g: torch.stack([s[g] for s in groups_list])
        for g in groups_list[0]
    }
    return groups_batch, torch.stack(contexts), torch.stack(labels)


# ==========================================================
# SECTION 4: GROUP ENCODERS
# ==========================================================

def make_mlp_encoder(in_dim, hidden_dim, out_dim, dropout=0.1):
    """
    Small MLP: in_dim -> hidden_dim -> out_dim
    BatchNorm for stable training on tabular features.
    Used for G1-G5.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class SPLTConvEncoder(nn.Module):
    """
    1D-Conv encoder for G6 (Temporal SPLT).

    Reshapes 30 flat features into [batch, 3 channels, 10 timesteps]:
      channel 0: splt_direction_0..9
      channel 1: splt_ps_0..9
      channel 2: splt_piat_ms_0..9

    Then applies 1D convolution to capture temporal patterns in
    the first 10 packets of each flow.

    WHY 1D-CONV instead of MLP for this group:
      SlowLoris: splt_piat_ms shows INCREASING inter-arrival times
                 (connection kept alive with slow headers)
      SYN_Flood: splt_piat_ms is near-zero (rapid fire packets)
      A convolutional filter naturally detects these patterns
      by looking at adjacent timesteps. An MLP on 30 flat numbers
      loses the timestep ordering information.
    """

    def __init__(self, d_token, dropout=0.1):
        super().__init__()
        # 3 input channels (dir, ps, piat), kernel_size=3 (look at 3 consecutive packets)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        # Global average pool across timesteps: (batch, 64, 10) -> (batch, 64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.proj  = nn.Linear(64, d_token)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, 30) — flattened SPLT features"""
        batch = x.shape[0]
        # Reshape: (batch, 30) -> (batch, 3, 10)
        # splt_direction_0..9 -> channel 0
        # splt_ps_0..9        -> channel 1
        # splt_piat_ms_0..9   -> channel 2
        x = x.view(batch, 3, 10)          # (batch, 3, 10)
        x = F.gelu(self.bn1(self.conv1(x)))  # (batch, 32, 10)
        x = F.gelu(self.bn2(self.conv2(x)))  # (batch, 64, 10)
        x = self.pool(x).squeeze(-1)          # (batch, 64)
        x = self.drop(x)
        return self.proj(x)                   # (batch, d_token)


# ==========================================================
# SECTION 5: ADST TRANSFORMER
# ==========================================================

class ADSTTransformer(nn.Module):
    """
    Attack-Domain Semantic Tokenization Transformer.

    Forward pass:
      1. Encode each of 6 semantic groups -> 6 tokens of shape d_token
      2. Encode global context (file_frag_rate) -> 1 token of shape d_token
      3. Stack: [G1, G2, G3, G4, G5, G6, CTX] -> (batch, 7, d_token)
      4. Apply N_LAYERS Transformer encoder blocks
         (self-attention across all 7 tokens)
      5. Mean-pool across 7 tokens -> (batch, d_token)
      6. Classification head -> (batch, n_classes)

    WHY MEAN-POOL instead of CLS token:
      With only 7 tokens (vs 131 for flat FT-Transformer), all 7
      tokens are informative — there is no "background" token that
      needs to be suppressed. Mean-pooling gives equal initial weight
      to all group representations, and the Transformer layers
      redistribute attention as needed. This also avoids the CLS
      token initialization issue that caused problems in the flat
      FT-Transformer (CLS started random and dominated early training).

    Inter-group attention examples the model should learn:
      - G3 (frag) HIGH + G7 (file_frag_rate) HIGH -> Fragmentation
      - G4 (GRE) ether_ratio HIGH + G1 (volume) active -> Mirai-greeth
      - G5 (recon) null_scan HIGH + fan_out_port HIGH -> Recon-PortScan
      - G2 (flags) SYN only + G5 fan_in HIGH -> DDoS-SYN_Flood
    """

    def __init__(self, group_sizes, d_token, n_heads, n_layers,
                 d_ff, n_classes, dropout):
        super().__init__()

        # ---- Group encoders ----
        # Smaller encoders — force the Transformer to do more work
        # by limiting group encoder capacity. Previously encoders
        # were too powerful and solved the task themselves.
        def hidden(n): return max(16, min(128, n * 2))

        self.encoders = nn.ModuleDict({
            "flow_volume":       make_mlp_encoder(group_sizes["flow_volume"],       hidden(group_sizes["flow_volume"]),       d_token, dropout),
            "tcp_flags":         make_mlp_encoder(group_sizes["tcp_flags"],         hidden(group_sizes["tcp_flags"]),         d_token, dropout),
            "fragmentation":     make_mlp_encoder(group_sizes["fragmentation"],     hidden(group_sizes["fragmentation"]),     d_token, dropout),
            "gre_header":        make_mlp_encoder(group_sizes["gre_header"],        hidden(group_sizes["gre_header"]),        d_token, dropout),
            "recon_cardinality": make_mlp_encoder(group_sizes["recon_cardinality"], hidden(group_sizes["recon_cardinality"]), d_token, dropout),
        })

        # G6 uses 1D-Conv for temporal structure
        self.splt_encoder = SPLTConvEncoder(d_token, dropout)

        # Global context: single scalar -> d_token
        self.context_encoder = nn.Sequential(
            nn.Linear(1, d_token),
            nn.GELU(),
        )

        # FIX 1: Learned group identity embeddings.
        # Each of the 7 groups gets a unique learnable vector added to
        # its token BEFORE the Transformer sees it. This breaks symmetry:
        # the Transformer can now distinguish "I am the fragmentation group"
        # from "I am the GRE group" even if their content is similar.
        # Without this, all 7 tokens are interchangeable and attention
        # collapses to uniform (1/7 = 0.1429 for all groups — exactly
        # what we observed). This is equivalent to positional encodings
        # in NLP Transformers but for semantic group identity.
        n_groups = 7  # 6 semantic groups + 1 global context
        self.group_id_embed = nn.Embedding(n_groups, d_token)

        # FIX 2: CLS token instead of mean-pool.
        # Mean-pool made the Transformer optional — if attention is
        # uniform, mean-pool gives the same result with or without
        # the Transformer (just averages group tokens). With a CLS token,
        # the model MUST use attention to aggregate group information into
        # the CLS representation. The CLS token has no content of its own
        # so it is forced to attend selectively to informative groups.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_token)

        # Classification head on CLS token output
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, groups, context):
        """
        groups:  dict {group_name: (batch, n_group_features)}
        context: (batch,) — file_frag_rate values

        Returns: (batch, n_classes) logits
        """
        tokens = []

        # Encode G1-G5 with MLP encoders
        for gname in ["flow_volume","tcp_flags","fragmentation",
                      "gre_header","recon_cardinality"]:
            tok = self.encoders[gname](groups[gname])  # (batch, d_token)
            tokens.append(tok)

        # Encode G6 with 1D-Conv
        tokens.append(self.splt_encoder(groups["temporal_splt"]))

        # Encode global context (file_frag_rate)
        ctx_tok = self.context_encoder(context.unsqueeze(-1))
        tokens.append(ctx_tok)

        # Stack: (batch, 7, d_token)
        x = torch.stack(tokens, dim=1)

        # FIX: Add group identity embeddings to break symmetry.
        # group_ids = [0, 1, 2, 3, 4, 5, 6] for the 7 groups.
        # Each group gets a unique learned vector added to its token.
        # The Transformer can now attend differently to different groups.
        batch_size = x.shape[0]
        group_ids  = torch.arange(7, device=x.device)               # (7,)
        group_embs = self.group_id_embed(group_ids).unsqueeze(0)     # (1, 7, d_token)
        x = x + group_embs                                           # (batch, 7, d_token)

        # Prepend CLS token — forces selective attention across groups
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_token)
        x   = torch.cat([cls, x], dim=1)                 # (batch, 8, d_token)

        # Transformer: CLS attends to all 7 group tokens
        x = self.transformer(x)   # (batch, 8, d_token)
        x = self.norm(x)

        # Extract CLS token (index 0) — contains selectively aggregated info
        cls_out = x[:, 0, :]      # (batch, d_token)

        return self.head(cls_out)  # (batch, n_classes)

    @torch.no_grad()
    def get_group_attention(self, groups, context):
        """
        Extract CLS attention weights from the FIRST Transformer layer.
        CLS token is at index 0; group tokens are at indices 1-7.
        Returns the CLS row of the attention matrix averaged over heads:
        attn[i] = how much CLS attended to group i (i=0..6)
        """
        tokens = []
        for gname in ["flow_volume","tcp_flags","fragmentation",
                      "gre_header","recon_cardinality"]:
            tokens.append(self.encoders[gname](groups[gname]))
        tokens.append(self.splt_encoder(groups["temporal_splt"]))
        tokens.append(self.context_encoder(context.unsqueeze(-1)))

        x = torch.stack(tokens, dim=1)

        # Add group identity embeddings
        group_ids  = torch.arange(7, device=x.device)
        group_embs = self.group_id_embed(group_ids).unsqueeze(0)
        x = x + group_embs

        # Prepend CLS
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x   = torch.cat([cls, x], dim=1)   # (batch, 8, d_token)

        # Get attention from first layer
        first_layer = self.transformer.layers[0]
        x_norm = first_layer.norm1(x)
        _, attn = first_layer.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True, average_attn_weights=True
        )
        # attn: (batch, 8, 8) — row 0 = CLS attention to each token
        # Columns 1-7 = group tokens (skip column 0 = CLS attending to itself)
        cls_attn = attn[:, 0, 1:].mean(dim=0)  # (7,) averaged over batch
        return cls_attn


# ==========================================================
# SECTION 6: LR SCHEDULE
# ==========================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * scale

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ==========================================================
# SECTION 7: TRAIN / EVAL
# ==========================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    device, use_amp, max_steps=None, print_every=100):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for step, (groups, context, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        groups  = {g: t.to(device) for g, t in groups.items()}
        context = context.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(groups, context)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % print_every == 0:
            n = max_steps or len(loader)
            print(f"    step {step+1:>4}/{n}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.4f}", flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names,
             use_amp=True, max_steps=None):
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    all_preds  = []
    all_labels = []
    total_loss = total = 0

    for step, (groups, context, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        groups  = {g: t.to(device) for g, t in groups.items()}
        context = context.to(device)
        labels  = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(groups, context)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=4, zero_division=0)
    return acc, total_loss / total, report, \
           np.array(all_labels), np.array(all_preds)


# ==========================================================
# SECTION 8: MAIN
# ==========================================================

def main():
    print("\n" + "=" * 65)
    print("ADST — ATTACK-DOMAIN SEMANTIC TOKENIZATION TRANSFORMER")
    print("=" * 65)

    # ---- Verify group definitions ----
    all_group_feats = []
    for feats in SEMANTIC_GROUPS.values():
        all_group_feats.extend(feats)
    all_group_feats.append(GLOBAL_CONTEXT_FEATURE)
    assert len(all_group_feats) == 130, \
        f"Expected 130 features, got {len(all_group_feats)}"
    assert len(set(all_group_feats)) == 130, \
        "Duplicate features in group definitions"
    print("Group definitions: 130 features, no gaps, no overlaps ✓")

    # ---- Load artifacts ----
    print("\nLoading training artifacts...")
    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)
    print(f"  Classes: {n_classes}, Total features: {len(feature_order)}")

    # Verify all group features exist in feature_order
    missing = [f for f in all_group_feats if f not in feature_order]
    if missing:
        raise ValueError(f"Features missing from feature_order: {missing}")
    print(f"  All group features present in feature_order ✓")

    # Print group summary
    print("\nSemantic groups:")
    for gname, feats in SEMANTIC_GROUPS.items():
        print(f"  {gname:<20}: {len(feats):>3} features")
    print(f"  {'global_context':<20}:   1 feature ({GLOBAL_CONTEXT_FEATURE})")

    # ---- Build datasets ----
    print("\nBuilding datasets...")
    train_ds = ADSTDataset(f"{TRAINING_DIR}/train.csv", feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    val_ds   = ADSTDataset(f"{TRAINING_DIR}/val.csv",   feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    test_ds  = ADSTDataset(f"{TRAINING_DIR}/test.csv",  feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"),
                              collate_fn=collate_adst)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"),
                              collate_fn=collate_adst)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"),
                              collate_fn=collate_adst)

    # ---- Build model ----
    print("\nBuilding ADST Transformer...")
    group_sizes = {g: len(feats) for g, feats in SEMANTIC_GROUPS.items()}

    model = ADSTTransformer(
        group_sizes=group_sizes,
        d_token=D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, n_classes=n_classes, dropout=DROPOUT
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Tokens: 7 (6 semantic groups + 1 global context)")
    print(f"  Transformer: {N_LAYERS} layers, {N_HEADS} heads, d_token={D_TOKEN}")

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = MAX_STEPS_PER_EPOCH or len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    use_amp = (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training ----
    print("\n" + "=" * 65)
    print("TRAINING")
    print("=" * 65)

    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    history          = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            DEVICE, use_amp, MAX_STEPS_PER_EPOCH, print_every=100)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, DEVICE, class_names,
            use_amp, MAX_EVAL_STEPS)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_lr()[0]:.2e}  time={elapsed:.0f}s")

        history.append(dict(epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "group_sizes": group_sizes,
                "class_names": class_names,
                "d_token": D_TOKEN, "n_heads": N_HEADS,
                "n_layers": N_LAYERS, "d_ff": D_FF,
            }, f"{OUTPUT_PREFIX}_model.pt")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Final evaluation ----
    print(f"\nBest model: epoch {best_epoch}, val_acc={best_val_acc:.4f}")
    ckpt = torch.load(f"{OUTPUT_PREFIX}_model.pt",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 65)
    print("VALIDATION SET RESULTS (full 216,000 rows)")
    print("=" * 65)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    val_acc, _, val_report, _, _ = evaluate(
        model, val_loader, DEVICE, class_names, use_amp)
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(val_report)

    print("=" * 65)
    print("TEST SET RESULTS (full 216,000 rows)")
    print("=" * 65)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    test_acc, _, test_report, y_true, y_pred = evaluate(
        model, test_loader, DEVICE, class_names, use_amp)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(test_report)

    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    print("\nPer-class recall (sorted, worst first):")
    for name, rec in sorted(zip(class_names, recalls), key=lambda x: x[1]):
        bar = "█" * int(rec * 30)
        print(f"  {name:<30}  {rec:.3f}  {bar}")

    # ---- Group attention analysis ----
    print("\nComputing semantic group attention weights...")
    model.eval()
    attn_accum = None
    count = 0
    with torch.no_grad():
        for step, (groups, context, _) in enumerate(val_loader):
            if step >= 20:
                break
            groups  = {g: t.to(DEVICE) for g, t in groups.items()}
            context = context.to(DEVICE)
            attn = model.get_group_attention(groups, context)  # (7,) CLS attention
            if attn_accum is None:
                attn_accum = attn.cpu()
            else:
                attn_accum += attn.cpu()
            count += 1

    if attn_accum is not None:
        group_importance = (attn_accum / count).numpy()  # (7,) CLS attention to each group
        group_importance = group_importance / group_importance.sum()

        print("\nCLS attention to each semantic group (higher = CLS relied on this group more):")
        for gname, imp in sorted(
            zip(GROUP_NAMES, group_importance), key=lambda x: -x[1]
        ):
            bar = "█" * int(imp * 50)
            print(f"  {gname:<20}  {imp:.4f}  {bar}")

        attn_df = pd.DataFrame({
            "group": GROUP_NAMES,
            "cls_attention": group_importance
        })
        attn_df.to_csv(f"{OUTPUT_PREFIX}_group_attention.csv", index=False)
        print(f"\nSaved group attention -> {OUTPUT_PREFIX}_group_attention.csv")

    # ---- Save outputs ----
    with open(f"{OUTPUT_PREFIX}_val_report.txt",  "w") as f:
        f.write(f"Validation Accuracy: {val_acc:.4f}\n\n{val_report}")
    with open(f"{OUTPUT_PREFIX}_test_report.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{test_report}")
    pd.DataFrame(history).to_csv(f"{OUTPUT_PREFIX}_history.csv", index=False)

    # ---- Final comparison ----
    print("\n" + "=" * 65)
    print("THREE-MODEL COMPARISON")
    print("=" * 65)
    print(f"  XGBoost (130 flat features):          90.06%")
    print(f"  Flat MLP (130 flat features):         83.93%")
    print(f"  ADST Transformer (7 semantic tokens): {test_acc*100:.2f}%")
    print()

    if test_acc > 0.9006:
        print("ADST OUTPERFORMS XGBoost ✓")
        print("Key finding: semantic grouping + Transformer beats tree-based method")
        print("with added interpretability (group-level attention explanations)")
    elif test_acc > 0.8393:
        print("ADST > MLP baseline ✓")
        gap = (0.9006 - test_acc) * 100
        print(f"ADST is {gap:.2f}% below XGBoost")
        print("Key finding: semantic grouping improves over flat neural network")
        print("XGBoost remains stronger, but ADST offers semantic interpretability")
    else:
        print("ADST <= MLP — semantic grouping did not help")
        print("Check: group definitions may need revision")
    print("\nDone.")


if __name__ == "__main__":
    main()