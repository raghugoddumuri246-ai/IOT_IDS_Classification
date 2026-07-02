# ================================================================
# KAGGLE NOTEBOOK: ADST — Attack-Domain Semantic Tokenization
# ================================================================
# GPU: P100 (sm_60) — requires PyTorch 2.4.0
# Run Cell 1 FIRST, then restart kernel, then run Cells 2-9.
# ================================================================


# ── CELL 1 ──────────────────────────────────────────────────────
# Install PyTorch 2.4.0 — last version with P100 (sm_60) support.
# PyTorch 2.5+ dropped sm_60 kernels causing cudaErrorNoKernelImageForDevice.
# IMPORTANT: After running this cell → Runtime → Restart Session
# Then run from Cell 2 (skip Cell 1 after restart).
# ────────────────────────────────────────────────────────────────
"""
!pip install -q torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118
"""


# ── CELL 2 ──────────────────────────────────────────────────────
# Imports, GPU verification, and reproducibility seeds
# ────────────────────────────────────────────────────────────────

import os, math, time, joblib, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, classification_report, recall_score)
from sklearn.exceptions import UndefinedMetricWarning

# Suppress known-safe warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Verify GPU works before anything else
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        _t = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
        DEVICE = torch.device("cuda")
        print(f"GPU             : {torch.cuda.get_device_name(0)}  ✓ working")
    except Exception as e:
        DEVICE = torch.device("cpu")
        print(f"GPU FAILED: {e}")
        print("→ Re-run Cell 1 and restart kernel to fix GPU.")
else:
    DEVICE = torch.device("cpu")
    print("No GPU — using CPU")

print(f"Training device : {DEVICE}")
torch.manual_seed(42)
np.random.seed(42)


# ── CELL 3 ──────────────────────────────────────────────────────
# Configuration — all hyperparameters in one place
# ────────────────────────────────────────────────────────────────

# ← UPDATE THIS PATH to where your TRAINING_DATA folder is on Kaggle
TRAINING_DIR  = "/kaggle/input/YOUR_DATASET/TRAINING_DATA"
OUTPUT_PREFIX = "adst"

# ADST token dimension — each semantic group encoder outputs d_token dims
D_TOKEN   = 64     # each group token size
N_HEADS   = 4      # Transformer attention heads (D_TOKEN must be divisible)
N_LAYERS  = 2      # Transformer encoder blocks
D_FF      = 256    # feedforward hidden dim inside Transformer
DROPOUT   = 0.1

# Training — same as MLP and flat FT-Transformer for fair comparison
BATCH_SIZE = 2048   # larger batch → stable gradients, faster on P100
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
PATIENCE      = 7

# Step caps — P100 is fast enough to run the full dataset per epoch
# With batch=2048 and 1,008,000 train rows → 493 batches per epoch
# Setting MAX_STEPS > 493 means the DataLoader exhausts naturally
# which is correct — all 1,008,000 training rows are used every epoch.
# The final "step 400/500" display is NOT a bug — training completes
# all 493 batches; the step counter just doesn't print the last partial
# chunk because print_every=100 and 493 is not a multiple of 100.
MAX_STEPS_PER_EPOCH = None  # None = use ALL data per epoch (P100 is fast)
MAX_EVAL_STEPS      = None  # None = evaluate on full val set each epoch

print(f"Config loaded.")
print(f"  d_token={D_TOKEN}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")
print(f"  batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")
print(f"  Using full dataset per epoch (no step cap on P100)")


# ── CELL 4 ──────────────────────────────────────────────────────
# Attack-Domain Semantic Group Definitions
# 130 features organized into 6 attack behavior domains + 1 global context
# Verified: no gaps, no overlaps, total = 130
# ────────────────────────────────────────────────────────────────

SEMANTIC_GROUPS = {

    # G1: HOW does this flow BEHAVE overall?
    # NFStream flow statistics + IP packet length stats + application info
    "flow_volume": [
        "protocol", "ip_version",
        "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
        "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
        "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
        "bidirectional_min_ps", "bidirectional_mean_ps",
        "bidirectional_stddev_ps", "bidirectional_max_ps",
        "src2dst_min_ps", "src2dst_mean_ps",
        "src2dst_stddev_ps", "src2dst_max_ps",
        "dst2src_min_ps", "dst2src_mean_ps",
        "dst2src_stddev_ps", "dst2src_max_ps",
        "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
        "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
        "src2dst_min_piat_ms", "src2dst_mean_piat_ms",
        "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
        "dst2src_min_piat_ms", "dst2src_mean_piat_ms",
        "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
        "application_name", "application_category_name", "application_confidence",
        "dpkt_packet_count",
        "dpkt_ip_total_len_min", "dpkt_ip_total_len_mean",
        "dpkt_ip_total_len_std", "dpkt_ip_total_len_max",
    ],

    # G2: WHICH TCP flags dominate?
    # Catches SYN floods (all SYN), PSHACK (PSH+ACK), RSTFINFlood (RST+FIN)
    "tcp_flags": [
        "bidirectional_syn_packets",  "bidirectional_cwr_packets",
        "bidirectional_ece_packets",  "bidirectional_urg_packets",
        "bidirectional_ack_packets",  "bidirectional_psh_packets",
        "bidirectional_rst_packets",  "bidirectional_fin_packets",
        "src2dst_syn_packets",  "src2dst_cwr_packets",
        "src2dst_ece_packets",  "src2dst_urg_packets",
        "src2dst_ack_packets",  "src2dst_psh_packets",
        "src2dst_rst_packets",  "src2dst_fin_packets",
        "dst2src_syn_packets",  "dst2src_cwr_packets",
        "dst2src_ece_packets",  "dst2src_urg_packets",
        "dst2src_ack_packets",  "dst2src_psh_packets",
        "dst2src_rst_packets",  "dst2src_fin_packets",
    ],

    # G3: Is this a FRAGMENTATION attack?
    # XGBoost rank #1, #3, #9 — these 5 features dominate classification
    "fragmentation": [
        "dpkt_frag_mf_count",
        "dpkt_frag_df_count",
        "dpkt_frag_offset_nonzero_count",
        "dpkt_frag_ratio",
        "dpkt_ip_options_present_count",
    ],

    # G4: Is this GRE-encapsulated Mirai traffic?
    # gre_inner_ether_ratio → greeth, gre_inner_ip_ratio → greip, 0 → udpplain
    "gre_header": [
        "dpkt_gre_packet_count",
        "dpkt_gre_inner_proto_ip_count",
        "dpkt_gre_inner_proto_ether_count",
        "dpkt_gre_ratio",
        "dpkt_gre_inner_ip_ratio",
        "dpkt_gre_inner_ether_ratio",
        "dpkt_ttl_min",  "dpkt_ttl_mean",
        "dpkt_ttl_std",  "dpkt_ttl_max",
        "dpkt_tcp_window_min",  "dpkt_tcp_window_mean",
        "dpkt_tcp_window_std",  "dpkt_tcp_window_max",
        "dpkt_header_byte_entropy",
    ],

    # G5: Is this RECONNAISSANCE / SCANNING?
    # NULL/FIN/XMAS scan patterns + fan-out cardinality for PortScan vs HostDiscovery
    "recon_cardinality": [
        "dpkt_tcp_null_scan_count",
        "dpkt_tcp_fin_scan_count",
        "dpkt_tcp_xmas_scan_count",
        "dpkt_icmp_echo_request_count",
        "dpkt_icmp_echo_reply_count",
        "dpkt_icmp_other_count",
        "fan_in_src_count",
        "fan_out_port_count",
        "fan_out_ip_count",
        "fan_out_proto_count",
        "fan_out_scope",
        "fan_app_diversity",
    ],

    # G6: What is the TEMPORAL PATTERN of the first 10 packets?
    # Treated as 10 timesteps × 3 channels → 1D-Conv encoder
    # SlowLoris: increasing inter-arrival times
    # SYN Flood: near-zero inter-arrival times, one direction
    "temporal_splt": [
        *[f"splt_direction_{i}" for i in range(10)],   # packet directions
        *[f"splt_ps_{i}"        for i in range(10)],   # packet sizes
        *[f"splt_piat_ms_{i}"   for i in range(10)],   # inter-arrival times
    ],
}

# G7: Global capture-level context (1 feature)
# file_frag_rate = fraction of ALL flows in this PCAP with MF-flag set
# Fragmentation PCAPs → near 1.0; other attack PCAPs → near 0.0
# This token proved to be the strongest single signal (0.97 attention
# for fragmentation attack classes in per-class attention analysis)
GLOBAL_CONTEXT_FEATURE = "file_frag_rate"
GROUP_NAMES = list(SEMANTIC_GROUPS.keys()) + ["global_context"]

# Verify: all 130 features assigned, no duplicates
_all = [f for feats in SEMANTIC_GROUPS.values() for f in feats]
_all.append(GLOBAL_CONTEXT_FEATURE)
assert len(_all) == 130,         f"Expected 130 features, got {len(_all)}"
assert len(set(_all)) == 130,    "Duplicate features detected"
print(f"Group definitions verified: {len(_all)} features, no duplicates ✓")
for g, feats in SEMANTIC_GROUPS.items():
    print(f"  {g:<20}: {len(feats):>3} features")
print(f"  {'global_context':<20}:   1 feature ({GLOBAL_CONTEXT_FEATURE})")


# ── CELL 5 ──────────────────────────────────────────────────────
# Dataset class — loads CSVs and organizes features into semantic groups
# Preprocessing: arcsinh(x) + clip[-15,15] for all features
# ────────────────────────────────────────────────────────────────

class ADSTDataset(Dataset):
    """
    Loads train/val/test CSV and organizes 130 features into
    6 semantic group tensors + 1 global context scalar.

    Preprocessing: arcsinh(x) + safety clip[-15,15]
    - arcsinh compresses large sparse values logarithmically
      (NOT hard clipping which destroyed 99.78% of fragmentation signal)
    - Preserves relative ordering within each sparse feature
    - Identical transform as MLP baseline for fair comparison
    """

    def __init__(self, csv_path, feature_order,
                 semantic_groups, global_ctx_feature):
        print(f"  Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Build feature index lookup
        feat_idx = {f: i for i, f in enumerate(feature_order)}

        # Load all 130 features → apply arcsinh → slice into groups
        X_np = df[feature_order].values.astype(np.float32)
        X    = torch.tensor(X_np, dtype=torch.float32)
        X    = torch.asinh(X)                           # log-compress outliers
        X    = torch.clamp(X, min=-15.0, max=15.0)     # safety ceiling

        clipped = ((X < -15) | (X > 15)).float().mean().item() * 100
        print(f"  arcsinh applied. Safety clip triggered: {clipped:.4f}% of values")

        # Slice each semantic group
        self.group_data = {}
        for gname, feats in semantic_groups.items():
            idxs = [feat_idx[f] for f in feats]
            self.group_data[gname] = X[:, idxs]

        # Global context token (single scalar per sample)
        self.context = X[:, feat_idx[global_ctx_feature]]

        self.n_samples = len(df)
        print(f"  Samples: {self.n_samples:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        groups  = {g: t[idx] for g, t in self.group_data.items()}
        context = self.context[idx]
        label   = self.labels[idx]
        return groups, context, label


def collate_adst(batch):
    """Stack per-sample group dicts into batch tensors."""
    groups_list, contexts, labels = zip(*batch)
    groups_batch = {
        g: torch.stack([s[g] for s in groups_list])
        for g in groups_list[0]
    }
    return groups_batch, torch.stack(contexts), torch.stack(labels)

print("Dataset class defined.")


# ── CELL 6 ──────────────────────────────────────────────────────
# ADST Model Architecture
#
# 7 tokens → Transformer → CLS → 24 classes
#
# G1 (43 feat) → MLP encoder → 64-dim token
# G2 (24 feat) → MLP encoder → 64-dim token
# G3 (5 feat)  → MLP encoder → 64-dim token
# G4 (15 feat) → MLP encoder → 64-dim token
# G5 (12 feat) → MLP encoder → 64-dim token
# G6 (30 feat) → Conv1D on [10 timesteps × 3 channels] → 64-dim token
# G7 (1 feat)  → Linear → 64-dim global context token
#
# + Group identity embeddings (break symmetry, prevent uniform attention)
# + CLS token (forces selective attention, not mean-pool)
# ────────────────────────────────────────────────────────────────

def make_mlp_encoder(in_dim, hidden_dim, out_dim, dropout=0.1):
    """Small MLP: in → hidden → out. BatchNorm for tabular stability."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class SPLTConvEncoder(nn.Module):
    """
    1D-Conv encoder for G6 (temporal SPLT features).
    Reshapes 30 flat features → [batch, 3 channels, 10 timesteps]:
      channel 0: splt_direction_0..9  (packet direction)
      channel 1: splt_ps_0..9         (packet size)
      channel 2: splt_piat_ms_0..9    (inter-arrival time)
    Conv filters capture temporal patterns across adjacent timesteps.
    """

    def __init__(self, d_token, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)   # (batch, 64, 10) → (batch, 64)
        self.proj  = nn.Linear(64, d_token)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.shape[0], 3, 10)              # (B, 3, 10)
        x = F.gelu(self.bn1(self.conv1(x)))         # (B, 32, 10)
        x = F.gelu(self.bn2(self.conv2(x)))         # (B, 64, 10)
        x = self.pool(x).squeeze(-1)                 # (B, 64)
        return self.proj(self.drop(x))               # (B, d_token)


class ADSTTransformer(nn.Module):
    """
    Attack-Domain Semantic Tokenization Transformer.

    Key design decisions:
    1. Group identity embeddings: each group gets a unique learnable
       vector added to its token. Without this, all 7 tokens look
       identical to the Transformer → attention stays uniform (1/7).
    2. CLS token instead of mean-pool: CLS MUST gather info by
       attending selectively to groups. Mean-pool made the Transformer
       optional (uniform attention = same result as mean of encoders).
    3. BatchNorm in group encoders (not LayerNorm): BatchNorm normalizes
       per-feature across the batch, preserving feature-scale info.
       LayerNorm would normalize across features and destroy scale signal.
    """

    def __init__(self, group_sizes, d_token, n_heads, n_layers,
                 d_ff, n_classes, dropout):
        super().__init__()

        # Group encoders — hidden dim proportional to group size
        def _h(n): return max(16, min(128, n * 2))

        self.encoders = nn.ModuleDict({
            "flow_volume":       make_mlp_encoder(
                group_sizes["flow_volume"],       _h(group_sizes["flow_volume"]),       d_token, dropout),
            "tcp_flags":         make_mlp_encoder(
                group_sizes["tcp_flags"],         _h(group_sizes["tcp_flags"]),         d_token, dropout),
            "fragmentation":     make_mlp_encoder(
                group_sizes["fragmentation"],     _h(group_sizes["fragmentation"]),     d_token, dropout),
            "gre_header":        make_mlp_encoder(
                group_sizes["gre_header"],        _h(group_sizes["gre_header"]),        d_token, dropout),
            "recon_cardinality": make_mlp_encoder(
                group_sizes["recon_cardinality"], _h(group_sizes["recon_cardinality"]), d_token, dropout),
        })
        self.splt_encoder    = SPLTConvEncoder(d_token, dropout)
        self.context_encoder = nn.Sequential(
            nn.Linear(1, d_token), nn.GELU())

        # Group identity embeddings (7 groups, each gets unique learned vector)
        # Fixes: uniform attention collapse seen in first ADST version
        self.group_id_embed = nn.Embedding(7, d_token)

        # CLS token — replaced mean-pool to force selective attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (Pre-LN for training stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm        = nn.LayerNorm(d_token)

        # Classification head on CLS output
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes))

        # Kaiming init for all linear layers
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
        # Encode each semantic group into a 64-dim token
        tokens = [
            self.encoders["flow_volume"](groups["flow_volume"]),
            self.encoders["tcp_flags"](groups["tcp_flags"]),
            self.encoders["fragmentation"](groups["fragmentation"]),
            self.encoders["gre_header"](groups["gre_header"]),
            self.encoders["recon_cardinality"](groups["recon_cardinality"]),
            self.splt_encoder(groups["temporal_splt"]),
            self.context_encoder(context.unsqueeze(-1)),
        ]

        x = torch.stack(tokens, dim=1)   # (B, 7, d_token)

        # Add group identity (breaks symmetry → non-uniform attention)
        gids  = torch.arange(7, device=x.device)
        x     = x + self.group_id_embed(gids).unsqueeze(0)

        # Prepend CLS token
        cls   = self.cls_token.expand(x.shape[0], -1, -1)
        x     = torch.cat([cls, x], dim=1)   # (B, 8, d_token)

        # Transformer: CLS attends to all 7 group tokens
        x     = self.transformer(x)
        x     = self.norm(x)

        # CLS output → classification
        return self.head(x[:, 0, :])

    @torch.no_grad()
    def get_group_attention(self, groups, context):
        """
        Returns CLS attention weight to each of 7 group tokens.
        Shape: (7,) — how much CLS relied on each group.
        Use for paper's semantic attention visualization.
        """
        tokens = [
            self.encoders["flow_volume"](groups["flow_volume"]),
            self.encoders["tcp_flags"](groups["tcp_flags"]),
            self.encoders["fragmentation"](groups["fragmentation"]),
            self.encoders["gre_header"](groups["gre_header"]),
            self.encoders["recon_cardinality"](groups["recon_cardinality"]),
            self.splt_encoder(groups["temporal_splt"]),
            self.context_encoder(context.unsqueeze(-1)),
        ]
        x    = torch.stack(tokens, dim=1)
        gids = torch.arange(7, device=x.device)
        x    = x + self.group_id_embed(gids).unsqueeze(0)
        cls  = self.cls_token.expand(x.shape[0], -1, -1)
        x    = torch.cat([cls, x], dim=1)

        layer = self.transformer.layers[0]
        xn    = layer.norm1(x)
        _, attn = layer.self_attn(xn, xn, xn,
                                  need_weights=True,
                                  average_attn_weights=True)
        # CLS row (index 0), columns 1-7 (group tokens)
        return attn[:, 0, 1:].mean(dim=0)   # (7,)

print("ADST model architecture defined.")
print(f"  7 tokens: 5× MLP encoder + 1× Conv1D (SPLT) + 1× global context")
print(f"  Transformer: {N_LAYERS} layers, {N_HEADS} heads, d_token={D_TOKEN}")


# ── CELL 7 ──────────────────────────────────────────────────────
# LR schedule + train/eval functions
# Plain float32 — no AMP (P100 Kaggle compatibility)
# ────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup → cosine annealing to LR/100."""

    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr_ratio=0.01):
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
            p = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * p))
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * scale

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def train_one_epoch(model, loader, optimizer, scheduler,
                    device, max_steps=None, print_every=50):
    """
    One training epoch, plain float32 (no AMP — P100 compatible).

    NOTE on step display: with batch=2048 and 1,008,000 train rows,
    the DataLoader produces 493 batches per epoch. If max_steps > 493,
    the loop ends naturally at batch 493 (all data used). The last
    printed step will be 450 (step+1=450) since 493 is not a multiple
    of 50. This is correct — all training data is processed.
    """
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
        logits = model(groups, context)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % print_every == 0:
            n = max_steps or len(loader)
            print(f"    step {step+1:>4}/{n}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.4f}", flush=True)

    # Always print final totals for the epoch (fixes the 400/500 display)
    print(f"    [epoch end] steps={step+1}  "
          f"loss={total_loss/total:.4f}  acc={correct/total:.4f}",
          flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names, max_steps=None):
    """Full evaluation, plain float32."""
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

print("LR scheduler and train/eval functions defined.")


# ── CELL 8 ──────────────────────────────────────────────────────
# Build datasets, model, and start training
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ADST — ATTACK-DOMAIN SEMANTIC TOKENIZATION TRANSFORMER")
    print("=" * 70)

    # Load artifacts saved by Script 06
    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)
    feat_idx      = {f: i for i, f in enumerate(feature_order)}

    print(f"\nClasses : {n_classes}")
    print(f"Features: {len(feature_order)}")
    print(f"Device  : {DEVICE}")

    # Verify all group features exist in the saved feature list
    all_group_feats = [
        f for feats in SEMANTIC_GROUPS.values() for f in feats]
    all_group_feats.append(GLOBAL_CONTEXT_FEATURE)
    missing = [f for f in all_group_feats if f not in feat_idx]
    if missing:
        raise ValueError(f"Features missing from feature_order: {missing}")
    print("All group features found in feature_order ✓")

    # Build datasets
    print("\nLoading datasets...")
    train_ds = ADSTDataset(
        f"{TRAINING_DIR}/train.csv", feature_order,
        SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    val_ds = ADSTDataset(
        f"{TRAINING_DIR}/val.csv", feature_order,
        SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    test_ds = ADSTDataset(
        f"{TRAINING_DIR}/test.csv", feature_order,
        SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=pin, collate_fn=collate_adst,
        persistent_workers=True)
    val_loader = DataLoader(
        val_ds, BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=pin, collate_fn=collate_adst,
        persistent_workers=True)
    test_loader = DataLoader(
        test_ds, BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=pin, collate_fn=collate_adst,
        persistent_workers=True)

    print(f"\nTrain batches per epoch: {len(train_loader)} "
          f"(all {len(train_ds):,} rows used)")

    # Build model
    print("\nBuilding ADST Transformer...")
    group_sizes = {g: len(feats) for g, feats in SEMANTIC_GROUPS.items()}
    model = ADSTTransformer(
        group_sizes, D_TOKEN, N_HEADS, N_LAYERS,
        D_FF, n_classes, DROPOUT).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Optimizer and LR schedule
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler       = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps)

    # Training loop
    print("\n" + "=" * 70)
    print(f"TRAINING  —  {EPOCHS} epochs × {steps_per_epoch} batches × "
          f"batch {BATCH_SIZE}")
    print("=" * 70)

    best_val_acc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            DEVICE, MAX_STEPS_PER_EPOCH, print_every=50)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, DEVICE, class_names, MAX_EVAL_STEPS)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_lr()[0]:.2e}  time={elapsed:.0f}s")

        history.append(dict(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "group_sizes": group_sizes,
                "class_names": class_names,
                "d_token": D_TOKEN, "n_heads": N_HEADS,
                "n_layers": N_LAYERS, "d_ff": D_FF,
            }, f"{OUTPUT_PREFIX}_model.pt")
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return model, train_loader, val_loader, test_loader, class_names, history


model, train_loader, val_loader, test_loader, class_names, history = main()


# ── CELL 9 ──────────────────────────────────────────────────────
# Final evaluation + group attention analysis
# Run after training completes (Cell 8)
# ────────────────────────────────────────────────────────────────

# Load best checkpoint
ckpt = torch.load(f"{OUTPUT_PREFIX}_model.pt",
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded best model from epoch {ckpt['epoch']}"
      f" (val_acc={ckpt['val_acc']:.4f})")

# Full validation evaluation
print("\n" + "=" * 70)
print("VALIDATION RESULTS  (full 216,000 rows)")
print("=" * 70)
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
val_acc, _, val_report, _, _ = evaluate(
    model, val_loader, DEVICE, class_names)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(val_report)

# Full test evaluation
print("=" * 70)
print("TEST RESULTS  (full 216,000 rows)")
print("=" * 70)
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
test_acc, _, test_report, y_true, y_pred = evaluate(
    model, test_loader, DEVICE, class_names)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(test_report)

# Per-class recall sorted worst → best
recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
print("Per-class recall (worst → best):")
for cls, rec in sorted(zip(class_names, recalls), key=lambda x: x[1]):
    bar = "█" * int(rec * 30)
    print(f"  {cls:<30}  {rec:.3f}  {bar}")

# Group attention analysis (paper's key visualization)
print("\n" + "=" * 70)
print("SEMANTIC GROUP ATTENTION (CLS → each group, averaged over val set)")
print("=" * 70)
model.eval()
attn_accum = None
count = 0
with torch.no_grad():
    for step, (groups, context, _) in enumerate(val_loader):
        if step >= 30:
            break
        groups  = {g: t.to(DEVICE) for g, t in groups.items()}
        context = context.to(DEVICE)
        attn    = model.get_group_attention(groups, context)  # (7,)
        attn_accum = attn.cpu() if attn_accum is None \
                     else attn_accum + attn.cpu()
        count += 1

if attn_accum is not None and count > 0:
    mean_attn = (attn_accum / count).numpy()
    mean_attn = mean_attn / mean_attn.sum()
    print(f"\n  {'Group':<22} {'Attention':>10}  {'vs uniform (1/7)':>18}")
    print("  " + "-" * 55)
    uniform = 1 / 7
    for gname, attn_val in sorted(
            zip(GROUP_NAMES, mean_attn), key=lambda x: -x[1]):
        bar  = "█" * int(attn_val * 70)
        diff = attn_val - uniform
        sign = "+" if diff > 0 else ""
        print(f"  {gname:<22} {attn_val:>10.4f}  {sign}{diff*100:>+6.1f}%  {bar}")

    # Save attention CSV for paper figures
    attn_df = pd.DataFrame({
        "group": GROUP_NAMES,
        "cls_attention": mean_attn
    })
    attn_df.to_csv(f"{OUTPUT_PREFIX}_group_attention.csv", index=False)
    print(f"\nSaved → {OUTPUT_PREFIX}_group_attention.csv")

# Save training history
pd.DataFrame(history).to_csv(f"{OUTPUT_PREFIX}_history.csv", index=False)
print(f"Saved → {OUTPUT_PREFIX}_history.csv")

# Final comparison table
print("\n" + "=" * 70)
print("PAPER COMPARISON TABLE")
print("=" * 70)
print(f"  XGBoost             (130 flat, trees):   90.06%")
print(f"  ADST Transformer    (7 semantic tokens):  {test_acc*100:.2f}%")
print(f"  Flat MLP            (130 flat, dense NN): 87.69%")
print(f"  Flat FT-Transformer (130 flat tokens):    TBD")
print()
if test_acc > 0.8769:
    print("✓ ADST outperforms flat MLP baseline")
    print("  Semantic grouping adds measurable value over flat neural approach")
if test_acc > 0.90:
    print("✓ ADST outperforms XGBoost — strong result for the paper")
print("\nDone.")