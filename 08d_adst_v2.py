# ================================================================
# KAGGLE NOTEBOOK: ADST v2 — Attack-Domain Semantic Tokenization
# ================================================================
# WHAT CHANGED FROM v1:
#   1. Parameter count is now printed explicitly and saved, so it can
#      be compared directly against the FT-Transformer's printed count.
#      (v1 computed n_params but the log you shared never showed it —
#      this version prints it right after model construction and again
#      in the final summary.)
#   2. FRAGMENTATION ENCODER CAPACITY ABLATION: in v1 the fragmentation
#      group (5 features) used hidden_dim = max(16, min(128, 5*2)) = 16,
#      the smallest encoder of any group — yet your XGBoost analysis
#      ranked fragmentation features #1, #3, #9 in importance, and v1's
#      attention analysis showed fragmentation getting the LEAST CLS
#      attention (0.0374, -10.5% vs uniform) of all 7 groups. This
#      version adds a toggle (FRAG_HIDDEN_OVERRIDE) to give that
#      encoder more capacity and see whether attention / accuracy on
#      DDoS-ICMP_Flood, DDoS-UDP_Flood shift as a result.
#   3. MULTI-SEED: runs seeds [42, 123, 2024] and reports mean +/- std
#      test accuracy, directly comparable to the FT-Transformer's
#      multi-seed summary.
#   Training protocol (batch=2048, full dataset/epoch, LR=1e-3) is
#   UNCHANGED — it was already the reference config the FT-Transformer
#   script was matched to.
# GPU: P100 (sm_60) — requires PyTorch 2.4.0
# Run Cell 1 FIRST, then restart kernel, then run Cells 2-9.
# ================================================================


# ── CELL 1 ──────────────────────────────────────────────────────
"""
!pip install -q torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118
"""


# ── CELL 2 ──────────────────────────────────────────────────────
import os, math, time, json, joblib, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, classification_report, recall_score)
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        _t = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
        DEVICE = torch.device("cuda")
        print(f"GPU             : {torch.cuda.get_device_name(0)}  working")
    except Exception as e:
        DEVICE = torch.device("cpu")
        print(f"GPU FAILED: {e}")
else:
    DEVICE = torch.device("cpu")
    print("No GPU — using CPU")

print(f"Training device : {DEVICE}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ── CELL 3 ──────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

TRAINING_DIR  = "/kaggle/input/YOUR_DATASET/TRAINING_DATA"
OUTPUT_PREFIX = "adst_v2"

D_TOKEN   = 64
N_HEADS   = 4
N_LAYERS  = 2
D_FF      = 256
DROPOUT   = 0.1

BATCH_SIZE = 2048
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
PATIENCE      = 7

MAX_STEPS_PER_EPOCH = None
MAX_EVAL_STEPS      = None

# Multi-seed: identical seed list to the FT-Transformer matched script,
# so the two mean+/-std summaries are directly comparable.
SEEDS = [42, 123, 2024]

# FRAGMENTATION ABLATION:
# v1 used hidden_dim=16 for this group (the formula-derived minimum),
# despite fragmentation features being XGBoost's top-ranked signal and
# receiving the LEAST attention of any group in v1's results.
# Set to None to reproduce v1 exactly. Set to an int (e.g. 64) to test
# whether more capacity here changes attention allocation and accuracy
# on DDoS-ICMP_Flood / DDoS-UDP_Flood (your weakest non-Mirai classes).
FRAG_HIDDEN_OVERRIDE = 64   # was implicitly 16 in v1

print("Config loaded.")
print(f"  d_token={D_TOKEN}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")
print(f"  batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, seeds={SEEDS}")
print(f"  fragmentation hidden_dim override: {FRAG_HIDDEN_OVERRIDE} "
      f"(v1 default would be 16)")


# ── CELL 4 ──────────────────────────────────────────────────────
# Semantic group definitions — UNCHANGED from v1
# ────────────────────────────────────────────────────────────────

SEMANTIC_GROUPS = {
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
    "fragmentation": [
        "dpkt_frag_mf_count",
        "dpkt_frag_df_count",
        "dpkt_frag_offset_nonzero_count",
        "dpkt_frag_ratio",
        "dpkt_ip_options_present_count",
    ],
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
    "temporal_splt": [
        *[f"splt_direction_{i}" for i in range(10)],
        *[f"splt_ps_{i}"        for i in range(10)],
        *[f"splt_piat_ms_{i}"   for i in range(10)],
    ],
}

GLOBAL_CONTEXT_FEATURE = "file_frag_rate"
GROUP_NAMES = list(SEMANTIC_GROUPS.keys()) + ["global_context"]

_all = [f for feats in SEMANTIC_GROUPS.values() for f in feats]
_all.append(GLOBAL_CONTEXT_FEATURE)
assert len(_all) == 130,      f"Expected 130 features, got {len(_all)}"
assert len(set(_all)) == 130, "Duplicate features detected"
print(f"Group definitions verified: {len(_all)} features, no duplicates")
for g, feats in SEMANTIC_GROUPS.items():
    print(f"  {g:<20}: {len(feats):>3} features")
print(f"  {'global_context':<20}:   1 feature ({GLOBAL_CONTEXT_FEATURE})")


# ── CELL 5 ──────────────────────────────────────────────────────
# Dataset class — UNCHANGED from v1
# ────────────────────────────────────────────────────────────────

class ADSTDataset(Dataset):
    def __init__(self, csv_path, feature_order,
                 semantic_groups, global_ctx_feature):
        print(f"  Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        feat_idx = {f: i for i, f in enumerate(feature_order)}

        X_np = df[feature_order].values.astype(np.float32)
        X    = torch.tensor(X_np, dtype=torch.float32)
        X    = torch.asinh(X)
        X    = torch.clamp(X, min=-15.0, max=15.0)

        clipped = ((X < -15) | (X > 15)).float().mean().item() * 100
        print(f"  arcsinh applied. Safety clip triggered: {clipped:.4f}% of values")

        self.group_data = {}
        for gname, feats in semantic_groups.items():
            idxs = [feat_idx[f] for f in feats]
            self.group_data[gname] = X[:, idxs]

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
    groups_list, contexts, labels = zip(*batch)
    groups_batch = {
        g: torch.stack([s[g] for s in groups_list])
        for g in groups_list[0]
    }
    return groups_batch, torch.stack(contexts), torch.stack(labels)

print("Dataset class defined.")


# ── CELL 6 ──────────────────────────────────────────────────────
# ADST Model Architecture — encoders now use a hidden-dim function that
# supports the fragmentation ablation override (FRAG_HIDDEN_OVERRIDE)
# ────────────────────────────────────────────────────────────────

def make_mlp_encoder(in_dim, hidden_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def group_hidden_dim(group_name, n_features):
    """
    v1 formula: max(16, min(128, n_features * 2))
    For fragmentation (5 features) this gives 16 — the smallest
    encoder of any group despite the highest XGBoost feature
    importance ranking. FRAG_HIDDEN_OVERRIDE lets you test a
    larger encoder for that group specifically; every other group
    is untouched.
    """
    if group_name == "fragmentation" and FRAG_HIDDEN_OVERRIDE is not None:
        return FRAG_HIDDEN_OVERRIDE
    return max(16, min(128, n_features * 2))


class SPLTConvEncoder(nn.Module):
    def __init__(self, d_token, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
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

        self.encoders = nn.ModuleDict({
            gname: make_mlp_encoder(
                group_sizes[gname],
                group_hidden_dim(gname, group_sizes[gname]),
                d_token, dropout)
            for gname in ["flow_volume", "tcp_flags", "fragmentation",
                          "gre_header", "recon_cardinality"]
        })
        self.splt_encoder    = SPLTConvEncoder(d_token, dropout)
        self.context_encoder = nn.Sequential(
            nn.Linear(1, d_token), nn.GELU())

        self.group_id_embed = nn.Embedding(7, d_token)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm        = nn.LayerNorm(d_token)

        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_tokens(self, groups, context):
        return [
            self.encoders["flow_volume"](groups["flow_volume"]),
            self.encoders["tcp_flags"](groups["tcp_flags"]),
            self.encoders["fragmentation"](groups["fragmentation"]),
            self.encoders["gre_header"](groups["gre_header"]),
            self.encoders["recon_cardinality"](groups["recon_cardinality"]),
            self.splt_encoder(groups["temporal_splt"]),
            self.context_encoder(context.unsqueeze(-1)),
        ]

    def forward(self, groups, context):
        tokens = self._encode_tokens(groups, context)
        x = torch.stack(tokens, dim=1)
        gids = torch.arange(7, device=x.device)
        x = x + self.group_id_embed(gids).unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])

    @torch.no_grad()
    def get_group_attention(self, groups, context):
        tokens = self._encode_tokens(groups, context)
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
        return attn[:, 0, 1:].mean(dim=0)

print("ADST model architecture defined.")
print(f"  7 tokens: 5x MLP encoder + 1x Conv1D (SPLT) + 1x global context")
print(f"  Transformer: {N_LAYERS} layers, {N_HEADS} heads, d_token={D_TOKEN}")
print(f"  fragmentation encoder hidden_dim: "
      f"{group_hidden_dim('fragmentation', 5)}")


# ── CELL 7 ──────────────────────────────────────────────────────
# LR schedule + train/eval — UNCHANGED from v1
# ────────────────────────────────────────────────────────────────

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
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    step = -1

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

    print(f"    [epoch end] steps={step+1}  "
          f"loss={total_loss/total:.4f}  acc={correct/total:.4f}", flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names, max_steps=None):
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
# Multi-seed training + attention analysis
# ────────────────────────────────────────────────────────────────

def run_one_seed(seed, train_loader, val_loader, test_loader,
                  group_sizes, n_classes, class_names):
    print("\n" + "#" * 70)
    print(f"# SEED {seed}")
    print("#" * 70)
    set_seed(seed)

    model = ADSTTransformer(
        group_sizes, D_TOKEN, N_HEADS, N_LAYERS,
        D_FF, n_classes, DROPOUT).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ADST Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler       = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []
    ckpt_path    = f"{OUTPUT_PREFIX}_seed{seed}_model.pt"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            DEVICE, MAX_STEPS_PER_EPOCH, print_every=100)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, DEVICE, class_names, MAX_EVAL_STEPS)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_lr()[0]:.2e}  time={elapsed:.0f}s")

        history.append(dict(epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": val_acc, "group_sizes": group_sizes,
                "class_names": class_names, "n_params": n_params,
            }, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model: epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    test_acc, _, test_report, y_true, y_pred = evaluate(
        model, test_loader, DEVICE, class_names)
    print(f"Seed {seed} TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")

    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    macro_p = classification_report(y_true, y_pred, target_names=class_names,
                                     output_dict=True, zero_division=0)["macro avg"]

    # Group attention (averaged over 30 val batches)
    model.eval()
    attn_accum = None
    count = 0
    with torch.no_grad():
        for step, (groups, context, _) in enumerate(val_loader):
            if step >= 30:
                break
            groups  = {g: t.to(DEVICE) for g, t in groups.items()}
            context = context.to(DEVICE)
            attn    = model.get_group_attention(groups, context)
            attn_accum = attn.cpu() if attn_accum is None else attn_accum + attn.cpu()
            count += 1
    mean_attn = (attn_accum / count).numpy()
    mean_attn = (mean_attn / mean_attn.sum()).tolist()

    pd.DataFrame(history).to_csv(f"{OUTPUT_PREFIX}_seed{seed}_history.csv", index=False)

    return {
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "macro_precision": macro_p["precision"],
        "macro_recall": macro_p["recall"],
        "macro_f1": macro_p["f1-score"],
        "per_class_recall": dict(zip(class_names, recalls.tolist())),
        "group_attention": dict(zip(GROUP_NAMES, mean_attn)),
    }


def main():
    print("=" * 70)
    print("ADST v2 — MULTI-SEED, FRAGMENTATION-ABLATION CAPABLE")
    print("=" * 70)

    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)
    feat_idx      = {f: i for i, f in enumerate(feature_order)}

    print(f"\nClasses : {n_classes}")
    print(f"Features: {len(feature_order)}")
    print(f"Device  : {DEVICE}")

    all_group_feats = [f for feats in SEMANTIC_GROUPS.values() for f in feats]
    all_group_feats.append(GLOBAL_CONTEXT_FEATURE)
    missing = [f for f in all_group_feats if f not in feat_idx]
    if missing:
        raise ValueError(f"Features missing from feature_order: {missing}")
    print("All group features found in feature_order - OK")

    print("\nLoading datasets (shared across all seeds)...")
    train_ds = ADSTDataset(f"{TRAINING_DIR}/train.csv", feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    val_ds   = ADSTDataset(f"{TRAINING_DIR}/val.csv", feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)
    test_ds  = ADSTDataset(f"{TRAINING_DIR}/test.csv", feature_order,
                           SEMANTIC_GROUPS, GLOBAL_CONTEXT_FEATURE)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, collate_fn=collate_adst,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=pin, collate_fn=collate_adst,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE * 2, shuffle=False,
                             num_workers=2, pin_memory=pin, collate_fn=collate_adst,
                             persistent_workers=True)

    print(f"\nTrain batches per epoch: {len(train_loader)} "
          f"(all {len(train_ds):,} rows used)")

    group_sizes = {g: len(feats) for g, feats in SEMANTIC_GROUPS.items()}

    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, train_loader, val_loader, test_loader,
                              group_sizes, n_classes, class_names)
        all_results.append(result)

    # ---- Summary across seeds ----
    test_accs = np.array([r["test_acc"] for r in all_results])
    f1s       = np.array([r["macro_f1"] for r in all_results])

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY — ADST v2")
    print(f"(fragmentation hidden_dim override = {FRAG_HIDDEN_OVERRIDE})")
    print("=" * 70)
    for r in all_results:
        print(f"  seed={r['seed']:<6} test_acc={r['test_acc']*100:.2f}%  "
              f"macro_f1={r['macro_f1']:.4f}  params={r['n_params']:,}")
    print(f"\n  Mean test_acc : {test_accs.mean()*100:.2f}%  "
          f"(+/- {test_accs.std()*100:.2f} pp std)")
    print(f"  Mean macro_f1 : {f1s.mean():.4f}  (+/- {f1s.std():.4f} std)")
    print(f"  Parameters    : {all_results[0]['n_params']:,} "
          f"(identical across seeds — architecture unchanged)")

    print("\n  Mean group attention across seeds:")
    mean_attn_by_group = {}
    for gname in GROUP_NAMES:
        vals = [r["group_attention"][gname] for r in all_results]
        mean_attn_by_group[gname] = (np.mean(vals), np.std(vals))
    for gname, (m, s) in sorted(mean_attn_by_group.items(), key=lambda x: -x[1][0]):
        print(f"    {gname:<20} {m:.4f} (+/- {s:.4f})")

    with open(f"{OUTPUT_PREFIX}_multiseed_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PREFIX}_multiseed_summary.json")
    print("\nCompare this mean +/- std directly against the FT-Transformer")
    print("multi-seed summary. Also compare fragmentation's mean attention")
    print("here against v1's 0.0374 to see if the capacity bump changed it.")

    return all_results


all_results = main()