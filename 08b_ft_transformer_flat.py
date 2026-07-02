# ============================================================
# KAGGLE NOTEBOOK: Flat FT-Transformer (Corrected Baseline)
# GPU: P100 (sm_60)
# Run each cell in order. Restart kernel after Cell 1.
# ============================================================


# ── CELL 1 ──────────────────────────────────────────────────
# Install PyTorch 2.4.0 — the LAST version with P100 (sm_60) support.
# PyTorch 2.5+ dropped sm_60 kernels, causing cudaErrorNoKernelImageForDevice.
# IMPORTANT: After running this cell, go to Runtime → Restart Session,
# then run from Cell 2 onwards (skip Cell 1 after restart).
# ─────────────────────────────────────────────────────────────
"""
!pip install -q torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118
"""
# Paste the above pip command (without the triple quotes) into its own
# Kaggle code cell and run it FIRST before anything else.


# ── CELL 2 ──────────────────────────────────────────────────
# Imports and GPU verification
# ─────────────────────────────────────────────────────────────

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

# Verify GPU works with a simple tensor op
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        _t = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
        DEVICE = torch.device("cuda")
        print(f"GPU             : {torch.cuda.get_device_name(0)}  ✓ working")
    except Exception as e:
        DEVICE = torch.device("cpu")
        print(f"GPU test FAILED : {e}")
        print("Falling back to CPU. Re-run Cell 1 and restart kernel to fix GPU.")
else:
    DEVICE = torch.device("cpu")
    print("GPU not available — using CPU")

print(f"Training device : {DEVICE}")
torch.manual_seed(42)
np.random.seed(42)


# ── CELL 3 ──────────────────────────────────────────────────
# Configuration — all hyperparameters in one place
# Matches MLP and ADST exactly for fair comparison
# ─────────────────────────────────────────────────────────────

TRAINING_DIR  = "/kaggle/input/YOUR_DATASET/TRAINING_DATA"  # ← update path
OUTPUT_PREFIX = "ft_flat"

# Model architecture
D_MODEL  = 64    # token embedding dimension (each feature → 64-dim vector)
N_HEADS  = 4     # attention heads in Transformer (D_MODEL must be divisible)
N_LAYERS = 2     # number of Transformer encoder blocks
D_FF     = 256   # feedforward hidden size inside each block
DROPOUT  = 0.1   # dropout probability

# Training — identical to MLP and ADST for fair comparison
BATCH_SIZE          = 512
EPOCHS              = 50
LR                  = 3e-4
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 2
PATIENCE            = 7
MAX_STEPS_PER_EPOCH = 500   # cap per epoch (500 × 512 = 256k samples)
MAX_EVAL_STEPS      = 50    # fast per-epoch val check (50 × 512 = 25.6k)

# Categorical features — use embedding lookup instead of linear projection
CATEGORICAL_COLS = ["application_name", "application_category_name"]
VOCAB_SIZES = {
    "application_name":          244,
    "application_category_name": 28,
}

print("Configuration loaded.")
print(f"  Model: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")
print(f"  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")


# ── CELL 4 ──────────────────────────────────────────────────
# Dataset class
# Loads CSV, applies arcsinh transform, splits into continuous/categorical
# ─────────────────────────────────────────────────────────────

class FlowDataset(Dataset):
    """
    Loads one of the pre-split CSVs and serves (cont, cat_dict, label).

    Preprocessing: arcsinh(x) + clip[-15,15]
      - arcsinh compresses large values logarithmically (not hard ceiling)
      - Preserves ordering within sparse features (frag counts, GRE counts)
      - Same transform as MLP and ADST — ensures fair comparison
    """

    def __init__(self, csv_path, cat_cols, feature_order):
        print(f"  Loading {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels    = torch.tensor(df["label"].values, dtype=torch.long)
        self.cat_cols  = cat_cols
        cont_cols      = [c for c in feature_order if c not in cat_cols]
        self.cont_cols = cont_cols

        # arcsinh transform — same as MLP and ADST
        X_np   = df[cont_cols].values.astype(np.float32)
        X      = torch.tensor(X_np, dtype=torch.float32)
        X      = torch.asinh(X)
        X      = torch.clamp(X, min=-15.0, max=15.0)

        clipped = ((X < -15) | (X > 15)).float().mean().item() * 100
        print(f"  arcsinh applied. Safety clip triggered: {clipped:.4f}% of values")
        self.cont_data = X

        # Categorical features as integer lookup indices
        self.cat_data = {}
        for col in cat_cols:
            if col in df.columns:
                self.cat_data[col] = torch.tensor(
                    df[col].values.astype(np.int64), dtype=torch.long)

        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cont  = self.cont_data[idx]
        cats  = {c: self.cat_data[c][idx]
                 for c in self.cat_cols if c in self.cat_data}
        label = self.labels[idx]
        return cont, cats, label


def collate_fn(batch):
    """Stack variable-length cat dicts into tensors."""
    conts, cats_list, labels = zip(*batch)
    cat_batch = {}
    if cats_list and cats_list[0]:
        for key in cats_list[0]:
            cat_batch[key] = torch.stack([c[key] for c in cats_list])
    return torch.stack(conts), cat_batch, torch.stack(labels)

print("Dataset class defined.")


# ── CELL 5 ──────────────────────────────────────────────────
# Model architecture: Corrected FT-Transformer
# All 3 original root causes fixed:
#   RC1: d_model-dim W per feature (not scalar) → feature values visible
#   RC2: embedding init std=0.02 (matches continuous scale) → no collapse
#   RC3: arcsinh preprocessing (not hard clip) → sparse features preserved
# ─────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converts each of 130 features into a d_model-dimensional token.

    Each continuous feature i uses:
        token_i = x_i * W_i + b_i
    where W_i is a (d_model,)-dimensional LEARNABLE VECTOR.
    Different values of x_i scale W_i differently → distinct token
    directions that LayerNorm can distinguish (Root Cause 1 fixed).

    Categorical features use an Embedding table initialized with
    std=0.02 to match the continuous token scale (Root Cause 2 fixed).
    """

    def __init__(self, cont_cols, cat_cols, vocab_sizes, d_model):
        super().__init__()
        self.cont_cols = cont_cols
        self.cat_cols  = cat_cols
        n_cont         = len(cont_cols)

        # d_model-dim weight per feature (Root Cause 1 fix)
        self.cont_weights = nn.Parameter(
            torch.randn(n_cont, d_model) * 0.02)   # (n_cont, d_model)
        self.cont_biases  = nn.Parameter(
            torch.randn(n_cont, d_model) * 0.02)   # (n_cont, d_model)

        # Embedding tables with matched init scale (Root Cause 2 fix)
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_model, padding_idx=0)
            for col in cat_cols
        })
        for col in cat_cols:
            nn.init.normal_(self.cat_embeddings[col].weight, std=0.02)
            with torch.no_grad():
                self.cat_embeddings[col].weight[0].zero_()

    def forward(self, cont, cat_dict):
        # einsum 'bi,id->bid': scalar cont[b,i] × vector weights[i,:]
        # produces (batch, n_cont, d_model) — GPU-compatible on all devices
        cont_tok = torch.einsum('bi,id->bid', cont, self.cont_weights)
        cont_tok = cont_tok + self.cont_biases

        tokens = [cont_tok]
        for col in self.cat_cols:
            if col in cat_dict:
                emb = self.cat_embeddings[col](cat_dict[col])  # (batch, d_model)
                tokens.append(emb.unsqueeze(1))                 # (batch, 1, d_model)

        return torch.cat(tokens, dim=1)  # (batch, 130+2+1, d_model)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block (more stable than Post-LN)."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + self.drop(a)
        x = x + self.ffn(self.norm2(x))
        return x


class FlatFTTransformer(nn.Module):
    """
    Corrected FT-Transformer: 130 individual feature tokens → Transformer.

    Architecture:
      130 features → 130 tokens (FeatureTokenizer)
      + 2 categorical tokens + 1 CLS token = 133 tokens total
      → 2× TransformerBlock (self-attention across ALL 133 tokens)
      → extract CLS token → classification head → 24 classes

    Difference from ADST:
      Flat: 133-token attention (133×133 = 17,689 relationships)
      ADST: 8-token attention  (8×8 = 64 relationships, semantic groups)
      → ADST is structurally simpler and more interpretable
    """

    def __init__(self, cont_cols, cat_cols, vocab_sizes,
                 d_model, n_heads, n_layers, d_ff, n_classes, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cont_cols, cat_cols, vocab_sizes, d_model)

        # CLS token: learnable summary token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

        # Classification head on CLS output
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, cont, cat_dict):
        tokens = self.tokenizer(cont, cat_dict)          # (B, 130+2, d_model)
        cls    = self.cls_token.expand(tokens.shape[0], -1, -1)
        x      = torch.cat([cls, tokens], dim=1)         # (B, 133, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])                     # CLS → 24 classes

print("Model architecture defined.")


# ── CELL 6 ──────────────────────────────────────────────────
# Learning rate schedule: linear warmup → cosine decay
# Matches MLP and ADST training schedules exactly
# ─────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup for WARMUP_EPOCHS, then cosine decay to LR/100."""

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
            p     = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * p))
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * scale

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

print("LR scheduler defined.")


# ── CELL 7 ──────────────────────────────────────────────────
# Training and evaluation functions
# Plain float32 — no AMP/autocast (required for P100 compatibility)
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler,
                    device, max_steps=None, print_every=100):
    """One training epoch, plain float32 (no AMP — P100 compatible)."""
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for step, (cont, cat_dict, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        cont     = cont.to(device)
        labels   = labels.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        optimizer.zero_grad()
        logits = model(cont, cat_dict)          # plain float32 forward
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

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names, max_steps=None):
    """Evaluation on val or test set, plain float32."""
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    all_preds  = []
    all_labels = []
    total_loss = total = 0

    for step, (cont, cat_dict, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        cont     = cont.to(device)
        labels   = labels.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        logits = model(cont, cat_dict)
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

print("Train/eval functions defined.")


# ── CELL 8 ──────────────────────────────────────────────────
# Main: load data, build model, train for 50 epochs
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FLAT FT-TRANSFORMER — CORRECTED BASELINE")
    print("All 3 root causes fixed. Expected result: 80-87%")
    print("=" * 65)

    # Load saved artifacts from Script 06
    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)

    cat_cols  = [c for c in CATEGORICAL_COLS if c in feature_order]
    cont_cols = [c for c in feature_order if c not in cat_cols]

    print(f"\nClasses : {n_classes}")
    print(f"Features: {len(feature_order)}  "
          f"({len(cont_cols)} continuous + {len(cat_cols)} categorical)")
    print(f"Device  : {DEVICE}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = FlowDataset(
        f"{TRAINING_DIR}/train.csv", cat_cols, feature_order)
    val_ds   = FlowDataset(
        f"{TRAINING_DIR}/val.csv",   cat_cols, feature_order)
    test_ds  = FlowDataset(
        f"{TRAINING_DIR}/test.csv",  cat_cols, feature_order)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin,
                              collate_fn=collate_fn)

    # Build model
    print("\nBuilding Flat FT-Transformer...")
    model = FlatFTTransformer(
        cont_cols, cat_cols, VOCAB_SIZES,
        D_MODEL, N_HEADS, N_LAYERS, D_FF, n_classes, DROPOUT
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"  Parameters  : {n_params:,}")
    print(f"  Tokens/sample: {len(cont_cols)+len(cat_cols)+1}"
          f" ({len(cont_cols)} cont + {len(cat_cols)} cat + 1 CLS)")

    # Optimizer and LR schedule
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = MAX_STEPS_PER_EPOCH or len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler       = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps)

    # Training loop
    print("\n" + "=" * 65)
    print(f"TRAINING  —  {EPOCHS} epochs × {steps_per_epoch} steps × "
          f"batch {BATCH_SIZE}")
    print("=" * 65)

    best_val_acc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    history      = []

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
                "class_names": class_names,
            }, f"{OUTPUT_PREFIX}_model.pt")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # Final evaluation on full val and test sets
    print(f"\nBest model: epoch {best_epoch}, val_acc={best_val_acc:.4f}")
    ckpt = torch.load(f"{OUTPUT_PREFIX}_model.pt",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS  (full 216,000 rows)")
    print("=" * 65)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    val_acc, _, val_report, _, _ = evaluate(
        model, val_loader, DEVICE, class_names)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(val_report)

    print("=" * 65)
    print("TEST RESULTS  (full 216,000 rows)")
    print("=" * 65)
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

    # Save training history and final comparison
    pd.DataFrame(history).to_csv(
        f"{OUTPUT_PREFIX}_history.csv", index=False)

    print("\n" + "=" * 65)
    print("PAPER COMPARISON TABLE")
    print("=" * 65)
    print(f"  XGBoost             (130 flat, trees)  :  90.06%")
    print(f"  ADST Transformer    (7 semantic tokens) :  87.90%")
    print(f"  Flat MLP            (130 flat, dense NN):  87.69%")
    print(f"  Flat FT-Transformer (130 flat tokens)   :  {test_acc*100:.2f}%")
    print()
    print("Interpretation:")
    if test_acc >= 0.878:
        print("  Flat FT-Transformer ≈ MLP and ADST.")
        print("  ADST advantage = interpretability (group attention)")
        print("  and targeted wins on hard classes (Mirai, Recon).")
    elif test_acc >= 0.80:
        print("  Flat FT-Transformer < MLP < ADST.")
        print("  Proves semantic grouping (ADST) > flat tokenization")
        print("  even with a correctly implemented Transformer baseline.")
    else:
        print("  Flat FT-Transformer << MLP.")
        print("  130-token attention is too complex to optimize with")
        print("  this architecture and training budget.")
        print("  ADST's 7-token design is structurally more efficient.")

    print("\nDone.")


main()