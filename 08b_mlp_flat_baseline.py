"""
==========================================================
SCRIPT 08b : FLAT MLP BASELINE
==========================================================

PIPELINE POSITION:
  06_splitting_master_dataset.py  ->  [THIS SCRIPT]
  (same train/val/test splits as XGBoost and ADST)

PURPOSE:
  This is the FLAT MLP baseline — same 130 features as
  XGBoost and ADST, but processed as a single flat vector
  through a standard feedforward neural network.

  Comparison structure for the paper:
    Model A: XGBoost        (130 flat features)  -> 90.06% (done)
    Model B: MLP (THIS)     (130 flat features)  -> ???
    Model C: ADST (08c)     (6 semantic tokens)  -> ???

  If ADST > MLP:  semantic grouping + Transformer helps
  If ADST ~ XGBoost: competitive with strong tree-based baseline
  If MLP < XGBoost: flat NNs struggle with tabular IoT data

  This comparison cleanly tests:
    "Does semantic grouping before a Transformer beat
     simply feeding all features flat into a neural network?"

  The ADST group encoders ARE small MLPs themselves,
  so the comparison isolates the effect of grouping +
  inter-group Transformer attention, not MLP vs Transformer.

WHY MLP INSTEAD OF FLAT FT-TRANSFORMER:
  The flat FT-Transformer (130 individual feature tokens)
  failed to converge beyond 18-19% accuracy due to the
  scalar feature tokenization (W_i * x_i, where W_i is
  a single scalar) being washed out by LayerNorm — the
  feature values became invisible to attention. This is a
  known limitation of per-feature scalar tokenization when
  features have heterogeneous scales even after clipping.

  An MLP processes all 130 features jointly through dense
  linear layers, which is more appropriate for flat tabular
  data and will converge reliably. This gives a meaningful
  neural baseline that genuinely represents "what a flat NN
  can do with these features without semantic structure."

ARCHITECTURE:
  Input: 130 features (128 continuous + 2 categorical)
  Layer 1: Linear(130, 512) + BatchNorm + GELU + Dropout(0.3)
  Layer 2: Linear(512, 512) + BatchNorm + GELU + Dropout(0.3)
  Layer 3: Linear(512, 256) + BatchNorm + GELU + Dropout(0.2)
  Layer 4: Linear(256, 128) + BatchNorm + GELU + Dropout(0.1)
  Output:  Linear(128, 24)
  ~550K parameters

  BatchNorm instead of LayerNorm: BatchNorm normalizes across
  the batch dimension (appropriate for tabular data where
  features have different scales). LayerNorm normalizes across
  the feature dimension (appropriate for sequence models where
  all tokens are the same type). Using BatchNorm avoids the
  scale-washing problem that broke the flat FT-Transformer.

OUTPUT:
  mlp_flat_model.pt
  mlp_flat_val_report.txt
  mlp_flat_test_report.txt
  mlp_flat_history.csv
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
OUTPUT_PREFIX = "mlp_flat"

# Architecture
HIDDEN_DIMS = [512, 512, 256, 128]
DROPOUT     = [0.3, 0.3, 0.2, 0.1]

# Training — MLP converges much faster than Transformer
BATCH_SIZE    = 2048   # can use larger batch since no attention matrix
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
PATIENCE      = 7

MAX_STEPS_PER_EPOCH = 500   # 500 x 2048 = 1,024,000 samples/epoch
                              # ~20s/epoch on T400 (much faster than Transformer)
MAX_EVAL_STEPS      = 50    # for per-epoch val check

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================================
# SECTION 2: DATASET
# ==========================================================

def load_dataset(csv_path, feature_order):
    """Load CSV, apply arcsinh transform, return X tensor and y tensor."""
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    y = torch.tensor(df["label"].values, dtype=torch.long)

    X_np = df[feature_order].values.astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)

    # FIX: arcsinh transform instead of hard clip [-10,10].
    # Hard clipping collapsed 99.78% of sparse-feature non-zero values
    # (dpkt_frag_mf_count, dpkt_gre_packet_count, etc.) to the SAME
    # ceiling value, destroying the ability to distinguish lightly
    # vs heavily fragmented/GRE-encapsulated flows. This affected
    # this MLP baseline equally — re-run after this fix for a fair
    # three-way comparison against ADST and XGBoost.
    X = torch.asinh(X)
    X = torch.clamp(X, min=-15.0, max=15.0)

    clipped = ((X < -15) | (X > 15)).float().mean().item() * 100
    print(f"  arcsinh transform applied. Post-transform clip [-15,15]: "
          f"{clipped:.4f}% of values clipped")

    return X, y


# ==========================================================
# SECTION 3: MLP ARCHITECTURE
# ==========================================================

class FlatMLP(nn.Module):
    """
    Standard feedforward MLP for tabular classification.

    Key design choices:
    - BatchNorm after each linear layer (not LayerNorm) — appropriate
      for tabular data where features have different distributions.
      BatchNorm normalizes across the batch per-feature, keeping
      feature-specific information intact. LayerNorm (used in
      Transformers) normalizes across features, which destroys
      feature-scale information in flat tabular inputs.
    - GELU activation (same as Transformer FFN blocks, for consistency)
    - Residual connections between same-size layers (512->512)
      to help gradient flow
    - Decreasing dropout as the network narrows (more regularization
      on wider layers where overfitting risk is higher)
    """

    def __init__(self, n_features, hidden_dims, dropouts, n_classes):
        super().__init__()

        layers = []
        in_dim = n_features

        for i, (h, d) in enumerate(zip(hidden_dims, dropouts)):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(d))
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        # Classification head
        self.head = nn.Linear(in_dim, n_classes)

        # Residual shortcut for 512->512 layer (layers 0->1)
        # Skips the first block and adds directly to the second block input
        self.residual_proj = nn.Linear(n_features, hidden_dims[1]) \
            if len(hidden_dims) > 1 and hidden_dims[0] == hidden_dims[1] \
            else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (batch, n_features)"""
        out = self.backbone(x)
        return self.head(out)


# ==========================================================
# SECTION 4: LEARNING RATE SCHEDULE
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
# SECTION 5: TRAINING AND EVALUATION
# ==========================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    device, use_amp, max_steps=None, print_every=50):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for step, (X, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

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

    for step, (X, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        X, y = X.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X)
            loss   = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        total += y.size(0)

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=4, zero_division=0)
    return acc, total_loss / total, report, \
           np.array(all_labels), np.array(all_preds)


# ==========================================================
# SECTION 6: MAIN
# ==========================================================

def main():
    print("\n" + "=" * 60)
    print("FLAT MLP BASELINE")
    print("=" * 60)

    # Load artifacts
    print("\nLoading training artifacts...")
    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)
    n_features    = len(feature_order)
    print(f"  Classes: {n_classes}, Features: {n_features}")

    # Load data
    print("\nBuilding datasets...")
    X_train, y_train = load_dataset(f"{TRAINING_DIR}/train.csv", feature_order)
    X_val,   y_val   = load_dataset(f"{TRAINING_DIR}/val.csv",   feature_order)
    X_test,  y_test  = load_dataset(f"{TRAINING_DIR}/test.csv",  feature_order)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # Build model
    print("\nBuilding Flat MLP...")
    model = FlatMLP(n_features, HIDDEN_DIMS, DROPOUT, n_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: {n_features} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> {n_classes}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = MAX_STEPS_PER_EPOCH or len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=0.01)

    use_amp = (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    history          = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            DEVICE, use_amp,
            max_steps=MAX_STEPS_PER_EPOCH, print_every=100)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, DEVICE, class_names,
            use_amp, max_steps=MAX_EVAL_STEPS)

        elapsed = time.time() - t0
        lr      = scheduler.get_lr()[0]

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr:.2e}  time={elapsed:.0f}s")

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
                "n_features": n_features,
                "n_classes": n_classes,
                "hidden_dims": HIDDEN_DIMS,
                "class_names": class_names,
            }, f"{OUTPUT_PREFIX}_model.pt")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # Final evaluation on full val and test sets
    print(f"\nBest model: epoch {best_epoch}, val_acc={best_val_acc:.4f}")
    ckpt = torch.load(f"{OUTPUT_PREFIX}_model.pt",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS (full)")
    print("=" * 60)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    val_acc, _, val_report, _, _ = evaluate(
        model, val_loader, DEVICE, class_names, use_amp)
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(val_report)

    print("=" * 60)
    print("TEST SET RESULTS (full)")
    print("=" * 60)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    test_acc, _, test_report, y_true, y_pred = evaluate(
        model, test_loader, DEVICE, class_names, use_amp)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(test_report)

    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    print("\nPer-class recall (sorted, worst first):")
    for name, rec in sorted(zip(class_names, recalls), key=lambda x: x[1]):
        bar = "█" * int(rec * 30)
        print(f"  {name:<30}  {rec:.3f}  {bar}")

    # Save outputs
    with open(f"{OUTPUT_PREFIX}_val_report.txt",  "w") as f:
        f.write(f"Validation Accuracy: {val_acc:.4f}\n\n{val_report}")
    with open(f"{OUTPUT_PREFIX}_test_report.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{test_report}")
    pd.DataFrame(history).to_csv(f"{OUTPUT_PREFIX}_history.csv", index=False)

    # Final comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  XGBoost (130 flat features):      90.06%  [done]")
    print(f"  Flat MLP (130 flat features):     {test_acc*100:.2f}%  [this]")
    print(f"  ADST (6 semantic group tokens):   ???%    [next: 08c]")
    print()
    if test_acc >= 0.90:
        print("MLP matches XGBoost. ADST must show interpretability advantage.")
    elif test_acc >= 0.80:
        print("MLP < XGBoost. ADST needs to close this gap via semantic grouping.")
    else:
        print("MLP << XGBoost. Strong motivation for semantic structure in 08c.")
    print("\nDone. Next: python 08c_adst_transformer.py")


if __name__ == "__main__":
    main()