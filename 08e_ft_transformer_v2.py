"""
==========================================================
SCRIPT 08a-v2 : FLAT FT-TRANSFORMER — MATCHED-CONFIG BASELINE
==========================================================

WHAT CHANGED FROM v1 (08a_ft_transformer_flat.py):

  This version exists ONLY to remove a confound in the ADST-vs-FT
  comparison. In v1, the FT-Transformer trained on 500 steps/epoch
  x batch 512 = 256,000 rows/epoch (25% of the training set), while
  ADST trained on the full 1,008,000 rows/epoch. Over 50 epochs that
  is a ~4x difference in total gradient updates seen, which alone
  could explain some or all of the accuracy gap between the two
  architectures. That is not a valid ablation.

  This version uses IDENTICAL training hyperparameters to the ADST
  script:
    - BATCH_SIZE          : 512  -> 2048
    - MAX_STEPS_PER_EPOCH  : 500  -> None (full dataset, ~493 batches)
    - MAX_EVAL_STEPS       : 50   -> None (full val/test set)
    - LR                   : 3e-4 -> 1e-3
  EPOCHS, WARMUP_EPOCHS, PATIENCE, and D_MODEL/N_HEADS/N_LAYERS/D_FF
  are unchanged (already matched ADST's D_TOKEN=64, heads=4, layers=2).

  Additionally, this version runs MULTIPLE SEEDS (42, 123, 2024) and
  reports mean +/- std test accuracy, so you can tell whether any
  observed gap vs ADST is a real architectural effect or just
  run-to-run noise from initialization / batch shuffling.

  Everything else (tokenizer, 3 root-cause fixes, architecture) is
  unchanged from v1 — only the training protocol changed.

USAGE:
  python 08a_ft_transformer_flat_matched.py
  (Trains 3 seeds sequentially, ~3x longer than a single run.
   If you only want one seed for a quick check, edit SEEDS below.)
==========================================================
"""

import os
import math
import time
import json
import joblib
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

# ==========================================================
# SECTION 1: CONFIGURATION — matched to ADST training protocol
# ==========================================================

TRAINING_DIR  = "TRAINING_DATA"
OUTPUT_PREFIX = "ft_flat_matched"

# Architecture — unchanged from v1, already matched ADST (d_token=64 etc.)
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 256
DROPOUT    = 0.1

# Training — NOW MATCHED TO ADST
BATCH_SIZE          = 2048   # was 512
EPOCHS               = 50
LR                    = 1e-3  # was 3e-4
WEIGHT_DECAY          = 1e-4
WARMUP_EPOCHS         = 2
PATIENCE              = 7
MAX_STEPS_PER_EPOCH   = None  # was 500 -> full dataset every epoch
MAX_EVAL_STEPS        = None  # was 50  -> full val/test set every eval

# Multiple seeds so the ADST-vs-FT gap can be checked against noise.
# Reduce to [42] for a single quick run.
SEEDS = [42, 123, 2024]

CATEGORICAL_COLS = ["application_name", "application_category_name"]
VOCAB_SIZES = {
    "application_name":          244,
    "application_category_name": 28,
}


def _gpu_works():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([1.0], device="cuda")
        _ = x + x
        return True
    except Exception:
        return False


if _gpu_works():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: GPU not usable. Falling back to CPU.")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# SECTION 2: DATASET  (unchanged from v1)
# ==========================================================

class FlowDataset(Dataset):
    def __init__(self, csv_path, cat_cols, feature_order):
        print(f"  Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels   = torch.tensor(df["label"].values, dtype=torch.long)
        self.cat_cols = cat_cols
        cont_cols     = [c for c in feature_order if c not in cat_cols]
        self.cont_cols = cont_cols

        X_np = df[cont_cols].values.astype(np.float32)
        X    = torch.tensor(X_np, dtype=torch.float32)
        X    = torch.asinh(X)
        X    = torch.clamp(X, min=-15.0, max=15.0)

        clipped = ((X < -15) | (X > 15)).float().mean().item() * 100
        print(f"  arcsinh + clip [-15,15]: {clipped:.4f}% clipped")
        self.cont_data = X

        self.cat_data = {}
        for col in cat_cols:
            if col in df.columns:
                self.cat_data[col] = torch.tensor(
                    df[col].values.astype(np.int64), dtype=torch.long)

        self.n_samples = len(df)

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        return (self.cont_data[idx],
                {c: self.cat_data[c][idx] for c in self.cat_cols
                 if c in self.cat_data},
                self.labels[idx])


def collate_fn(batch):
    conts, cats_list, labels = zip(*batch)
    cat_batch = {}
    if cats_list and cats_list[0]:
        for key in cats_list[0]:
            cat_batch[key] = torch.stack([c[key] for c in cats_list])
    return torch.stack(conts), cat_batch, torch.stack(labels)


# ==========================================================
# SECTION 3: ARCHITECTURE  (unchanged from v1 — both root-cause fixes intact)
# ==========================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, cont_cols, cat_cols, vocab_sizes, d_model):
        super().__init__()
        self.cont_cols = cont_cols
        self.cat_cols  = cat_cols
        n_cont         = len(cont_cols)

        self.cont_weights = nn.Parameter(torch.randn(n_cont, d_model) * 0.02)
        self.cont_biases  = nn.Parameter(torch.randn(n_cont, d_model) * 0.02)

        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_model, padding_idx=0)
            for col in cat_cols
        })
        for col in cat_cols:
            nn.init.normal_(self.cat_embeddings[col].weight, std=0.02)
            with torch.no_grad():
                self.cat_embeddings[col].weight[0].zero_()

    def forward(self, cont, cat_dict):
        cont_tok = torch.einsum('bi,id->bid', cont, self.cont_weights)
        cont_tok = cont_tok + self.cont_biases

        tokens = [cont_tok]
        for col in self.cat_cols:
            if col in cat_dict:
                emb = self.cat_embeddings[col](cat_dict[col])
                tokens.append(emb.unsqueeze(1))

        return torch.cat(tokens, dim=1)


class TransformerBlock(nn.Module):
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
        n      = self.norm1(x)
        a, _   = self.attn(n, n, n)
        x      = x + self.drop(a)
        x      = x + self.ffn(self.norm2(x))
        return x


class FlatFTTransformer(nn.Module):
    def __init__(self, cont_cols, cat_cols, vocab_sizes,
                 d_model, n_heads, n_layers, d_ff, n_classes, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cont_cols, cat_cols, vocab_sizes, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, cont, cat_dict):
        tokens = self.tokenizer(cont, cat_dict)
        cls    = self.cls_token.expand(tokens.shape[0], -1, -1)
        x      = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])


# ==========================================================
# SECTION 4: LR SCHEDULE  (unchanged)
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
            p     = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * p))
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * scale

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ==========================================================
# SECTION 5: TRAIN / EVAL  (unchanged logic, max_steps now None -> full data)
# ==========================================================

def train_one_epoch(model, loader, optimizer, scheduler,
                    device, max_steps=None, print_every=100):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    step = -1

    for step, (cont, cat_dict, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        cont     = cont.to(device)
        labels   = labels.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        optimizer.zero_grad()
        logits = model(cont, cat_dict)
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


# ==========================================================
# SECTION 6: SINGLE-SEED TRAINING RUN
# ==========================================================

def run_one_seed(seed, train_ds, val_ds, test_ds, cont_cols, cat_cols,
                  class_names, n_classes):
    print("\n" + "#" * 65)
    print(f"# SEED {seed}")
    print("#" * 65)
    set_seed(seed)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=pin, persistent_workers=True)

    model = FlatFTTransformer(
        cont_cols, cat_cols, VOCAB_SIZES,
        D_MODEL, N_HEADS, N_LAYERS, D_FF, n_classes, DROPOUT
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Train batches/epoch: {len(train_loader)} "
          f"(all {len(train_ds):,} rows used — matched to ADST)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)   # full dataset, same as ADST
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

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
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc}, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest: epoch {best_epoch}, val_acc={best_val_acc:.4f}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    test_acc, _, test_report, y_true, y_pred = evaluate(
        model, test_loader, DEVICE, class_names)
    print(f"Seed {seed} TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")

    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    macro_p = classification_report(y_true, y_pred, target_names=class_names,
                                     output_dict=True, zero_division=0)["macro avg"]

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
    }


# ==========================================================
# SECTION 7: MAIN — multi-seed orchestration
# ==========================================================

def main():
    print("\n" + "=" * 65)
    print("FLAT FT-TRANSFORMER — MATCHED-CONFIG, MULTI-SEED")
    print(f"Seeds: {SEEDS}")
    print("=" * 65)

    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)

    cat_cols  = [c for c in CATEGORICAL_COLS if c in feature_order]
    cont_cols = [c for c in feature_order if c not in cat_cols]

    print(f"\nClasses: {n_classes}  Features: {len(feature_order)} "
          f"({len(cont_cols)} cont + {len(cat_cols)} cat)")

    print("\nBuilding datasets (shared across all seeds)...")
    train_ds = FlowDataset(f"{TRAINING_DIR}/train.csv", cat_cols, feature_order)
    val_ds   = FlowDataset(f"{TRAINING_DIR}/val.csv",   cat_cols, feature_order)
    test_ds  = FlowDataset(f"{TRAINING_DIR}/test.csv",  cat_cols, feature_order)

    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, train_ds, val_ds, test_ds,
                              cont_cols, cat_cols, class_names, n_classes)
        all_results.append(result)

    # ---- Summary across seeds ----
    test_accs = np.array([r["test_acc"] for r in all_results])
    f1s       = np.array([r["macro_f1"] for r in all_results])

    print("\n" + "=" * 65)
    print("MULTI-SEED SUMMARY — Flat FT-Transformer (matched config)")
    print("=" * 65)
    for r in all_results:
        print(f"  seed={r['seed']:<6} test_acc={r['test_acc']*100:.2f}%  "
              f"macro_f1={r['macro_f1']:.4f}  params={r['n_params']:,}")
    print(f"\n  Mean test_acc : {test_accs.mean()*100:.2f}%  "
          f"(+/- {test_accs.std()*100:.2f} pp std)")
    print(f"  Mean macro_f1 : {f1s.mean():.4f}  (+/- {f1s.std():.4f} std)")
    print(f"  Parameters    : {all_results[0]['n_params']:,} "
          f"(identical across seeds — architecture unchanged)")

    with open(f"{OUTPUT_PREFIX}_multiseed_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PREFIX}_multiseed_summary.json")
    print("\nCompare this mean +/- std directly against the ADST multi-seed")
    print("summary. If the two ranges overlap, the architectural gap is")
    print("not distinguishable from seed noise at this sample size.")


if __name__ == "__main__":
    main()