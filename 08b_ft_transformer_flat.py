"""
==========================================================
SCRIPT 08b : FLAT FT-TRANSFORMER BASELINE
==========================================================

PIPELINE POSITION:
  06_splitting_master_dataset.py  ->  [THIS SCRIPT]
  (same train/val/test splits as 08_xgboost_model_creation.py)

PURPOSE:
  This is the FLAT (non-semantic) Transformer baseline.
  Every one of the 130 features is treated as an equal,
  independent token. The Transformer learns feature
  interactions through self-attention with NO prior
  knowledge of which features belong together.

  This script is Step 1 of three:
    Step 1 (THIS):  Flat FT-Transformer  <- you are here
    Step 2 (08c):   ADST (manual semantic group tokens)
    Step 3 (08d):   Learned grouping (data-driven groups)

  Comparing Step 1 vs Step 2 is the core paper experiment.
  If ADST beats flat, the grouping matters. If flat matches
  ADST, it means XGBoost-level feature engineering already
  provides enough structure without explicit grouping.

ARCHITECTURE — FT-Transformer (Feature Tokenization Transformer):
  Each feature (continuous or categorical) is projected
  independently into a d_model-dimensional embedding.
  These 130 embeddings become 130 tokens. A CLS token is
  prepended, giving 131 tokens total. Multi-head self-
  attention is applied across all 131 tokens for L layers.
  The final CLS token representation is passed to a
  classification head that outputs 24 class logits.

  Why FT-Transformer over TabTransformer for this data:
    - TabTransformer applies attention ONLY to categorical
      columns; your data has only 2 categorical cols out
      of 130, so TabTransformer would ignore 98% of features
      in its attention mechanism.
    - FT-Transformer applies attention to ALL features,
      which is correct when the key signals are continuous
      (frag_ratio, gre_ratio, fan_in_src_count, ttl_mean).

  Why no positional encoding:
    Features are not a sequence — there is no natural
    "order" to protocol, ip_version, bidirectional_packets,
    etc. Positional encodings would add spurious structure.

OUTPUT:
  ft_flat_model.pt              <- saved model weights
  ft_flat_val_report.txt        <- validation metrics
  ft_flat_test_report.txt       <- test metrics
  ft_flat_feature_importance.csv <- attention-based feature importance
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
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score
)
from sklearn.exceptions import UndefinedMetricWarning

# Suppress sklearn warnings during early training epochs when
# model hasn't learned all 24 classes yet (expected behaviour)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ==========================================================
# SECTION 1: CONFIGURATION
# ==========================================================

TRAINING_DIR  = "TRAINING_DATA"
OUTPUT_PREFIX = "ft_flat"          # all output files start with this

# Model hyperparameters — tuned for 2GB GPU (NVIDIA T400)
# Full analysis showed batch=512 with d_model=64, 2 layers fits in ~530MB
# keeping ~800MB headroom for gradients and PyTorch overhead.
# This is a VALID FT-Transformer — smaller d_model is compensated by
# the large training set (1.44M rows gives enough gradient signal).
D_MODEL    = 64      # token embedding dimension (was 128, reduced for 2GB GPU)
N_HEADS    = 4       # attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS   = 2       # Transformer encoder blocks (was 3, reduced for 2GB GPU)
D_FF       = 256     # feedforward hidden dimension (was 512)
DROPOUT    = 0.1

# Training hyperparameters
BATCH_SIZE    = 512
EPOCHS        = 30
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 1
PATIENCE      = 5

# With clipped features [-10,10], the model converges much faster.
# 400 steps x 512 = 204,800 samples per epoch is sufficient.
# Each epoch ~8-10 min on T400. 30 epochs = ~4-5 hours.
MAX_STEPS_PER_EPOCH = 400

# Cap validation rows per eval — 50 batches = 25,600 rows, enough for
# stable accuracy tracking during full-epoch training.
MAX_EVAL_STEPS = 50

# Categorical columns that get embedding tables instead of linear projection
# These have discrete integer IDs (from app_name_vocab.json etc.)
CATEGORICAL_COLS = ["application_name", "application_category_name"]
# Vocabulary sizes (must match app_name_vocab.json and app_category_vocab.json)
VOCAB_SIZES = {
    "application_name": 244,    # 243 unique + 0 for unknown
    "application_category_name": 28,  # 27 unique + 0 for unknown
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==========================================================
# SECTION 2: DATASET CLASS
# ==========================================================

class FlowDataset(Dataset):
    """
    Loads one of the pre-split CSVs (train/val/test) and serves
    (features, label) pairs for the DataLoader.

    Categorical columns are separated and returned as integer
    tensors for embedding lookup. Continuous columns are returned
    as float tensors for linear projection.

    The label column is the integer-encoded class (0-23) saved
    by 06_splitting_master_dataset.py.
    """

    def __init__(self, csv_path, cat_cols, feature_order):
        print(f"  Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Separate categorical and continuous features
        # Categorical: integer tensors for embedding lookup
        # Continuous:  float32 tensors for linear projection
        self.cat_data  = {}
        cont_cols = [c for c in feature_order if c not in cat_cols]

        for col in cat_cols:
            if col in df.columns:
                self.cat_data[col] = torch.tensor(
                    df[col].values.astype(np.int64), dtype=torch.long
                )

        self.cont_data = torch.tensor(
            df[cont_cols].values.astype(np.float32), dtype=torch.float32
        )

        # CRITICAL: Clip features to [-10, 10] after RobustScaler.
        # RobustScaler uses median/IQR which works well for XGBoost
        # (tree splits don't care about absolute magnitude). But for
        # Transformers, W*x + b token magnitude is proportional to x.
        # Sparse features like dpkt_frag_mf_count (mostly 0, occasionally
        # 50,000+) create tokens 50,000x larger than other features,
        # causing attention to collapse onto whichever features have
        # the largest values in each sample — different for each class.
        # Clipping to [-10, 10] bounds all tokens to comparable magnitudes
        # without losing the ordering information (positive vs negative,
        # large vs small) that matters for classification.
        self.cont_data = torch.clamp(self.cont_data, min=-10.0, max=10.0)

        clipped_pct = (
            (torch.tensor(df[cont_cols].values.astype(np.float32)) < -10).sum() +
            (torch.tensor(df[cont_cols].values.astype(np.float32)) > 10).sum()
        ).item() / (len(df) * len(cont_cols)) * 100
        print(f"  Feature clip [-10,10]: {clipped_pct:.2f}% of values clipped")

        self.cat_cols  = cat_cols
        self.cont_cols = cont_cols
        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cont = self.cont_data[idx]
        cats = {col: self.cat_data[col][idx] for col in self.cat_cols if col in self.cat_data}
        label = self.labels[idx]
        return cont, cats, label


def collate_fn(batch):
    """Custom collate: handle dict of categorical tensors."""
    conts, cats_list, labels = zip(*batch)
    cont_batch  = torch.stack(conts)
    label_batch = torch.stack(labels)
    cat_batch   = {}
    if cats_list and cats_list[0]:
        for key in cats_list[0]:
            cat_batch[key] = torch.stack([c[key] for c in cats_list])
    return cont_batch, cat_batch, label_batch


# ==========================================================
# SECTION 3: FT-TRANSFORMER ARCHITECTURE
# ==========================================================

class FeatureTokenizer(nn.Module):
    """
    Converts each feature into a d_model-dimensional token.

    For continuous features:
        token_i = W_i * x_i + b_i
        Each feature has its OWN weight W_i (scalar) and bias b_i (vector).
        This is the key idea of FT-Transformer: per-feature learned projections
        rather than a shared linear layer across all features.
        The weight is scalar (single feature) multiplied by a d_model-dim bias.

    For categorical features:
        token_i = Embedding(x_i)  [nn.Embedding lookup]
        The integer value indexes into a learned embedding table.

    Output: tensor of shape (batch, n_tokens, d_model)
      where n_tokens = n_continuous + n_categorical
    """

    def __init__(self, cont_cols, cat_cols, vocab_sizes, d_model):
        super().__init__()
        self.cont_cols = cont_cols
        self.cat_cols  = cat_cols
        self.d_model   = d_model

        n_cont = len(cont_cols)

        # Per-feature weight: one scalar weight per continuous feature
        # Shape: (n_cont, 1) — broadcast multiplied with x_i
        self.cont_weights = nn.Parameter(torch.randn(n_cont, 1) * 0.02)

        # Per-feature bias: one d_model vector per continuous feature
        # Shape: (n_cont, d_model)
        # Small normal init (not zeros) gives each feature a distinct
        # starting direction in embedding space, helping early training
        self.cont_biases = nn.Parameter(torch.randn(n_cont, d_model) * 0.02)

        # Categorical embedding tables
        # CRITICAL: Initialize with std=0.02 to match the continuous token
        # magnitude (cont_weights are randn*0.02). Default PyTorch embedding
        # init is Normal(0,1) which produces tokens ~51x larger than
        # continuous tokens, causing attention to collapse entirely onto
        # the 2 categorical features and ignore all 128 continuous features.
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_model, padding_idx=0)
            for col in cat_cols
        })
        # Rescale all embedding weights to match continuous token scale
        for col in cat_cols:
            nn.init.normal_(self.cat_embeddings[col].weight, std=0.02)
            # Zero out the padding index row (keeps padding_idx=0 semantics)
            with torch.no_grad():
                self.cat_embeddings[col].weight[0].zero_()

    def forward(self, cont, cat_dict):
        """
        cont:     (batch, n_cont)  float32 — scaled continuous features
        cat_dict: dict {col_name: (batch,) int64} — categorical feature values

        Returns:  (batch, n_tokens, d_model)
        """
        batch_size = cont.shape[0]

        # Continuous: (batch, n_cont, 1) * (n_cont, 1) -> (batch, n_cont, 1)
        # then add bias: (n_cont, d_model) -> (batch, n_cont, d_model)
        cont_unsq = cont.unsqueeze(-1)              # (batch, n_cont, 1)
        cont_tok  = cont_unsq * self.cont_weights   # per-feature scaling
        cont_tok  = cont_tok + self.cont_biases     # (batch, n_cont, d_model)

        tokens = [cont_tok]

        # Categorical: embedding lookup -> (batch, 1, d_model) per column
        for col in self.cat_cols:
            if col in cat_dict:
                emb = self.cat_embeddings[col](cat_dict[col])  # (batch, d_model)
                tokens.append(emb.unsqueeze(1))                 # (batch, 1, d_model)

        return torch.cat(tokens, dim=1)   # (batch, n_tokens, d_model)


class TransformerBlock(nn.Module):
    """
    One Transformer encoder block using Pre-LayerNorm:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Pre-LN is used instead of Post-LN because:
      - More stable gradient flow during training
      - Does not require learning-rate warmup as strictly
      - Standard in modern Transformer implementations
        (GPT-2, BERT large, etc. switched to Pre-LN)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True    # (batch, seq, d_model) convention
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),          # GELU > ReLU for Transformers empirically
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class FlatFTTransformer(nn.Module):
    """
    Full FT-Transformer for multiclass IoT IDS classification.

    Architecture (in order):
      1. FeatureTokenizer:   130 features -> 130 tokens of shape (batch, 130, d_model)
      2. CLS token:          prepend learnable CLS token -> (batch, 131, d_model)
      3. TransformerBlock x N_LAYERS:  self-attention across all 131 tokens
      4. Final LayerNorm:    applied to the output sequence
      5. CLS extraction:     take token 0 -> (batch, d_model)
      6. Classification head: linear -> (batch, n_classes)

    The CLS token acts as the global "summary" of the flow.
    After N_LAYERS of attention, the CLS token has aggregated
    information from all 130 feature tokens.

    This is equivalent to how BERT uses [CLS] for classification —
    the CLS token attends to all features across all layers, learning
    a weighted summary of which features are most relevant for
    distinguishing attack classes.

    In Phase 2 (ADST), the 130 individual feature tokens will be
    REPLACED by 6 group-level tokens (one per semantic group).
    The architecture shell (blocks, CLS, head) stays identical —
    only the tokenizer changes. This makes the flat vs. ADST
    comparison a clean ablation: same architecture, different input.
    """

    def __init__(
        self,
        cont_cols, cat_cols, vocab_sizes,
        d_model, n_heads, n_layers, d_ff,
        n_classes, dropout
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            cont_cols, cat_cols, vocab_sizes, d_model
        )

        # Learnable CLS token embedding (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head with one hidden layer for regularisation
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, cont, cat_dict):
        """
        cont:     (batch, n_cont)  scaled continuous features
        cat_dict: dict of categorical tensors

        Returns:  (batch, n_classes) logits
        """
        # Tokenize all features
        tokens = self.tokenizer(cont, cat_dict)     # (batch, n_tokens, d_model)

        # Prepend CLS token
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls, tokens], dim=1)              # (batch, n_tokens+1, d_model)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token (index 0) for classification
        cls_out = x[:, 0, :]                    # (batch, d_model)
        return self.head(cls_out)               # (batch, n_classes)

    def get_attention_weights(self, cont, cat_dict):
        """
        Returns attention weights from the LAST layer for feature importance.
        Shape: (batch, n_heads, n_tokens+1, n_tokens+1)

        The row corresponding to CLS token (index 0) shows how much
        attention CLS paid to each feature token — this is a natural
        feature importance measure.
        """
        tokens = self.tokenizer(cont, cat_dict)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        for i, block in enumerate(self.blocks):
            normed = block.norm1(x)
            if i == len(self.blocks) - 1:
                # Capture attention weights from last layer
                _, attn_weights = block.attn(
                    normed, normed, normed,
                    need_weights=True,
                    average_attn_weights=False   # keep per-head weights
                )
                return attn_weights
            else:
                attn_out, _ = block.attn(normed, normed, normed)
                x = x + block.drop(attn_out)
                x = x + block.ffn(block.norm2(x))

        return None


# ==========================================================
# SECTION 4: LEARNING RATE SCHEDULE
# ==========================================================

class WarmupCosineScheduler:
    """
    Linear warmup for WARMUP_EPOCHS, then cosine annealing to LR/100.

    Why warmup:
      At init, the CLS token and feature embeddings are random.
      High LR immediately would cause destructive updates.
      Ramping up gradually lets the model settle into a good region.

    Why cosine decay:
      Smooth decay avoids the sharp drop of step-decay schedules.
      The model keeps refining as LR decreases — important for
      distinguishing fine-grained classes (Mirai subtypes, Recon).
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer      = optimizer
        self.warmup_steps   = warmup_steps
        self.total_steps    = total_steps
        self.min_lr_ratio   = min_lr_ratio
        self.base_lrs       = [g["lr"] for g in optimizer.param_groups]
        self._step          = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * scale

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ==========================================================
# SECTION 5: TRAINING AND EVALUATION FUNCTIONS
# ==========================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp,
                    max_steps=None, print_every=50):
    """
    One training epoch with mixed-precision (AMP) for GPU efficiency.

    max_steps: if set, stop after this many batches (used to cap epoch
               length on slow GPUs — e.g. max_steps=400 means each epoch
               sees 400 * 512 = 204,800 samples, completing in ~1-2 min
               on T400 instead of 10-20 min for the full 1,968 batches).
               The DataLoader shuffles every epoch so all training data
               is eventually seen across epochs.

    print_every: print progress every N batches so you can monitor
                 training without waiting for the full epoch.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    criterion = nn.CrossEntropyLoss()

    for step, (cont, cat_dict, labels) in enumerate(loader):

        # Cap epoch length for slow GPUs
        if max_steps is not None and step >= max_steps:
            break

        cont   = cont.to(device)
        labels = labels.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        optimizer.zero_grad()

        # Mixed precision forward pass — new API
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(cont, cat_dict)
            loss   = criterion(logits, labels)

        # Scaled backward + optimizer step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Print within-epoch progress so training feels responsive
        if (step + 1) % print_every == 0:
            running_acc  = correct / total
            running_loss = total_loss / total
            n_done = max_steps if max_steps else len(loader)
            print(f"    step {step+1:>4}/{n_done}  "
                  f"loss={running_loss:.4f}  acc={running_acc:.4f}",
                  flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names, use_amp=True, max_steps=None):
    """Full evaluation with classification report.
    max_steps: if set, evaluate only this many batches (fast per-epoch check).
               Set to None for full evaluation (used at end of training only).
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    total      = 0

    criterion = nn.CrossEntropyLoss()

    for step, (cont, cat_dict, labels) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        cont   = cont.to(device)
        labels = labels.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(cont, cat_dict)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    avg_loss = total_loss / total

    return acc, avg_loss, report, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def compute_attention_importance(model, loader, device, feature_names, n_batches=10):
    """
    Compute feature importance from CLS attention weights.

    For each batch, extract the attention weights from the last Transformer
    layer at the CLS token position (index 0), averaged across heads.
    This gives a score per feature token showing how much the CLS token
    attended to each feature when making its prediction.

    This is NOT SHAP — it is attention-based importance, which is faster
    but less precise. Use it as a quick sanity check that the model is
    paying attention to the right features (fragmentation, GRE, scan patterns).
    """
    model.eval()
    attn_accumulate = None
    count = 0

    for i, (cont, cat_dict, labels) in enumerate(loader):
        if i >= n_batches:
            break

        cont     = cont.to(device)
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}

        attn_weights = model.get_attention_weights(cont, cat_dict)
        if attn_weights is None:
            continue

        # attn_weights: (batch, heads, seq, seq)
        # CLS row (index 0): how much CLS attends to each token
        cls_attn = attn_weights[:, :, 0, 1:]  # (batch, heads, n_tokens)
        cls_attn = cls_attn.mean(dim=(0, 1))   # average over batch and heads
        cls_attn = cls_attn.cpu().numpy()

        if attn_accumulate is None:
            attn_accumulate = cls_attn
        else:
            attn_accumulate += cls_attn
        count += 1

    if attn_accumulate is None or count == 0:
        return None

    attn_accumulate /= count
    attn_accumulate /= attn_accumulate.sum()  # normalise to sum to 1

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Attention_Importance": attn_accumulate
    }).sort_values("Attention_Importance", ascending=False)

    return importance_df


# ==========================================================
# SECTION 6: MAIN TRAINING LOOP
# ==========================================================

def main():
    print("\n" + "=" * 60)
    print("FT-TRANSFORMER — FLAT BASELINE")
    print("=" * 60)

    # ---- Load artifacts from Script 06 ----
    print("\nLoading training artifacts...")
    le            = joblib.load(f"{TRAINING_DIR}/label_encoder.pkl")
    feature_order = joblib.load(f"{TRAINING_DIR}/feature_order.pkl")
    class_names   = list(le.classes_)
    n_classes     = len(class_names)

    print(f"  Classes: {n_classes}")
    print(f"  Features: {len(feature_order)}")

    # Separate continuous and categorical columns
    cat_cols  = [c for c in CATEGORICAL_COLS if c in feature_order]
    cont_cols = [c for c in feature_order if c not in cat_cols]
    print(f"  Continuous features: {len(cont_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")

    # ---- Build Datasets ----
    print("\nBuilding datasets...")
    train_dataset = FlowDataset(f"{TRAINING_DIR}/train.csv", cat_cols, feature_order)
    val_dataset   = FlowDataset(f"{TRAINING_DIR}/val.csv",   cat_cols, feature_order)
    test_dataset  = FlowDataset(f"{TRAINING_DIR}/test.csv",  cat_cols, feature_order)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # 0 = main process only, no forking
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,      # same as train batch — NOT *4, OOM on 2GB GPU
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,      # same as train batch
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ---- Build Model ----
    print("\nBuilding FT-Transformer...")
    model = FlatFTTransformer(
        cont_cols   = cont_cols,
        cat_cols    = cat_cols,
        vocab_sizes = VOCAB_SIZES,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        d_ff        = D_FF,
        n_classes   = n_classes,
        dropout     = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")
    print(f"  d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")

    # ---- Optimizer and Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    steps_per_epoch = MAX_STEPS_PER_EPOCH if MAX_STEPS_PER_EPOCH else len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = steps_per_epoch * WARMUP_EPOCHS

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01
    )

    # Mixed precision scaler (GPU speedup) — using new API for PyTorch >= 2.0
    # Falls back gracefully if cuda not available
    use_amp = (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training Loop ----
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_acc    = 0.0
    best_epoch      = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, DEVICE, use_amp,
            max_steps=MAX_STEPS_PER_EPOCH, print_every=20
        )

        # Free reserved GPU memory before evaluation
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Fast eval during training (subset of val set for speed)
        val_acc, val_loss, val_report, _, _ = evaluate(
            model, val_loader, DEVICE, class_names, use_amp,
            max_steps=MAX_EVAL_STEPS
        )

        elapsed = time.time() - t_start
        current_lr = scheduler.get_lr()[0]

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  time={elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "cont_cols": cont_cols,
                "cat_cols": cat_cols,
                "class_names": class_names,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
            }, f"{OUTPUT_PREFIX}_model.pt")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ---- Load Best Model and Final Evaluation ----
    print(f"\nBest model: epoch {best_epoch}, val_acc={best_val_acc:.4f}")
    print("Loading best model for final evaluation...")

    checkpoint = torch.load(f"{OUTPUT_PREFIX}_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Validation evaluation
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS")
    print("=" * 60)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    val_acc, val_loss, val_report, _, _ = evaluate(
        model, val_loader, DEVICE, class_names, use_amp
    )
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(val_report)

    # Test evaluation
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    test_acc, test_loss, test_report, y_true, y_pred = evaluate(
        model, test_loader, DEVICE, class_names, use_amp
    )
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(test_report)

    # Per-class recall sorted worst to best
    recalls = recall_score(y_true, y_pred, average=None)
    print("\nPer-class recall (sorted, worst first):")
    for name, rec in sorted(zip(class_names, recalls), key=lambda x: x[1]):
        bar = "█" * int(rec * 30)
        print(f"  {name:<30}  {rec:.3f}  {bar}")

    # ---- Attention-based Feature Importance ----
    print("\nComputing attention-based feature importance...")
    all_feature_names = cont_cols + cat_cols
    importance_df = compute_attention_importance(
        model, val_loader, DEVICE, all_feature_names, n_batches=20
    )
    if importance_df is not None:
        print("\nTop 20 features by attention (CLS → feature, last layer):")
        print(importance_df.head(20).to_string(index=False))
        importance_df.to_csv(f"{OUTPUT_PREFIX}_feature_importance.csv", index=False)
        print(f"Saved -> {OUTPUT_PREFIX}_feature_importance.csv")

    # ---- Save Reports and History ----
    with open(f"{OUTPUT_PREFIX}_val_report.txt", "w") as f:
        f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
        f.write(val_report)

    with open(f"{OUTPUT_PREFIX}_test_report.txt", "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(test_report)

    pd.DataFrame(history).to_csv(f"{OUTPUT_PREFIX}_history.csv", index=False)

    print(f"\nSaved -> {OUTPUT_PREFIX}_val_report.txt")
    print(f"Saved -> {OUTPUT_PREFIX}_test_report.txt")
    print(f"Saved -> {OUTPUT_PREFIX}_history.csv")

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:         Flat FT-Transformer (no semantic grouping)")
    print(f"Features:      {len(feature_order)} (all treated equally)")
    print(f"Parameters:    {n_params:,}")
    print(f"Best epoch:    {best_epoch}")
    print(f"Val accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print()
    print("XGBoost baseline for comparison:")
    print("  Test accuracy: 90.06%")
    print()
    if test_acc > 0.9006:
        print("FT-Transformer OUTPERFORMS XGBoost baseline.")
    elif test_acc > 0.8800:
        print("FT-Transformer is close to XGBoost — semantic grouping (08c) may close the gap.")
    else:
        print("FT-Transformer UNDERPERFORMS XGBoost — check training curves for underfitting.")
    print()
    print("Next step: run 08c_ft_transformer_adst.py (semantic group tokens)")
    print("Done.")


if __name__ == "__main__":
    main()
