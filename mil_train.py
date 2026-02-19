"""
Train a Gated Attention Multiple Instance Learning (MIL) classifier
on slide-level .npz feature bags to predict FL subtype.

Features bags are produced by feature_extraction.py.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from collections import Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────
FEAT_DIR          = "/kaggle/working/slide_feats"
MAX_PATCHES_TRAIN = 800
MAX_PATCHES_VAL   = 1200
EPOCHS            = 10
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
HIDDEN            = 256
DROPOUT           = 0.25
SEED              = 42
CLASS_NAMES       = ["NotFL1", "FL1"]
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SlideBagDataset(Dataset):
    """
    Loads pre-computed slide feature bags from .npz files.

    Each .npz contains:
      feats  — [N, d] patch embeddings
      coords — [N, 2] level-0 coordinates
      label  — int binary label
    """
    def __init__(self, npz_paths: list, max_patches: int = None, train: bool = True):
        self.paths = npz_paths
        self.max_patches = max_patches
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        data = np.load(self.paths[i])
        h = data["feats"]      # [N, d]
        y = int(data["label"])
        c = data["coords"]     # [N, 2]
        if self.train and self.max_patches and h.shape[0] > self.max_patches:
            idx = np.random.choice(h.shape[0], self.max_patches, replace=False)
            h, c = h[idx], c[idx]
        return torch.from_numpy(h), torch.tensor(y), torch.from_numpy(c)


def collate_bags(batch):
    """Custom collate for variable-length bags."""
    xs, ys, coords = zip(*batch)
    return xs, torch.stack(ys), coords


# ── Model ────────────────────────────────────────────────────────────────────

class AttnMIL(nn.Module):
    """
    Gated Attention MIL classifier.

    Architecture:
      V(h) = Tanh(W_V · h)        — content projection
      U(h) = Sigmoid(W_U · h)     — gating
      A    = Softmax(w · (V ⊙ U)) — per-patch attention weights
      z    = Σ A_i · h_i           — weighted slide representation
      logits = Dropout → Linear(z)

    Args:
        in_dim:      patch embedding dimension
        hidden:      attention hidden dimension
        num_classes: number of output classes
        gated:       whether to use gated (V ⊙ U) or plain attention
        dropout:     dropout rate on classifier head
    """
    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 num_classes: int = 2, gated: bool = True, dropout: float = DROPOUT):
        super().__init__()
        self.V   = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U   = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid()) if gated else None
        self.w   = nn.Linear(hidden, 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [N, d] patch embeddings (float32, single slide)
        Returns:
            logits: [num_classes]
            A:      [N] attention weights
        """
        v = self.V(h)
        a = v * self.U(h) if self.U else v
        A = torch.softmax(self.w(a).squeeze(1), dim=0)          # [N]
        z = torch.sum(A.unsqueeze(1) * h, dim=0, keepdim=True)  # [1, d]
        logits = self.cls(z).squeeze(0)                          # [C]
        return logits, A


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, num_classes: int):
    y_true, y_pred, y_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for xs, ys, _ in loader:
            (h,), (y,) = xs, ys
            h = h.to(device).float()
            logits, _ = model(h)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            y_true.append(int(y.item()))
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1]) if num_classes == 2 \
              else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return acc, cm, y_true, y_pred, y_prob, auc


def plot_results(cm, y_true, y_prob, num_classes, class_names):
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    th = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    plt.tight_layout(); plt.show()

    # ROC curve
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.show()


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    train_paths = sorted(glob.glob(os.path.join(FEAT_DIR, "train", "*.npz")))
    val_paths   = sorted(glob.glob(os.path.join(FEAT_DIR, "val",   "*.npz")))
    assert train_paths, f"No .npz files found in {FEAT_DIR}/train"

    train_loader = DataLoader(
        SlideBagDataset(train_paths, max_patches=MAX_PATCHES_TRAIN, train=True),
        batch_size=1, shuffle=True, collate_fn=collate_bags
    )
    val_loader = DataLoader(
        SlideBagDataset(val_paths, max_patches=MAX_PATCHES_VAL, train=False),
        batch_size=1, shuffle=False, collate_fn=collate_bags
    )

    # Infer embedding dimension from first file
    in_dim = np.load(train_paths[0])["feats"].shape[1]
    num_classes = len(CLASS_NAMES)
    print(f"Encoder dim: {in_dim} | Classes: {CLASS_NAMES}")

    model = AttnMIL(in_dim, hidden=HIDDEN, num_classes=num_classes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xs, ys, _ in train_loader:
            (h,), (y,) = xs, ys
            h = h.to(device).float()
            y = y.to(device).long()
            logits, _ = model(h)
            loss = crit(logits.unsqueeze(0), y.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        acc, _, _, _, _, auc = evaluate(model, val_loader, num_classes)
        print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | "
              f"val_acc={acc:.3f} | val_auc={auc:.3f}")

    # Final evaluation and plots
    acc, cm, y_true, y_pred, y_prob, auc = evaluate(model, val_loader, num_classes)
    print(f"\nFinal  val_acc={acc:.3f} | val_auc={auc:.3f}")
    plot_results(cm, y_true, y_prob, num_classes, CLASS_NAMES)

    # Save model
    save_path = os.path.join(FEAT_DIR, "attn_mil.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    return model


if __name__ == "__main__":
    train()
