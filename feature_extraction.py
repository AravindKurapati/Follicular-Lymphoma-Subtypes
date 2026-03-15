"""
Encode WSI patches into slide-level feature bags (.npz files).

Runs all three encoders sequentially, each writing to its own output directory:
  slide_feats_resnet/  — ResNet-50 SSL (2048-dim)
  slide_feats_vit/     — ViT-B/16 ImageNet (768-dim)
  slide_feats_dino/    — DINO ViT-B/16 (768-dim)

To run a single encoder set ENCODERS = ["resnet"] (or "vit" / "dino") below.

Each .npz contains:
  feats:    [N_patches, embed_dim] float32
  coords:   [N_patches, 2]        int32   (level-0 x,y coordinates)
  label:    int32                 binary label (0=NotFL1, 1=FL1)
  slide_id: str
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── CONFIG ───────────────────────────────────────────────────────────────────
PATCH_DIR    = "/kaggle/working/supervised_patches"   # <label>/<slide_id>/*.png
BASE_FEAT_DIR = "/kaggle/working"
SLIDE_KEY    = "/kaggle/input/flslidekey/fl slide key.xlsx"
RESNET_CKPT  = "/kaggle/working/barlow_encoder_resnet50.pth"
ENCODERS     = ["resnet", "vit", "dino"]   # set to a subset to run only those
BATCH_SIZE   = 64
RANDOM_STATE = 42
CLASS_NAMES  = ["NotFL1", "FL1"]
LABEL_TO_ID  = {c: i for i, c in enumerate(CLASS_NAMES)}

ENCODER_FEAT_DIRS = {
    "resnet": os.path.join(BASE_FEAT_DIR, "slide_feats_resnet"),
    "vit":    os.path.join(BASE_FEAT_DIR, "slide_feats_vit"),
    "dino":   os.path.join(BASE_FEAT_DIR, "slide_feats_dino"),
}
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ── Encoder factory ──────────────────────────────────────────────────────────

def get_encoder(encoder: str):
    """
    Returns (model, transform, out_dim) for the chosen encoder.

      "resnet" — ResNet-50 backbone loaded from barlow_encoder_resnet50.pth (SSL)
      "vit"    — ViT-B/16 pretrained on ImageNet via torchvision
      "dino"   — DINO ViT-B/16 from facebookresearch/dino on torch.hub
    """
    if encoder == "resnet":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = nn.Identity()
        sd = torch.load(RESNET_CKPT, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print(f"[resnet] Loaded SSL weights from {RESNET_CKPT}")
        tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            _IMAGENET_NORM,
        ])
        out_dim = 2048

    elif encoder == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        tfms = weights.transforms()
        out_dim = 768

    elif encoder == "dino":
        model = torch.hub.load(
            "facebookresearch/dino:main", "dino_vitb16", pretrained=True
        )
        print("[dino] Loaded DINO ViT-B/16 from torch.hub")
        tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            _IMAGENET_NORM,
        ])
        out_dim = 768

    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: resnet, vit, dino.")

    model.eval().to(device)
    return model, tfms, out_dim


# ── Patch dataset ────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """Loads all patches for a single slide."""
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slide_patch_paths(slide_id: str, label_str: str) -> list:
    d = os.path.join(PATCH_DIR, label_str, slide_id)
    return sorted(glob.glob(os.path.join(d, "*.png")))


def _coords_from_paths(paths: list) -> np.ndarray:
    """Parse (x, y) level-0 coordinates from patch filenames 'x_y.png'."""
    coords = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        x, y = name.split("_")
        coords.append((int(x), int(y)))
    return np.asarray(coords, dtype=np.int32)


def slide_label_str(row) -> str:
    return "FL1" if str(row["FL subtype"]) == "FL1" else "NotFL1"


# ── Per-slide encoding ───────────────────────────────────────────────────────

def encode_slide(paths: list, model: nn.Module, tfms, out_dim: int) -> np.ndarray:
    """Batched inference over all patches for one slide. Returns [N, out_dim]."""
    ds = PatchDataset(paths, tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = model(batch).detach().cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0).astype(np.float32)


# ── Build NPZs for one encoder ───────────────────────────────────────────────

def build_npz_for_encoder(df: pd.DataFrame, encoder: str):
    feat_dir = ENCODER_FEAT_DIRS[encoder]
    model, tfms, out_dim = get_encoder(encoder)

    for split_name, split_df in df:
        out_dir = os.path.join(feat_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        existing = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(out_dir, "*.npz"))
        }

        for _, row in split_df.iterrows():
            slide_id  = str(row["SLIDE"])
            label_str = slide_label_str(row)
            label_id  = LABEL_TO_ID[label_str]

            if slide_id in existing:
                print(f"  [{encoder}/{split_name}] {slide_id}: already exists — skipping")
                continue

            paths = _slide_patch_paths(slide_id, label_str)
            if not paths:
                print(f"  [{encoder}/{split_name}] {slide_id}: no patches found — skipping")
                continue

            feats  = encode_slide(paths, model, tfms, out_dim)
            coords = _coords_from_paths(paths)
            out    = os.path.join(out_dir, f"{slide_id}.npz")
            np.savez_compressed(
                out,
                feats=feats,
                coords=coords,
                label=np.int32(label_id),
                slide_id=np.array(slide_id),
            )
            print(f"  [{encoder}/{split_name}] {slide_id}: {feats.shape[0]} patches "
                  f"({out_dim}-dim) → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    slide_key = pd.read_excel(SLIDE_KEY)
    slide_key["SLIDE"] = slide_key["SLIDE"].astype(str)

    train_df, val_df = train_test_split(
        slide_key, test_size=0.2,
        stratify=slide_key["FL subtype"],
        random_state=RANDOM_STATE,
    )
    splits = [("train", train_df), ("val", val_df)]

    for encoder in ENCODERS:
        print(f"\n{'='*60}")
        print(f"Encoder: {encoder}")
        print(f"{'='*60}")
        build_npz_for_encoder(splits, encoder)

    print("\nDone.")


if __name__ == "__main__":
    main()
