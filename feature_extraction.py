"""
Encode WSI patches into slide level feature bags (.npz files) using a
chosen patch encoder: ViT-B/16, ResNet-50 (SSL) and DINO ViT.

Each .npz contains:
  feats: [N_patches, embed_dim] float32
  coords: [N_patches, 2] int32  (level-0 x,y coordinates)
  label: int32 binary label
  slide_id: str

Set ENCODER = "vit" | "resnet" | "dino" in CONFIG below.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── CONFIG ───────────────────────────────────────────────────────────────────
PATCH_DIR    = "/kaggle/working/supervised_patches"   # <label>/<slide_id>/*.png
FEAT_DIR     = "/kaggle/working/slide_feats"
SLIDE_KEY    = "/kaggle/input/flslidekey/fl slide key.xlsx"
ENCODER      = "vit"       # "vit" | "resnet" | "dino"
RESNET_CKPT  = "/kaggle/working/barlow_encoder_resnet50.pth"  # only used if ENCODER == "resnet"
DINO_CKPT    = None        # path to DINO ViT checkpoint if using local weights
RANDOM_STATE = 42
CLASS_NAMES  = ["NotFL1", "FL1"]
LABEL_TO_ID  = {c: i for i, c in enumerate(CLASS_NAMES)}
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slide_label_str(row) -> str:
    return "FL1" if str(row["FL subtype"]) == "FL1" else "NotFL1"


def get_encoder(encoder: str = "vit", resnet_ckpt: str = None, dino_ckpt: str = None):
    """
    Returns (model, transforms, output_dim) for the chosen encoder.

    Supported encoders:
      "vit"    — ViT-B/16 pretrained on ImageNet (torchvision)
      "resnet" — ResNet-50 with optional SSL checkpoint
      "dino"   — DINO ViT-S/8 or ViT-B/8 via torch.hub (self-supervised)
    """
    if encoder == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        tfms = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        out_dim = 768

    elif encoder == "resnet":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        if resnet_ckpt and os.path.exists(resnet_ckpt):
            sd = torch.load(resnet_ckpt, map_location="cpu")
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                # tolerate keys prefixed with "backbone."
                sd_clean = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
                model.load_state_dict(sd_clean, strict=False)
            print(f"Loaded SSL weights from {resnet_ckpt}")
        tfms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        out_dim = 2048

    elif encoder == "dino":
        # DINO ViT-S/8 via torch.hub — replace with a pathology foundation model
        # checkpoint (CONCH, UNI, etc.) for best results
        if dino_ckpt and os.path.exists(dino_ckpt):
            # Load local DINO checkpoint
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits8", pretrained=False)
            sd = torch.load(dino_ckpt, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            print(f"Loaded DINO weights from {dino_ckpt}")
        else:
            # Fallback: download pretrained DINO ViT-S/8 from torch.hub
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
            print("Loaded DINO ViT-S/8 (ImageNet pretrained via torch.hub)")
        from torchvision import transforms
        tfms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        out_dim = 384   # ViT-S/8; use 768 for ViT-B/8

    else:
        raise ValueError(f"Unknown encoder: {encoder}. Choose 'vit', 'resnet', or 'dino'.")

    model.eval().to(device)
    return model, tfms, out_dim


def _slide_patch_paths(slide_id: str, label_str: str) -> list:
    d = os.path.join(PATCH_DIR, label_str, slide_id)
    return sorted(glob.glob(os.path.join(d, "*.png")))


def _coords_from_name(path: str):
    """Parse (x, y) level-0 coordinates from patch filename 'x_y.png'."""
    name = os.path.splitext(os.path.basename(path))[0]
    x_str, y_str = name.split("_")
    return int(x_str), int(y_str)


def build_npz_for_split(df_split: pd.DataFrame, split_name: str,
                        encoder: str = "vit", resnet_ckpt: str = None, dino_ckpt: str = None):
    out_dir = os.path.join(FEAT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    model, tfms, _ = get_encoder(encoder, resnet_ckpt, dino_ckpt)

    for _, row in df_split.iterrows():
        slide_id  = str(row["SLIDE"])
        label_str = slide_label_str(row)
        label_id  = LABEL_TO_ID[label_str]
        paths     = _slide_patch_paths(slide_id, label_str)
        if not paths:
            print(f"  [{split_name}] No patches found for slide {slide_id}")
            continue

        feats, coords = [], []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                inp = tfms(img).unsqueeze(0).to(device)
                f   = model(inp).squeeze(0).detach().cpu().numpy()
                feats.append(f)
                coords.append(_coords_from_name(p))

        feats  = np.asarray(feats,  dtype=np.float32)
        coords = np.asarray(coords, dtype=np.int32)
        out    = os.path.join(out_dir, f"{slide_id}.npz")
        np.savez_compressed(out, feats=feats, coords=coords,
                            label=np.int32(label_id), slide_id=np.array(slide_id))
        print(f"  [{split_name}] {slide_id}: {feats.shape[0]} patches → {out}")


def build_all_npz(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Build .npz files, skipping slides that already exist."""
    for split_name, df_split in [("train", train_df), ("val", val_df)]:
        exist = set(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(FEAT_DIR, split_name, "*.npz"))
        )
        need = [r for _, r in df_split.iterrows() if str(r["SLIDE"]) not in exist]
        if not need:
            print(f"[{split_name}] All NPZs already present — skipping.")
            continue
        print(f"[{split_name}] Building {len(need)} NPZs with encoder='{ENCODER}'...")
        build_npz_for_split(pd.DataFrame(need), split_name, ENCODER, RESNET_CKPT, DINO_CKPT)


def main():
    slide_key = pd.read_excel(SLIDE_KEY)
    slide_key["SLIDE"] = slide_key["SLIDE"].astype(str)

    train_df, val_df = train_test_split(
        slide_key, test_size=0.2,
        stratify=slide_key["FL subtype"],
        random_state=RANDOM_STATE,
    )
    build_all_npz(train_df, val_df)


if __name__ == "__main__":
    main()
