"""
data_prep.py
------------
Extract and filter 224x224 tissue patches from whole slide images (.svs).
Saves patches to:
  ssl_patches/      <slide_id>/*.png           (for SSL pretraining)
  supervised_patches/<label>/<slide_id>/*.png  (for supervised training)
"""

import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import openslide

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_DIR  = "/kaggle/input/mydataset"
SLIDE_KEY    = "/kaggle/input/flslidekey/fl slide key.xlsx"
PATCH_SIZE   = 224
STRIDE       = 224
LEVEL        = 0
MAX_PATCHES  = 800
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_slide_key(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["SLIDE"] = df["SLIDE"].astype(str)
    df["filename"] = df["SLIDE"] + ".svs"
    print(df["FL subtype"].value_counts())
    return df


def tissue_mask(patch: Image.Image, white_thresh: float = 0.6, black_thresh: float = 0.2) -> bool:
    """
    Returns True if patch contains sufficient tissue content.

    Args:
        patch: PIL Image (RGB)
        white_thresh: max fraction of pixels considered white (>0.9 intensity)
        black_thresh: max fraction of pixels considered black (<0.1 intensity)
    """
    gray = patch.convert("L")
    arr = np.array(gray) / 255.0
    white_frac = np.mean(arr > 0.9)
    black_frac = np.mean(arr < 0.1)
    return (white_frac < white_thresh) and (black_frac < black_thresh)


def extract_patches(slide_path: str, slide_id: str, label: str, max_patches: int = MAX_PATCHES):
    """
    Extract tissue-containing patches from a WSI and save to disk.

    Args:
        slide_path: Path to .svs file
        slide_id:   Unique slide identifier
        label:      Binary label string ("FL1" or "NotFL1")
        max_patches: Maximum patches to extract per slide
    """
    ssl_dir  = os.path.join("/kaggle/working/ssl_patches", slide_id)
    sup_dir  = os.path.join("/kaggle/working/supervised_patches", label, slide_id)
    os.makedirs(ssl_dir, exist_ok=True)
    os.makedirs(sup_dir, exist_ok=True)

    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[LEVEL]

    count = 0
    for x in range(0, w, STRIDE):
        for y in range(0, h, STRIDE):
            if count >= max_patches:
                return
            patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            if tissue_mask(patch):
                fname = f"{x}_{y}.png"
                patch.save(os.path.join(ssl_dir, fname))
                patch.save(os.path.join(sup_dir, fname))
                count += 1


def main():
    slide_key = load_slide_key(SLIDE_KEY)

    train_df, val_df = train_test_split(
        slide_key,
        test_size=0.2,
        stratify=slide_key["FL subtype"],
        random_state=RANDOM_STATE,
    )

    for split_name, df in [("train", train_df), ("val", val_df)]:
        print(f"\nExtracting {split_name} patches...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            slide_id     = row["SLIDE"]
            binary_label = "FL1" if row["FL subtype"] == "FL1" else "NotFL1"
            slide_path   = os.path.join(DATASET_DIR, f"{slide_id}.svs")
            if os.path.exists(slide_path):
                extract_patches(slide_path, slide_id, binary_label)
            else:
                print(f"  [skip] {slide_path} not found")


if __name__ == "__main__":
    main()
