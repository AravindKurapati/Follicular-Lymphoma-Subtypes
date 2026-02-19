"""
Visualize which WSI regions drove the MIL classifier's prediction
by overlaying top-k attention weights onto a slide thumbnail.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import openslide

from mil_train import AttnMIL, CLASS_NAMES

# ── CONFIG ───────────────────────────────────────────────────────────────────
FEAT_DIR    = "/kaggle/working/slide_feats"
DATASET_DIR = "/kaggle/input/mydataset"
MODEL_CKPT  = "/kaggle/working/slide_feats/attn_mil.pth"
PATCH_SIZE  = 224
TOP_K       = 200    # number of high-attention patches to highlight
THUMB_LEVEL = 2      # WSI pyramid level for thumbnail
SAT_THRESH  = 30     # saturation threshold for tissue mask
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: str, feat_dir: str) -> AttnMIL:
    """Load trained AttnMIL model from checkpoint."""
    import glob
    sample_npz = glob.glob(os.path.join(feat_dir, "val", "*.npz"))[0]
    in_dim = np.load(sample_npz)["feats"].shape[1]
    model = AttnMIL(in_dim, hidden=256, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def get_tissue_mask(slide: openslide.OpenSlide, level: int = 2, sat_thresh: int = 30):
    """
    Returns a tissue mask and thumbnail dimensions at the given pyramid level.
    Uses HSV saturation to separate tissue from glass/background.
    """
    from PIL import Image as PILImage
    import cv2

    dims = slide.level_dimensions[level]
    thumb = slide.get_thumbnail(dims).convert("RGB")
    img_hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
    mask = img_hsv[:, :, 1] > sat_thresh
    return mask, dims, thumb


def show_attention_on_thumbnail(
    slide_path: str,
    npz_path: str,
    model: AttnMIL,
    level: int = THUMB_LEVEL,
    k: int = TOP_K,
):
    """
    Overlay top-k attention patch locations on a WSI thumbnail.

    High-attention patches are marked with red squares.
    Reveals which tissue regions drove the model's FL subtype prediction.

    Args:
        slide_path: path to .svs WSI file
        npz_path:   path to pre-computed slide .npz feature bag
        model:      trained AttnMIL model
        level:      WSI pyramid level for thumbnail rendering
        k:          number of top-attention patches to highlight
    """
    data   = np.load(npz_path)
    h      = torch.from_numpy(data["feats"]).float().to(device)
    coords = data["coords"]   # [N, 2] level-0 (x, y)
    label  = int(data["label"])

    with torch.no_grad():
        logits, A = model(h)
        pred = int(torch.argmax(logits).item())

    A = A.cpu().numpy()

    with openslide.OpenSlide(slide_path) as slide:
        _, (tw, th), _ = get_tissue_mask(slide, level=level, sat_thresh=SAT_THRESH)
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")
        W0, H0 = slide.level_dimensions[0]

    sx, sy = tw / W0, th / H0
    thumb_arr = np.array(thumb).astype(np.float32)

    # Draw top-k attention patches as red squares
    idx = np.argsort(A)[-k:]
    for i in idx:
        cx = int((coords[i, 0] + PATCH_SIZE // 2) * sx)
        cy = int((coords[i, 1] + PATCH_SIZE // 2) * sy)
        r  = 6
        y0, y1 = max(0, cy - r), min(th, cy + r)
        x0, x1 = max(0, cx - r), min(tw, cx + r)
        thumb_arr[y0:y1, x0:x1, 0] = 255   # red channel
        thumb_arr[y0:y1, x0:x1, 1] = 0
        thumb_arr[y0:y1, x0:x1, 2] = 0

    true_label = CLASS_NAMES[label]
    pred_label = CLASS_NAMES[pred]

    plt.figure(figsize=(8, 8))
    plt.imshow(thumb_arr.astype(np.uint8))
    plt.axis("off")
    plt.title(
        f"Top-{k} attention regions\n"
        f"True: {true_label}  |  Predicted: {pred_label}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def run_all_val_slides():
    """Visualize attention maps for all validation slides."""
    import glob
    import pandas as pd
    from sklearn.model_selection import train_test_split

    slide_key = pd.read_excel("/kaggle/input/flslidekey/fl slide key.xlsx")
    slide_key["SLIDE"] = slide_key["SLIDE"].astype(str)
    _, val_df = train_test_split(
        slide_key, test_size=0.2,
        stratify=slide_key["FL subtype"], random_state=42
    )

    model = load_model(MODEL_CKPT, FEAT_DIR)

    for _, row in val_df.iterrows():
        slide_id   = str(row["SLIDE"])
        npz_path   = os.path.join(FEAT_DIR, "val", f"{slide_id}.npz")
        slide_path = os.path.join(DATASET_DIR, f"{slide_id}.svs")
        if os.path.exists(npz_path) and os.path.exists(slide_path):
            print(f"\nSlide: {slide_id}")
            show_attention_on_thumbnail(slide_path, npz_path, model)
        else:
            print(f"  [skip] {slide_id}: missing npz or svs")


if __name__ == "__main__":
    run_all_val_slides()
