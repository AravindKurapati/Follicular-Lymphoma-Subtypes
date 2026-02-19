"""
DINO ViT encoder for patch-level feature extraction.

DINO (Self-DIstillation with NO labels) produces spatially coherent
attention maps that map well to semantically meaningful regions in
histology: follicles, immune infiltrates, stroma. Without any labels.

This module is a drop in replacement for the ImageNet ViT used in
feature_extraction.py. Set ENCODER = "dino" there to use it.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms


# ── CONFIG ───────────────────────────────────────────────────────────────────
# Choose model size: "vits8" (384-dim), "vitb8" (768-dim)
DINO_MODEL   = "dino_vits8"
# Optional: path to local DINO or pathology foundation model weights
LOCAL_CKPT   = None
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet normalization (matches DINO pretraining)
DINO_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Output dims per model
EMBED_DIMS = {
    "dino_vits8":  384,
    "dino_vits16": 384,
    "dino_vitb8":  768,
    "dino_vitb16": 768,
}


def load_dino_encoder(model_name: str = DINO_MODEL, local_ckpt: str = None) -> nn.Module:
    """
    Load a DINO ViT encoder from torch.hub or a local checkpoint.

    For production use on pathology data, replace the hub model with a
    pathology-specific foundation model:
      - CONCH: https://github.com/mahmoodlab/CONCH
      - UNI:   https://github.com/mahmoodlab/UNI
      - PLIP:  https://github.com/PathologyFoundation/plip

    Args:
        model_name: DINO model identifier (e.g. "dino_vits8")
        local_ckpt:  optional path to a local state_dict (.pth)

    Returns:
        model: eval-mode encoder on device (classification head removed)
    """
    model = torch.hub.load("facebookresearch/dino:main", model_name, pretrained=(local_ckpt is None))

    if local_ckpt and os.path.exists(local_ckpt):
        sd = torch.load(local_ckpt, map_location="cpu")
        # Handle common checkpoint formats
        if "model" in sd:
            sd = sd["model"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded DINO weights from: {local_ckpt}")
    else:
        print(f"Loaded pretrained {model_name} from torch.hub (facebookresearch/dino)")

    model.eval().to(device)
    return model


def get_dino_encoder_and_transforms(model_name: str = DINO_MODEL, local_ckpt: str = None):
    """
    Convenience function returning (model, transforms, embed_dim).
    Compatible with the get_encoder() interface in feature_extraction.py.
    """
    model   = load_dino_encoder(model_name, local_ckpt)
    out_dim = EMBED_DIMS.get(model_name, 384)
    return model, DINO_TRANSFORMS, out_dim


def visualize_dino_attention(model: nn.Module, img_tensor: torch.Tensor, patch_size: int = 8):
    """
    Extract and visualize DINO self-attention maps for a single image.

    DINO's attention heads reveal which spatial regions the model considers
    important — in histology this often corresponds to follicle boundaries,
    immune cell clusters, and stromal regions.

    Args:
        model:       DINO ViT model (eval mode)
        img_tensor:  [1, 3, H, W] preprocessed image tensor
        patch_size:  ViT patch size (8 for vits8/vitb8, 16 for vits16/vitb16)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    img_tensor = img_tensor.to(device)
    w_featmap = img_tensor.shape[-1] // patch_size
    h_featmap = img_tensor.shape[-2] // patch_size

    with torch.no_grad():
        attentions = model.get_last_selfattention(img_tensor)  # [1, heads, N+1, N+1]

    nh = attentions.shape[1]
    # Keep only attention from [CLS] token to patches
    attentions = attentions[0, :, 0, 1:].reshape(nh, h_featmap, w_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0),
        scale_factor=patch_size,
        mode="nearest"
    )[0].cpu().numpy()

    fig, axes = plt.subplots(1, nh + 1, figsize=(4 * (nh + 1), 4))

    # Original image
    img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np); axes[0].set_title("Input patch"); axes[0].axis("off")

    for i in range(nh):
        axes[i + 1].imshow(attentions[i], cmap="inferno")
        axes[i + 1].set_title(f"Head {i + 1}")
        axes[i + 1].axis("off")

    plt.suptitle("DINO Self-Attention Maps", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick sanity check
    model, tfms, out_dim = get_dino_encoder_and_transforms(DINO_MODEL, LOCAL_CKPT)
    print(f"Encoder: {DINO_MODEL} | Output dim: {out_dim}")

    # Test with a random tensor
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feat = model(dummy)
    print(f"Feature shape: {feat.shape}")   # should be [1, out_dim]
