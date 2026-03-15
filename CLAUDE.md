# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end weakly-supervised deep learning pipeline to predict transcriptional subtypes of follicular lymphoma (FL) from H&E histology whole-slide images (WSIs). Goal: replace expensive RNA-seq subtyping with routine pathology image analysis.

**Task**: Binary classification — FL1 vs NotFL1 (collapsed due to small class sizes for FL4/FL6).

---

## Pipeline (run in order)

```bash
python data_prep.py          # Extract 224×224 patches from .svs WSIs
python ssl_train.py          # Barlow Twins SSL pretraining → barlow_encoder_resnet50.pth
python clustering.py         # Validate encoder: Leiden clustering + UMAP on patch embeddings
python feature_extraction.py # Encode patches → per-slide .npz bags
python mil_train.py          # Train Gated Attention MIL → slide_feats/attn_mil.pth
python attention_viz.py      # Overlay top-k attention patches on WSI thumbnails
```

To run all encoders: feature_extraction.py now runs ResNet SSL, ViT-B/16, and DINO sequentially,
outputting to slide_feats_resnet/, slide_feats_vit/, slide_feats_dino/ respectively.
To run a specific encoder set `ENCODER = "resnet" | "vit" | "dino"` at the top.
---

## Architecture

```
WSI (.svs) → data_prep.py → patches (PNG)
                               ↓
              ssl_train.py (Barlow Twins, ResNet-50)
                               ↓
         feature_extraction.py → slide_feats/*.npz
              (encoder: ViT-B/16, ResNet-50 SSL, or DINO ViT-S/8)
                               ↓
                    mil_train.py (Gated Attention MIL)
                               ↓
                    attention_viz.py (heatmaps)
```

### Key models

| Component | File | Details |
|-----------|------|---------|
| SSL pretraining | `ssl_train.py` | Barlow Twins, ResNet-50, 30 epochs, bs=128, lr=3e-4, λ=5e-3 |
| DINO encoder | `dino_encoder.py` | Drop-in ViT replacement via `torch.hub` |
| MIL classifier | `mil_train.py` | Gated Attention MIL, hidden=256, dropout=0.25, 10 epochs, lr=1e-4 |
| Interpretability | `attention_viz.py` | Overlays top-k attention patches as red squares on WSI thumbnail |

### Feature bags (.npz schema)
```python
{ 'feats': [N, d], 'coords': [N, 2], 'label': int, 'slide_id': str }
```

---

## Data & Paths

All paths are hardcoded for Kaggle. Update these constants at the top of each script before running locally:

| Script | Key path constants |
|--------|--------------------|
| `data_prep.py` | `SLIDE_DIR`, `SSL_OUT`, `SUP_OUT`, slide key Excel path |
| `ssl_train.py` | `PATCH_DIR`, `SAVE_PATH` |
| `feature_extraction.py` | `PATCH_ROOT`, `FEAT_DIR`, `SSL_MODEL_PATH` |
| `mil_train.py` | `FEAT_DIR`, `MODEL_SAVE_PATH` |
| `attention_viz.py` | `FEAT_DIR`, `MODEL_PATH`, `SLIDE_DIR` |

Slide metadata lives in `fl slide key.xlsx` with columns `SLIDE` and `FL subtype`.

---

## Dependencies

```
openslide-python torch torchvision pandas openpyxl scikit-learn
umap-learn leidenalg igraph opencv-python matplotlib seaborn tqdm
```

---

## Results

- ResNet-50 (SSL, in-domain): **78% acc, 0.65 AUC** — outperforms ViT because of domain adaptation
- ViT-B/16 (ImageNet): 71% acc, 0.58 AUC
- Cluster validation: Spearman ρ = 0.42, p < 0.001 (encoder captures real biology)

---

## Planned Work

- Replace ImageNet ViT with pathology foundation model (CONCH, UNI, or PLIP)
- Multimodal fusion: H&E embeddings + RNA-seq features (target >90% AUC)
- Multi-class classification as dataset grows
- TransMIL as alternative aggregator to Gated Attention MIL (in progress)
- Fix tissue_mask to use HSV saturation + Laplacian blur filtering (replacing grayscale threshold)
- Add checkpoint saving to ssl_train.py every 5 epochs

## Known Issues (being fixed)
- tissue_mask in data_prep.py currently uses grayscale thresholding — produces noisy patches.
  Fix: HSV saturation filter + Laplacian blur check.
- ssl_train.py has no checkpoint saving — weights lost on Kaggle session end.
- feature_extraction.py encodes patches one-by-one — needs batching (bs=64).