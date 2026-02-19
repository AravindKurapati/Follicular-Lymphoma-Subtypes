# Pathology Meets Genomics: Predicting FL Transcriptional Subtypes from H&E Histology

**Can deep learning infer molecular subtypes from routine pathology images?**

This project investigates whether **H&E-stained whole slide images (WSIs)** can predict the **transcriptional subtypes of follicular lymphoma (FL)** traditionally defined using RNA sequencing.

---

## Motivation

Follicular lymphoma is clinically heterogeneous. ~20% of patients progress rapidly with poor outcomes, while others remain indolent for years. RNA-seq reveals transcriptional subtypes that predict prognosis, but:

- **RNA-seq is expensive, slow, and unavailable in most labs**
- **H&E slides are free, instant, and everywhere**

The question: Can we infer molecular subtypes from morphology alone?

---

## Repository Structure

```
Follicular-Lymphoma-Subtypes/
│
├── data_prep.py          # WSI patch extraction & tissue filtering
├── ssl_train.py          # Barlow Twins self-supervised pretraining (ResNet-50)
├── clustering.py         # Leiden clustering + UMAP visualization
├── feature_extraction.py # Encode patches → .npz slide feature bags (ViT or ResNet)
├── mil_train.py          # Gated Attention MIL training & evaluation
├── dino_encoder.py       # DINO ViT encoder (planned replacement for ImageNet ViT)
├── attention_viz.py      # Attention heatmap overlay on WSI thumbnails
└── README.md
```

---

## Pipeline Overview

```
Raw WSI (gigapixel)
  ↓
data_prep.py        — Extract & filter 224×224 patches
  ↓
ssl_train.py        — Barlow Twins + ResNet-50 (self-supervised, no labels)
  ↓
clustering.py       — Leiden clustering + UMAP to validate learned morphology
  ↓
feature_extraction.py — Encode patches → slide-level .npz bags
                        [Encoder: ViT-B/16 | ResNet-50 SSL | DINO ViT (planned)]
  ↓
mil_train.py        — Gated Attention MIL → slide-level FL subtype prediction
  ↓
attention_viz.py    — Visualize which WSI regions drove the prediction
  ↓
[In progress]       — Multimodal fusion with RNA-seq features
```

---

## Architecture

### 1. Patch Extraction & Filtering (`data_prep.py`)

Extract 224×224 patches from gigapixel WSIs with aggressive tissue filtering:

- Discard patches >80% white (background/glass)
- Discard patches that are too dark (artifacts, out-of-focus)
- Grayscale entropy heuristics for borderline cases

**Result**: ~800–1000 high-quality tissue patches per slide.

### 2. Self-Supervised Learning — Barlow Twins (`ssl_train.py`)

Train a ResNet-50 backbone on unlabeled patches using Barlow Twins:

```
For each patch:
  1. Create two strongly augmented views (crop, color jitter, blur, flip)
  2. Pass both through ResNet-50 backbone (no FC) → MLP projector
  3. Barlow Twins loss:
     - Diagonal of cross-correlation matrix → 1 (invariance)
     - Off-diagonal → 0 (redundancy reduction)

Result: 2048-dim patch encoder adapted to H&E stain and FL morphology
```

**Why SSL?** Labeling 800K+ patches is infeasible. SSL lets the model learn tissue structure entirely unsupervised, adapting to domain-specific H&E characteristics that ImageNet pretraining never sees.

### 3. Clustering & Biological Validation (`clustering.py`)

After SSL training, cluster patch embeddings to verify the encoder learned real biology:

```
1. Build k-NN graph of patch embeddings (k=15)
2. Run Leiden community detection
3. Reduce to 2D with UMAP, color by cluster
4. Overlay FL1 / NotFL1 labels to check subtype enrichment
```

**Results**: Spearman ρ = 0.42 (p < 0.001) between cluster membership and transcriptional subtype — confirming the encoder captured meaningful morphological structure, not noise.

Identified morphological neighborhoods include:
- **Immune-dense regions**: High cellularity, T-cell-rich
- **Follicular structures**: Distinct gland-like formations
- **Sparse/fibrotic regions**: Low cellularity, stromal tissue

### 4. Feature Extraction (`feature_extraction.py`)

Encode all patches into `.npz` slide-level feature bags:

```python
ENCODER = "vit"     # "vit" | "resnet" | "dino" (planned)
```

Each `.npz` stores:
- `feats`: `[N_patches, embed_dim]` — patch embeddings
- `coords`: `[N_patches, 2]` — (x, y) coordinates in level-0 space
- `label`: binary FL subtype label
- `slide_id`: slide identifier

### 5. Gated Attention MIL (`mil_train.py`)

Slide-level classification using Multiple Instance Learning:

```
~800 patches per slide (each with d-dim embedding)
  ↓
Gated Attention:
  V(h) = Tanh(W_V · h)     # content projection
  U(h) = Sigmoid(W_U · h)  # gating
  A = Softmax(w · (V ⊙ U)) # per-patch attention weights
  z = Σ A_i · h_i           # weighted slide representation
  ↓
Linear classifier → FL1 / NotFL1
```

Attention weights reveal **which tissue regions drive the prediction** — not just what the model decided, but where it looked.

### 6. DINO ViT Encoder (`dino_encoder.py`) — Planned

The current ViT-B/16 uses generic ImageNet pretraining. DINO-pretrained ViTs are significantly better suited for pathology because:

- DINO's self-attention heads naturally attend to **semantically meaningful regions** — in histology this maps well to follicles, stroma, immune infiltrates
- DINO was shown to produce spatially coherent attention maps without any supervision
- Domain-specific DINO models (e.g. [CONCH](https://github.com/mahmoodlab/CONCH), [UNI](https://github.com/mahmoodlab/UNI)) pretrained on pathology data are available and expected to outperform generic ImageNet ViT

**Planned**: replace `vit_b_16(IMAGENET1K_V1)` with a DINO-pretrained ViT backbone, either via `torch.hub` or a pathology foundation model checkpoint.

---

## Encoder Comparison

| Encoder | Pretraining | Domain | Accuracy | AUC |
|---|---|---|---|---|
| ResNet-50 (SSL) | Barlow Twins on FL patches | In-domain | 78% | 0.65 |
| ViT-B/16 | ImageNet supervised | Out-of-domain | 71% | 0.58 |
| ViT-B/16 (alt) | ImageNet supervised | Out-of-domain | 69% | 0.55 |
| DINO ViT | Pathology SSL (planned) | In-domain | TBD | TBD |

**Key insight**: ResNet won because it was domain-adapted via SSL on FL patches specifically, while the ViT used generic ImageNet pretraining. This comparison is not ResNet vs ViT architecturally — it's *domain-adapted* vs *generic*. A DINO ViT pretrained on pathology data is the natural next step.

---

## Classification Design: FL1 vs NotFL1

Ideally a 7-way classifier across all FL subtypes, but data constraints made this impractical:

- FL6 and FL4 had only 3–4 samples each — deep learning on tiny classes is a recipe for overfitting
- **Pragmatic choice**: Binary FL1 vs NotFL1 gives a working, generalizable pipeline

Multi-class classification becomes tractable as data grows.

---

## Multimodal Fusion (In Progress)

Combining WSI-derived embeddings with RNA-seq features:

1. Aggregate patch embeddings → slide vector via attention-MIL
2. Combine with RNA-seq features (gene expression, pathway scores, cell-type signatures)
3. Train joint model on image + transcriptomic inputs

**Early results:**

| Modality | Accuracy |
|---|---|
| H&E only | 71% |
| RNA-seq only | 85% |
| H&E + RNA (in progress) | expected >90% |

---

## Setup

```bash
pip install openslide-python torch torchvision tqdm scikit-learn \
            umap-learn leidenalg igraph albumentations pandas openpyxl
apt-get install -y openslide-tools
```

**Data**: WSI `.svs` files + slide key Excel (`fl slide key.xlsx`) with FL subtype labels.

---

## Usage

```bash
# 1. Extract patches from WSIs
python data_prep.py

# 2. SSL pretraining (ResNet-50 + Barlow Twins)
python ssl_train.py

# 3. Cluster patch embeddings to validate encoder
python clustering.py

# 4. Encode patches into slide-level .npz bags
#    Set ENCODER = "vit" | "resnet" | "dino" in feature_extraction.py
python feature_extraction.py

# 5. Train MIL classifier
python mil_train.py

# 6. Visualize attention maps
python attention_viz.py
```

---

## Citation

If you use this code or reference the approach, please cite:

```
@article{kurapati2024pathology,
  title   = {Pathology Meets Genomics: Predicting FL Transcriptional Subtypes from H&E Histology},
  author  = {Kurapati, Aravind S. and Park, Christopher and others},
  journal = {TBD},
  year    = {2024}
}
```

---

## Medium

For the full write-up and intuition behind this work:

**[Pathology Meets Genomics: A Multimodal Approach to FL Classification](https://medium.com/@aravind.kurapati/pathology-meets-genomics)**
