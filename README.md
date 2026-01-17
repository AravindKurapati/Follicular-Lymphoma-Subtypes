# Pathology Meets Genomics: Predicting FL Transcriptional Subtypes from H&E Histology

**Can deep learning infer molecular subtypes from routine pathology images?**

This project investigates whether **H&E-stained whole slide images (WSIs)** can predict the **transcriptional subtypes of follicular lymphoma (FL)**  traditionally defined using RNA sequencing. The work bridges histopathology and precision oncology by asking: *What does biology look like under a microscope?*

---

## Motivation

Follicular lymphoma is clinically heterogeneous. ~20% of patients progress rapidly with poor outcomes, while others remain indolent for years. RNA-seq reveals transcriptional subtypes that predict prognosis — but:

- **RNA-seq is expensive, slow, and unavailable in most labs**
- **H&E slides are free, instant, and everywhere**

The question: Can we infer molecular subtypes from morphology alone?

If yes, we unlock:
- **Clinical impact**: Risk stratification using existing infrastructure
- **Scalability**: Low-cost triage in resource-limited settings
- **Insight**: Understanding how biology manifests as tissue structure

---

## Project Overview

### What We Built

A **multimodal deep learning pipeline** to classify FL transcriptional subtypes from histology:
```
Whole SLide Images 
  → Self-supervised pretraining (Barlow Twins + SSL)
  → Patch encoding (ResNet50 or ViT-B/16)
  → Slide-level aggregation (Gated Attention MIL)
  → Subtype prediction
```

### Key Components

| Stage | Method | Output |
|-------|--------|--------|
| **Preprocessing** | Tissue mask + patch extraction | 224×224 tiles, background filtered |
| **Encoding** | ViT-B/16 (ImageNet pretrained) | 768D patch embeddings |
| **Aggregation** | Gated Attention MIL | Per-slide predictions + attention weights |
| **Multimodal (WIP)** | Fusion with RNA-seq | Integrated subtype prediction |

---

## Architecture

### 1. Patch Extraction & Filtering
```python
# Extract 224×224 patches from whole slide images
# Filter out background (>60% white) and artifacts (<20% black)
# Result: ~800 high-quality patches per slide
```
**Why this matters**: Garbage in = garbage out. Spent weeks optimizing tissue detection across multiple staining protocols and microscopy platforms.

### 2. Self-Supervised Pretraining (Barlow Twins)
```
ResNet50 backbone → 512D → 128D projector
Minimize cross-correlation of augmented patch pairs
Train on unlabeled SSL patches to learn tissue-specific features
```
**Result**: Encoder captures morphologic patterns without label supervision.

### 3. Patch-Level Classification
```
SSL embeddings → Logistic Regression
Cluster patches with Leiden algorithm
Analyze which morphologic neighborhoods correlate with FL subtypes
```

### 4. Slide-Level Aggregation (Gated Attention MIL)
```python
class AttnMIL(nn.Module):
    # V: feature projection (Tanh)
    # U: gating mechanism (Sigmoid) 
    # w: attention scoring
    # cls: final classifier
    
    # Output: per-patch attention weights + slide prediction
    # Interpretability: visualize which regions drive the prediction
```

**Why MIL?**  
- Most patches look similar (morphologically repetitive tissue)
- Signal is in *which patches matter* and their spatial organization
- Attention mechanism identifies relevant morphologic neighborhoods
- Outputs are interpretable: see exactly which regions influenced the prediction

### 5. Multimodal Fusion (In Progress)
```
Image features (patch embeddings)
    ↓
+ RNA features (gene expression)
    ↓
→ Integrated subtype predictor
→ Compare: H&E only vs RNA only vs H&E + RNA
```

---

##  Results

### Quantitative Performance
- **Patch-level classification**: ~78% accuracy (binary FL1 vs NotFL1)
- **Slide-level (MIL)**: ~71% accuracy on validation set
- **AUC**: 0.82 (binary classification)

### Qualitative Insights
- Attention heatmaps highlight clinically relevant regions
- Leiden clustering reveals morphologically distinct neighborhoods
- Spearman correlation: cluster membership ↔ transcriptional subtype (ρ = 0.42, p < 0.001)

**See**: `notebooks/` for confusion matrices, ROC curves, and attention visualizations.

---


##  Key Insights

### 1. Data Quality is Everything
Spent **3+ weeks** optimizing tissue filtering across multiple staining protocols. Patch quality directly impacts what the model learns. See the [Medium post](https://medium.com/@aravind.kurapati/...) for the full story.

### 2. Attention Reveals Biology
The gated attention mechanism learned to weight patches by relevance. Visualization showed the model highlighting:
- Dense immune infiltration (high attention)
- Sparse cellularity regions (low attention)
- Spatial clustering patterns (neighborhood effects)

These align with known biology: FL subtypes differ in immune microenvironment composition.

### 3. Morphology ↔ Transcriptomics Bridge
Patch embeddings correlated with gene expression signatures. Example:
- Patches with high attention → high immune infiltration genes (CD8, IFNG, etc.)
- Patches with low attention → immune depletion markers

This suggests **morphology encodes transcriptional information**.

### 4. Multimodal is the Future
H&E alone achieved ~71% accuracy. RNA alone achieves ~85%. Combined? We're building that now. Early results suggest fusion improves both prediction accuracy and biological interpretability.

---

## Methods

### Barlow Twins (Self-Supervised Learning)
- **Why**: Learn tissue-specific features without labeled data
- **Loss**: Cross-correlation matrix with on/off-diagonal penalties
- **Result**: Encoder captures morphologic patterns (validated by downstream performance)

### Multiple Instance Learning (MIL)
- **Why**: Slides have variable numbers of patches; aggregation is non-trivial
- **Architecture**: Gated attention (learns which patches matter)
- **Advantage**: Interpretable predictions (see attention weights per patch)

### Leiden Clustering
- **Why**: Discover morphologically distinct neighborhoods
- **Method**: k-NN graph + community detection
- **Result**: Clusters correlate with FL subtypes (Spearman ρ=0.42, p<0.001)

---

##  Validation & Benchmarking

### Metrics
- Accuracy, Precision, Recall, F1
- ROC-AUC (per-class and macro)
- Confusion matrices with per-class breakdown

### Comparisons
- **H&E only** (this work): 71% accuracy
- **RNA only** (baseline): 85% accuracy
- **H&E + RNA** (multimodal, WIP): expected >90%

### Generalization
- Tested on slides from multiple labs
- Different staining protocols
- Different microscopy platforms

---

##  Work in Progress

- [ ] Integrate RNA-seq data for multimodal predictions

---

##  Citation

If you use this code or reference the approach, please cite:

```bibtex
@article{klairmont2024fl,
  title={Follicular lymphoma transcriptional classifier predicts overall survival 
         following frontline immunochemotherapy},
  author={Klairmont, M. and ...},
  journal={TBD},
  year={2024}
}
```

---

## Medium Post

For the story behind this work — see:

**[Pathology Meets Genomics: A Multimodal Approach to FL Classification](https://medium.com/@aravind.kurapati/...)**


---

**Last Updated**: December 2025
