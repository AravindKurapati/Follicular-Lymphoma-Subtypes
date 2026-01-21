# Pathology Meets Genomics: Predicting FL Transcriptional Subtypes from H&E Histology

**Can deep learning infer molecular subtypes from routine pathology images?**

This project investigates whether **H&E-stained whole slide images (WSIs)** can predict the **transcriptional subtypes of follicular lymphoma (FL)** traditionally defined using RNA sequencing.

---

## Motivation

Follicular lymphoma is clinically heterogeneous. ~20% of patients progress rapidly with poor outcomes, while others remain indolent for years. RNA-seq reveals transcriptional subtypes that predict prognosis, but:

- **RNA-seq is expensive, slow, and unavailable in most labs**
- **H&E slides are free, instant, and everywhere**

The question: Can we infer molecular subtypes from morphology alone?


## The Problem Nobody Warns You About

Building models on pathology images is messier than papers suggest. Most of the  work happens in data prep, not model architecture:

- **Patch extraction**: Gigapixel WSIs (100,000 × 100,000 pixels) must be chunked into 224×224 tiles
- **Filtering**: Most patches are garbage—white space (background), pen marks, out-of-focus regions, scanner artifacts
- **Tissue quality**: "Too white" and "too black" are subjective; thresholds vary wildly across slides from different labs

**Core insight**: Patch quality is not just "data cleaning." 

---

## What We Built

```
Raw WSI (gigapixel)
  ↓
Extract & filter patches (224×224, background removed)
  ↓
Self-supervised learning (Barlow Twins + ResNet-50)
  → Learn tissue patterns without labels
  ↓
Cluster & validate (Leiden + UMAP)
  → Verify the model learned real biology
  ↓
Slide-level classification (Gated Attention MIL)
  → Predict FL subtype + show which regions matter
  ↓
Multimodal fusion (In progress)
  → Combine image embeddings with RNA-seq
```

---

## Architecture

### 1. Patch Extraction & Filtering

Extract 224×224 patches from WSI and filter aggressively:
- Discard patches >80% white (background)
- Discard patches that are too black (artifacts, out of focus)
- Use grayscale distribution and entropy heuristics for borderline cases

**Result**: ~800-1000 high-quality patches per slide

### 2. Self-Supervised Learning (Barlow Twins + ResNet-50)

Train without labels on unlabeled patches:

```
For each patch:
  1. Create two strongly augmented views (crops, color jitter, blur, flip)
  2. Pass both through ResNet-50 backbone → remove final FC layer
  3. Pass outputs through small MLP projector
  4. Barlow Twins loss: make embeddings of same patch similar, 
     but keep feature dimensions uncorrelated

Result: 2048D patch encoder capturing morphology
```

**Why SSL?** Labeling 800k patches is impossible. SSL lets the model discover tissue structure on its own.

### 3. Clustering & Validation (Leiden + UMAP)

After SSL training, cluster patch embeddings to validate what the model learned:

```
1. Build k-nearest neighbors graph of patch embeddings
2. Run Leiden clustering (community detection algorithm)
3. Reduce to 2D using UMAP, color by cluster ID
4. Manually inspect representative patches from each cluster
```

**What to look for:**
- Each color represents a distinct morphological neighborhood
- If clusters are spatially organized and interpretable (immune-dense, fibrotic, follicles), the SSL encoder learned real patterns
- If it's a uniform blob, the model learned noise

**Key insight**: If the UMAP were random, all 10,000 patches would scatter uniformly with no coherent clusters. Instead, clear topological separation tells us the model picked up on real biology.

### 4. Slide-Level Classification (Gated Attention MIL)

 Multiple Instance Learning learns which patches matter:

**Patch aggregation explained:**
- You have ~10,000 patches per slide, each with a 2048-dim embedding
- MIL learns attention weights for each patch
- Slide representation = weighted sum of patch embeddings
- Goes from gigapixel image (billions of pixel values) → single 2048-dim vector

**Gated Attention MIL:**
```python
# V: feature projection (Tanh)
# U: gating mechanism (Sigmoid)
# w: learned attention weights
# Output: per-patch attention + slide-level prediction
```


-  Attention maps show which regions drove the prediction
-  Immune-dense regions get high attention, sparse regions get low
-  Not all biopsy regions are equally informative

---

## Classification Decision: FL1 vs NotFL1

Ideally, we'd build a 7-way classifier for all FL subtypes. But the data didn't support it:
- FL6 and FL4 had only 3-4 samples each
- Deep learning on tiny datasets is a classic recipe for overfitting

**Pragmatic choice**: Binary classification. FL1 (most common, sufficient data) vs everything else (NotFL1). This gives us a working pipeline; multi-class classification scales as data grows.

---

## ResNet vs ViT: What Actually Happened

Tested two encoders with the same MIL aggregation:

| Backbone | Accuracy | AUC |
|----------|----------|-----|
| SSL ResNet-50 (domain-adapted) | 78% | 0.65 |
| ImageNet ViT (generic pretrain) | 71% | 0.58 |
| ImageNet ViT (alternative) | 69% | 0.55 |

**Why ResNet won**: It has strong inductive bias for local texture—exactly what histology needs. I trained ResNet self-supervised on my own FL patches, so it adapted to H&E stain, scanner style, and FL-specific morphology. The ViT used generic ImageNet pretraining with no domain-specific learning.

**Honest caveat**: This wasn't ResNet vs ViT fundamentally. It was domain-adapted ResNet vs generic ViT. If I pretrain a ViT with SSL on FL patches, the story might change. Domain adaptation matters more than architecture at this scale.

---

## Clustering Results

Leiden clustering on SSL embeddings revealed distinct morphological neighborhoods. Example clusters:

- **Immune-dense clusters**: High cellularity, T-cell markers visible
- **Sparse/fibrotic clusters**: Low cellularity, stromal regions
- **Follicle clusters**: Specific gland structures

**Biological validation:**
- Spearman correlation between cluster membership and transcriptional subtype: ρ = 0.42 (p < 0.001)
- When overlaying FL1 vs NotFL1 labels: some clusters enriched for one subtype, others present in both

This proves the model learned meaningful morphological structure, not random noise.

---

## Multimodal Fusion (In Progress)

Current approach:
1. Aggregate patch embeddings into slide-level vector (via attention-MIL)
2. Combine with RNA-seq features (gene expression, pathway scores, cell-type signatures)
3. Train multimodal model on image + transcriptomic features

**Early results:**
- H&E only: 71% accuracy
- RNA only: 85% accuracy
- H&E + RNA (WIP): Expected >90%

---



---




---

## Citation

If you use this code or reference the approach, please cite:

```bibtex
@article{kurapati2024pathology,
  title={Pathology Meets Genomics: Predicting FL Transcriptional Subtypes from H&E Histology},
  author={Kurapati, Aravind S. and Park, Christopher and others},
  journal={TBD},
  year={2024}
}
```

---

## Medium Post

For the full story behind this work, please see my medium:

**[Pathology Meets Genomics: A Multimodal Approach to FL Classification](https://medium.com/@aravind.kurapati/pathology-meets-genomics)**

