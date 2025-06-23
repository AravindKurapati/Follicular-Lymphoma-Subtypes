# Predicting FL Transcriptional Subtypes from H&E Histology

##  Overview

This project investigates whether **H&E-stained whole slide images (WSIs)** can be used to predict the **transcriptional subtypes of follicular lymphoma (FL)** — subtypes that are traditionally defined using RNA sequencing.

The goal is to develop a **deep learning model** that classifies FL subtypes from histology alone, without access to transcriptomic data. This model complements the research presented in:

> **Klairmont et al.**  
> *Follicular lymphoma transcriptional classifier predicts overall survival following frontline immunochemotherapy*

---

##  Why Predict from H&E Alone?

My professor’s objective for this project was to test an important hypothesis:

> “Can we infer molecular subtypes from routine pathology images, without using RNA-seq?”

###  1. Real-World Clinical Applicability
- RNA-seq is expensive, slow, and not widely available.
- H&E slides are cheap, fast, and standard in all pathology labs.
- A successful model could immediately improve clinical decision-making using existing infrastructure.

###  2. Testing the Power of Morphology
- RNA-based subtypes reflect tumor biology and the tumor microenvironment.
- If these biological differences are visible in tissue architecture, a CNN should detect them — even if a pathologist can’t.

### ⚖ 3. Establishing a Benchmark
- This H&E-only model serves as a baseline.
- It can be compared against:
  - RNA-only models
  - Multimodal models (H&E + RNA)
- This helps assess when RNA data meaningfully adds predictive value.

###  4. Enabling Scalable, Low-Cost Triage
- In low-resource settings, the model could screen patients using just H&E.
- High-risk cases could then be flagged for further testing or clinical trials.

###  5. Bridging Morphologic and Molecular Worlds
- This work explores whether RNA-defined subtypes manifest as detectable morphologic patterns — creating a bridge between histopathology and precision oncology.

---

##  What This Notebook Does

### Objective:
Train a convolutional neural network (CNN) to classify FL transcriptional subtypes using H&E whole slide images.

### Pipeline:

1. **Data Loading**
   - Loads slide metadata (`fl slide key.xlsx`) containing subtype labels.
   - Uses `openslide` to read `.svs` format WSIs.

2. **Image Preprocessing**
   - Extracts tiles from slides using `PIL` and `numpy`.
   - Applies transformations with `torchvision`.

3. **Modeling**
   - Fine-tunes a pre-trained CNN (e.g., ResNet) using `PyTorch`.
   - Optimizes with cross-entropy loss and evaluates performance with standard metrics.

4. **Visualization**
   - Reduces learned embeddings using UMAP (`umap-learn`) for cluster visualization.
   - Plots subtype-specific tile distributions.

---

##  Outputs

- Confusion matrix
- ROC-AUC score
- Classification report
- UMAP visualizations of learned feature space

---


##  Citation

If using this code or referencing the subtyping approach, please cite:

> Klairmont et al. *Follicular lymphoma transcriptional classifier predicts overall survival following frontline immunochemotherapy*. [Manuscript in review].

---

##  Author Note

This work was conducted as part of a research investigation into low-cost, scalable, and accessible precision oncology tools. The project aligns with clinical translation efforts and aims to reduce reliance on high-cost molecular diagnostics.

