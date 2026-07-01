# NETRA — ANUDRISHTI 🛰️
### Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data
**Bhartiya Antariksh Hackathon — Team Submission**

An AI pipeline for cross-modal satellite image retrieval. Given a SAR (radar) query image, NETRA finds the most geographically similar optical images from a satellite archive — and vice versa — bridging the sensing gap between Sentinel-1 (SAR) and Sentinel-2 (Optical) modalities using a Siamese contrastive learning architecture built on DINOv2.

> 📦 Dataset used: [TUM Sentinel-1/2 on Kaggle](https://www.kaggle.com/datasets/shambac/tum-sentinel-1-2)

---

## The Problem

Satellite imagery comes from multiple sensors — optical cameras, SAR (Synthetic Aperture Radar), and multispectral instruments. Each sees the world differently:

- **Optical images** (Sentinel-2) are visually interpretable but become blind under cloud cover.
- **SAR images** (Sentinel-1) penetrate clouds but look nothing like an optical image.

Today, analysts work in sensor silos — optical archives are queried with optical images, SAR archives with SAR metadata. There is no system that lets you ask:

> *"Find me satellite images that show the same geographic scene as this query — regardless of which sensor captured them."*

**NETRA closes this gap.**

---

## What's Actually Built (Current State)

The current repository contains a single end-to-end notebook: `01_Data_Exploration.ipynb`.

---

## The VISION: What NETRA Does

NETRA is a cross-modal satellite image retrieval system. Given a query image from **any sensor**, it retrieves the most geographically similar images from a **different sensor modality** — in under one second.
**A pipeline rather than a feature.**

```
Cloudy optical image (LISS-IV / Sentinel-2)
        ↓
  Cloud Removal (Pix2Pix GAN)
        ↓
  Shared Encoder (ResNet-50 + projection head)
        ↓
  128-dim embedding vector
        ↓
  FAISS nearest-neighbour search
        ↓
  Top-5 SAR matches from same geographic region

```
## Why It Matters for India

India's satellite constellation — Resourcesat-2 (LISS-IV), RISAT-1S, Cartosat — produces multi-sensor data daily. Disaster response teams during floods in Assam or cyclones along the eastern coast need to correlate SAR and optical imagery in near-real-time. NETRA makes cross-sensor search as fast as a Google image search.

The system is sensor-agnostic by design. Adding a new sensor (hyperspectral, thermal) requires only fine-tuning the projection head on new paired data — no architectural changes.

---

Here is what's implemented inside it:

### Step 1 — Data Loading & Pre-processing
- Custom `SEN12Dataset` PyTorch class that loads matched Sentinel-1 (SAR) and Sentinel-2 (Optical) image pairs from disk.
- SAR images (single-channel grayscale) are converted to 3-channel RGB to match DINOv2's input format.
- Standard ImageNet-style normalization transforms applied (`224×224`, mean/std from DINOv2).
- A secondary `FullSatelliteDataset` class enables recursive scanning across the full dataset directory structure (all `s2_*/s1_*` folder pairs), supporting the full TUM Sentinel-1/2 dataset scale.
- 80/20 train/validation split using `random_split` with a fixed seed (42) for reproducibility.

### Step 2 — DINOv2 Feature Extraction (Baseline Similarity Test)
- DINOv2 ViT-Base/14 (`vit_base_patch14_dinov2`) loaded via `timm` as a **frozen** feature extractor.
- Optical and SAR image batches passed through DINOv2 to obtain raw `[batch, 768]` feature embeddings.
- L2-normalized cosine similarity matrix computed between all optical and SAR embeddings in the batch.
- Results printed to verify whether out-of-the-box DINOv2 embeddings can distinguish cross-modal pairs — they cannot reliably, motivating the next step.

### Step 3 — Siamese Network with Projection Head
- `SAROptSiameseNetwork`: a shared Siamese architecture wrapping the **frozen DINOv2 backbone**.
- A trainable **projection head** (MLP: `768 → 512 → 256`) maps both optical and SAR embeddings into a shared 256-dimensional space.
- Both branches are L2-normalized for cosine similarity alignment.

### Step 4 — Contrastive Training (InfoNCE / NT-Xent)
- **Loss function**: symmetric InfoNCE (contrastive) loss with a temperature of `0.1`.
- True matches (same geographic location) are the diagonal in the batch similarity matrix.
- Only the projection head parameters are trained; DINOv2 backbone weights stay frozen throughout.
- Optimizer: Adam (`lr = 0.001`).
- Training loop includes per-epoch train/validation loss tracking.
- Best model weights automatically saved as `best_siamese_model.pth`.

### Step 5 — Visual Retrieval Demo
- `run_hackathon_demo()`: A visual end-to-end demonstration function.
- Picks a random SAR image as a query from the validation set.
- Retrieves the **Top-3 most similar optical images** using cosine similarity over projected embeddings.
- Displays query + top matches as an annotated matplotlib figure, showing match rank, similarity score, and whether the true pair was retrieved.

---

## Repository Structure

```
Netra-Anudrishti/
└── 01_Data_Exploration.ipynb   # Complete end-to-end notebook (data → model → demo)
```

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision torchaudio timm matplotlib Pillow
```

Or run the `%pip install` cells at the top of the notebook.

### Dataset Setup

1. Download the [TUM Sentinel-1/2 dataset from Kaggle](https://www.kaggle.com/datasets/shambac/tum-sentinel-1-2).
2. Update the `ROOT_DATA_DIR` variable in the notebook to point to your local dataset folder.

> ⚠️ **The dataset is large (8+ GB). We strongly recommend running this notebook on Google Colab or Kaggle Notebooks with the dataset mounted directly.**

Expected folder structure inside your dataset directory:
```
ROIs1158_spring/
├── s2_43/    ← Optical (Sentinel-2) images
├── s1_43/    ← SAR (Sentinel-1) images
├── s2_44/
├── s1_44/
...
```

### Run the Notebook

Open `01_Data_Exploration.ipynb` in Jupyter or Google Colab and run cells sequentially from top to bottom.

---

## Technical Stack

| Layer | Library |
|---|---|
| Deep learning | PyTorch + torchvision |
| Backbone | DINOv2 ViT-B/14 (frozen, via `timm`) |
| Contrastive learning | InfoNCE / NT-Xent loss (custom implementation) |
| Data loading | PyTorch Dataset + DataLoader |
| Visualization | Matplotlib |
| Runtime | Python 3.9+ |
| Hardware support | CUDA / Apple Silicon (MPS) / CPU (auto-detected) |

---

## Results

*Quantitative evaluation metrics (F1@5, F1@10) are being finalised. Training and validation loss curves confirm the projection head is learning a shared cross-modal embedding space — diagonal similarity scores increase meaningfully after training.*

---

## Scope for Future Work

The following components are planned but **not yet implemented** in the current repository. These represent the full technical vision for NETRA:

### 🔲 Module 1 — Cloud-Robust Optical Reconstruction (Pix2Pix GAN)
India's optical satellites are blind during the monsoon season (June–September). A retrieval system that only works on cloud-free imagery is unusable for 4–5 months of the year.

**Planned**:
- **Generator**: U-Net architecture — encodes the cloudy image, decodes it to a cloud-free reconstruction with skip connections for spatial detail preservation.
- **Discriminator**: PatchGAN (70×70 receptive field) — evaluates local texture realism, not just global plausibility.
- **Loss**: Adversarial + L1 pixel reconstruction (λ=100) — prevents hallucination and blurring.
- **Training data**: SEN12MS-CR (pre-aligned cloudy/cloud-free Sentinel-2 pairs from Zenodo).
- **Targets**: RMSE < 0.1, PSNR > 25 dB, SSIM > 0.8.

This GAN would serve as **Stage 1** of the full NETRA pipeline — cleaning the optical query before it reaches the cross-modal retrieval encoder.

### 🔲 Module 2 — FAISS Vector Search Index
- Replace in-notebook cosine similarity with a proper **FAISS IndexFlatIP** vector index.
- Pre-compute and store embeddings for the entire satellite archive.
- Enable sub-second nearest-neighbour search at retrieval time.

### 🔲 Module 3 — Evaluation & Benchmarking
- Implement F1@5 and F1@10 metrics for both same-modal and cross-modal retrieval.
- Quantitative benchmark on held-out test split.
- Multi-seasonal generalization testing (spring / summer / winter datasets).

### 🔲 Module 4 — Indian Satellite Data Integration
- Fine-tune the encoder on Indian satellite data: Resourcesat-2 (LISS-IV), RISAT-1S, Cartosat.
- Source: NRSC Bhoonidhi portal.
- Manual co-registration pipeline for India-specific scenes.

### 🔲 Module 5 — Production Pipeline & UI
- Modularize into `src/cloud_removal/`, `src/retrieval/`, `src/pipeline/` packages.
- Command-line inference script: `python src/pipeline/main.py --query image.tif --modality sar --top_k 5`
- Drag-and-drop web interface for uploading a SAR image and visualizing retrieved optical matches.

---

## Full Vision Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NETRA Pipeline                      │
│                                                         │
│  [Optical Query] → [Cloud Removal GAN] ─┐               │
│                                          ├→ [Siamese  ] → [FAISS] → [Top-K Results]
│  [SAR Query] ────────────────────────────┘   [Encoder ]
│                                                         │
│  Database: pre-computed embeddings (SAR + optical)      │
└─────────────────────────────────────────────────────────┘
```

*Currently implemented: the Siamese Encoder.*
*Not yet implemented: Cloud Removal GAN, FAISS indexing, production pipeline.*

---


*Built for Bhartiya Antariksh Hackathon*

---

