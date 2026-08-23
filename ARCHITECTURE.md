# 🏗️ ARCHITECTURE.md
## SugarcaneAI — System Architecture Deep Dive

> This document explains **every architectural decision** in the system in plain language.  
> Written for: Research scholars, collaborators, and reviewers of the Ph.D. work.

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy)
2. [The Disease Triangle Model](#2-the-disease-triangle-model)
3. [Full Pipeline Overview](#3-full-pipeline-overview)
4. [Layer 1 — Input Streams](#4-layer-1--input-streams)
5. [Layer 2 — Preprocessing](#5-layer-2--preprocessing)
6. [Layer 3 — Feature Extraction](#6-layer-3--feature-extraction)
7. [Layer 4 — Cross-Attention Fusion (Obj. 1 Core)](#7-layer-4--cross-attention-fusion-objective-1)
8. [Layer 5 — Stochastic Inference (Obj. 2)](#8-layer-5--stochastic-inference-objective-2)
9. [Layer 6 — Decision Heads](#9-layer-6--decision-heads)
10. [Layer 7 — Output & Explainability (Obj. 5)](#10-layer-7--output--explainability-objective-5)
11. [Knowledge Distillation (Obj. 4)](#11-knowledge-distillation-objective-4)
12. [Data Flow Diagram](#12-data-flow-diagram)
13. [Why These Design Choices?](#13-why-these-design-choices)
14. [Objective-Wise Architecture Map](#14-objective-wise-architecture-map)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)

---

## 1. System Philosophy

The central philosophy of this system is captured in one sentence:

> **"An image shows you WHAT the disease looks like. The environment tells you WHY and HOW SURE you can be."**

Traditional plant disease models only ask:  
> *"Given this leaf image, what disease is it?"*

Our model asks:  
> *"Given this leaf image AND the current weather, what disease is it, how severe is it, and how confident should I be?"*

This is the **Disease Triangle** — a well-established biological principle that disease occurrence requires three things: a susceptible **Host**, a viable **Pathogen**, and a favorable **Environment**. We encode all three.

---

## 2. The Disease Triangle Model

```
        HOST (Leaf Visual Features)
             ┌────────────┐
             │  CNN + ViT │
             │  Encoder   │
             └─────┬──────┘
                   │
                   │ Query (Q)
                   ▼
           ┌───────────────┐
           │ Cross-Attention│◄── Key (K), Value (V)
           │    Fusion      │         │
           └───────┬────────┘         │
                   │         ┌────────┴──────┐
                   │         │ ENVIRONMENT   │
                   │         │ MLP Encoder   │
                   │         │ [T, RH, Sm, P]│
                   │         └───────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  PATHOGEN (Disease) │
        │  Classifier + Grade │
        └─────────────────────┘
```

The cross-attention mechanism literally makes the visual features "ask" the environmental features: *"Given what you know about the weather, which disease pattern am I seeing?"*

---

## 3. Full Pipeline Overview

The system has **four processing stages** (from the Synopsis, Section 7):

```
Stage 1: INPUT STREAMS
  ├── Visual Stream (X_img):   RGB leaf images, 224×224
  └── Contextual Stream (X_env): [Temperature, Humidity, Soil Moisture, Rainfall]

Stage 2: VALIDATION & QUALITY ASSURANCE
  ├── Visual QA:   BRISQUE quality score → reject blurry/overexposed
  ├── Visual QA:   pHash deduplication → remove near-duplicates
  └── Metadata QA: Range-check (Humidity < 100%, Temp < 50°C)
                   KNN imputation for missing sensor values

Stage 3: PREPROCESSING & ALIGNMENT
  ├── Image:       CLAHE enhancement (Grade 1 faint lesion boost)
  ├── Image:       Albumentations augmentation (random flip, rotate, color jitter)
  ├── Metadata:    Min-Max normalization → [0, 1] range
  ├── Metadata:    Linear projection → 128-dim Environmental Token
  └── Sync:        GPS + EXIF timestamp → match image to weather entry (±15 min)

Stage 4: FEATURE EXTRACTION + FUSION
  ├── CNN Backbone (EfficientNet-B3): Spatial feature extraction
  ├── Swin-Transformer Encoder:       Global visual tokens (128-dim vectors, 196 patches)
  ├── MLP Metadata Encoder:           Environmental tokens (128-dim)
  └── Cross-Attention Fusion:         Image queries environment context → Fused tokens

Stage 5: INFERENCE
  ├── MC-Dropout (T=50 passes):        Stochastic uncertainty estimation
  ├── Disease Classifier:              5-class softmax
  ├── Severity Grader:                 Grade 0–4 (Ordinal Regression)
  └── Uncertainty Head:                ECE score, entropy flag

Stage 6: OUTPUT
  ├── Disease class + confidence score
  ├── Severity grade (0 = Healthy, 4 = Terminal)
  ├── Uncertainty flag ("High/Low Confidence")
  └── Grad-CAM++ heatmap (XAI pathology report)
```

---

## 4. Layer 1 — Input Streams

### 4.1 Visual Stream (X_img)

- **Format:** High-resolution RGB JPEG images of sugarcane leaves and stalks
- **Capture:** Smartphone (12MP+) or DSLR across three Maharashtra zones:  
  Kolhapur, Sangli, Satara (Panchganga + Krishna river basins)
- **Sampling strategy:** Stratified Purposive
  - Equal representation across 5 disease classes
  - Equal representation across 5 severity grades (0–4)
  - Sampling across weather profiles: High Humidity vs Dry periods
- **Regional varieties:** Co 86032, CoM 0265

### 4.2 Contextual Metadata Stream (X_env)

The Environmental Vector at time `t` for location `L`:

```
E_tL = [T, H, Sm, P]
```

| Symbol | Feature | Unit | Source | Normal Range |
|--------|---------|------|--------|-------------|
| T | Temperature | °C | DHT22 sensor / OpenWeatherMap API | 10°C – 50°C |
| H | Relative Humidity | % | DHT22 sensor / OpenWeatherMap API | 20% – 100% |
| Sm | Soil Moisture | % | Capacitive soil sensor | 0% – 100% |
| P | Precipitation/Rainfall | mm | Weather API / rain gauge | 0 – 300mm |

**Metadata Synchronization:** Image EXIF timestamp is matched to the nearest weather data entry within a **±15-minute window** at the GPS location of capture.

---

## 5. Layer 2 — Preprocessing

### 5.1 Visual Preprocessing Pipeline

```
Raw Image
    │
    ▼
BRISQUE Quality Filter ──[Score > Threshold]──► REJECT (blurry/overexposed)
    │ (Score ≤ Threshold = Good quality)
    ▼
pHash Deduplication ──[Too similar to existing]──► REJECT (near-duplicate)
    │ (Unique image)
    ▼
CLAHE Enhancement
  • Contrast Limited Adaptive Histogram Equalization
  • Selectively boosts contrast of Grade 1 faint lesions
  • Does NOT amplify background noise
    │
    ▼
Albumentations Augmentation (Training only)
  • RandomHorizontalFlip, RandomVerticalFlip
  • RandomRotation (±45°)
  • ColorJitter (brightness, contrast, saturation)
  • RandomCrop to 224×224
    │
    ▼
Normalize to ImageNet stats ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

**Why BRISQUE?** Blurry field images with motion artifacts are a major source of noise. BRISQUE is a no-reference image quality metric — it does not need a "perfect" reference image.

**Why pHash deduplication?** Without this, near-identical images across train/test splits cause **data leakage** — the model memorizes instead of learning.

### 5.2 Metadata Preprocessing Pipeline

```
Raw Metadata Vector [T, H, Sm, P]
    │
    ▼
Range Check Thresholding
  • Humidity ∈ [0, 100]    → reject outliers
  • Temperature ∈ [0, 50]  → reject outliers (biological plausibility)
    │
    ▼
KNN Imputation (k=5)
  • Missing sensor values predicted from k nearest temporal neighbors
    │
    ▼
Min-Max Normalization to [0, 1]:
  X_norm = (X - X_min) / (X_max - X_min)
    │
    ▼
Linear Projection Layer (10 → 128-dim Environmental Token)
```

---

## 6. Layer 3 — Feature Extraction

### 6.1 Image Encoder (Hybrid CNN-ViT)

```python
# Pseudocode — do NOT copy as final code
image_patches = cnn_backbone(image)          # EfficientNet-B3 spatial features
visual_tokens = swin_transformer(image_patches)  # Shape: (B, 196, 128)
```

**CNN Backbone — EfficientNet-B3:**
- Pretrained on ImageNet (transfer learning)
- Extracts fine-grained spatial features (lesion texture, color, spread pattern)
- Output: feature maps → flattened patches

**Swin-Transformer Encoder:**
- Self-attention over image patches → **196 visual tokens**, each 128-dim
- Captures global context (where on the leaf is the lesion?)
- "Biologically-Informed Reasoning" layer: self-attention captures macro-disease patterns

**Why EfficientNet + Swin together?**  
CNN captures local spatial features (lesion texture). Swin captures global relationships (distribution across the leaf). Together they form a richer visual representation than either alone.

### 6.2 Metadata Encoder (MLP)

```python
# Pseudocode
meta_tokens = mlp_encoder(metadata_vector)   # Shape: (B, 1, 128)
```

- Simple Multi-Layer Perceptron: `[4 → 64 → 128]` with ReLU + Dropout
- Projects the 4 environmental values into the **same 128-dim embedding space** as visual tokens
- This is essential — Cross-Attention requires Q, K, V to be in the same dimensional space

---

## 7. Layer 4 — Cross-Attention Fusion (Objective 1)

This is the **core contribution** of Objective 1.

### The Attention Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

Where:
- **Q (Query)** = Visual tokens from Swin-Transformer → *"What do I look like?"*
- **K (Key)** = Environmental tokens from MLP → *"What conditions am I in?"*
- **V (Value)** = Environmental tokens from MLP → *"What context should I pick up?"*

### Why Cross-Attention (not self-attention or simple concatenation)?

| Method | What it does | Problem |
|--------|-------------|---------|
| **Image-only** | Classify from image alone | Cannot disambiguate look-alike diseases |
| **Concatenation** | Append metadata to image features | Simple but no contextual weighting |
| **Self-attention** | Attend within same modality | Doesn't learn cross-modal relationships |
| **Cross-Attention (ours)** | Visual tokens query environmental context | Visual features are UPDATED by weather context |

**Real example:**  
- Visual features see: *"Red spots on leaf"*  
- Cross-Attention asks environmental context: *"Is humidity > 80%?"*  
- If YES → Red spots + High humidity → **Red Rot (Confident)**  
- If NO → Red spots + Low humidity → **Leaf Scorch / Abiotic stress (Lower confidence)**

### Architecture Detail

```python
# Pseudocode
fused_tokens = cross_attention(
    query=visual_tokens,   # (B, 196, 128) — image patches ask questions
    key=meta_tokens,       # (B, 1, 128)   — environment answers
    value=meta_tokens      # (B, 1, 128)   — environment provides context
)
# fused_tokens shape: (B, 196, 128)
# Each visual patch is now "aware" of the environmental context
```

**Residual Connection:** `fused = LayerNorm(visual_tokens + cross_attention_output)`  
**Feed-Forward:** Post-attention FFN for non-linear transformation  
**Stacking:** 2 cross-attention layers (configurable)

---

## 8. Layer 5 — Stochastic Inference (Objective 2)

### Monte Carlo Dropout (MC-Dropout)

Instead of running inference once, we run it **T=50 times** with Dropout kept ON:

```
For t = 1 to 50:
    prediction_t = model_with_dropout_ON(image, metadata)

mean_prediction = average(all 50 predictions)
uncertainty = variance(all 50 predictions)
```

**If the 50 predictions agree** → Low uncertainty → Model is confident  
**If the 50 predictions disagree wildly** → High uncertainty → "Flag for expert review"

### Expected Calibration Error (ECE)

ECE measures how honest the model is:
- A model saying "90% confident" should be right ~90% of the time
- Current vision-only models are overconfident (say 95% but are wrong 30% of the time)
- Our target: **ECE ≤ 0.05** (5% calibration error or less)

---

## 9. Layer 6 — Decision Heads

### 9.1 Disease Classifier
- `Linear(128 → 5)` + Softmax
- 5 classes: healthy, red_rot, grassy_shoot, smut, pokkah_boeng
- Loss: Cross-Entropy

### 9.2 Severity Grader
- Grades 0–4 (Healthy → Early → Moderate → Severe → Terminal)
- Method: Ordinal Regression (respects the natural ordering of severity)
- Loss: Mean Absolute Error (MAE) for grade prediction

### 9.3 Uncertainty Head
- Computed from MC-Dropout variance
- Outputs: Entropy score, calibration flag

---

## 10. Layer 7 — Output & Explainability (Objective 5)

Every prediction produces a **Diagnostic Report**:

```
┌────────────────────────────────────────────────┐
│           SUGARCANE DIAGNOSTIC REPORT           │
├────────────────────────────────────────────────┤
│ Disease:      Red Rot (Colletotrichum falcatum)│
│ Confidence:   87.3%                            │
│ Severity:     Grade 2 (Moderate)               │
│ Uncertainty:  LOW — prediction is reliable     │
│ Key Visual:   Red internal discoloration       │
│               spotted across mid-leaf          │
│ Key Context:  Humidity=84% (HIGH), Temp=29°C   │
├────────────────────────────────────────────────┤
│  [Grad-CAM++ Heatmap attached]                 │
│  → High activation on mid-leaf lesion zones    │
└────────────────────────────────────────────────┘
```

**Grad-CAM++ Heatmap:** Visualizes WHICH pixels the model used for its decision. Used for pathological validation by plant disease experts.

---

## 11. Knowledge Distillation (Objective 4)

The Teacher-Student framework compresses the full model for edge deployment:

```
Teacher Model (Full)              Student Model (Lightweight)
┌─────────────────────┐          ┌──────────────────────────┐
│ EfficientNet-B3     │   KL     │ MobileViT / MobileNet    │
│ + Swin-Transformer  │──────────│ + Lightweight MLP        │
│ + Cross-Attention   │Divergence│ + Distilled Cross-Attn   │
│ ~16M params         │  Loss    │ ~4-6M params             │
└─────────────────────┘          └──────────────────────────┘
                                   Target: <30ms inference on phone
                                   Target: <3% accuracy drop
                                   Target: ≥40% FLOPs reduction
```

---

## 12. Data Flow Diagram

```
FIELD CAPTURE
    Farmer takes photo of diseased sugarcane leaf
    GPS + timestamp embedded in EXIF metadata
         │
         ▼
UPLOAD / SYNC
    Image → Server
    GPS + timestamp → Weather API (OpenWeatherMap / IBM Weather)
    → Pulls historical T, H, Sm, P for that location & time
         │
         ▼
QUALITY GATE
    BRISQUE filter → rejects blurry images
    pHash → rejects duplicates
    Range check → validates sensor readings
    KNN imputation → fills missing sensor gaps
         │
         ▼
PREPROCESSING
    CLAHE enhancement → boosts faint Grade 1 lesions
    Min-Max normalization of metadata
    Resize + normalize image to 224×224
         │
         ▼
FEATURE EXTRACTION
    CNN + Swin → 196 visual tokens (128-dim each)
    MLP → 1 environmental token (128-dim)
         │
         ▼
CROSS-ATTENTION FUSION (★ Core Objective 1)
    Visual tokens query environmental context
    → Fused context-aware visual representation
         │
         ▼
MC-DROPOUT INFERENCE (50 passes) (Objective 2)
    → Mean prediction + Uncertainty score
         │
         ├──────────────────┬─────────────────┐
         ▼                  ▼                 ▼
  Disease Class       Severity Grade    Uncertainty
  (5 classes)          (0-4 scale)       (ECE score)
         │
         ▼
EXPLAINABILITY
    Grad-CAM++ → attention heatmap overlay
    Score-CAM → diagnostic visualization
         │
         ▼
OUTPUT REPORT
    {disease, confidence, severity, uncertainty, heatmap}
```

---

## 13. Why These Design Choices?

| Decision | Alternative Considered | Why We Chose This |
|----------|----------------------|-------------------|
| **EfficientNet-B3** | ResNet50, VGG | Best accuracy/efficiency tradeoff for field images |
| **Swin-Transformer** | ViT-Base | Handles variable-scale lesions via shifted windows |
| **Cross-Attention** | Concatenation, FiLM | Learns adaptive weighting — image queries environment |
| **MC-Dropout** | Deep Ensembles, Bayesian NN | Computationally efficient, single model |
| **ECE metric** | Brier Score | Standard calibration benchmark in medical/agricultural AI |
| **Grad-CAM++** | LIME, SHAP | Works directly with CNN features, real-time capable |
| **uv** | pip, conda, poetry | Fastest resolver, reproducible lockfile, Rust-backed |
| **Stratified Purposive Sampling** | Random sampling | Prevents class imbalance and ensures Grade 0-4 coverage |

---

## 14. Objective-Wise Architecture Map

```
Objective 1 — Cross-Attention Fusion
  Files: src/models/fusion/cross_attention.py
         src/models/encoders/image_encoder.py
         src/models/encoders/metadata_encoder.py
         src/models/multimodal_model.py
  Config: configs/obj1_cross_attention.yaml

Objective 2 — Uncertainty Quantification
  Files: src/models/heads/uncertainty_head.py
         src/evaluation/calibration.py
  Config: configs/obj2_uncertainty.yaml

Objective 3 — Maharashtra Dataset
  Files: src/data/preprocessing/metadata_sync.py
         src/data/preprocessing/quality_filter.py
         src/data/preprocessing/deduplication.py
         scripts/sync_metadata.py
  Config: (dataset collection protocol — see docs/objectives/obj3_dataset.md)

Objective 4 — Knowledge Distillation
  Files: src/models/baselines/ (student models)
         src/training/losses.py (KL-divergence loss)
  Config: configs/obj4_distillation.yaml

Objective 5 — Explainability
  Files: src/explainability/gradcam_plus.py
         src/explainability/attention_viz.py
         src/evaluation/ablation.py

Objective 6 — Baseline Comparison
  Files: src/models/baselines/image_only.py
         src/models/baselines/concat_baseline.py
         scripts/run_baselines.py
         scripts/run_ablation.py
```

---

## 15. Known Limitations & Future Work

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No dataset yet (Obj 3 in progress) | Cannot train full model | Use PlantVillage + synthetic metadata for Obj 1 prototyping |
| Weather API rate limits | May miss metadata for some images | Cache API results locally, use DHT22 hardware for field sites |
| Single-leaf input | Missing stalk-level context | Future: multi-view input pipeline |
| Inference latency (Teacher model) | Not real-time on mobile yet | Obj 4 (distillation) addresses this |
| Western Maharashtra focus | Limited generalizability | Cross-farm validation planned in Obj 5 |

---

*This document will be updated as each research objective is completed.*  
*Last updated: August 2026 — Objective 1 in progress.*
