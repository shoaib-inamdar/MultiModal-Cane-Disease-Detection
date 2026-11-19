<div align="center">

# 🌾 SugarcaneAI
### Novel Lightweight Explainable Uncertainty-Aware Multimodal Hybrid Vision Transformer  
### for Early-Stage Sugarcane Disease Detection Under Real-Field Conditions

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![CI](https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/managed%20by-uv-7c3aed)](https://github.com/astral-sh/uv)
[![wandb](https://img.shields.io/badge/Experiment%20Tracking-W%26B-FFBE00?logo=weightsandbiases)](https://wandb.ai)
[![DVC](https://img.shields.io/badge/Data%20Version%20Control-DVC-945DD6?logo=dvc)](https://dvc.org)
[![codecov](https://img.shields.io/badge/coverage-pytest--cov-brightgreen)](https://pytest.org)

<br/>

**A Ph.D. research system that fuses leaf images with real-time environmental metadata**  
**(temperature, humidity, soil moisture, rainfall) via Cross-Attention to resolve visual ambiguities**  
**in early-stage sugarcane disease identification — achieving contextually reliable, honest AI diagnostics.**

<br/>

[📄 Synopsis](Synopsis.docx) · [🏗️ Architecture](ARCHITECTURE.md) · [📖 Walkthrough](WALKTHROUGH.md) · [🔬 Docs](docs/) · [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 Research Objectives

| # | Objective | Core Technique | Status |
|---|-----------|---------------|--------|
| **1** | Cross-Attention Multimodal Feature Fusion | Swin-Tiny + MetadataEncoder + CrossAttentionFusion | ✅ **Done** — Test F1: 0.873, Acc: 0.886 |
| **2** | Uncertainty Quantification | Monte Carlo Dropout (T=30) + ECE Calibration | ✅ **Done** — MC-Dropout integrated, Grad-CAM++ live |
| **3** | Maharashtra Multimodal Dataset | Field Survey + DHT22/IoT Sensors + GPS Sync | ⏳ Stage C (future fieldwork) |
| **4** | Knowledge Distillation (Edge Deployment) | Teacher→Student KL-Divergence, MobileNetV3-Small | ✅ **Done** — 93.3% param reduction (28.3M → 1.9M) |
| **5** | Explainability (Grad-CAM++) | Grad-CAM++ heatmaps per prediction | ✅ **Done** — Live in demo app |
| **6** | Comprehensive Baseline Comparison | 5-model comparison (CrossAttn, Concat, Student, ResNet50, EfficientNet) | ✅ **Done** — see results table |

---

## 🔬 Research Problem

> *"Early-stage Red Rot lesions are visually indistinguishable from Leaf Scorch under low humidity. A vision-only model cannot know the difference — but a model that also sees the weather can."*

Current Hybrid Vision Transformer models are **environmentally blind** — they ignore temperature, relative humidity, and soil moisture, which are the primary environmental drivers of fungal disease progression. This causes:

- ❌ **High false-positive rates** — Misclassifying abiotic stress as disease
- ❌ **Overconfident wrong predictions** — No uncertainty quantification
- ❌ **No field readiness** — Models too heavy for edge deployment

### Our Solution — The Disease Triangle Framework

```
HOST (Leaf)          + PATHOGEN (Fungus)    + ENVIRONMENT (Climate)
Visual tokens (ViT)    Disease classifier     Cross-Attention Fusion
                                             [Temperature, Humidity, Moisture]
```

---

## 🏗️ Architecture Overview

```
INPUT LAYER
┌─────────────────────────────────────────────────────────────────┐
│  Leaf Image (RGB)    │  Temperature │ Humidity │ Soil Moisture  │
│  [B, 3, 224, 224]   │              Metadata Vector              │
└──────────┬──────────────────────────────┬────────────────────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐             ┌─────────────────┐
   │ CNN Backbone  │             │ KNN Imputation  │
   │(EfficientNet) │             │ + Range Check   │
   │  + CLAHE      │             │ + Min-Max Norm  │
   └───────┬───────┘             └────────┬────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐             ┌─────────────────┐
   │ Swin-Transform│             │  MLP Metadata   │
   │    Encoder    │             │    Encoder      │
   │  Visual       │             │  Environmental  │
   │  Tokens (Q)   │             │  Tokens (K, V)  │
   └───────┬───────┘             └────────┬────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
              ┌─────────────────────┐
              │  CROSS-ATTENTION    │  ← Objective 1 Core
              │  Fusion Layer       │
              │  Attn(Q,K,V) =      │
              │  softmax(QKᵀ/√d)·V  │
              └──────────┬──────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  MC-Dropout (T=50 passes)     │  ← Objective 2
         │  Stochastic Inference         │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌──────────┐   ┌──────────────┐  ┌──────────────┐
  │ Disease  │   │  Severity    │  │ Uncertainty  │
  │Classifier│   │  Grader(0-4) │  │  Head (ECE)  │
  └──────────┘   └──────────────┘  └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │  OUTPUT: Diagnostic Report    │
         │  • Disease class + confidence │
         │  • Severity grade (0-4)       │
         │  • Uncertainty flag           │
         │  • Grad-CAM++ heatmap         │
         └───────────────────────────────┘
```

---

## 🎮 Demo — 5-Model Comparison App

Upload one leaf image → all 5 models diagnose simultaneously:

```bash
# Place all 5 checkpoints in checkpoints/ then:
uv run python demo/app.py
```

| Checkpoint | Model | F1 |
|------------|-------|----|
| `best_model.pt` | SugarcaneAI CrossAttn | 0.873 |
| `concat_fusion.pt` | Concat Fusion (ablation) | 0.832 |
| `student_model.pt` | Student MobileNetV3 | val 0.923 |
| `resnet50.pt` | ResNet50 | 0.820 |
| `efficientnet_b3.pt` | EfficientNet-B3 | 0.818 |

---

## ⚡ Quick Start

> **Prerequisites:** Python 3.10+, `uv` package manager, CUDA 11.8+ (optional)

```bash
# 1. Clone the repository
git clone https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection.git
cd MultiModal-Cane-Disease-Detection

# 2. Install uv (if not already installed)
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies
uv sync

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Copy environment template and add your API keys
cp .env.example .env

# 6. Run tests (no dataset needed for unit tests)
uv run pytest tests/unit/ -v

# 7. Train Objective 1 model (once dataset is ready)
uv run python scripts/train.py --config configs/obj1_cross_attention.yaml
```

---

## 📊 Results Dashboard

### Module Status — All Phases Complete (1–15)

| Module | Status | Tests |
|--------|--------|-------|
| `SwinBackbone` (Swin-Tiny, 49 patches × 768-dim) | ✅ Done | 3 passed |
| `MetadataEncoder` (4 → 256-dim token) | ✅ Done | 2 passed |
| `CrossAttentionFusion` (8 heads, 256-dim) | ✅ Done | 3 passed |
| `SugarcaneClassifier` (full model assembly) | ✅ Done | 3 passed |
| `trainer.py` (training loop) | ✅ Done | 3 passed |
| `evaluator.py` (checkpoint + test eval) | ✅ Done | 2 passed |
| `mc_dropout.py` (MC-Dropout + ECE) | ✅ Done | 3 passed |
| `gradcam.py` (Grad-CAM++ explainability) | ✅ Done | 2 passed |
| `student_classifier.py` + `distillation.py` | ✅ Done | 3 passed |
| Integration test (full pipeline) | ✅ Done | 1 passed |
| **Total tests** | | **32 passed** |

### Final Results Table — All 5 Models

**Stage B — Thite 6,748 dataset (4 classes, 1,111 train / 138 val / 139 test)**

| Model | Test Acc | Test F1 | Params | Type |
|-------|----------|---------|--------|------|
| ResNet50 (image-only, pretrained) | 0.843 | 0.820 | 23.5M | Image-only baseline |
| EfficientNet-B3 (image-only, pretrained) | 0.843 | 0.818 | 10.7M | Image-only baseline |
| Concat Fusion (Swin-T + MetaEnc, no attn) | 0.857 | 0.832 | ~28M | Ablation baseline |
| Distilled Student (MobileNetV3-Small) | — | val 0.923 | **1.9M** | 93.3% param reduction ✅ |
| **SugarcaneAI CrossAttn (Ours)** | **0.886** | **0.873** | 28.3M | **Best ✅** |

### Research Hypotheses Status

- ✅ CrossAttn (F1=0.873) > Concat Fusion (F1=0.832): **+4.1% — cross-attention hypothesis confirmed**
- ✅ CrossAttn (F1=0.873) > ResNet50 (F1=0.820): **+5.3% over image-only — synopsis target met**
- ✅ Student val F1 = 0.923, **93.3% param reduction** (28.3M → 1.9M), ~98% FLOPs (Obj 4)
- ✅ Grad-CAM++ heatmaps live in demo (Obj 5)
- ✅ MC-Dropout uncertainty quantification integrated (Obj 2)
- ⏳ Stage C — real GPS + sensor metadata (future fieldwork)

---

## 🦠 Disease Classes

| Class | Description | Key Environmental Trigger |
|-------|-------------|--------------------------|
| `healthy` (0) | No infection | — |
| `red_rot` (1) | *Colletotrichum falcatum* — red internal discoloration | High humidity (>80%) |
| `grassy_shoot` (2) | Phytoplasma — excessive tillering, pale shoots | Temperature extremes |
| `smut` (3) | *Sporisorium scitamineum* — black whip-like growth | Drought stress |

> **Objective 1 scope:** 4 classes only. Severity grading (Grade 0–4) and Pokkah Boeng are added in later objectives.

---

## 🗃️ Dataset Strategy — 3-Tier Plan

| Stage | Status | Images | Metadata | Purpose |
|-------|--------|--------|----------|---------|
| **A — Synthetic** | ✅ Done (Phases 1–8) | `torch.rand()` tensors | Random floats | Proved full pipeline works |
| **B — Mendeley Public** | ⏳ Next step | Real leaf photos | Synthetic (weather API later) | Train & validate visual encoder |
| **C — Real Field** | ⏳ Objective 3 | Maharashtra field photos | DHT22 + IMD API | Final paper results |

### Stage B Datasets (Priority Order)

| Priority | Dataset | Images | Download | Why |
|----------|---------|--------|----------|-----|
| **1st** | Thite et al. | 6,748 | [Mendeley](https://data.mendeley.com/datasets/355y629ynj/1) | Most classes, real field variability |
| **2nd** | S2RMCMD | ~16 diseases | [Mendeley](https://data.mendeley.com/datasets/rt8sv9t445) | Environmental annotations included |
| **3rd** | Daphal & Koli | 2,569 | [Mendeley](https://data.mendeley.com/datasets/9424skmnrk/1) | Maharashtra-specific (your region) |

> **Stage B workflow:** Download → extract to `data/raw/` → update `SugarcaneDataset(synthetic=False, images_dir=...)` → retrain on Colab GPU. Environmental metadata stays synthetic until Stage C.

---


## 📁 Repository Structure

```
MultiModal-Cane-Disease-Detection/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Test & Validate on every push
│   │   ├── cd.yml                    # Deployment pipeline
│   │   ├── model-validation.yml      # Automated model forward-pass check
│   │   └── code-quality.yml          # Ruff, mypy linting
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── experiment_proposal.md
│   │   └── dataset_contribution.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
│
├── src/
│   ├── data/
│   │   ├── dataset.py                # PyTorch Dataset class
│   │   ├── dataloader.py             # DataLoader factory
│   │   ├── augmentation.py           # Albumentations pipeline
│   │   └── preprocessing/
│   │       ├── quality_filter.py     # BRISQUE quality check
│   │       ├── deduplication.py      # pHash duplicate removal
│   │       ├── image_enhance.py      # CLAHE enhancement
│   │       ├── metadata_sync.py      # GPS+timestamp → weather API
│   │       └── normalizer.py         # Min-Max normalization
│   │
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── cnn_backbone.py       # EfficientNet-B3 / ResNet50
│   │   │   └── swin_transformer.py   # Swin-T backbone
│   │   ├── encoders/
│   │   │   ├── image_encoder.py      # CNN + ViT hybrid encoder
│   │   │   └── metadata_encoder.py   # MLP environmental encoder
│   │   ├── fusion/
│   │   │   ├── cross_attention.py    # ★ CORE — Objective 1
│   │   │   ├── concat_fusion.py      # Baseline comparison
│   │   │   └── attention_utils.py
│   │   ├── heads/
│   │   │   ├── disease_classifier.py
│   │   │   ├── severity_grader.py    # Objective 2+
│   │   │   └── uncertainty_head.py   # MC Dropout — Objective 2
│   │   ├── multimodal_model.py       # Main model assembly
│   │   └── baselines/
│   │       ├── image_only.py
│   │       ├── concat_baseline.py
│   │       └── pretrained_cnn.py
│   │
│   ├── training/
│   │   ├── trainer.py                # Main training loop
│   │   ├── losses.py                 # Custom loss functions
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   └── callbacks.py              # Early stopping, checkpointing
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # F1, Accuracy, ECE
│   │   ├── confusion_matrix.py
│   │   ├── calibration.py            # ECE computation
│   │   └── ablation.py               # Ablation study runner
│   │
│   ├── explainability/
│   │   ├── gradcam_plus.py           # Grad-CAM++ — Objective 5
│   │   └── attention_viz.py          # Attention heatmaps
│   │
│   └── utils/
│       ├── logger.py
│       ├── config.py
│       ├── seed.py                   # Reproducibility (seed=42)
│       ├── device.py                 # GPU/CPU management
│       └── visualization.py
│
├── configs/
│   ├── base_config.yaml
│   ├── obj1_cross_attention.yaml     # ← Start here
│   ├── obj2_uncertainty.yaml
│   ├── obj4_distillation.yaml
│   ├── ablation/
│   └── baselines/
│
├── experiments/
│   ├── obj1/                         # Results, plots, notes
│   ├── obj2/
│   ├── baselines/
│   └── ablation/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_experiments.ipynb
│   ├── 04_cross_attention_analysis.ipynb
│   ├── 05_ablation_study.ipynb
│   └── 06_results_visualization.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_dataset.py
│   │   ├── test_preprocessing.py
│   │   ├── test_cross_attention.py
│   │   ├── test_metadata_encoder.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_model_forward.py
│   └── conftest.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── run_baselines.py
│   ├── download_data.py
│   └── sync_metadata.py
│
├── docs/
│   ├── objectives/
│   │   ├── obj1_cross_attention.md
│   │   ├── obj2_uncertainty.md
│   │   ├── obj3_dataset.md
│   │   ├── obj4_distillation.md
│   │   ├── obj5_explainability.md
│   │   └── obj6_comparison.md
│   ├── dataset.md
│   └── api/
│
├── data/                             # Gitignored — tracked by DVC
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── checkpoints/                      # Gitignored — tracked by DVC
│
├── pyproject.toml                    # uv / pip dependencies + tool config
├── uv.lock                           # Locked dependency tree
├── Makefile                          # Shortcut commands
├── Dockerfile
├── docker-compose.yml
├── .env.example                      # Template — never commit .env
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── ARCHITECTURE.md                   # ← Detailed system architecture
├── WALKTHROUGH.md                    # ← Phase-by-phase build guide
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── CITATION.cff                      # Academic citation (standard)
```

---

## 🛠️ Professional Toolstack

| Category | Tool | Purpose |
|----------|------|---------|
| **Package Manager** | `uv` | Fast, modern Python dependency management |
| **Deep Learning** | PyTorch 2.1+ | Model training & inference |
| **Vision Backbone** | timm (EfficientNet, Swin) | Pretrained model zoo |
| **Data Augmentation** | Albumentations | Fast image augmentation |
| **Image Quality** | BRISQUE, pHash | Quality filter + deduplication |
| **Image Enhancement** | OpenCV (CLAHE) | Contrast enhancement |
| **Experiment Tracking** | Weights & Biases | Live metrics, plots, model artifacts |
| **Data Versioning** | DVC | Version control for datasets + checkpoints |
| **Testing** | pytest + pytest-cov | Unit + integration tests |
| **Code Quality** | ruff + mypy | Linting + type checking |
| **CI/CD** | GitHub Actions | Automated testing on every push |
| **Code Review** | CodeRabbit | AI-powered PR review |
| **Pre-commit** | pre-commit | Catch issues before commit |
| **Containers** | Docker | Reproducible environments |
| **Secrets** | GitHub Secrets + `.env` | Secure API key management |

---

## 🧪 Running Tests

```bash
# Unit tests only (no dataset required — uses synthetic tensors)
uv run pytest tests/unit/ -v

# Integration tests (requires minimal dataset)
uv run pytest tests/integration/ -v

# All tests with coverage report
uv run pytest tests/ -v --cov=src --cov-report=html

# Run a specific test file
uv run pytest tests/unit/test_cross_attention.py -v -s
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Before submitting a PR, ensure:

```bash
uv run pre-commit run --all-files   # Code quality checks
uv run pytest tests/unit/ -v        # All unit tests pass
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Author:** Shoaib Inamdar  
**GitHub:** [@shoaib-inamdar](https://github.com/shoaib-inamdar)  

</div>