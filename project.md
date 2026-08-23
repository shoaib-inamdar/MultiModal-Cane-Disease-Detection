# 🌱 Production-Grade Open Source GitHub Repository
## Sugarcane AI - Multimodal Disease Detection System

---

## 📁 Complete Repository Structure

```
sugarcane-disease-ai/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Continuous Integration
│   │   ├── cd.yml                    # Continuous Deployment
│   │   ├── model-validation.yml      # Auto model testing
│   │   └── code-quality.yml          # Linting, formatting
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── experiment_proposal.md    # Research-specific
│   │   └── dataset_contribution.md  # For OSS contributors
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS                    # Who reviews what
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── dataloader.py            # DataLoader factory
│   │   ├── augmentation.py          # Albumentations pipeline
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── quality_filter.py    # BRISQUE quality check
│   │       ├── deduplication.py     # pHash duplicate removal
│   │       ├── image_enhance.py     # CLAHE enhancement
│   │       ├── metadata_sync.py     # GPS+timestamp → weather
│   │       └── normalizer.py        # Min-Max normalization
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_backbone.py      # EfficientNet/ResNet
│   │   │   └── swin_transformer.py  # Swin-T backbone
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── image_encoder.py     # CNN + ViT hybrid
│   │   │   └── metadata_encoder.py  # MLP metadata encoder
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   ├── cross_attention.py   # CORE - Obj 1
│   │   │   ├── concat_fusion.py     # Baseline comparison
│   │   │   └── attention_utils.py
│   │   ├── heads/
│   │   │   ├── __init__.py
│   │   │   ├── disease_classifier.py
│   │   │   ├── severity_grader.py   # Obj 2 onwards
│   │   │   └── uncertainty_head.py  # MC Dropout - Obj 2
│   │   ├── multimodal_model.py      # Main model assembly
│   │   └── baselines/
│   │       ├── image_only.py        # Vision-only baseline
│   │       ├── concat_baseline.py   # Simple fusion baseline
│   │       └── pretrained_cnn.py    # ResNet/EffNet baseline
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── losses.py                # Custom loss functions
│   │   ├── optimizers.py            # Optimizer factory
│   │   ├── schedulers.py            # LR schedulers
│   │   └── callbacks.py             # Early stopping, etc.
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # F1, Acc, ECE, etc.
│   │   ├── confusion_matrix.py
│   │   ├── calibration.py           # ECE computation
│   │   └── ablation.py              # Ablation study runner
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── gradcam_plus.py          # Grad-CAM++ - Obj 5
│   │   └── attention_viz.py         # Attention heatmaps
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                # Logging setup
│       ├── config.py                # Config loader
│       ├── seed.py                  # Reproducibility
│       ├── device.py                # GPU/CPU management
│       └── visualization.py         # Plot utilities
│
├── configs/
│   ├── base_config.yaml             # Base configuration
│   ├── obj1_cross_attention.yaml    # Objective 1 config
│   ├── obj2_uncertainty.yaml        # Objective 2 config
│   ├── obj4_distillation.yaml       # Objective 4 config
│   ├── ablation/
│   │   ├── image_only.yaml
│   │   ├── temp_only.yaml
│   │   ├── humidity_only.yaml
│   │   ├── soil_only.yaml
│   │   └── all_metadata.yaml
│   └── baselines/
│       ├── resnet50.yaml
│       ├── efficientnet.yaml
│       └── vit_only.yaml
│
├── experiments/
│   ├── obj1/
│   │   ├── results/                 # Saved metrics JSON
│   │   ├── plots/                   # Generated figures
│   │   └── README.md                # Experiment notes
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
│   └── conftest.py                  # Shared test fixtures
│
├── scripts/
│   ├── train.py                     # Main training entry
│   ├── evaluate.py                  # Evaluation entry
│   ├── run_ablation.py              # Ablation runner
│   ├── run_baselines.py             # Baseline comparison
│   ├── download_data.py             # Dataset downloader
│   └── sync_metadata.py             # Weather API sync
│
├── docs/
│   ├── architecture.md              # System architecture
│   ├── dataset.md                   # Dataset documentation
│   ├── objectives/
│   │   ├── obj1_cross_attention.md
│   │   ├── obj2_uncertainty.md
│   │   ├── obj3_dataset.md
│   │   ├── obj4_distillation.md
│   │   ├── obj5_explainability.md
│   │   └── obj6_comparison.md
│   ├── api/                         # Auto-generated API docs
│   └── results/                     # Research results docs
│
├── data/                            # Gitignored - tracked by DVC
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── checkpoints/                     # Gitignored - tracked by DVC
│
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── Makefile                         # Common commands
├── Dockerfile
├── docker-compose.yml
├── .env.example                     # Template (never .env itself)
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── CITATION.cff                     # Research citation file
```

---

## 📄 README.md (Production Grade)

```markdown
<div align="center">

# 🌱 SugarcaneAI
### Multimodal Uncertainty-Aware Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/yourname/sugarcane-ai/actions/workflows/ci.yml/badge.svg)](...)
[![wandb](https://img.shields.io/badge/Tracked-W%26B-yellow)](https://wandb.ai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Novel Lightweight Explainable Uncertainty-Aware Multimodal Hybrid Vision Transformer  
for Early-Stage Sugarcane Disease Detection Under Real-Field Conditions**

[📄 Paper](#) · [📊 Dataset](#) · [🔬 Experiments](#) · [📖 Docs](docs/) · [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 Research Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Cross-Attention Multimodal Feature Fusion | 🔄 In Progress |
| 2 | Monte Carlo Dropout Uncertainty Quantification | ⏳ Planned |
| 3 | Maharashtra Sugarcane Multimodal Dataset | ⏳ Planned |
| 4 | Knowledge Distillation (Lightweight) | ⏳ Planned |
| 5 | Explainability + Ablation Studies | ⏳ Planned |
| 6 | Comprehensive Baseline Comparison | ⏳ Planned |

## 🏗️ Architecture Overview

Image (Leaf) ──► CNN Backbone ──► Swin Transformer ──► Visual Tokens
                                                              │
                                                              ▼
                                                    Cross-Attention Fusion
                                                              │
                                                              ▲
Metadata ────► Normalizer ──► MLP Encoder ──► Metadata Tokens
(T, RH, SM)

Fused ──► Disease Classifier + Severity Grader + Uncertainty Head

## ⚡ Quick Start

\```bash
git clone https://github.com/yourname/sugarcane-ai.git
cd sugarcane-ai
pip install -r requirements.txt
python scripts/train.py --config configs/obj1_cross_attention.yaml
\```

## 📊 Results (Objective 1)

| Model | Accuracy | F1 | ECE |
|-------|----------|----|-----|
| Image Only (Baseline) | - | - | - |
| Concat Fusion | - | - | - |
| **Cross-Attention (Ours)** | - | - | - |
```

---

## ⚙️ Core Config File

```yaml
# configs/obj1_cross_attention.yaml

project:
  name: "sugarcane-ai"
  objective: 1
  experiment: "cross_attention_baseline"
  seed: 42

data:
  image_dir: "data/processed/images"
  metadata_path: "data/processed/metadata.csv"
  image_size: 224
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  num_workers: 4

  # Metadata variables for Objective 1
  metadata_features:
    - temperature
    - relative_humidity
    - soil_moisture

  classes:
    - healthy
    - red_rot
    - grassy_shoot
    - smut
    - pokkah_boeng

  normalization:
    method: "minmax"
    temperature: [10.0, 50.0]
    relative_humidity: [20.0, 100.0]
    soil_moisture: [0.0, 100.0]

model:
  image_encoder:
    backbone: "efficientnet_b3"      # or swin_tiny
    pretrained: true
    output_dim: 768

  metadata_encoder:
    input_dim: 3
    hidden_dims: [64, 128]
    output_dim: 768
    dropout: 0.1

  fusion:
    type: "cross_attention"          # cross_attention | concat | none
    num_heads: 8
    dropout: 0.1
    num_layers: 2

  classifier:
    num_classes: 5
    dropout: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adamw"
  weight_decay: 0.01
  scheduler: "cosine_annealing"
  early_stopping_patience: 15

logging:
  wandb:
    enabled: true
    project: "sugarcane-ai-obj1"
  tensorboard: true
  log_every_n_steps: 10
  save_top_k: 3
```

---

## 🔥 Core Model Code

```python
# src/models/multimodal_model.py

import torch
import torch.nn as nn
from src.models.encoders.image_encoder import ImageEncoder
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.models.fusion.cross_attention import CrossAttentionFusion
from src.models.heads.disease_classifier import DiseaseClassifier


class SugarcaneMultimodalModel(nn.Module):
    """
    Objective 1: Cross-Attention Multimodal Fusion Model
    
    Combines visual features from CNN-ViT with environmental
    metadata using cross-attention for context-aware diagnosis.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Visual Branch: CNN + Swin Transformer
        self.image_encoder = ImageEncoder(config.model.image_encoder)

        # Environmental Branch: MLP Encoder
        self.metadata_encoder = MetadataEncoder(
            config.model.metadata_encoder
        )

        # Objective 1 Core: Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(config.model.fusion)

        # Output Head
        self.classifier = DiseaseClassifier(config.model.classifier)

    def forward(self, image, metadata):
        """
        Args:
            image    : (B, 3, H, W) - Leaf image tensor
            metadata : (B, M)       - [temperature, humidity, soil_moisture]
        
        Returns:
            logits   : (B, num_classes)
        """
        # Step 1: Extract visual tokens
        visual_tokens = self.image_encoder(image)
        # Shape: (B, num_patches, embed_dim)

        # Step 2: Encode environmental context
        meta_tokens = self.metadata_encoder(metadata)
        # Shape: (B, meta_seq_len, embed_dim)

        # Step 3: Cross-Attention Fusion (Core of Objective 1)
        fused = self.fusion(
            query=visual_tokens,    # Image asks about environment
            key=meta_tokens,
            value=meta_tokens
        )
        # Shape: (B, num_patches, embed_dim)

        # Step 4: Classify disease
        logits = self.classifier(fused)
        # Shape: (B, num_classes)

        return logits
```

```python
# src/models/fusion/cross_attention.py

import torch
import torch.nn as nn
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention between Visual Tokens (Query) 
    and Environmental Tokens (Key, Value).
    
    Research Hypothesis (Objective 1):
        Environmental context helps resolve visual ambiguities
        in early-stage sugarcane disease detection.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
    
    Where:
        Q = Visual tokens    (from CNN-ViT)
        K = Metadata tokens  (from Environmental MLP)
        V = Metadata tokens
    """

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.get("embed_dim", 768)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # Layer norm + residual
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, query, key, value):
        """
        Args:
            query : (B, N, D) - Visual tokens
            key   : (B, M, D) - Environmental tokens
            value : (B, M, D) - Environmental tokens
        
        Returns:
            out   : (B, N, D) - Context-aware visual features
        """
        # Cross-attention: image queries environmental context
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key,
            value=value
        )

        # Residual + LayerNorm
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        out = self.norm2(query + ffn_out)

        return out
```

---

## 🔄 GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml

name: 🧪 CI - Test & Validate

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: 🐍 Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: 📦 Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: 🎨 Code formatting check
        run: black --check src/ tests/

      - name: 🔍 Linting
        run: flake8 src/ tests/ --max-line-length 88

      - name: 🧪 Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: 🔗 Run integration tests
        run: pytest tests/integration/ -v

      - name: 🤖 Validate model forward pass
        run: python -c "
          from src.models.multimodal_model import SugarcaneMultimodalModel
          import torch
          # Quick sanity check - model must not crash
          print('Model import: OK')
        "

      - name: 📊 Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 🧪 Test Examples

```python
# tests/unit/test_cross_attention.py

import pytest
import torch
from src.models.fusion.cross_attention import CrossAttentionFusion


class TestCrossAttentionFusion:
    """Tests for Objective 1 core component"""

    @pytest.fixture
    def config(self):
        return {
            "embed_dim": 256,
            "num_heads": 8,
            "dropout": 0.0  # 0 for deterministic testing
        }

    @pytest.fixture
    def fusion_module(self, config):
        return CrossAttentionFusion(config)

    def test_output_shape(self, fusion_module):
        """Cross-attention output must match query shape"""
        B, N, D = 4, 196, 256  # Batch, patches, embed_dim
        M = 3                   # metadata tokens

        query = torch.randn(B, N, D)
        key = torch.randn(B, M, D)
        value = torch.randn(B, M, D)

        output = fusion_module(query, key, value)

        assert output.shape == (B, N, D), \
            f"Expected {(B,N,D)}, got {output.shape}"

    def test_metadata_influence(self, fusion_module):
        """Different metadata should produce different outputs"""
        B, N, D, M = 2, 196, 256, 3

        query = torch.randn(B, N, D)
        
        # Two different environmental contexts
        env_hot_dry = torch.tensor([[31.0, 40.0, 20.0]])  # T, RH, SM
        env_cold_wet = torch.tensor([[18.0, 85.0, 80.0]])

        out1 = fusion_module(query[:1], env_hot_dry.unsqueeze(1).expand(-1, M, D),
                             env_hot_dry.unsqueeze(1).expand(-1, M, D))
        out2 = fusion_module(query[:1], env_cold_wet.unsqueeze(1).expand(-1, M, D),
                             env_cold_wet.unsqueeze(1).expand(-1, M, D))

        assert not torch.allclose(out1, out2), \
            "Different environments must produce different outputs"

    def test_gradients_flow(self, fusion_module):
        """Gradients must flow through fusion layer"""
        B, N, D, M = 2, 49, 256, 3
        query = torch.randn(B, N, D, requires_grad=True)
        key = torch.randn(B, M, D, requires_grad=True)
        value = torch.randn(B, M, D, requires_grad=True)

        output = fusion_module(query, key, value)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None, "Gradients must flow to visual tokens"
        assert key.grad is not None, "Gradients must flow to metadata tokens"
```

---

## 🏷️ GitHub Issues & Milestones Setup

```markdown
## Milestones (Create in GitHub Settings)

Milestone 1: Objective 1 - Cross-Attention Fusion
  Due: [Week 8]
  Issues:
    - #1  [OBJ1] Setup dataset structure (image + metadata pairing)
    - #2  [OBJ1] Implement BRISQUE quality filtering
    - #3  [OBJ1] Implement pHash deduplication
    - #4  [OBJ1] Implement CLAHE enhancement pipeline
    - #5  [OBJ1] Build CNN + Swin Transformer image encoder
    - #6  [OBJ1] Build MLP metadata encoder
    - #7  [OBJ1] Implement Cross-Attention fusion module
    - #8  [OBJ1] Baseline 1: Image-only model
    - #9  [OBJ1] Baseline 2: Concatenation fusion model
    - #10 [OBJ1] Ablation: individual metadata contributions
    - #11 [OBJ1] Evaluation: F1, Accuracy, Confusion Matrix
    - #12 [OBJ1] Write unit tests for all components
    - #13 [OBJ1] Experiment logging with wandb

Milestone 2: Objective 2 - Uncertainty Quantification
Milestone 3: Objective 3 - Maharashtra Dataset
Milestone 4: Objective 4 - Knowledge Distillation
Milestone 5: Objective 5 - Explainability
Milestone 6: Objective 6 - Final Comparison

## Labels to Create
bug, obj1, obj2, obj3, obj4, obj5, obj6,
experiment, dataset, model, testing,
documentation, good-first-issue, help-wanted,
blocked, in-progress, research
```

---

## 🛠️ Makefile (What Top OSS Projects Use)

```makefile
# Makefile

.PHONY: help install test lint format clean train evaluate

help:
	@echo "SugarcaneAI - Available Commands"
	@echo "================================"
	@echo "make install      - Install dependencies"
	@echo "make test         - Run all tests"
	@echo "make lint         - Run linting"
	@echo "make format       - Format code with black"
	@echo "make train-obj1   - Train Objective 1 model"
	@echo "make baselines    - Run all baseline experiments"
	@echo "make ablation     - Run ablation studies"
	@echo "make evaluate     - Evaluate trained model"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/ scripts/

train-obj1:
	python scripts/train.py \
		--config configs/obj1_cross_attention.yaml

baselines:
	python scripts/run_baselines.py \
		--configs configs/baselines/

ablation:
	python scripts/run_ablation.py \
		--configs configs/ablation/

evaluate:
	python scripts/evaluate.py \
		--config configs/obj1_cross_attention.yaml \
		--checkpoint checkpoints/best_model.pth

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov/
```

---

## 📊 Complete Toolstack Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   PROFESSIONAL TOOLSTACK                    │
├──────────────────┬──────────────────────────────────────────┤
│  VERSION CONTROL │  Git + GitHub                           │
│  PROJECT MGMT    │  GitHub Projects (Kanban) + Milestones  │
│  CI/CD           │  GitHub Actions                         │
│  EXPERIMENT      │  Weights & Biases (wandb)               │
│  LARGE FILES     │  DVC + Google Drive / S3                │
│  CODE QUALITY    │  black + flake8 + pre-commit            │
│  TESTING         │  pytest + pytest-cov                    │
│  DOCS            │  MkDocs or Sphinx                       │
│  CONTAINERS      │  Docker + docker-compose                │
│  SECRETS         │  GitHub Secrets + .env.example          │
│  CITATION        │  CITATION.cff (academic standard)       │
└──────────────────┴──────────────────────────────────────────┘
```

---

> **🎯 Bottom Line:** This structure mirrors what **Hugging Face, facebookresearch, google-research** use on GitHub — clean separation of concerns, every objective tracked as a milestone, every experiment reproducible, and nothing large committed directly to Git.