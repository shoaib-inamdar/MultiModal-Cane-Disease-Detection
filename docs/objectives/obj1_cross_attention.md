# Objective 1 — Cross-Attention Multimodal Feature Fusion

## Overview

**Research Question:**  
*"Can Cross-Modal Attention between leaf visual features and environmental metadata resolve the visual ambiguity between early-stage Red Rot lesions and abiotic stresses like Leaf Scorch under field conditions?"*

**Core Hypothesis:**  
Cross-Attention fusion will improve macro F1-score by ≥5% over vision-only baselines, because environmental context (humidity, temperature, soil moisture) acts as a diagnostic prior that resolves visually identical symptoms.

---

## Architecture

```
Visual Tokens (Q)     Environmental Tokens (K, V)
[B, 196, 128]         [B, 1, 128]
      │                     │
      └──────┬──────────────┘
             ▼
    Cross-Attention Layer
    Attn(Q,K,V) = softmax(QKᵀ/√d) · V
             │
             ▼
    Fused Context-Aware Features
    [B, 196, 128]
             │
             ▼
    Disease Classifier → 5 classes
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/fusion/cross_attention.py` | **Core module** — the cross-attention mechanism |
| `src/models/encoders/image_encoder.py` | CNN + Swin visual feature extraction |
| `src/models/encoders/metadata_encoder.py` | MLP environmental feature encoding |
| `src/models/multimodal_model.py` | Full model assembly |
| `configs/obj1_cross_attention.yaml` | Training configuration |
| `tests/unit/test_cross_attention.py` | Unit tests for fusion module |

---

## Baselines (What We Compare Against)

| Model | Description | Config |
|-------|-------------|--------|
| Image Only (ResNet50) | Standard CNN, no metadata | `configs/baselines/resnet50.yaml` |
| Image Only (EfficientNet-B3) | Efficient CNN, no metadata | `configs/baselines/efficientnet.yaml` |
| Concat Fusion | Simple feature concatenation | `configs/baselines/concat.yaml` |
| **Cross-Attention (Ours)** | Novel fusion mechanism | `configs/obj1_cross_attention.yaml` |

---

## Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Macro F1-Score | Balanced accuracy across all 5 classes | >5% over baseline |
| Accuracy | Overall correct predictions | Track, not primary |
| ECE | Model calibration (honesty) | ≤0.05 |
| Confusion Matrix | Per-class performance | Visualize |

---

## Ablation Study Plan

To prove Cross-Attention adds value, we run ablations:

| Ablation | Removed Component | Purpose |
|----------|-------------------|---------|
| Image Only | All metadata | Lower bound |
| Temperature Only | Humidity + Moisture | Single-feature test |
| Humidity Only | Temperature + Moisture | Single-feature test |
| Soil Moisture Only | Temperature + Humidity | Single-feature test |
| All Metadata (Concat) | Cross-Attention (use concat) | Fusion method test |
| **Full Model (Ours)** | Nothing | Upper bound |

---

## Results (To Be Filled)

| Model | Accuracy | Macro F1 | ECE |
|-------|----------|----------|-----|
| ResNet50 Baseline | — | — | — |
| EfficientNet Baseline | — | — | — |
| Concat Fusion | — | — | — |
| **Cross-Attention (Ours)** | **—** | **—** | **—** |

---

## Training Commands

```bash
# Train baseline models first
uv run python scripts/run_baselines.py --configs configs/baselines/

# Train our cross-attention model
uv run python scripts/train.py --config configs/obj1_cross_attention.yaml

# Run ablation studies
uv run python scripts/run_ablation.py --configs configs/ablation/

# Evaluate best checkpoint
uv run python scripts/evaluate.py \
    --config configs/obj1_cross_attention.yaml \
    --checkpoint checkpoints/obj1_best.pth
```

---

## Status

- [ ] Preprocessing pipeline implemented
- [ ] Synthetic dataset class implemented  
- [ ] CNN backbone implemented
- [ ] Swin-Transformer encoder implemented
- [ ] MLP metadata encoder implemented
- [ ] Cross-Attention fusion implemented
- [ ] Training loop implemented
- [ ] Baseline models implemented
- [ ] Evaluation metrics implemented
- [ ] Unit tests written
- [ ] Ablation study run
- [ ] Results documented
- [ ] CI/CD configured
