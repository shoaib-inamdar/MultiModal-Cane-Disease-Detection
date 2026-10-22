# Objective 1 -- Cross-Attention Multimodal Feature Fusion

## Overview

**Research Question:**
*"Can Cross-Modal Attention between leaf visual features and environmental metadata resolve the visual ambiguity between early-stage Red Rot lesions and abiotic stresses like Leaf Scorch under field conditions?"*

**Core Hypothesis:**
Cross-Attention fusion will improve macro F1-score by >=5% over vision-only baselines, because environmental context (humidity, temperature, soil moisture) acts as a diagnostic prior that resolves visually identical symptoms.

---

## Architecture (As Built -- Phases 1-8)

```
Visual Tokens (Q)       Environmental Tokens (K, V)
[B, 49, 768]            [B, 1, 256]
      |                      |
      |   proj 768->256      |
      v                      |
[B, 49, 256] -----------> Cross-Attention (8 heads)
                             |
                             v
                    Fused Context-Aware Features [B, 49, 256]
                             |
                        mean(dim=1)
                             v
                    [B, 256] fused vector
                             |
                      nn.Linear(256, num_classes)
                             v
                    Logits [B, num_classes]
```

**Actual tensor dimensions in this repo:**

| Stage | Shape | Module |
|-------|-------|--------|
| Image input | `[B, 3, 224, 224]` | — |
| Visual patches | `[B, 49, 768]` | `SwinBackbone` |
| Projected patches | `[B, 49, 256]` | `CrossAttentionFusion.proj` |
| Env token | `[B, 1, 256]` | `MetadataEncoder` |
| Fused vector | `[B, 256]` | `CrossAttentionFusion` (mean pool) |
| Logits | `[B, num_classes]` | `nn.Linear(256, num_classes)` |

---

## Key Files (Current Repo Structure)

| File | Purpose |
|------|---------|
| `src/models/backbone/swin_backbone.py` | Swin-Tiny encoder -- `[B,3,224,224]` -> `[B,49,768]` |
| `src/models/encoders/metadata_encoder.py` | MLP env encoder -- `[B,4]` -> `[B,1,256]` |
| `src/models/fusion/cross_attention.py` | **Core module** -- CrossAttentionFusion |
| `src/models/sugarcane_classifier.py` | Full model assembly + classifier head |
| `src/training/trainer.py` | `train_one_epoch`, `evaluate`, `save_checkpoint` |
| `src/training/metrics.py` | `compute_metrics` -- accuracy + macro F1 |
| `src/evaluation/evaluator.py` | `evaluate_on_test`, `load_checkpoint`, `print_report` |
| `configs/obj1_cross_attention.yaml` | Training hyperparameters |
| `notebooks/01_evaluate.ipynb` | Colab evaluation notebook (keep in repo for reproducibility) |

---

## Baselines (What We Compare Against)

| Model | Description | How trained |
|-------|-------------|------------|
| **SugarcaneAI (Cross-Attention)** | Full model -- image + weather | Trained from scratch |
| Image Only (Swin-Tiny) | Same backbone, no metadata branch | **Separate** training run |
| Metadata Only (MLP) | MetadataEncoder + Linear head, no image branch | **Separate** training run |
| Concat Fusion | Concatenate visual + env features, no attention | **Separate** training run |

> **Important:** Baselines must be trained independently from scratch on the same data split.
> Do NOT ablate from the already-trained multimodal model -- that does not constitute a valid baseline.

---

## Dataset Stages

| Stage | Dataset | Metadata | Status |
|-------|---------|----------|--------|
| **A -- Synthetic** | `torch.rand()` tensors | Random floats | Done (Phases 1-8) |
| **B -- Kaggle/Mendeley** | Thite 6,748 + Daphal 2,569 leaf images | Synthetic (API-sourced later) | Next step |
| **C -- Real Field** | Maharashtra field photos | DHT22 + IMD weather API | Objective 3 |

**Stage B recommended datasets (in priority order):**
1. Thite 6,748 -- https://data.mendeley.com/datasets/355y629ynj/1 (primary training)
2. Daphal & Koli 2,569 -- https://data.mendeley.com/datasets/9424skmnrk/1 (Maharashtra validation)
3. S2RMCMD -- https://data.mendeley.com/datasets/rt8sv9t445 (multimodal annotations)

---

## Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Macro F1-Score | Balanced accuracy across all 4 classes | >5% over baseline |
| Accuracy | Overall correct predictions | Track, not primary |
| Confusion Matrix | Per-class performance | Visualize |
| ECE | Model calibration honesty | <=0.05 (Objective 2) |

---

## Build Status (Phase by Phase)

- [x] Phase 1-2: Project scaffold, CI/CD, synthetic dataset
- [x] Phase 3: Swin-Tiny backbone (`test_swin_backbone.py` -- 3 passed)
- [x] Phase 4: MLP metadata encoder (`test_metadata_encoder.py` -- 2 passed)
- [x] Phase 5: Cross-Attention fusion (`test_cross_attention.py` -- 3 passed)
- [x] Phase 6: SugarcaneClassifier assembly (`test_sugarcane_classifier.py` -- 3 passed)
- [x] Phase 7: Training loop + metrics (`test_trainer.py` -- 3 passed)
- [x] Phase 8: Evaluator + checkpoint (`test_evaluator.py` -- 2 passed)
- [x] Unit tests: **24 passing** (all modules covered)
- [ ] Phase 9: Integration tests + coverage report
- [ ] Phase 10: Final CI/CD hardening + documentation freeze
- [ ] Stage B: Real image dataset integration
- [ ] Baseline comparison (3 separately trained models)
- [ ] Ablation study
- [ ] Results table with real F1 numbers
