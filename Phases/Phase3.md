# 🌾 SugarcaneAI — Objective 1 Build Walkthrough
## Phase-by-Phase Teaching Guide

> **Your Role:** You build. I guide. Say **"Phase X done"** when finished → I unlock the next phase.

---

## 📋 What Objective 1 Builds

A working multimodal model that:
1. Takes a **sugarcane leaf image** + **environmental metadata** [Temperature, Humidity, Soil Moisture, Rainfall]
2. Fuses them using **Cross-Attention** (your novel contribution)
3. Outputs one of **4 disease classes**: Healthy, Red Rot, Grassy Shoot, Smut

> ⚠️ Severity grading, MC-Dropout, Explainability, and Knowledge Distillation are **NOT** part of Objective 1. They come in Objectives 2–5.

---

## 🗺️ Phase Map — Objective 1

| Phase | Name | Time Estimate | Status |
|-------|------|--------------|--------|
| **1** | Setup & Workspace | — | ✅ Done |
| **2** | Dummy Dataset + DataLoader | — | ✅ Done |
| **3** | Pretrained Visual Backbone | 2–3 hours | ⏳ In Progress |
| **4** | Metadata MLP Encoder | 1–2 hours | 🔒 Locked |
| **5** | Cross-Attention Fusion ⭐ | 3–4 hours | 🔒 Locked |
| **6** | Full Model Assembly | 1–2 hours | 🔒 Locked |
| **7** | Training on Colab/Kaggle | 3–4 hours | 🔒 Locked |
| **8** | Evaluation & Baselines | 2–3 hours | 🔒 Locked |
| **9** | Testing (pytest) | 2–3 hours | 🔒 Locked |
| **10** | Final Cleanup & Docs | 1–2 hours | 🔒 Locked |

**Total estimated time: ~20–26 hours of focused work**

---

## 🗃️ Dataset Strategy — 3 Stages

| Stage | When | Images | Environmental Data | Purpose |
|-------|------|--------|--------------------|---------| 
| **A — Dummy** | Now (Phases 2–7) | `torch.rand(3,224,224)` synthetic tensors | Randomly generated [T, H, Sm, R] | Test pipeline architecture |
| **B — Kaggle Public** | After Stage A works | Real leaf images from Kaggle | **Still synthetic metadata** | Validate visual encoder |
| **C — Real Field** | Final (Objective 3) | Your own Maharashtra field photos | Real DHT22 sensor readings synced by timestamp | Final research results |

---

---

# ✅ PHASE 1 — COMPLETED

- ✅ Python 3.14, uv 0.8.14
- ✅ All folders + `__init__.py` files
- ✅ `pyproject.toml` — torch 2.13, timm 1.0.28 installed
- ✅ `.gitignore`, `.env.example`, `.pre-commit-config.yaml`
- ✅ `.github/workflows/ci.yml`
- ✅ `tests/conftest.py`

---

---

# ✅ PHASE 2 — COMPLETED

- ✅ `configs/obj1_cross_attention.yaml` — ML training config
- ✅ `src/utils/config.py` — `load_config()` working
- ✅ `src/utils/seed.py` — `set_seed()` working
- ✅ `src/data/augmentation.py` — `get_train_transforms()` + `get_val_transforms()`
- ✅ `src/data/dataset.py` — `SugarcaneDataset` with synthetic data
- ✅ `src/data/dataloader.py` — `get_dataloaders()` working
- ✅ `tests/unit/test_dataset.py` — 4/4 tests PASS
- ✅ CI: `ruff format --check` ✅ `ruff check` ✅ `pytest 8 passed` ✅

---

---

# ⏳ PHASE 3 — Pretrained Visual Backbone (Swin-Tiny)
**Time: 2–3 hours**

---

## 🎯 Goal
Load **Swin-Tiny** from `timm` (a pretrained model zoo). It will be your "visual eye" — it looks at the leaf image and produces **49 feature tokens**, one per image patch. These tokens are later fed into Cross-Attention.

## 🤔 Why Pretrained — NOT From Scratch

Your dataset will have only a few thousand images. A Swin Transformer trained from scratch needs millions. But Swin-Tiny was **already trained on ImageNet-21k** (14 million images) — it already knows textures, edges, colors, shapes. You just "redirect" that knowledge toward disease features. This is called **Transfer Learning**.

> 🧠 Think of it like hiring an expert photographer who already knows how to spot visual patterns — you just tell them what specific disease signs to look for.

---

## 🪜 Step 3.1 — Create `src/models/backbone/swin_backbone.py`

PSEUDOCODE — Do NOT copy. Write the Python yourself using this as a guide:

```
--- IMPORTS ---

import timm
import torch
import torch.nn as nn


--- CLASS: SwinBackbone(nn.Module) ---

  Why this class?
    We wrap timm's model in our own class so the rest of the code
    doesn't need to know which backbone is being used.
    Later you can swap Swin-Tiny for anything else by changing one line.


  __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):

    self.model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,      ← removes the 1000-class ImageNet head
        global_pool="",     ← disables pooling → keeps all 49 patch tokens
    )

    self.embed_dim = self.model.num_features   ← will be 768 for Swin-Tiny

    Why num_classes=0?
      By default timm loads Swin with a final Linear(768 → 1000) layer for
      ImageNet classification. Setting num_classes=0 removes that last layer
      so we get raw 768-dim features instead of class probabilities.

    Why global_pool=""?
      Without this, even with num_classes=0, timm still averages all 49 patch
      tokens into one [B, 768] vector. We need all 49 tokens separately for
      Cross-Attention (each image patch individually queries environmental
      context). So we disable pooling completely.


  ---


  forward(self, x):

    Args:
      x : Tensor of shape [B, 3, 224, 224]   ← normalised RGB leaf image

    Returns:
      tokens : Tensor of shape [B, 49, 768]
                   ↑         ↑    ↑
                   batch    patches  feature dim
                            (7×7)

    return self.model(x)   ← timm handles everything internally
```

> 💡 **Why 49 patches?**
> Swin-Tiny uses 4×4 pixel patch embeddings on a 224×224 image.
> After its internal downsampling: `224/4/4/2 = 7` → 7×7 = **49 spatial patches**.
> Each patch = one 768-dim vector describing that region of the leaf.

---

## 🪜 Step 3.2 — Create `tests/unit/test_swin_backbone.py`

PSEUDOCODE — Write 3 tests:

```
--- IMPORTS ---

import torch
import pytest
from src.models.backbone.swin_backbone import SwinBackbone


--- FIXTURE ---

  backbone (scope="module"):
    Return SwinBackbone(pretrained=False)

    Why pretrained=False?
      In tests we only check shapes and gradient flow — not accuracy.
      pretrained=False skips the 112MB download so CI tests run instantly.


---


  test_output_shape(backbone):

    Purpose: Verify the backbone returns exactly [B, 49, 768]

    B = 2
    x = torch.rand(B, 3, 224, 224)
    out = backbone(x)

    assert out.shape == torch.Size([B, 49, 768])


  ---


  test_gradient_flows(backbone):

    Purpose: Verify backward pass works (no dead/broken layers)

    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    out = backbone(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None

    Why?
      If any layer blocks gradients, x.grad stays None.
      This would break training — the backbone weights could never update.


  ---


  test_no_classification_output(backbone):

    Purpose: Confirm output is 3D tokens, NOT 2D class logits

    x = torch.rand(1, 3, 224, 224)
    out = backbone(x)

    assert out.dim() == 3

    Why?
      If global_pool="" was forgotten, output would be [B, 768] (2D).
      That would silently break Cross-Attention which expects [B, 49, 768].
```

## ✅ Done When
```powershell
uv run pytest tests/unit/test_swin_backbone.py -v
# 3 passed ✅
```

> ⚠️ First run downloads Swin-Tiny weights (~112 MB, once only). Use Colab if your internet is slow. In CI this is skipped because `pretrained=False`.

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `global_pool=""` | Output is `[B, 768]` instead of `[B, 49, 768]` — breaks Cross-Attention |
| Using `pretrained=True` in tests | CI will try to download 112MB every run — always use `pretrained=False` in fixtures |
| Building Swin from scratch | Don't. `timm.create_model(...)` is all you need |

---

## Key Tensor Shape After Phase 3

```
Image input:      [B, 3, 224, 224]
       ↓
  SwinBackbone
       ↓
Patch tokens:     [B, 49, 768]   ← 49 = 7×7 spatial regions of the leaf
```

---

> Say **"Phase 3 done"** when tests pass and I'll unlock Phase 4.

---

---

# 🔒 PHASE 4 — Locked
*Unlocks after Phase 3 is complete.*

---

# 🔒 PHASE 5 — Locked
*Unlocks after Phase 4 is complete.*

---

# 🔒 PHASE 6 — Locked
*Unlocks after Phase 5 is complete.*

---

# 🔒 PHASE 7 — Locked
*Unlocks after Phase 6 is complete.*

---

# 🔒 PHASE 8 — Locked
*Unlocks after Phase 7 is complete.*

---

# 🔒 PHASE 9 — Locked
*Unlocks after Phase 8 is complete.*

---

# 🔒 PHASE 10 — Locked
*Unlocks after Phase 9 is complete.*

---

## 📚 Quick Reference

### Key tensor shapes so far
```
Image input:      [B, 3, 224, 224]
Swin features:    [B, 49, 768]      ← 49 = 7×7 patches  (Phase 3)
Metadata raw:     [B, 4]            ← [T, H, Sm, R]     (Phase 2)
```

### uv commands
```powershell
uv sync --extra dev                  # Install all deps
uv run pytest tests/unit/ -v         # Run all tests
uv run ruff format --check src/ tests/ --exclude .venv
uv run ruff check src/ tests/ --exclude .venv
```

### Disease Classes
```
0 = Healthy
1 = Red Rot
2 = Grassy Shoot
3 = Smut
```
