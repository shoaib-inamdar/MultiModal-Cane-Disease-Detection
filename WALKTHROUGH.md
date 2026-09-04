# 🌾 SugarcaneAI — Objective 1 Build Walkthrough
## Phase-by-Phase Teaching Guide

> **Your Role:** You build. I guide. Read each phase → try it → say **"Phase X done"** → we continue.

---

## 📋 What Objective 1 Builds

A working multimodal model that:
1. Takes a **sugarcane leaf image** + **environmental metadata** [Temperature, Humidity, Soil Moisture, Rainfall]
2. Fuses them using **Cross-Attention** (your novel contribution)
3. Outputs one of **4 disease classes**: Healthy, Red Rot, Grassy Shoot, Smut

> ⚠️ Severity grading, MC-Dropout, Explainability, and Knowledge Distillation are **NOT** part of Objective 1. They come in Objectives 2–5.

---

## 🗺️ Phase Map — Objective 1

| Phase | Name | Time Estimate | What You Build |
|-------|------|--------------|----------------|
| **1** | Setup & Workspace | ✅ Done | Folders, tools, dependencies |
| **2** | Dummy Dataset + DataLoader | 2–3 hours | Synthetic dataset class for testing without real data |
| **3** | Pretrained Visual Backbone | 2–3 hours | Swin-Tiny from `timm` as feature extractor |
| **4** | Metadata MLP Encoder | 1–2 hours | MLP that converts [T, H, Sm, R] → 256-dim token |
| **5** | Cross-Attention Fusion ⭐ | 3–4 hours | Core contribution: visual tokens attend to env tokens |
| **6** | Full Model Assembly | 1–2 hours | Connect backbone → fusion → classifier head |
| **7** | Training on Colab/Kaggle | 3–4 hours | Train loop + Google Drive checkpoints + W&B |
| **8** | Evaluation & Baselines | 2–3 hours | F1, Accuracy vs image-only baseline |
| **9** | Testing (pytest) | 2–3 hours | Unit + integration tests |
| **10** | Final Cleanup & Docs | 1–2 hours | CI green, README updated, results documented |

**Total estimated time: ~20–26 hours of focused work**

---

## 🗃️ Dataset Strategy — 3 Stages

This project uses a progressive dataset approach. You don't need real data to start.

| Stage | When | Images | Environmental Data | Purpose |
|-------|------|--------|--------------------|---------|
| **A — Dummy** | Now (Phases 2–7) | `torch.rand(3,224,224)` synthetic tensors | Randomly generated [T, H, Sm, R] | Test pipeline architecture |
| **B — Kaggle Public** | After Stage A works | Real leaf images from Kaggle | **Still fake/synthetic metadata** | Validate visual encoder |
| **C — Real Field** | Final (Objective 3) | Your own Maharashtra field photos | Real DHT22 sensor readings synced by timestamp | Final research results |

### Stage B — Kaggle Datasets to Use
When ready to move from dummy to real images:
1. **[Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/swapnildaphal/sugarcane-leaf-disease-dataset)** — Daphal & Koli (Maharashtra, ~2,569 images)
2. **[Sugarcane Leaf Dataset](https://data.mendeley.com/datasets/355y629ynj/1)** — Thite et al. (6,748 images, 9 classes)

For Stage B, set `synthetic=False` in `SugarcaneDataset` but keep `synthetic_metadata=True` — real images but fake weather values.

> ⚠️ You will NOT have real environmental metadata until Stage C (your own field collection). The model still trains on real images, just with synthetic weather values as a placeholder.

### Stage C Transition
Replace the metadata CSV path in `configs/obj1_cross_attention.yaml`:
```yaml
data:
  images_dir: "data/raw/maharashtra_field/"    # ← your real photos
  metadata_csv: "data/raw/field_metadata.csv"  # ← real DHT22 sensor readings
  synthetic: false
  synthetic_metadata: false
```

---


---

# ✅ PHASE 1 — COMPLETED

- ✅ Python 3.14, uv 0.8.14
- ✅ All folders + `__init__.py` files
- ✅ `pyproject.toml` — 88 packages installed (torch 2.13, timm 1.0.28)
- ✅ `.gitignore`, `.env.example`, `.pre-commit-config.yaml`
- ✅ `.github/workflows/ci.yml`
- ✅ `tests/conftest.py`

---

---

# ✅ PHASE 2 — Dummy Dataset + DataLoader
**Time: 2–3 hours**

---

## 🎯 Goal
Build a `SugarcaneDataset` that generates **fake data** so you can test the full pipeline without waiting for real images.

## 🤔 Why
You can't train a model without data — but you also can't wait months to collect field data before even verifying the architecture works. A dummy dataset lets you develop and test the full pipeline now. When real data arrives, you swap one line: `synthetic=False`.

## 🪜 Steps

### Step 2.1 — Create `configs/obj1_cross_attention.yaml`
YAML config that controls all settings without touching Python code.

Create the file with these sections:
```yaml
experiment:
  name: "obj1_cross_attention_v1"
  seed: 42

data:
  num_classes: 4          # Healthy, Red Rot, Grassy Shoot, Smut
  class_names: ["Healthy", "Red Rot", "Grassy Shoot", "Smut"]
  image_size: 224
  metadata_dim: 4         # [Temperature, Humidity, Soil Moisture, Rainfall]
  batch_size: 16          # Small batch for Colab T4 GPU
  num_synthetic_samples: 300

model:
  backbone: "swin_tiny_patch4_window7_224"   # timm model name
  embed_dim: 768          # Swin-Tiny output dim
  metadata_hidden_dim: 128
  metadata_out_dim: 256
  num_attention_heads: 8
  fusion_dropout: 0.1

training:
  num_epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  patience: 5             # Early stopping

colab:
  use_drive: true
  drive_checkpoint_path: "/content/drive/MyDrive/SugarcaneAI/checkpoints"

logging:
  use_wandb: false        # Set true when you have API key
  project: "sugarcane-ai-obj1"
```

### Step 2.2 — Create `src/utils/config.py`
A function `load_config(path: str) -> dict` that reads a YAML file.
Use `yaml.safe_load()`. Raise `FileNotFoundError` if path doesn't exist.

### Step 2.3 — Create `src/utils/seed.py`
A function `set_seed(seed: int = 42)` that seeds `random`, `numpy`, `torch`, and `torch.cuda`.

### Step 2.4 — Create `src/data/augmentation.py`
Two functions:
- `get_train_transforms(image_size)` — Resize, RandomHFlip, Rotate(±30°), ColorJitter, CLAHE(p=0.3), Normalize(ImageNet stats), ToTensorV2
- `get_val_transforms(image_size)` — Resize, Normalize, ToTensorV2

### Step 2.5 — Create `src/data/dataset.py` ← Main file

**Class:** `SugarcaneDataset(torch.utils.data.Dataset)`

**`__init__` args:** `config: dict`, `split: str`, `synthetic: bool = True`

**One sample returns a dict:**
```python
{
    "image":    torch.Tensor,  # [3, 224, 224] float32, values in [0,1]
    "metadata": torch.Tensor,  # [4] float32, normalized to [0,1]
    "label":    torch.Tensor,  # [] torch.long, value 0-3
}
```

**Synthetic metadata normalization:**
```
Temperature: random(15, 45) / 50
Humidity:    random(30, 100) / 100
Soil Moist:  random(10, 90) / 100
Rainfall:    random(0, 200) / 300
```

Use `synthetic=True` always for now. Add real-file loading path for later.

### Step 2.6 — Create `src/data/dataloader.py`
Function `get_dataloaders(config: dict) -> dict` returning `{"train": ..., "val": ..., "test": ...}`.

DataLoader settings:
- `batch_size` from config
- `shuffle=True` for train only
- `num_workers=0` (important for Windows + Colab compatibility)
- `pin_memory=False`

### Step 2.7 — Create `tests/unit/test_dataset.py`

Write 4 tests:
1. `test_dataset_length` — assert `len(dataset) == config["data"]["num_synthetic_samples"]`
2. `test_sample_shapes` — image `[3,224,224]`, metadata `[4]`, label is `torch.long`
3. `test_metadata_range` — metadata min ≥ 0.0, max ≤ 1.0
4. `test_dataloader_batch` — batch image shape `[batch_size, 3, 224, 224]`

## ✅ Done When
```powershell
uv run pytest tests/unit/test_dataset.py -v
# 4 passed ✅
```

---

---

# ⏳ PHASE 3 — Pretrained Visual Backbone (Swin-Tiny)
**Time: 2–3 hours**

---

## 🎯 Goal
Load **Swin-Tiny** from `timm` as a frozen/fine-tunable feature extractor. It outputs a feature vector — NOT a classification. We use it purely as a "visual understanding engine".

## 🤔 Why Pretrained — NOT From Scratch
- Your dataset will have only a few thousand images
- Swin-Tiny trained on ImageNet-21k already "knows" textures, edges, color patterns
- Training from scratch with few images → overfitting, poor generalization
- Using pretrained = 90% of the work is already done by ImageNet training

## 🪜 Steps

### Step 3.1 — Create `src/models/backbone/swin_backbone.py`

**What to write:**
- A class `SwinBackbone(nn.Module)`
- In `__init__`: load `timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)`
  - `num_classes=0` removes the classification head — gives you features instead of predictions
- In `forward(x)`: pass image through model, return the output feature vector

**Output shape:** `[B, 768]` — a 768-dim feature vector per image

> 💡 **What is `num_classes=0`?** When you load a model normally, the last layer maps features to 1000 ImageNet classes. Setting `num_classes=0` removes that last layer, so you get the raw 768-dim feature representation instead. This is exactly what we want for fusion.

### Step 3.2 — Create `tests/unit/test_swin_backbone.py`

Write 2 tests:
1. `test_output_shape` — `backbone(batch_image).shape == [B, 768]`
2. `test_gradient_flows` — `loss.backward()` runs without error (no dead layers)

## ⚠️ Common Mistakes
| Mistake | Fix |
|---------|-----|
| Forgetting `num_classes=0` | The output will be wrong shape |
| Trying to build Swin from scratch | Don't. Just `timm.create_model(...)` |
| `pretrained=True` downloading on slow internet | Run once on Colab, weights cache to `~/.cache/torch` |

## ✅ Done When
```python
backbone = SwinBackbone()
out = backbone(torch.rand(2, 3, 224, 224))
assert out.shape == torch.Size([2, 768])
```

---

---

# ⏳ PHASE 4 — Metadata MLP Encoder
**Time: 1–2 hours**

---

## 🎯 Goal
Build a small MLP that converts the 4 environmental values into a 256-dim vector that can be used in Cross-Attention.

## 🪜 Steps

### Step 4.1 — Create `src/models/encoders/metadata_encoder.py`

**Class:** `MetadataEncoder(nn.Module)`

**Architecture:**
```
Input: [B, 4]
  → Linear(4, 64) → ReLU → Dropout(0.1)
  → Linear(64, 128) → ReLU → Dropout(0.1)
  → Linear(128, 256)
Output: [B, 1, 256]   ← unsqueeze(1) to add sequence dimension for attention
```

Why `[B, 1, 256]`? Cross-Attention expects sequence format. The metadata becomes a single "token" — one environmental context vector that all 196 image patches can attend to.

### Step 4.2 — Create `tests/unit/test_metadata_encoder.py`

2 tests:
1. `test_output_shape` — output shape `[B, 1, 256]`
2. `test_gradient_flows` — backward pass succeeds

## ✅ Done When
```python
encoder = MetadataEncoder()
out = encoder(torch.rand(4, 4))
assert out.shape == torch.Size([4, 1, 256])
```

---

---

# ⏳ PHASE 5 — Cross-Attention Fusion ⭐
**Time: 3–4 hours — This is the core of your research**

---

## 🎯 Goal
Implement the Cross-Attention layer that makes visual features "aware" of environmental context.

## 🤔 The Big Idea

The Swin backbone gives you 768-dim visual features. But the metadata encoder gives 256-dim vectors. They need to be in the **same dimension** to do attention. So first project visual features to 256-dim, then do Cross-Attention.

```
After projection:
  Visual features (Q): [B, 196, 256]  ← 196 image patches, each 256-dim
  Metadata tokens (K): [B, 1, 256]    ← 1 environmental context token, 256-dim
  Metadata tokens (V): [B, 1, 256]

Cross-Attention:
  Attn(Q, K, V) = softmax(QKᵀ / √256) · V
  Output: [B, 196, 256]   ← each patch is now "aware" of the environment
```

**In plain English:**  
Each of the 196 image patches asks the weather data: *"Given what you know about temperature and humidity, how should I adjust my disease prediction?"*

## 🪜 Steps

### Step 5.1 — Create `src/models/fusion/cross_attention.py`

**Class:** `CrossAttentionFusion(nn.Module)`

**`__init__` args:** `visual_dim=768`, `metadata_dim=256`, `num_heads=8`, `dropout=0.1`

**What to write:**
1. A projection layer: `nn.Linear(visual_dim, metadata_dim)` — maps 768 → 256 so dimensions match
2. `nn.MultiheadAttention(embed_dim=metadata_dim, num_heads=num_heads, batch_first=True)`
3. Layer norm: `nn.LayerNorm(metadata_dim)`
4. A feed-forward: `nn.Sequential(Linear(256, 512), ReLU, Dropout, Linear(512, 256))`

**`forward(visual_features, metadata_tokens)` steps:**
1. Project: `visual_proj = self.proj(visual_features)` → `[B, 196, 256]`

> 🤔 **Wait — visual_features from Swin is `[B, 768]`, not `[B, 196, 256]`!**  
> You're right. Swin-Tiny with `num_classes=0` returns a **globally pooled** `[B, 768]` vector, not patch tokens.  
> Fix: use `global_pool=''` when loading timm model to get patch tokens instead.  
> Add this to `SwinBackbone`: `timm.create_model(..., num_classes=0, global_pool='')` → output: `[B, 49, 768]`  
> Then go back and update Phase 3 accordingly. (49 = 7×7 patches for Swin-Tiny)

2. Unsqueeze projected features to sequence: `[B, 49, 256]`
3. Cross-attention: `attn_out, _ = self.attention(query=visual_proj, key=metadata_tokens, value=metadata_tokens)`
4. Residual + LayerNorm: `out = self.norm(visual_proj + attn_out)`
5. Feed-forward + residual: `out = out + self.ffn(out)`
6. Pool: `fused = out.mean(dim=1)` → `[B, 256]`
7. Return fused: `[B, 256]`

### Step 5.2 — Create `tests/unit/test_cross_attention.py`

3 tests:
1. `test_output_shape` — output `[B, 256]`
2. `test_gradient_flows` — backward succeeds, no NaN
3. `test_attention_weights_valid` — attention weights sum ≈ 1.0

## ✅ Done When
```python
fusion = CrossAttentionFusion()
visual = torch.rand(2, 49, 768)
meta   = torch.rand(2, 1, 256)
out    = fusion(visual, meta)
assert out.shape == torch.Size([2, 256])
```

---

---

# ⏳ PHASE 6 — Full Model Assembly
**Time: 1–2 hours**

---

## 🎯 Goal
Connect all 3 modules into one `SugarcaneClassifier` model.

## 🪜 Steps

### Step 6.1 — Update `SwinBackbone` (from Phase 3)
Change `global_pool=''` as noted in Phase 5. Output is now `[B, 49, 768]`.

### Step 6.2 — Create `src/models/multimodal_model.py`

**Class:** `SugarcaneClassifier(nn.Module)`

**`__init__` takes:** `config: dict`

**What it contains:**
- `self.backbone = SwinBackbone()` — visual features
- `self.meta_encoder = MetadataEncoder()` — env tokens
- `self.fusion = CrossAttentionFusion()` — fuse
- `self.classifier = nn.Linear(256, num_classes)` — 256 → 4 classes

**`forward(image, metadata)`:**
1. `visual = self.backbone(image)` → `[B, 49, 768]`
2. `meta = self.meta_encoder(metadata)` → `[B, 1, 256]`
3. `fused = self.fusion(visual, meta)` → `[B, 256]`
4. `logits = self.classifier(fused)` → `[B, 4]`
5. Return `logits`

### Step 6.3 — Create `tests/unit/test_full_model.py`

2 tests:
1. `test_forward_pass` — `model(image, metadata).shape == [B, 4]`
2. `test_end_to_end_backward` — full forward + `loss.backward()` completes

## ✅ Done When
```python
model = SugarcaneClassifier(config)
img   = torch.rand(2, 3, 224, 224)
meta  = torch.rand(2, 4)
out   = model(img, meta)
assert out.shape == torch.Size([2, 4])
```

---

---

# ⏳ PHASE 7 — Training on Google Colab / Kaggle
**Time: 3–4 hours**

---

## 🎯 Goal
Train the full model end-to-end. Since your laptop has no strong GPU, all training runs on **Google Colab (T4 GPU)** or **Kaggle Notebooks (P100 GPU)**.

## 🪜 Steps

### Step 7.1 — Create `src/training/trainer.py`

**Function:** `train_one_epoch(model, dataloader, optimizer, criterion, device) -> float`
- Loops over batches
- `optimizer.zero_grad()` → forward → loss → `loss.backward()` → `optimizer.step()`
- Returns average epoch loss

**Function:** `validate(model, dataloader, criterion, device) -> tuple[float, float]`
- No `torch.no_grad()` loop for validation
- Returns (val_loss, val_accuracy)

### Step 7.2 — Create `src/training/callbacks.py`

**Class:** `EarlyStopping`
- `__init__(patience=5)`
- `__call__(val_loss) -> bool` — returns True if should stop
- Saves best val_loss, counts patience

**Function:** `save_checkpoint(model, optimizer, epoch, path)`
- Saves `{"model_state": ..., "optimizer_state": ..., "epoch": epoch}` with `torch.save`

### Step 7.3 — Create `scripts/train.py`

The main training script. It should:
1. Parse `--config` argument
2. Load config + set seed
3. Get dataloaders
4. Build model
5. Set optimizer (AdamW), scheduler (CosineAnnealingLR), criterion (CrossEntropyLoss)
6. Training loop with early stopping + checkpoint saving
7. Log to W&B if `config["logging"]["use_wandb"]`

### Step 7.4 — Create `notebooks/train_colab.ipynb`

This is a Jupyter notebook for running on Google Colab. It must have:

```python
# Cell 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone repo (if not already)
!git clone https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection.git
%cd MultiModal-Cane-Disease-Detection

# Cell 3 — Install dependencies
!pip install uv
!uv sync

# Cell 4 — Train
!uv run python scripts/train.py --config configs/obj1_cross_attention.yaml

# Cell 5 — Copy checkpoint to Drive
import shutil
shutil.copy("checkpoints/best_model.pth",
            "/content/drive/MyDrive/SugarcaneAI/checkpoints/best_model.pth")
```

> 💡 **Kaggle alternative:** Same steps but use Kaggle's "Add Data" to mount your repo, and save to `/kaggle/working/` then download manually.

### Step 7.5 — Update `configs/obj1_cross_attention.yaml`

Already has `colab.drive_checkpoint_path`. Make sure `training.batch_size: 16` — Colab T4 has 16GB VRAM, batch 16 is safe.

## ⚠️ Common Mistakes
| Mistake | Fix |
|---------|-----|
| `batch_size: 32` crashes Colab | Keep at 16 for Swin-Tiny |
| Forgetting to save checkpoint | Set checkpoint path in config, call `save_checkpoint` after each epoch if val_loss improved |
| W&B login fails on Colab | Run `!wandb login YOUR_API_KEY` in notebook cell |

## ✅ Done When
- Training runs for 5 epochs without crashing
- Loss decreases (even slightly — it's dummy data, so don't expect high accuracy)
- Checkpoint saved to Drive at `SugarcaneAI/checkpoints/best_model.pth`

---

---

# ⏳ PHASE 8 — Evaluation & Baselines
**Time: 2–3 hours**

---

## 🎯 Goal
Measure model performance and compare against a simple image-only baseline to prove Cross-Attention adds value.

## 🪜 Steps

### Step 8.1 — Create `src/evaluation/metrics.py`

Functions to write:
- `compute_accuracy(preds, labels) -> float`
- `compute_macro_f1(preds, labels, num_classes) -> float` — use `sklearn.metrics.f1_score`
- `compute_per_class_f1(preds, labels, class_names) -> dict`

### Step 8.2 — Create `src/evaluation/confusion_matrix.py`

Function `plot_confusion_matrix(preds, labels, class_names, save_path)`:
- Use `sklearn.metrics.confusion_matrix` + `seaborn.heatmap`
- Save to `experiments/obj1/plots/confusion_matrix.png`

### Step 8.3 — Create `src/models/baselines/image_only.py`

**Class:** `ImageOnlyClassifier(nn.Module)`
- Same Swin-Tiny backbone
- Global pool → `Linear(768, 4)` directly (no metadata, no fusion)
- This is the baseline we beat

### Step 8.4 — Create `scripts/evaluate.py`

Load a checkpoint → run on val/test set → print accuracy + macro F1 + confusion matrix.

### Step 8.5 — Create `scripts/run_baselines.py`

Train and evaluate `ImageOnlyClassifier` using the same config + dummy dataset.

## ✅ Done When
Results table populated (even on dummy data — just to verify the pipeline works):

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Image Only (Swin-Tiny) | ~25% (random) | ~0.25 |
| **Cross-Attention Ours** | ~25% (random) | ~0.25 |

> On dummy data both will be ~25% (4 classes, random). That's expected! Real improvement comes with real data. The point is the **pipeline works end-to-end**.

---

---

# ⏳ PHASE 9 — Testing (pytest)
**Time: 2–3 hours**

---

## 🎯 Goal
Write comprehensive unit + integration tests so CI stays green.

## 🪜 Steps

### Step 9.1 — Fill `tests/conftest.py`

Add shared fixtures:
```python
import pytest, torch
from src.utils.config import load_config

@pytest.fixture
def config():
    return load_config("configs/obj1_cross_attention.yaml")

@pytest.fixture
def fake_image_batch():
    return torch.rand(2, 3, 224, 224)

@pytest.fixture
def fake_meta_batch():
    return torch.rand(2, 4)
```

### Step 9.2 — Complete all unit tests
By this phase you should already have:
- `test_dataset.py` ← Phase 2
- `test_swin_backbone.py` ← Phase 3
- `test_metadata_encoder.py` ← Phase 4
- `test_cross_attention.py` ← Phase 5
- `test_full_model.py` ← Phase 6

Add:
- `tests/unit/test_metrics.py` — F1 with known inputs
- `tests/unit/test_callbacks.py` — EarlyStopping logic

### Step 9.3 — Create `tests/integration/test_full_pipeline.py`

One test that does everything:
```
load_config → set_seed → get_dataloaders → build_model → one_forward_pass → compute_loss → backward
```
If this passes, the full pipeline is validated.

## ✅ Done When
```powershell
uv run pytest tests/ -v --cov=src --cov-report=term-missing
# All tests PASS
# Coverage ≥ 70%
```

---

---

# ⏳ PHASE 10 — Final Cleanup & Documentation
**Time: 1–2 hours**

---

## 🎯 Goal
Make the repo professional, reproducible, and ready for supervisors/reviewers to inspect.

## 🪜 Steps

### Step 10.1 — Verify CI is green
```powershell
# Push to GitHub — CI must pass
git add .
git commit -m "feat: complete objective 1 pipeline"
git push origin main
```
Check Actions tab — all 3 jobs (ruff, mypy, pytest) must be ✅.

### Step 10.2 — Update `experiments/obj1/`
Create `experiments/obj1/results/obj1_results.json`:
```json
{
  "model": "SugarcaneClassifier (Cross-Attention)",
  "dataset": "Dummy Multimodal (300 samples)",
  "backbone": "swin_tiny_patch4_window7_224 (pretrained)",
  "classes": ["Healthy", "Red Rot", "Grassy Shoot", "Smut"],
  "train_accuracy": null,
  "val_accuracy": null,
  "macro_f1": null,
  "note": "Placeholder — to be filled with real dataset results"
}
```

### Step 10.3 — Write `notebooks/01_data_exploration.ipynb`
A simple notebook that:
- Loads config
- Creates dummy dataset
- Visualizes 8 sample images (random tensors with class labels)
- Shows metadata distribution plots

### Step 10.4 — Verify pre-commit locally
```powershell
uv run pre-commit run --all-files
```
Fix any issues before final push.

## ✅ Objective 1 is DONE When

- [ ] All 10 phases complete
- [ ] CI badge green on GitHub
- [ ] `uv run pytest tests/` → all pass, ≥70% coverage
- [ ] Model forward pass: `SugarcaneClassifier(image, metadata) → [B, 4]` ✅
- [ ] Checkpoint saved to Google Drive
- [ ] `experiments/obj1/results/obj1_results.json` exists
- [ ] Dummy dataset pipeline proven end-to-end
- [ ] `WALKTHROUGH.md` Phase 10 checkbox filled

**Next step after Objective 1:** Begin Objective 3 (real Maharashtra dataset collection) in parallel with Objective 2 (MC-Dropout uncertainty).

---

## 📚 Quick Reference

### Running on Google Colab
```python
# Always start Colab session with:
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection.git
%cd MultiModal-Cane-Disease-Detection
!pip install uv -q && uv sync -q
```

### Key tensor shapes to memorize
```
Image input:          [B, 3, 224, 224]
Swin features:        [B, 49, 768]      ← 49 = 7×7 patches
Projected visual:     [B, 49, 256]
Metadata raw:         [B, 4]
Metadata encoded:     [B, 1, 256]
Fused output:         [B, 256]
Logits:               [B, 4]            ← 4 disease classes
```

### Disease Classes
```
0 = Healthy
1 = Red Rot
2 = Grassy Shoot
3 = Smut
```

### uv commands
```powershell
uv sync --extra dev          # Install all deps
uv run pytest tests/ -v      # Run all tests
uv run python scripts/train.py --config configs/obj1_cross_attention.yaml
uv run pre-commit run --all-files
```
