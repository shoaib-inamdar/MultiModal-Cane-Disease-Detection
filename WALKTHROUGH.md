# 🌾 SugarcaneAI — Complete Build Walkthrough
## Phase-by-Phase Teaching Guide for Beginners

> **Your Role:** You are the builder. I am your guide. I explain; you code.  
> **Rule:** Read each phase carefully → try it → come back and say **"Phase X done"** → we continue.

---

## 📋 Project Overview (What We're Building)

You are building a **Ph.D.-level AI research system** that:

1. Takes a **photo of a sugarcane leaf** + **weather data** (temperature, humidity, etc.)
2. Uses an AI model to ask: *"Given this weather context, what disease is this leaf showing?"*
3. Outputs: **disease name + confidence + severity grade (0-4) + uncertainty score**

This is real, novel research. You'll learn PyTorch, transformers, and production software engineering along the way. 🚀

---

## 🗺️ Phase Map (All Phases)

| Phase | Name | What You Build |
|-------|------|---------------|
| **1** | Setup & Workspace | Folder structure, tools, API keys |
| **2** | Data Pipeline (Synthetic) | Dataset class, dataloader, augmentation (no real data needed yet!) |
| **3** | Image Encoder | CNN Backbone + Swin-Transformer |
| **4** | Metadata Encoder | MLP for weather data |
| **5** | Cross-Attention Fusion ⭐ | The core research contribution |
| **6** | Training Loop | Train the model, log to W&B |
| **7** | Evaluation | F1, ECE, Confusion Matrix |
| **8** | Testing (pytest) | Unit + Integration tests |
| **9** | CI/CD | GitHub Actions, pre-commit hooks |
| **10** | Explainability | Grad-CAM++ heatmaps |

---

---

# 🟢 PHASE 1 — Setup & Workspace

---

## 🎯 Goal

Set up your entire development environment so that when we write code in Phase 2, everything "just works". A good setup prevents 90% of beginner frustrations.

---

## 🤔 Why This Matters

Imagine building a house without a blueprint or tools. You'd be lost. Phase 1 is your blueprint + toolbox. Every professional project starts here. Top open-source repositories like HuggingFace and PyTorch start with clean setups — and so do we.

---

## 🪜 Steps

### Step 1.1 — Verify Python Version

Open your terminal (PowerShell on Windows) and check:

```powershell
python --version
```

You need **Python 3.10 or higher**. If you see 3.8 or 3.9, you'll need to install 3.10+.

> ⚠️ **Common mistake:** Your workspace's `pyproject.toml` currently says `requires-python = ">=3.14"`. This is a typo — Python 3.14 doesn't exist yet! You'll need to fix this to `">=3.10"`.

**Fix it:** Open `pyproject.toml` and change line 6 to:
```toml
requires-python = ">=3.10"
```

---

### Step 1.2 — Verify uv is Installed

You said you're using `uv`. Check it's working:

```powershell
uv --version
```

If you don't see a version number, install it:

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal.

> 💡 **What is uv?** `uv` is like a supercharged version of `pip` — it's written in Rust so it's 10-100x faster. It also manages your virtual environment automatically. Think of it as `pip + virtualenv + pip-tools` combined.

---

### Step 1.3 — Create the Project Folder Structure

You need to create all the folders before writing any code. Here's the structure you need for Phase 1 and 2:

```
MultiModal-Cane-Disease-Detection/   ← (already exists)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing/
│   │       └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone/
│   │   │   └── __init__.py
│   │   ├── encoders/
│   │   │   └── __init__.py
│   │   ├── fusion/
│   │   │   └── __init__.py
│   │   └── heads/
│   │       └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── configs/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── docs/
│   └── objectives/
├── experiments/
│   └── obj1/
├── notebooks/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── .github/
│   └── workflows/
└── checkpoints/
```

**How to create folders in PowerShell (Windows):**

```powershell
# Navigate to your project
cd "C:\Users\HP\Documents\MultiModal-Cane-Disease-Detection"

# Create all directories at once
New-Item -ItemType Directory -Force -Path @(
    "src\data\preprocessing",
    "src\models\backbone",
    "src\models\encoders",
    "src\models\fusion",
    "src\models\heads",
    "src\models\baselines",
    "src\training",
    "src\evaluation",
    "src\explainability",
    "src\utils",
    "configs\ablation",
    "configs\baselines",
    "tests\unit",
    "tests\integration",
    "scripts",
    "docs\objectives",
    "experiments\obj1\results",
    "experiments\obj1\plots",
    "notebooks",
    "data\raw",
    "data\processed",
    "data\splits",
    "checkpoints",
    ".github\workflows"
)
```

---

### Step 1.4 — Add Python Dependencies to pyproject.toml

Open `pyproject.toml` and add the core dependencies. I'll tell you what each one does:

```toml
[project]
name = "multimodal-cane-disease-detection"
version = "0.1.0"
description = "Novel Lightweight Explainable Uncertainty-Aware Multimodal Hybrid ViT for Sugarcane Disease Detection"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

dependencies = [
    # Deep Learning Core
    "torch>=2.1.0",           # PyTorch — the brain of the system
    "torchvision>=0.16.0",    # PyTorch image utilities
    "timm>=0.9.0",            # Pretrained model zoo (EfficientNet, Swin)

    # Image Processing
    "Pillow>=10.0.0",         # Image loading
    "opencv-python>=4.8.0",   # CLAHE enhancement, image ops
    "albumentations>=1.3.0",  # Data augmentation

    # Data & Science
    "numpy>=1.24.0",          # Numerical computing
    "pandas>=2.0.0",          # Metadata CSV handling
    "scikit-learn>=1.3.0",    # Metrics, KNN imputation

    # Configuration
    "pyyaml>=6.0",            # YAML config files
    "python-dotenv>=1.0.0",   # Load .env API keys

    # Experiment Tracking
    "wandb>=0.16.0",          # Weights & Biases logging

    # Utilities
    "tqdm>=4.66.0",           # Progress bars
    "rich>=13.0.0",           # Beautiful terminal output
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",

    # Code Quality
    "ruff>=0.1.0",            # Linter + formatter (replaces black + flake8)
    "mypy>=1.5.0",            # Type checking
    "pre-commit>=3.5.0",      # Git hook manager

    # Image Quality (for preprocessing)
    "imagehash>=4.3.0",       # pHash deduplication
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]  # Error, Flake8, Isort, Naming, Warning

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

> 💡 **Why separate `dev` dependencies?** Core dependencies are needed to RUN the project. Dev dependencies are only needed to DEVELOP it (testing, linting). This keeps the production install lean.

---

### Step 1.5 — Install Dependencies with uv

```powershell
# Install core + dev dependencies
uv sync --extra dev
```

This will:
1. Read your `pyproject.toml`
2. Create a `.venv` virtual environment automatically
3. Install all packages
4. Generate a `uv.lock` file (pinned versions for reproducibility)

> ⚠️ **PyTorch note:** The `torch` package is large (~2GB). Be patient on the first install. Make sure you have a good internet connection.

---

### Step 1.6 — Install Arduino IDE

Since your project involves sensors, you'll need Arduino IDE for the hardware side:

1. Go to: https://www.arduino.cc/en/software
2. Download **Arduino IDE 2.x** (the newer one)
3. Install it on Windows
4. Open Arduino IDE and install the following libraries (Tools → Manage Libraries):
   - `DHT sensor library` by Adafruit (for DHT22)
   - `Adafruit Unified Sensor` (dependency)

> **Wait — does this project need Arduino?**  
> Looking at your Synopsis, the actual PhD research uses sensors to COLLECT data (DHT22 for temperature/humidity). But the ML model runs in Python. The Arduino just sends sensor readings to a CSV or database. We'll set up the Python side now and tackle the Arduino hardware connection later in a separate data collection phase.

---

### Step 1.7 — Set Up API Keys

Create a `.env.example` file (this goes in your repo — safe to commit):

```bash
# .env.example — Copy this to .env and fill in your keys
# NEVER commit .env to Git

# Weights & Biases (free at wandb.ai)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=sugarcane-ai-obj1
WANDB_ENTITY=your_username_here

# OpenWeatherMap (free tier at openweathermap.org)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Optional: IBM Weather Company (premium)
IBM_WEATHER_API_KEY=your_ibm_key_here
```

Then create your actual `.env` file:

```powershell
# Copy the template
Copy-Item .env.example .env
```

Edit `.env` and fill in real keys.

**Getting API keys (all free):**

| Service | URL | What it's for |
|---------|-----|---------------|
| Weights & Biases | https://wandb.ai/authorize | Experiment tracking (sign up, get API key from settings) |
| OpenWeatherMap | https://openweathermap.org/api | Historical weather data for metadata sync |

---

### Step 1.8 — Set Up Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ["--maxkb=500"]         # Block accidental large file commits

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff                       # Linting
        args: [--fix]
      - id: ruff-format                # Formatting

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

Install the hooks:

```powershell
uv run pre-commit install
```

From now on, every time you `git commit`, these checks run automatically. They will catch mistakes before they enter the repository.

---

### Step 1.9 — Create .gitignore

Create `.gitignore` to prevent accidentally committing sensitive or large files:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.Python
*.egg-info/
dist/
build/

# Virtual environment
.venv/
venv/
env/

# Environment variables (NEVER commit this!)
.env

# Large files (managed by DVC)
data/raw/
data/processed/
data/splits/
checkpoints/
*.pth
*.pt
*.onnx

# Experiment outputs
wandb/
experiments/*/results/*.json
experiments/*/plots/*.png

# Jupyter
.ipynb_checkpoints/
*.ipynb_meta

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/settings.json
.idea/
*.swp
```

---

### Step 1.10 — Create conftest.py for Tests

Create `tests/conftest.py`:

```python
# tests/conftest.py
# This file contains shared fixtures for all your tests.
# You will fill in the actual fixtures in Phase 8.
# For now, just create an empty file.
```

And `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` — all empty files.

---

## 📁 Files to Create in Phase 1

| File | Action |
|------|--------|
| `pyproject.toml` | Update with full dependencies |
| `.env.example` | Create (commit this) |
| `.env` | Create (NEVER commit) |
| `.gitignore` | Create |
| `.pre-commit-config.yaml` | Create |
| `tests/conftest.py` | Create (empty for now) |
| All `__init__.py` files | Create (empty) |
| All folders | Create via PowerShell |

---

## ⚠️ Common Mistakes in Phase 1

| Mistake | How to Avoid |
|---------|-------------|
| Committing `.env` with real API keys | Always check `.gitignore` has `.env` before `git add .` |
| Using Python 3.8 or 3.9 | Run `python --version` — need 3.10+ |
| Forgetting `uv.lock` | Run `uv sync` to generate it; commit `uv.lock` |
| Creating folders but forgetting `__init__.py` | Every Python package folder needs `__init__.py` |
| Installing PyTorch without CUDA when GPU is available | Check if you have an NVIDIA GPU: run `nvidia-smi` |
| Typo in `pyproject.toml` breaks uv | Run `uv sync` and fix any errors shown |

---

## ✅ How to Know Phase 1 is Done

Run through this checklist:

```powershell
# Check Python version
python --version                          # Should say 3.10+

# Check uv works
uv --version                              # Should show a version

# Check dependencies installed
uv run python -c "import torch; print(torch.__version__)"    # Should print 2.x.x
uv run python -c "import timm; print(timm.__version__)"      # Should print version
uv run python -c "import wandb; print(wandb.__version__)"    # Should print version

# Check pre-commit installed
uv run pre-commit --version               # Should show version

# Check folder structure exists
ls src/models/fusion/                     # Should show __init__.py
ls tests/unit/                            # Should exist
```

If all commands run without errors — **Phase 1 is done! 🎉**

---

## 🎉 Phase 1 Celebration

You've just done what every professional engineer does before writing a single line of real code:
- ✅ Set up a reproducible environment
- ✅ Configured code quality tools  
- ✅ Protected your secrets
- ✅ Created a clean folder structure

This is how HuggingFace, PyTorch, and every top open-source project starts.

---

# 🛑 STOP HERE

👉 **Come back and say "Phase 1 done"** when you've completed all steps above.

👉 When Phase 1 is done, we'll start **Phase 2: Data Pipeline** — where you build the PyTorch Dataset class that works WITHOUT any real dataset (using synthetic tensors for testing).

---

---

# ✅ PHASE 1 — COMPLETED

All of the following are now done:
- ✅ Python 3.14, uv 0.8.14 verified
- ✅ All folders created
- ✅ All `__init__.py` files created
- ✅ `pyproject.toml` fixed and updated (88 packages installed)
- ✅ `.env.example` + `.env` created
- ✅ `.gitignore` created
- ✅ `.pre-commit-config.yaml` created
- ✅ `tests/conftest.py` created
- ✅ `uv sync --extra dev` — torch 2.13, timm 1.0.28, cv2 5.0 all working

---

---

# 🟡 PHASE 2 — Data Pipeline (Synthetic First!)

---

## 🎯 Goal

Build a PyTorch `Dataset` class and `DataLoader` that:
1. **Works right now** with fake (synthetic) tensors — no real images needed yet
2. **Will work identically** when real leaf images + sensor CSVs arrive later
3. Includes an **augmentation pipeline** for training (random flips, CLAHE, etc.)

---

## 🤔 Why This Matters

Think of a Dataset class like a **filing cabinet**. Your model says: *"Give me sample #42."* The Dataset class opens the cabinet, pulls out the image file and its metadata row, preprocesses them, and hands them to the model in the right format.

Building this with synthetic data first means:
- You can test the entire model pipeline **before** collecting a single real photo
- If a shape is wrong, you find out instantly — not after months of data collection
- This is standard practice at places like Google DeepMind and HuggingFace

---

## 🪜 Steps

### Step 2.1 — Understand What a PyTorch Dataset Is

Before writing code, understand the **contract** every PyTorch Dataset must fulfil:

```python
# Every Dataset must have exactly these 3 things:
class MyDataset:
    def __init__(self):
        # Called once — set up the data paths, transforms, etc.
        pass

    def __len__(self):
        # Called when Python asks: "how big is this dataset?"
        # Must return an integer (the total number of samples)
        return 100

    def __getitem__(self, index):
        # Called when Python asks: "give me sample number `index`"
        # Must return one sample — for us that's (image, metadata, label)
        pass
```

PyTorch's `DataLoader` calls `__len__` and `__getitem__` automatically.  
You never call them yourself — the DataLoader handles it.

---

### Step 2.2 — Understand What One "Sample" Looks Like

For your sugarcane model, one sample = three things:

```
Sample #42
├── image:    a tensor of shape [3, 224, 224]   (RGB image, 3 colour channels, 224×224 pixels)
├── metadata: a tensor of shape [4]             ([temperature, humidity, soil_moisture, rainfall])
└── label:    an integer 0–4                    (0=healthy, 1=red_rot, 2=grassy_shoot, 3=smut, 4=pokkah_boeng)
```

This is your design. Every file you write from now on must produce or consume samples in this exact format.

> 💡 **Why shape `[3, 224, 224]`?**  
> PyTorch uses **CHW format** — Channels, Height, Width. So `[3, 224, 224]` means 3 colour channels (RGB), each 224 pixels tall and 224 pixels wide. EfficientNet and Swin-Transformer both expect exactly this size.

---

### Step 2.3 — Create `src/utils/config.py`

Before building the Dataset, you need a config loader. This reads your YAML config files and makes them available as Python dictionaries.

**What to write:**
- A function `load_config(path: str) -> dict` that reads a YAML file and returns its contents as a dictionary
- Use `yaml.safe_load()` — NOT `yaml.load()` (the safe version prevents code injection)
- Handle the case where the file doesn't exist (raise a clear `FileNotFoundError`)

**Pseudocode (you write the real code):**
```
function load_config(path):
    if path does not exist:
        raise FileNotFoundError with a helpful message
    open the file
    return yaml.safe_load(file)
```

**File to create:** `src/utils/config.py`

---

### Step 2.4 — Create `configs/obj1_cross_attention.yaml`

This is your experiment configuration. YAML is like a settings file — it controls your model without you having to change the Python code.

**Create the file** `configs/obj1_cross_attention.yaml` with these sections:

```yaml
# configs/obj1_cross_attention.yaml
# Experiment configuration for Objective 1: Cross-Attention Fusion

experiment:
  name: "obj1_cross_attention_v1"
  seed: 42                          # Fixed seed for reproducibility

data:
  num_classes: 5                    # healthy, red_rot, grassy_shoot, smut, pokkah_boeng
  image_size: 224                   # Input image size (pixels)
  metadata_dim: 4                   # [temperature, humidity, soil_moisture, rainfall]
  train_split: 0.7                  # 70% of data for training
  val_split: 0.15                   # 15% for validation
  test_split: 0.15                  # 15% for testing
  # When real dataset exists, change these paths:
  images_dir: "data/raw/images"
  metadata_csv: "data/raw/metadata.csv"

model:
  embed_dim: 128                    # Embedding dimension for all tokens
  num_attention_heads: 4            # Number of heads in cross-attention
  num_fusion_layers: 2              # How many cross-attention layers to stack
  dropout_rate: 0.1                 # For MC-Dropout (Objective 2)

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  patience: 10                      # Early stopping patience (epochs)

logging:
  use_wandb: false                  # Set to true when you have your W&B API key
  project: "sugarcane-ai-obj1"
  log_every_n_steps: 10
```

> 💡 **Why a YAML config?** Without YAML configs, you'd have to change Python code to try different settings. With YAML configs, you just change the YAML file and re-run. This is how research teams track experiments — each experiment gets its own config file.

---

### Step 2.5 — Create `src/data/dataset.py`

This is the main file of Phase 2. You are building the `SugarcaneDataset` class.

**Here is the full specification — you write the code:**

**Class name:** `SugarcaneDataset`  
**Inherits from:** `torch.utils.data.Dataset`

**`__init__` method takes:**
- `config: dict` — the loaded YAML config
- `split: str` — one of `"train"`, `"val"`, or `"test"`
- `synthetic: bool = True` — when True, generate fake data instead of loading real files

**What `__init__` must do:**
1. Store `config`, `split`, `synthetic` as attributes
2. If `synthetic=True`: generate `N=200` fake samples using random tensors (explained below)
3. If `synthetic=False`: load a CSV metadata file, build a list of `(image_path, metadata_row, label)` tuples

**`__len__` returns:** total number of samples

**`__getitem__(index)` must return a Python dictionary:**
```python
{
    "image": torch.Tensor,      # shape [3, 224, 224], float32, values in [0, 1]
    "metadata": torch.Tensor,   # shape [4], float32, values in [0, 1]
    "label": torch.Tensor,      # shape [], dtype=torch.long (a single integer)
}
```

**How to make synthetic data:**
```python
# For a fake image:
image = torch.rand(3, 224, 224)   # Random pixels in [0, 1]

# For fake metadata: [temperature, humidity, soil_moisture, rainfall]
# Min-Max normalize to [0, 1]:
# temperature: real range [0°C, 50°C] → divide by 50
# humidity: real range [0%, 100%] → divide by 100
# soil_moisture: real range [0%, 100%] → divide by 100
# rainfall: real range [0mm, 300mm] → divide by 300
raw_meta = torch.tensor([
    random float between 10 and 45,   # temperature
    random float between 30 and 100,  # humidity
    random float between 10 and 90,   # soil_moisture
    random float between 0 and 200,   # rainfall
])
# normalize each value by its max

# For a fake label:
label = torch.randint(0, num_classes, (1,)).squeeze()
```

**Apply transforms only during training:**
- If `split == "train"`: apply the augmentation pipeline (Step 2.6)
- If `split == "val"` or `"test"`: only resize and normalize

**File to create:** `src/data/dataset.py`

---

### Step 2.6 — Create `src/data/augmentation.py`

Augmentation randomly transforms training images so the model learns to be robust to lighting, orientation, and camera angle differences.

**What to write:**
- A function `get_train_transforms(image_size: int)` → returns an Albumentations `Compose` pipeline
- A function `get_val_transforms(image_size: int)` → returns a simpler pipeline (just resize + normalize)

**Training augmentation pipeline (in this order):**
1. `A.Resize(image_size, image_size)` — resize to 224×224
2. `A.RandomHorizontalFlip(p=0.5)` — 50% chance of flipping left-right
3. `A.RandomVerticalFlip(p=0.2)` — 20% chance of flipping up-down
4. `A.Rotate(limit=45, p=0.5)` — random rotation up to ±45 degrees
5. `A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)` — random colour changes
6. `A.CLAHE(clip_limit=2.0, p=0.3)` — CLAHE enhancement (30% of the time — helps Grade 1 lesions)
7. `A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet normalisation
8. `ToTensorV2()` — convert numpy array to PyTorch tensor

**Validation pipeline:**
1. `A.Resize(image_size, image_size)`
2. `A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
3. `ToTensorV2()`

> 💡 **Why ImageNet mean/std?** EfficientNet and Swin-Transformer were pretrained on ImageNet. Using the same normalization as ImageNet training makes transfer learning more effective. It's the standard values used by almost all vision models.

> ⚠️ **Important import:** Albumentations needs `from albumentations.pytorch import ToTensorV2`. Don't forget this.

**File to create:** `src/data/augmentation.py`

---

### Step 2.7 — Create `src/data/dataloader.py`

The DataLoader wraps your Dataset and adds:
- **Batching:** groups samples into batches of `batch_size` (e.g., 32 at a time)
- **Shuffling:** randomizes order each epoch (for training only)
- **Parallel loading:** uses multiple CPU workers to load data while GPU trains

**What to write:**
- A function `get_dataloaders(config: dict)` that:
  1. Creates `SugarcaneDataset` for train, val, test splits
  2. Wraps each in `torch.utils.data.DataLoader`
  3. Returns a dict: `{"train": ..., "val": ..., "test": ...}`

**DataLoader settings:**
- `batch_size`: from config
- `shuffle=True` for train, `shuffle=False` for val and test
- `num_workers=2` (use 2 CPU cores for loading) — on Windows, add `pin_memory=False`
- `drop_last=True` for train (drops last incomplete batch — avoids batch norm issues)

**File to create:** `src/data/dataloader.py`

---

### Step 2.8 — Create `src/utils/seed.py`

Reproducibility is critical in research. The same code must give the same results every time.

**What to write:**
- A function `set_seed(seed: int = 42)` that sets the random seed for:
  - `random` (Python standard library)
  - `numpy`
  - `torch`
  - `torch.cuda` (if GPU available)

```python
# Hint: these are the 4 lines you need to call
import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

**File to create:** `src/utils/seed.py`

---

### Step 2.9 — Write a Smoke Test

A **smoke test** is a quick sanity check — it "runs smoke" through the system to see if anything catches fire.

**Create** `tests/unit/test_dataset.py` with these test functions:

**Test 1 — `test_dataset_length`:**  
Create a `SugarcaneDataset(config, split="train", synthetic=True)` and assert that `len(dataset) == 200` (or whatever N you chose for synthetic data).

**Test 2 — `test_single_sample_shapes`:**  
Get sample `dataset[0]` and assert:
- `sample["image"].shape == torch.Size([3, 224, 224])`
- `sample["metadata"].shape == torch.Size([4])`
- `sample["label"].dtype == torch.int64`

**Test 3 — `test_metadata_normalized`:**  
Get sample `dataset[0]` and assert that all metadata values are in `[0, 1]` range:
- `sample["metadata"].min() >= 0.0`
- `sample["metadata"].max() <= 1.0`

**Test 4 — `test_dataloader_batch`:**  
Create a DataLoader from `get_dataloaders(config)["train"]`, get the first batch, and assert:
- `batch["image"].shape == torch.Size([batch_size, 3, 224, 224])`
- `batch["metadata"].shape == torch.Size([batch_size, 4])`

**File to create:** `tests/unit/test_dataset.py`

---

## 📁 Files to Create in Phase 2

| File | What it does |
|------|-------------|
| `configs/obj1_cross_attention.yaml` | Experiment configuration |
| `src/utils/config.py` | YAML config loader |
| `src/utils/seed.py` | Reproducibility seed setter |
| `src/data/augmentation.py` | Albumentations pipeline |
| `src/data/dataset.py` | `SugarcaneDataset` class ← main file |
| `src/data/dataloader.py` | DataLoader factory |
| `tests/unit/test_dataset.py` | Smoke tests |

---

## ⚠️ Common Mistakes in Phase 2

| Mistake | How to Avoid |
|---------|-------------|
| Forgetting to inherit from `torch.utils.data.Dataset` | First line of class: `class SugarcaneDataset(Dataset):` |
| Applying augmentation to val/test data | Only use `get_train_transforms` when `split == "train"` |
| Metadata values outside `[0, 1]` | Always divide by the max value (50, 100, 100, 300) |
| Wrong tensor dtype for labels | Labels must be `torch.long` (int64), not float |
| Image shape in wrong order | PyTorch uses CHW `[3, 224, 224]`, not HWC `[224, 224, 3]` |
| `num_workers > 0` crashing on Windows | Add `if __name__ == "__main__"` guard around DataLoader calls in scripts |
| Forgetting `ToTensorV2()` import | `from albumentations.pytorch import ToTensorV2` |

---

## ✅ How to Know Phase 2 is Done

Run your tests:

```powershell
uv run pytest tests/unit/test_dataset.py -v
```

Expected output:
```
tests/unit/test_dataset.py::test_dataset_length        PASSED ✅
tests/unit/test_dataset.py::test_single_sample_shapes  PASSED ✅
tests/unit/test_dataset.py::test_metadata_normalized   PASSED ✅
tests/unit/test_dataset.py::test_dataloader_batch      PASSED ✅

4 passed in X.Xs
```

If all 4 pass — **Phase 2 is done! 🎉**

---

## 🎉 Phase 2 Summary

You just built the **data foundation** of your research system. When real data arrives later, all you do is:
1. Put images in `data/raw/images/`
2. Put sensor readings in `data/raw/metadata.csv`
3. Change `synthetic=False` in one line

The rest of the pipeline — model, training, evaluation — stays identical.

---

# 🛑 STOP HERE

👉 **Come back and say "Phase 2 done"** when all 4 tests pass.

👉 When Phase 2 is done, we'll start **Phase 3** — building the **Image Encoder** (CNN backbone + Swin-Transformer).

---

---

# ⏳ Phase 3 — Coming After Phase 2 Complete

*(Unlock by completing Phase 2)*

---

# ⏳ Phase 4 onwards...

*(To be unlocked phase by phase)*


---

---

## 📚 Key Concepts Reference

### What is `uv`?
A Python package manager written in Rust. Much faster than `pip`. It creates virtual environments automatically and generates a lockfile for reproducibility.

```powershell
uv sync            # Install all deps from pyproject.toml
uv run pytest      # Run command inside the virtual environment
uv add numpy       # Add a new dependency
uv remove numpy    # Remove a dependency
```

### What is `pre-commit`?
A tool that runs checks automatically every time you type `git commit`. If any check fails, the commit is blocked until you fix the issue.

```
git commit -m "Add cross-attention module"
    │
    ▼
pre-commit runs:
    • trailing-whitespace check ✅
    • end-of-file-fixer ✅
    • ruff linting ✅
    • ruff formatting ✅
    │
    ▼
Commit goes through ✅  (or is blocked if any check fails ❌)
```

### What is `pytest`?
A testing framework. You write functions that test your code, and pytest runs them all and tells you which pass and which fail.

```python
# A simple test example
def test_addition():
    assert 2 + 2 == 4    # This will pass ✅

def test_subtraction():
    assert 10 - 3 == 8   # This will FAIL ❌ (should be 7)
```

### What is `wandb`?
Weights & Biases — it's like a diary for your AI experiments. Every time you train a model, it logs:
- Loss curves (automatically)
- Accuracy graphs
- Model checkpoints
- Your config/hyperparameters

You can see all your experiments in a browser dashboard and compare them.

### Should I use pytest AND CodeRabbit?

**Yes — and here's why:**

| Tool | What it does | When to use |
|------|-------------|-------------|
| `pytest` | **You** write tests that check your code logic | Every time you write a new function |
| `CodeRabbit` | **AI** reviews your Pull Requests for quality issues | When you open a PR on GitHub |

They complement each other. pytest catches logic errors. CodeRabbit catches code quality, security, and design issues. Top open-source repos like FastAPI, LangChain, and HuggingFace use both.

**To add CodeRabbit:** Sign up at https://coderabbit.ai, connect your GitHub repo, and it automatically reviews every PR. Free for public repos.

### Do I need a dataset right now?

**NO.** This is the beauty of good software engineering:
- Phase 2 will teach you to write a Dataset class that works with **random (synthetic) data** for testing
- You can build, test, and validate the entire model architecture **without real data**
- Real data (photos + sensor readings) is collected during Objective 3
- The model code will work the same way when real data arrives — you just swap the data path

This is exactly what the PyTorch team and HuggingFace do — they write tests with fake data so the architecture can be validated independently of dataset collection.

### What's CI/CD and do I need it?

**CI (Continuous Integration):** Every time you push code to GitHub, GitHub Actions automatically runs your pytest tests. If tests fail, you get an email. This catches bugs early.

**CD (Continuous Deployment):** Automatically deploys your model/API when tests pass. For a research project, this might mean automatically running evaluation on a test set.

**Should you add it?** YES — it's what separates a professional research repository from a hobby project. We'll set it up in Phase 9.

---

## 🔗 Useful Links

| Resource | URL | Why |
|---------|-----|-----|
| PyTorch Docs | https://pytorch.org/docs | Reference for all `torch` operations |
| timm Model List | https://huggingface.co/timm | See all available pretrained models |
| Albumentations Docs | https://albumentations.ai/docs | Image augmentation reference |
| W&B Quickstart | https://docs.wandb.ai/quickstart | Set up experiment tracking |
| uv Docs | https://docs.astral.sh/uv | Package management reference |
| Ruff Docs | https://docs.astral.sh/ruff | Code quality tool |

---

*Good luck! You've got this. One phase at a time. 🌾*
