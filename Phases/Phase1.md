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

# ⏳ Phase 2 — Coming After Phase 1 Complete

*(Not revealed yet — complete Phase 1 first)*

---

# ⏳ Phase 3 — Coming After Phase 2 Complete

*(Not revealed yet)*

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
