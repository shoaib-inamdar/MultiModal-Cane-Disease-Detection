# 🤝 Contributing to SugarcaneAI

Thank you for your interest in contributing to this research project!

---

## 🌿 Who Can Contribute?

- **Research collaborators** — fellow researchers adding new experiments or baselines
- **Dataset contributors** — field workers adding labeled images + metadata
- **Code contributors** — developers improving the pipeline, preprocessing, or tools
- **Documentation contributors** — improving explanations, fixing typos

---

## 🛠️ Development Setup

```bash
# Clone the repo
git clone https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection.git
cd MultiModal-Cane-Disease-Detection

# Install all dependencies including dev tools
uv sync --extra dev

# Install pre-commit hooks (REQUIRED before first commit)
uv run pre-commit install

# Verify everything works
uv run pytest tests/unit/ -v
```

---

## 📋 Contribution Workflow

### 1. Find or Create an Issue

- Check [Issues](../../issues) for existing tasks
- Use labels: `obj1`, `obj2`, `dataset`, `bug`, `documentation`
- For new ideas, open an **Experiment Proposal** issue first

### 2. Create a Branch

```bash
# Always branch from develop, not main
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Examples:
# git checkout -b feature/clahe-preprocessing
# git checkout -b fix/cross-attention-shape-bug
# git checkout -b experiment/obj1-swin-large
```

### 3. Write Code + Tests

- **Every new function must have a unit test** in `tests/unit/`
- Tests must pass before opening a PR
- Follow the existing code style (ruff handles formatting)

```bash
# Run tests
uv run pytest tests/unit/ -v

# Check code style
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add CLAHE preprocessing to image pipeline"
```

**Commit message conventions:**

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or module |
| `fix:` | Bug fix |
| `test:` | Adding or fixing tests |
| `docs:` | Documentation changes |
| `refactor:` | Code restructure (no feature change) |
| `experiment:` | New experiment or model variant |
| `data:` | Dataset or preprocessing changes |

Pre-commit hooks will run automatically. Fix any failures before the commit is accepted.

### 5. Open a Pull Request

- Target branch: `develop` (not `main`)
- Fill in the PR template completely
- Link the related Issue
- CodeRabbit will automatically review your PR — address its suggestions

---

## 🧪 Testing Guidelines

### Unit Tests (Required for all new code)

Unit tests use **synthetic (fake) tensors** — no real dataset required:

```python
# Example structure
import pytest
import torch

class TestYourModule:
    @pytest.fixture
    def module(self):
        # Create your module with test config
        ...

    def test_output_shape(self, module):
        # Always test that output shapes are correct
        ...

    def test_gradient_flow(self, module):
        # Always test that gradients flow through
        ...
```

### What Must Be Tested

| Component | What to Test |
|-----------|-------------|
| Every model layer | Output shape is correct |
| Every model layer | Gradients flow (no dead gradients) |
| Preprocessing | Normalization stays in [0, 1] range |
| Preprocessing | BRISQUE correctly rejects bad images |
| Metrics | ECE is in [0, 1] range |
| Data loader | No data leakage between splits |

---

## 📊 Adding a New Experiment

1. Create a config in `configs/` (copy `obj1_cross_attention.yaml` as template)
2. Create a results folder in `experiments/`
3. Run the experiment: `uv run python scripts/train.py --config configs/your_config.yaml`
4. Log results to W&B (automatic if `wandb.enabled: true`)
5. Save results summary in `experiments/obj1/results/your_experiment.json`
6. Add notes to `experiments/obj1/README.md`

---

## 🌾 Adding Dataset Samples

> **Important:** Raw images and data files are NEVER committed to Git. They are managed by DVC.

If you're contributing field-collected samples:

1. Contact the research team to get DVC remote access
2. Add your images to `data/raw/images/`
3. Add your metadata CSV rows to `data/raw/metadata.csv`
4. Run the sync script: `uv run python scripts/sync_metadata.py`
5. Push to DVC: `dvc push`

Label format for images: `{disease_class}_{severity_grade}_{location}_{timestamp}.jpg`  
Example: `red_rot_grade2_kolhapur_20260315_0935.jpg`

---

## ❌ What NOT to Do

- ❌ Do NOT commit `.env` or any file with real API keys
- ❌ Do NOT commit files > 500KB to Git (use DVC for data/models)
- ❌ Do NOT push directly to `main`
- ❌ Do NOT merge your own PR (request a review)
- ❌ Do NOT break existing tests without a documented reason
- ❌ Do NOT add new dependencies without discussing in an Issue first

---

## 🆘 Getting Help

- Open a GitHub Discussion for questions
- Tag `@shoaib-inamdar` for urgent issues
- For research questions: Contact the research guide

---

*Thank you for helping advance sugarcane disease detection research! 🌾*
