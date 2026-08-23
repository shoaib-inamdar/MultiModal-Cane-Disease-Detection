# 📋 CHANGELOG

All notable changes to SugarcaneAI will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — Objective 1 In Progress

### Added
- `README.md` — Production-grade project documentation with badges, architecture overview, and quick start
- `ARCHITECTURE.md` — Detailed system architecture documentation covering all 6 objectives
- `WALKTHROUGH.md` — Phase-by-phase beginner teaching guide
- `CONTRIBUTING.md` — Contribution guidelines and workflow
- `CHANGELOG.md` — This file
- `pyproject.toml` — Full dependency specification with `uv` support
- `.pre-commit-config.yaml` — Code quality hooks (ruff, mypy)
- `.gitignore` — Comprehensive ignore rules for Python/PyTorch projects
- `.env.example` — Environment variable template
- Complete folder structure for all 6 research objectives

### Planned (Objective 1)
- `src/data/preprocessing/quality_filter.py` — BRISQUE quality filtering
- `src/data/preprocessing/deduplication.py` — pHash deduplication
- `src/data/preprocessing/image_enhance.py` — CLAHE enhancement
- `src/data/preprocessing/normalizer.py` — Min-Max metadata normalization
- `src/data/dataset.py` — PyTorch Dataset class (with synthetic fallback)
- `src/models/backbone/cnn_backbone.py` — EfficientNet-B3 backbone
- `src/models/backbone/swin_transformer.py` — Swin-Transformer encoder
- `src/models/encoders/image_encoder.py` — Hybrid CNN+ViT image encoder
- `src/models/encoders/metadata_encoder.py` — MLP environmental encoder
- `src/models/fusion/cross_attention.py` — **Core contribution: Cross-Attention Fusion**
- `src/models/multimodal_model.py` — Full model assembly
- `src/training/trainer.py` — Training loop with W&B logging
- `src/evaluation/metrics.py` — F1, ECE, Accuracy computation
- `tests/unit/test_cross_attention.py` — Unit tests for fusion module
- `configs/obj1_cross_attention.yaml` — Experiment configuration
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline

---

## [0.1.0] — 2026-08-23 — Project Bootstrap

### Added
- Initial project scaffold with `uv` package manager
- Basic `main.py` placeholder
- `pyproject.toml` initial configuration
- Git repository initialization
- `Synopsis.docx` — Ph.D. research synopsis

### Project Info
- **Author:** Shoaib Inamdar (`@shoaib-inamdar`)
- **Repository:** `shoaib-inamdar/MultiModal-Cane-Disease-Detection`

---

[Unreleased]: https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shoaib-inamdar/MultiModal-Cane-Disease-Detection/releases/tag/v0.1.0
