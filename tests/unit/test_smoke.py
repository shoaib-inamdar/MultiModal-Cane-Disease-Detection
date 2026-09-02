# tests/unit/test_smoke.py
# Minimal smoke tests that always pass.
# These validate the project skeleton is intact (no src imports needed).
# Real tests are added phase-by-phase as modules are built (Phase 2+).


def test_project_structure():
    """Verify core project directories exist."""
    import os

    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/models/backbone",
        "src/models/encoders",
        "src/models/fusion",
        "src/models/heads",
        "src/training",
        "src/evaluation",
        "src/utils",
        "tests/unit",
        "tests/integration",
        "configs",
        "scripts",
        "notebooks",
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"


def test_pyproject_toml_exists():
    """pyproject.toml must exist and contain project name."""
    import os

    assert os.path.isfile("pyproject.toml"), "pyproject.toml not found"
    content = open("pyproject.toml").read()
    assert "multimodal-cane-disease-detection" in content


def test_env_example_exists():
    """.env.example must exist (never commit real .env)."""
    import os

    assert os.path.isfile(".env.example"), ".env.example not found"


def test_gitignore_blocks_env():
    """.gitignore must block .env from being committed."""
    import os

    assert os.path.isfile(".gitignore"), ".gitignore not found"
    content = open(".gitignore").read()
    assert ".env" in content, ".env not in .gitignore — security risk!"
