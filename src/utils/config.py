from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path.resolve()}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
