"""Configuration loading shared by solver entry points."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load a solver configuration from a YAML file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
