"""Load and validate the central ml_logger runtime configuration."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

from ..terminal import parse_utc_offset

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
PROJECT_CONFIG_NAME = "logger_config.yaml"
CONFIG_ENVIRONMENT_VARIABLE = "ML_LOGGER_CONFIG"
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_RECORDING_LEVELS = {"summary", "actions", "full"}


def load_config(config_path: str | Path | None = None, run_type=None) -> dict:
    """Load defaults, user configuration, and an optional run-type override."""
    default_config = _read_yaml(DEFAULT_CONFIG_PATH)
    selected_path = _select_config_path(config_path)
    config = deepcopy(default_config)
    if selected_path != DEFAULT_CONFIG_PATH:
        _deep_merge(config, _read_yaml(selected_path))
    overrides = config.get("run_type_overrides", {}).get(run_type, {})
    _deep_merge(config, overrides)
    config["config_path"] = str(selected_path)
    _validate_config(config)
    return config


def _select_config_path(config_path):
    """Resolve explicit, environment, project, then packaged configuration."""
    if config_path:
        return Path(config_path)
    environment_path = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if environment_path:
        return Path(environment_path)
    project_path = Path.cwd() / PROJECT_CONFIG_NAME
    return project_path if project_path.exists() else DEFAULT_CONFIG_PATH


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _deep_merge(destination, source):
    """Recursively copy ``source`` values over a destination mapping."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_merge(destination[key], value)
        else:
            destination[key] = deepcopy(value)
    return destination


def _validate_config(config):
    """Normalize and validate settings that affect runtime behavior."""
    logging_level = str(config["logging"]["level"]).upper()
    if logging_level not in VALID_LEVELS:
        raise ValueError(f"Unsupported logging level: {logging_level}")
    config["logging"]["level"] = logging_level
    recording_level = config["games"]["recording_level"]
    if recording_level not in VALID_RECORDING_LEVELS:
        raise ValueError(f"Unsupported recording level: {recording_level}")
    parse_utc_offset(config["terminal"]["timezone"])
