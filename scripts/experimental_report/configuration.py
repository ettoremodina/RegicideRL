"""Configuration helpers for the experimental report pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

import yaml


def load_report_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the experimental-report section of the YAML config."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        project_config = yaml.safe_load(stream) or {}
    report_config = project_config.get("experimental_report")
    if not isinstance(report_config, dict):
        raise ValueError("Missing 'experimental_report' section in config")
    _validate_protocol(report_config.get("protocol", {}))
    _validate_agents(report_config.get("agents", {}))
    return deepcopy(report_config)


def select_agents(
    report_config: dict[str, Any],
    requested: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return enabled or explicitly requested agent specifications."""
    agents = report_config["agents"]
    if requested:
        unknown = sorted(set(requested) - set(agents))
        if unknown:
            raise ValueError(f"Unknown agent(s): {', '.join(unknown)}")
        return {name: deepcopy(agents[name]) for name in requested}
    selected = {
        name: deepcopy(spec)
        for name, spec in agents.items()
        if spec.get("enabled", False)
    }
    if not selected:
        raise ValueError("No agents are enabled in the experimental-report config")
    return selected


def apply_protocol_overrides(
    report_config: dict[str, Any],
    games: int | None = None,
    base_seed: int | None = None,
) -> dict[str, Any]:
    """Apply CLI overrides without mutating the loaded configuration."""
    effective = deepcopy(report_config)
    if games is not None:
        effective["protocol"]["games_per_agent"] = games
    if base_seed is not None:
        effective["protocol"]["base_seed"] = base_seed
    _validate_protocol(effective["protocol"])
    return effective


def snapshot_report_config(
    source_path: str | Path,
    report_config: dict[str, Any],
    run_dir: str | Path,
) -> None:
    """Save the source and effective configurations beside a run."""
    destination = Path(run_dir)
    shutil.copy2(source_path, destination / "config.yaml")
    with (destination / "experimental_report_config.yaml").open(
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(
            {"experimental_report": report_config},
            stream,
            sort_keys=False,
            allow_unicode=True,
        )


def _validate_protocol(protocol: dict[str, Any]) -> None:
    required_positive = (
        "games_per_agent",
        "max_decisions_per_game",
        "bootstrap_samples",
    )
    for key in required_positive:
        if int(protocol.get(key, 0)) <= 0:
            raise ValueError(f"experimental_report.protocol.{key} must be positive")
    confidence = float(protocol.get("confidence_level", 0.0))
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "experimental_report.protocol.confidence_level must be between 0 and 1"
        )
    if "base_seed" not in protocol:
        raise ValueError("experimental_report.protocol.base_seed is required")


def _validate_agents(agents: dict[str, Any]) -> None:
    if not agents:
        raise ValueError("experimental_report.agents must not be empty")
    for name, spec in agents.items():
        missing = {"class_path", "label", "description"} - set(spec)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Agent '{name}' is missing: {fields}")
