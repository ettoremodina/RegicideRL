"""Safe configuration reading, validation, diffing, backup, and saving."""

from __future__ import annotations

import difflib
import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConfigDefinition:
    """Describe one repository configuration visible in the control panel."""

    config_id: str
    title: str
    path: str
    description: str
    editable: bool = True
    restart_note: str = "New values apply to jobs started after the save."

    def as_dict(self) -> dict[str, Any]:
        """Serialize configuration metadata for the browser."""
        return {
            "id": self.config_id,
            "title": self.title,
            "path": self.path,
            "description": self.description,
            "editable": self.editable,
            "restart_note": self.restart_note,
        }


CONFIG_DEFINITIONS = (
    ConfigDefinition(
        "main",
        "Main solver configuration",
        "config.yaml",
        "Environment, PPO, AlphaZero, tuning, and comparison settings.",
    ),
    ConfigDefinition(
        "smoke",
        "Smoke-test configuration",
        "config_test.yaml",
        "Reduced PPO and tuning settings for quick checks.",
    ),
    ConfigDefinition(
        "logger",
        "Logger configuration",
        "logger_config.yaml",
        "Logging, saving, telemetry, reports, metric filters, and run overrides.",
        restart_note=(
            "The logger snapshots this file when a run starts; active runs do not "
            "hot-reload it."
        ),
    ),
    ConfigDefinition(
        "logger-defaults",
        "Packaged logger defaults",
        "ml_logger/configs/default_config.yaml",
        "Read-only fallback values shipped with the logger package.",
        editable=False,
        restart_note="This reference file is intentionally read-only in the panel.",
    ),
)


class ConfigurationService:
    """Manage only the explicitly allowlisted repository configuration files."""

    def __init__(self, repository_root: Path, state_root: Path):
        self.repository_root = repository_root.resolve()
        self.state_root = state_root.resolve()
        self._definitions = {
            definition.config_id: definition for definition in CONFIG_DEFINITIONS
        }

    def definitions(self) -> list[dict[str, Any]]:
        """Return all visible configuration definitions."""
        return [definition.as_dict() for definition in CONFIG_DEFINITIONS]

    def read(self, config_id: str) -> dict[str, Any]:
        """Read one allowlisted file with its content hash and parsed structure."""
        definition = self._definition(config_id)
        path = self.repository_root / definition.path
        text = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
        return {
            **definition.as_dict(),
            "text": text,
            "sha256": _sha256(text),
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime,
                timezone.utc,
            ).isoformat(),
            "top_level_keys": list(parsed) if isinstance(parsed, dict) else [],
        }

    def preview(
        self,
        config_id: str,
        proposed_text: str,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Validate a proposed edit and return a unified diff without writing."""
        definition = self._definition(config_id)
        if not definition.editable:
            raise PermissionError(f"{definition.title} is read-only")
        current = self.read(config_id)
        if expected_sha256 and current["sha256"] != expected_sha256:
            raise RuntimeError(
                "The file changed after it was opened. Reload before saving."
            )
        errors, warnings = self._validate(definition, proposed_text)
        diff = "".join(
            difflib.unified_diff(
                current["text"].splitlines(keepends=True),
                proposed_text.splitlines(keepends=True),
                fromfile=definition.path,
                tofile=f"{definition.path} (proposed)",
            )
        )
        return {
            "valid": not errors,
            "changed": proposed_text != current["text"],
            "errors": errors,
            "warnings": warnings,
            "diff": diff or "No changes.",
            "current_sha256": current["sha256"],
        }

    def save(
        self,
        config_id: str,
        proposed_text: str,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """Atomically save a validated edit after retaining the previous version."""
        preview = self.preview(config_id, proposed_text, expected_sha256)
        if not preview["valid"]:
            raise ValueError("Configuration validation failed")
        if not preview["changed"]:
            return {**self.read(config_id), "backup": None, "changed": False}
        definition = self._definition(config_id)
        target = self.repository_root / definition.path
        backup = self._backup(target)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(proposed_text, encoding="utf-8")
        temporary.replace(target)
        return {
            **self.read(config_id),
            "backup": str(backup.relative_to(self.repository_root)),
            "changed": True,
        }

    def _definition(self, config_id: str) -> ConfigDefinition:
        try:
            return self._definitions[config_id]
        except KeyError as error:
            raise KeyError(f"Unknown configuration: {config_id}") from error

    def _validate(
        self,
        definition: ConfigDefinition,
        proposed_text: str,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            parsed = yaml.safe_load(proposed_text)
        except yaml.YAMLError as error:
            return [str(error)], warnings
        if not isinstance(parsed, dict):
            return ["The YAML document must contain a mapping at its root."], warnings
        if definition.config_id == "main":
            _validate_required_sections(
                parsed,
                ("env", "ppo", "training", "tuning", "alphazero", "experimental_report"),
                errors,
            )
            warnings.extend(self._model_warnings(parsed))
        elif definition.config_id == "smoke":
            _validate_required_sections(
                parsed,
                ("env", "ppo", "training", "tuning"),
                errors,
            )
        elif definition.config_id == "logger":
            _validate_logger(parsed, errors)
        return errors, warnings

    def _model_warnings(self, config: dict[str, Any]) -> list[str]:
        """Report enabled learned agents whose configured model is absent."""
        warnings: list[str] = []
        agents = config.get("experimental_report", {}).get("agents", {})
        path_keys = {"ppo": "model_path", "alphazero": "checkpoint_path"}
        for agent_name, path_key in path_keys.items():
            settings = agents.get(agent_name, {})
            if not settings.get("enabled"):
                continue
            configured = settings.get("kwargs", {}).get(path_key)
            if configured and not (self.repository_root / configured).exists():
                warnings.append(
                    f"Enabled agent '{agent_name}' references a missing path: {configured}"
                )
        return warnings

    def _backup(self, target: Path) -> Path:
        """Save the current source text beneath the ignored panel state tree."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = self.state_root / "config-backups" / target.stem
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{timestamp}-{uuid.uuid4().hex[:8]}{target.suffix}"
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        return backup


def _validate_required_sections(
    config: dict[str, Any],
    required: tuple[str, ...],
    errors: list[str],
) -> None:
    """Append errors for missing or non-mapping top-level sections."""
    for section in required:
        if section not in config:
            errors.append(f"Missing top-level section: {section}")
        elif not isinstance(config[section], dict):
            errors.append(f"Section '{section}' must be a mapping")


def _validate_logger(config: dict[str, Any], errors: list[str]) -> None:
    """Validate logger values not fully covered by the package loader."""
    dashboard = config.get("dashboard", {})
    raw_mode = dashboard.get("mode", "auto")
    mode = ("live" if raw_mode else "off") if isinstance(raw_mode, bool) else str(raw_mode).lower()
    if mode not in {"auto", "live", "compact", "off"}:
        errors.append("dashboard.mode must be auto, live, compact, or off")
    try:
        if float(dashboard.get("refresh_rate", 4)) <= 0:
            errors.append("dashboard.refresh_rate must be positive")
    except (TypeError, ValueError):
        errors.append("dashboard.refresh_rate must be numeric")
    telemetry = config.get("telemetry", {})
    try:
        if float(telemetry.get("sample_interval_sec", 5)) <= 0:
            errors.append("telemetry.sample_interval_sec must be positive")
    except (TypeError, ValueError):
        errors.append("telemetry.sample_interval_sec must be numeric")
    for index, highlight in enumerate(config.get("highlights", [])):
        try:
            re.compile(str(highlight.get("pattern", "")))
        except re.error as error:
            errors.append(f"highlights[{index}].pattern is invalid: {error}")


def _sha256(text: str) -> str:
    """Return a stable UTF-8 content hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
