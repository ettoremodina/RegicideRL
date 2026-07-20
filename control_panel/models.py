"""Declarative command and parameter models used by the control panel."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

ParameterKind = Literal[
    "boolean",
    "choice",
    "identifier",
    "integer",
    "multi_choice",
    "number",
    "path",
    "text",
]
PathMode = Literal[
    "artifact_output",
    "existing_dir",
    "existing_file",
    "repo_path",
]


@dataclass(frozen=True)
class ParameterSpec:
    """Describe one validated form field and its CLI representation."""

    key: str
    label: str
    flag: str | None = None
    kind: ParameterKind = "text"
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""
    placeholder: str = ""
    positional: bool = False
    path_mode: PathMode | None = None

    def normalize(self, value: Any, repository_root: Path) -> Any:
        """Validate a submitted value and return its CLI-safe form."""
        if value is None or value == "" or value == []:
            if self.required:
                raise ValueError(f"{self.label} is required")
            return None
        normalizers = {
            "boolean": self._normalize_boolean,
            "choice": self._normalize_choice,
            "identifier": self._normalize_identifier,
            "integer": self._normalize_integer,
            "multi_choice": self._normalize_multi_choice,
            "number": self._normalize_number,
            "path": lambda item: self._normalize_path(item, repository_root),
            "text": lambda item: str(item).strip(),
        }
        return normalizers[self.kind](value)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the field definition for the browser client."""
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "required": self.required,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "help": self.help,
            "placeholder": self.placeholder,
        }

    def _normalize_boolean(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{self.label} must be true or false")

    def _normalize_choice(self, value: Any) -> str:
        normalized = str(value).strip()
        if normalized not in self.choices:
            raise ValueError(
                f"{self.label} must be one of: {', '.join(self.choices)}"
            )
        return normalized

    def _normalize_identifier(self, value: Any) -> str:
        normalized = str(value).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", normalized):
            raise ValueError(f"{self.label} contains unsupported characters")
        return normalized

    def _normalize_integer(self, value: Any) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{self.label} must be an integer") from error
        self._validate_range(normalized)
        return normalized

    def _normalize_multi_choice(self, value: Any) -> list[str]:
        values = value if isinstance(value, list) else str(value).split(",")
        normalized = [str(item).strip() for item in values if str(item).strip()]
        unsupported = [item for item in normalized if item not in self.choices]
        if unsupported:
            raise ValueError(
                f"Unsupported {self.label.lower()}: {', '.join(unsupported)}"
            )
        return normalized

    def _normalize_number(self, value: Any) -> float:
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{self.label} must be a number") from error
        self._validate_range(normalized)
        return normalized

    def _normalize_path(self, value: Any, repository_root: Path) -> str:
        resolved = _resolve_repository_path(str(value), repository_root)
        if self.path_mode == "existing_file" and not resolved.is_file():
            raise ValueError(f"{self.label} does not exist or is not a file")
        if self.path_mode == "existing_dir" and not resolved.is_dir():
            raise ValueError(f"{self.label} does not exist or is not a directory")
        if self.path_mode == "artifact_output":
            artifacts_root = (repository_root / "artifacts").resolve()
            if not _is_within(resolved, artifacts_root):
                raise ValueError(f"{self.label} must be inside artifacts/")
        return str(resolved.relative_to(repository_root.resolve())) or "."

    def _validate_range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum:g}")


@dataclass(frozen=True)
class CommandSpec:
    """Define one allowlisted repository command exposed by the panel."""

    command_id: str
    title: str
    category: str
    description: str
    base_argv: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    risk: Literal["light", "standard", "heavy", "maintenance", "desktop"] = (
        "standard"
    )
    confirmation: str = ""
    source: str = ""
    creates_run: bool = True
    quick_action: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def build_argv(
        self,
        python_executable: Path,
        submitted: Mapping[str, Any],
        repository_root: Path,
    ) -> list[str]:
        """Build a list-form subprocess command from validated inputs."""
        supported = {parameter.key for parameter in self.parameters}
        unknown = set(submitted) - supported
        if unknown:
            raise ValueError(f"Unsupported parameters: {', '.join(sorted(unknown))}")
        argv = [str(python_executable), *self.base_argv]
        for parameter in self.parameters:
            value = submitted.get(parameter.key, parameter.default)
            normalized = parameter.normalize(value, repository_root)
            if normalized is None or normalized is False:
                continue
            if parameter.kind == "boolean":
                if parameter.flag:
                    argv.append(parameter.flag)
                continue
            values = normalized if isinstance(normalized, list) else [normalized]
            if parameter.flag:
                argv.append(parameter.flag)
            argv.extend(str(item) for item in values)
        return argv

    def as_dict(self) -> dict[str, Any]:
        """Serialize command metadata and form fields for the browser."""
        return {
            "id": self.command_id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "parameters": [field.as_dict() for field in self.parameters],
            "risk": self.risk,
            "confirmation": self.confirmation,
            "source": self.source,
            "creates_run": self.creates_run,
            "quick_action": self.quick_action,
            "tags": list(self.tags),
        }


def _resolve_repository_path(value: str, repository_root: Path) -> Path:
    """Resolve a user path and reject traversal outside the repository."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    resolved = candidate.resolve(strict=False)
    if not _is_within(resolved, repository_root.resolve()):
        raise ValueError("Paths outside the repository are not allowed")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by another resolved path."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
