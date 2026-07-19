"""Canonical storage for runs, metrics, results, and their catalog."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configs.config_loader import load_config
from .serialization import json_safe

SCHEMA_VERSION = 1
DEFAULT_ARTIFACTS_DIR = "artifacts"


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _git_metadata() -> dict[str, Any]:
    """Capture the current commit and dirty flag without requiring Git."""
    metadata: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata.update(commit=commit.stdout.strip(), dirty=bool(status.stdout))
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


class RunCatalog:
    """SQLite index for fast run and game discovery."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        """Create idempotent run/game tables and lookup indexes."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    victory INTEGER,
                    bosses_defeated INTEGER,
                    turns INTEGER,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    path TEXT NOT NULL,
                    summary_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_games_run_id ON games(run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
                """
            )

    def upsert_run(self, manifest: dict[str, Any], path: Path) -> None:
        """Insert a run or refresh its mutable status and manifest fields."""
        payload = json.dumps(json_safe(manifest), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, run_type, name, status, started_at, ended_at,
                    path, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    ended_at=excluded.ended_at,
                    manifest_json=excluded.manifest_json
                """,
                (
                    manifest["run_id"],
                    manifest["run_type"],
                    manifest["name"],
                    manifest["status"],
                    manifest["started_at"],
                    manifest.get("ended_at"),
                    str(path),
                    payload,
                ),
            )

    def upsert_game(
        self,
        game_id: str,
        run_id: str,
        status: str,
        started_at: str,
        path: Path,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update one game summary in the catalog.

        The initial row may represent a running game; finalization updates the
        status, outcome, turn count, and serialized summary in place.
        """
        summary = summary or {}
        payload = json.dumps(json_safe(summary), ensure_ascii=False) if summary else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO games (
                    game_id, run_id, status, victory, bosses_defeated,
                    turns, started_at, ended_at, path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    status=excluded.status,
                    victory=excluded.victory,
                    bosses_defeated=excluded.bosses_defeated,
                    turns=excluded.turns,
                    ended_at=excluded.ended_at,
                    summary_json=excluded.summary_json
                """,
                (
                    game_id,
                    run_id,
                    status,
                    summary.get("victory"),
                    summary.get("bosses_defeated"),
                    summary.get("turns"),
                    started_at,
                    summary.get("ended_at"),
                    str(path),
                    payload,
                ),
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the newest cataloged runs up to ``limit``."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one catalog row by run identifier, if present."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_games(self, run_id: str) -> list[dict[str, Any]]:
        """Return games belonging to a run in start-time order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM games WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        """Return one catalog row by game identifier, if present."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        return dict(row) if row else None


@dataclass
class RunContext:
    """Lifecycle and filesystem context for one executable run."""

    run_id: str
    run_type: str
    name: str
    root_dir: Path
    run_dir: Path
    manifest: dict[str, Any]
    catalog: RunCatalog
    settings: dict[str, Any]
    _metrics_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(
        cls,
        run_type: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        root_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> "RunContext":
        """Create run directories, manifest, configuration snapshot, and catalog row.

        Args:
            run_type: Stable workflow category used by configuration overrides.
            name: Optional human-readable run name.
            config: Effective workflow configuration to snapshot.
            root_dir: Artifact root override.
            metadata: Additional JSON-compatible provenance.
            settings: Preloaded logger settings; loaded automatically if omitted.

        Returns:
            Running context ready for logs, metrics, games, and results.
        """
        effective_settings = settings or load_config(run_type=run_type)
        started_at = utc_now()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_name = name or run_type
        run_id = f"{run_type}-{timestamp}-{uuid.uuid4().hex[:8]}"
        artifacts_root = Path(
            root_dir
            or os.environ.get("REGICIDE_ARTIFACTS_DIR")
            or effective_settings.get("artifacts", {}).get(
                "root_dir", DEFAULT_ARTIFACTS_DIR
            )
        )
        (artifacts_root / "datasets").mkdir(parents=True, exist_ok=True)
        (artifacts_root / "promoted_models").mkdir(parents=True, exist_ok=True)
        date_path = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_dir = artifacts_root / "runs" / date_path / run_id
        for child in (
            "logs",
            "metrics",
            "games",
            "models",
            "checkpoints",
            "analysis",
            "datasets",
        ):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        manifest = _build_manifest(
            run_id,
            run_type,
            run_name,
            started_at,
            config,
            metadata,
            effective_settings,
        )
        _atomic_json_write(run_dir / "manifest.json", manifest)
        if config is not None:
            _atomic_json_write(run_dir / "config.json", config)
        catalog = RunCatalog(artifacts_root / "catalog.sqlite")
        context = cls(
            run_id,
            run_type,
            run_name,
            artifacts_root,
            run_dir,
            manifest,
            catalog,
            effective_settings,
        )
        catalog.upsert_run(manifest, run_dir)
        return context

    @classmethod
    def attach(
        cls,
        run_id: str,
        run_dir: str | Path,
        root_dir: str | Path,
    ) -> "RunContext":
        """Attach a process-local context to an existing run directory.

        This is used by worker processes and does not create a new manifest or
        run identifier.
        """
        path = Path(run_dir)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            run_id=run_id,
            run_type=manifest["run_type"],
            name=manifest["name"],
            root_dir=Path(root_dir),
            run_dir=path,
            manifest=manifest,
            catalog=RunCatalog(Path(root_dir) / "catalog.sqlite"),
            settings=manifest.get("logger_settings") or load_config(
                run_type=manifest["run_type"]
            ),
        )

    def descriptor(self) -> dict[str, str]:
        """Return a process-safe descriptor for worker attachment."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "root_dir": str(self.root_dir),
        }

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one timestamped metric record when metric saving is enabled."""
        if not self.saving_enabled("metrics"):
            return
        entry = {"timestamp": utc_now(), "step": step, **json_safe(metrics)}
        path = self.run_dir / "metrics" / "metrics.jsonl"
        with self._metrics_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_telemetry(self, telemetry: dict[str, Any]) -> None:
        """Append one timestamped hardware sample when telemetry is enabled."""
        if not self.saving_enabled("telemetry"):
            return
        entry = {"timestamp": utc_now(), **json_safe(telemetry)}
        path = self.run_dir / "metrics" / "telemetry.jsonl"
        with self._metrics_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_result(
        self,
        filename: str,
        result: dict[str, Any],
        category: str = "analysis",
    ) -> Path:
        """Atomically save a JSON result below a run category directory.

        Returns the intended path even when result saving is disabled, allowing
        callers to keep a stable control flow.
        """
        output_path = self.run_dir / category / filename
        if not self.saving_enabled("results"):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(output_path, result)
        return output_path

    def saving_enabled(self, category: str) -> bool:
        """Return whether the global and category-specific save switches are on."""
        saving = self.settings.get("saving", {})
        return bool(saving.get("enabled", True) and saving.get(category, True))

    @property
    def game_recording_enabled(self) -> bool:
        """Return whether optional game artifacts are enabled for this run."""
        games = self.settings.get("games", {})
        saving_enabled = self.settings.get("saving", {}).get("enabled", True)
        return bool(saving_enabled and games.get("enabled", True))

    @property
    def game_recording_level(self) -> str:
        """Return the configured ``summary``, ``actions``, or ``full`` level."""
        return self.settings.get("games", {}).get("recording_level", "actions")

    def complete(self, metadata: dict[str, Any] | None = None) -> None:
        """Finalize the run successfully and merge optional result metadata."""
        self._finish("completed", metadata)

    def fail(self, error: BaseException | str) -> None:
        """Finalize the run as failed with a serialized error message."""
        self._finish("failed", {"error": str(error)})

    def _finish(self, status: str, metadata: dict[str, Any] | None) -> None:
        self.manifest["status"] = status
        self.manifest["ended_at"] = utc_now()
        if metadata:
            self.manifest.setdefault("result", {}).update(json_safe(metadata))
        _atomic_json_write(self.run_dir / "manifest.json", self.manifest)
        self.catalog.upsert_run(self.manifest, self.run_dir)


def _build_manifest(
    run_id: str,
    run_type: str,
    name: str,
    started_at: str,
    config: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    logger_settings: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the versioned, JSON-safe provenance record for a new run."""
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_type": run_type,
        "name": name,
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "command": command,
        "python": sys.version,
        "git": _git_metadata(),
        "config": json_safe(config or {}),
        "metadata": json_safe(metadata or {}),
        "logger_settings": json_safe(logger_settings),
    }
