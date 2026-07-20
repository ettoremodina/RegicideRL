"""Read-only repository, catalog, run, artifact, and system inspection."""

from __future__ import annotations

import ast
import json
import mimetypes
import os
import sqlite3
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml

IGNORED_DIRECTORIES = {
    ".agents",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "venv",
}
BROWSER_DENY = {".git", "venv", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class RepositoryService:
    """Expose a safe read model over the repository and logger catalog."""

    def __init__(self, repository_root: Path, registered_sources: set[str]):
        self.repository_root = repository_root.resolve()
        self.artifacts_root = self.repository_root / "artifacts"
        self.catalog_path = self.artifacts_root / "catalog.sqlite"
        self.registered_sources = registered_sources
        self._cache_lock = threading.Lock()
        self._cache: dict[str, tuple[float, Any]] = {}

    def overview(self) -> dict[str, Any]:
        """Return the compact repository, catalog, system, and health snapshot."""
        catalog = self.catalog_summary()
        inventory = self.inventory()
        system = self.system_snapshot()
        git = self.git_snapshot()
        return {
            "catalog": catalog,
            "inventory": inventory,
            "system": system,
            "git": git,
            "health": self._health(catalog, inventory, system, git),
        }

    def catalog_summary(self) -> dict[str, Any]:
        """Summarize run and game states without trusting status as liveness."""
        if not self.catalog_path.exists():
            return _empty_catalog_summary()
        with self._connect_catalog() as connection:
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM runs GROUP BY status"
            ).fetchall()
            total_runs = sum(row["count"] for row in status_rows)
            game_counts = _game_status_counts(connection)
            running_rows = connection.execute(
                """
                SELECT r.run_id, r.path, r.started_at,
                       MAX(e.timestamp) AS last_event_at
                FROM runs r
                LEFT JOIN events e ON e.run_id = r.run_id
                WHERE r.status = 'running'
                GROUP BY r.run_id
                """
            ).fetchall()
            paths = connection.execute("SELECT path FROM runs").fetchall()
        missing_paths = sum(not self._catalog_path(row["path"]).exists() for row in paths)
        stale_running = sum(
            _is_stale(row["last_event_at"] or row["started_at"])
            for row in running_rows
        )
        return {
            "total_runs": total_runs,
            "statuses": {row["status"]: row["count"] for row in status_rows},
            "games": game_counts,
            "missing_paths": missing_paths,
            "stale_running": stale_running,
            "catalog_bytes": self.catalog_path.stat().st_size,
        }

    def list_runs(
        self,
        limit: int = 100,
        status: str | None = None,
        run_type: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return catalog runs enriched with path, event, and liveness signals."""
        if not self.catalog_path.exists():
            return []
        clauses: list[str] = []
        parameters: list[Any] = []
        if status:
            clauses.append("r.status = ?")
            parameters.append(status)
        if run_type:
            clauses.append("r.run_type = ?")
            parameters.append(run_type)
        if query:
            clauses.append("(r.run_id LIKE ? OR r.name LIKE ? OR r.run_type LIKE ?)")
            search = f"%{query}%"
            parameters.extend((search, search, search))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 500)))
        sql = f"""
            SELECT r.*,
                   (SELECT MAX(timestamp) FROM events e WHERE e.run_id = r.run_id)
                       AS last_event_at,
                   (SELECT payload_json FROM events t
                    WHERE t.run_id = r.run_id AND t.kind = 'telemetry'
                    ORDER BY t.sequence DESC LIMIT 1) AS telemetry_json
            FROM runs r
            {where}
            ORDER BY r.started_at DESC
            LIMIT ?
        """
        with self._connect_catalog() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._decorate_run(row) for row in rows]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        """Return one run's manifest, metrics, telemetry, games, files, and log."""
        if not self.catalog_path.exists():
            raise KeyError(f"Unknown run: {run_id}")
        with self._connect_catalog() as connection:
            row = connection.execute(
                """
                SELECT r.*,
                       (SELECT MAX(timestamp) FROM events e WHERE e.run_id = r.run_id)
                           AS last_event_at,
                       (SELECT payload_json FROM events t
                        WHERE t.run_id = r.run_id AND t.kind = 'telemetry'
                        ORDER BY t.sequence DESC LIMIT 1) AS telemetry_json
                FROM runs r WHERE r.run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            metric_rows = connection.execute(
                """
                SELECT sequence, timestamp, step, payload_json
                FROM events
                WHERE run_id = ? AND kind = 'metrics'
                ORDER BY sequence DESC LIMIT 2000
                """,
                (run_id,),
            ).fetchall()
            telemetry_rows = connection.execute(
                """
                SELECT sequence, timestamp, payload_json
                FROM events
                WHERE run_id = ? AND kind = 'telemetry'
                ORDER BY sequence DESC LIMIT 50
                """,
                (run_id,),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT sequence, kind, timestamp, step, payload_json
                FROM events
                WHERE run_id = ?
                ORDER BY sequence DESC LIMIT 200
                """,
                (run_id,),
            ).fetchall()
            games = _list_games(connection, run_id)
        run = self._decorate_run(row)
        run_path = self._catalog_path(row["path"])
        metrics = [_event_payload(item) for item in reversed(metric_rows)]
        telemetry = [_event_payload(item) for item in reversed(telemetry_rows)]
        events = [_timeline_event(item) for item in reversed(event_rows)]
        return {
            **run,
            "manifest": _load_json(row["manifest_json"]),
            "metrics": metrics,
            "metric_series": _metric_series(metrics),
            "telemetry": telemetry,
            "events": events,
            "games": games,
            "files": _walk_files(run_path, self.repository_root, limit=500),
            "log": _read_text_tail(run_path / "logs" / "run.log"),
        }

    def game_detail(self, game_id: str) -> dict[str, Any]:
        """Return a recorded game's summary, initial state, and replay events."""
        if not self.catalog_path.exists():
            raise KeyError(f"Unknown game: {game_id}")
        with self._connect_catalog() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='games'"
            ).fetchone()
            if not exists:
                raise KeyError(f"Unknown game: {game_id}")
            row = connection.execute(
                "SELECT * FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown game: {game_id}")
        payload = dict(row)
        payload["summary"] = _load_json(payload.pop("summary_json"))
        game_path = self._catalog_path(payload["path"])
        payload["path"] = str(game_path)
        payload["path_exists"] = game_path.is_dir()
        payload["initial_state"] = _read_json_file(game_path / "initial_state.json")
        payload["events"], payload["events_truncated"] = _read_jsonl(
            game_path / "events.jsonl",
            limit=2000,
        )
        return payload

    def inventory(self) -> dict[str, Any]:
        """Scan source files, configs, docs, entry points, and artifact size."""
        cached = self._cached("inventory", 20)
        if cached is not None:
            return cached
        area_counts: Counter[str] = Counter()
        extension_counts: Counter[str] = Counter()
        entrypoints: list[dict[str, Any]] = []
        config_files: list[str] = []
        source_files = 0
        source_bytes = 0
        for root, directories, filenames in os.walk(self.repository_root):
            directories[:] = [
                name for name in directories if name not in IGNORED_DIRECTORIES
            ]
            root_path = Path(root)
            for filename in filenames:
                path = root_path / filename
                relative = path.relative_to(self.repository_root)
                source_files += 1
                try:
                    source_bytes += path.stat().st_size
                except OSError:
                    continue
                area_counts[relative.parts[0] if len(relative.parts) > 1 else "root"] += 1
                extension_counts[path.suffix.lower() or "[none]"] += 1
                if path.suffix.lower() in {".yaml", ".yml", ".toml", ".ini"}:
                    config_files.append(relative.as_posix())
                if path.suffix == ".py" and _has_main_guard(path):
                    source = relative.as_posix()
                    entrypoints.append(
                        {
                            "source": source,
                            "registered": source in self.registered_sources,
                        }
                    )
        artifact_files, artifact_bytes = _directory_size(self.artifacts_root)
        payload = {
            "source_files": source_files,
            "source_bytes": source_bytes,
            "artifact_files": artifact_files,
            "artifact_bytes": artifact_bytes,
            "areas": dict(area_counts.most_common()),
            "extensions": dict(extension_counts.most_common(20)),
            "entrypoints": sorted(entrypoints, key=lambda item: item["source"]),
            "registered_entrypoints": sum(item["registered"] for item in entrypoints),
            "config_files": sorted(config_files),
        }
        self._store_cache("inventory", payload)
        return payload

    def git_snapshot(self) -> dict[str, Any]:
        """Return branch, commit, and working-tree changes without mutating Git."""
        cached = self._cached("git", 3)
        if cached is not None:
            return cached
        branch = self._git("branch", "--show-current")
        commit = self._git("rev-parse", "--short", "HEAD")
        changes = self._git("status", "--short").splitlines()
        payload = {
            "branch": branch or "detached",
            "commit": commit,
            "dirty": bool(changes),
            "change_count": len(changes),
            "changes": changes[:100],
        }
        self._store_cache("git", payload)
        return payload

    def system_snapshot(self) -> dict[str, Any]:
        """Return immediate host resource usage without a blocking sample."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.repository_root))
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "logical_cpus": psutil.cpu_count(logical=True),
            "memory_percent": memory.percent,
            "memory_used": memory.used,
            "memory_total": memory.total,
            "disk_percent": disk.percent,
            "disk_used": disk.used,
            "disk_total": disk.total,
        }

    def list_directory(self, scope: str, relative_path: str = "") -> dict[str, Any]:
        """List one approved repository subtree for the artifact browser."""
        root = self._scope_root(scope)
        path = self._safe_scope_path(root, relative_path)
        if not path.is_dir():
            raise NotADirectoryError(relative_path)
        entries: list[dict[str, Any]] = []
        for child in path.iterdir():
            if child.name in BROWSER_DENY or child.name.startswith(".trash-"):
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            child_relative = child.relative_to(root).as_posix()
            mime_type = mimetypes.guess_type(child.name)[0]
            entries.append(
                {
                    "name": child.name,
                    "path": child_relative,
                    "kind": "directory" if child.is_dir() else "file",
                    "size": stat.st_size if child.is_file() else None,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        timezone.utc,
                    ).isoformat(),
                    "mime_type": mime_type,
                    "previewable": _previewable(child, mime_type),
                }
            )
        entries.sort(key=lambda item: (item["kind"] != "directory", item["name"].lower()))
        return {
            "scope": scope,
            "path": path.relative_to(root).as_posix() if path != root else "",
            "entries": entries[:1000],
        }

    def read_text_file(self, scope: str, relative_path: str) -> dict[str, Any]:
        """Read a bounded approved text file for inline inspection."""
        path = self.resolve_view_file(scope, relative_path)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError("This file type is not available as text")
        if path.stat().st_size > 2_000_000:
            raise ValueError("Text preview is limited to 2 MB")
        return {
            "scope": scope,
            "path": relative_path,
            "content": path.read_text(encoding="utf-8", errors="replace"),
        }

    def resolve_view_file(self, scope: str, relative_path: str) -> Path:
        """Resolve one approved regular file for the HTTP viewer."""
        root = self._scope_root(scope)
        path = self._safe_scope_path(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        return path

    def _decorate_run(self, row: sqlite3.Row) -> dict[str, Any]:
        """Add derived liveness and path health without changing catalog state."""
        payload = dict(row)
        payload.pop("manifest_json", None)
        payload.pop("telemetry_json", None)
        telemetry = _load_json(row["telemetry_json"]) if row["telemetry_json"] else {}
        process_payload = telemetry.get("process", {}) if isinstance(telemetry, dict) else {}
        pid = process_payload.get("pid")
        process_alive = _pid_alive(pid)
        path = self._catalog_path(row["path"])
        stale = row["status"] == "running" and _is_stale(
            row["last_event_at"] or row["started_at"]
        )
        payload.update(
            {
                "path": str(path.relative_to(self.repository_root))
                if _is_within(path, self.repository_root)
                else str(path),
                "path_exists": path.exists(),
                "stale": stale,
                "telemetry": telemetry,
                "telemetry_pid": pid,
                "process_alive": process_alive,
                "effective_state": _effective_run_state(
                    row["status"], path.exists(), stale, process_alive
                ),
                "duration_seconds": _duration_seconds(
                    row["started_at"], row["ended_at"]
                ),
            }
        )
        return payload

    def _catalog_path(self, stored_path: str) -> Path:
        """Resolve catalog paths written as either relative or absolute strings."""
        path = Path(stored_path)
        return (self.repository_root / path).resolve() if not path.is_absolute() else path.resolve()

    def _connect_catalog(self) -> sqlite3.Connection:
        """Open a query-only connection compatible with concurrent WAL writers."""
        connection = sqlite3.connect(self.catalog_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _scope_root(self, scope: str) -> Path:
        """Resolve an approved browser scope by stable identifier."""
        roots = {
            "artifacts": self.artifacts_root,
            "docs": self.repository_root / "docs",
            "papers": self.repository_root / "papers",
            "repo": self.repository_root,
            "rules": self.repository_root / "rules",
        }
        try:
            return roots[scope].resolve()
        except KeyError as error:
            raise KeyError(f"Unknown browser scope: {scope}") from error

    def _safe_scope_path(self, root: Path, relative_path: str) -> Path:
        """Reject traversal and denied directories in browser paths."""
        requested = (root / relative_path).resolve(strict=False)
        if not _is_within(requested, root):
            raise PermissionError("Path traversal is not allowed")
        relative_parts = requested.relative_to(root).parts
        if any(part in BROWSER_DENY for part in relative_parts):
            raise PermissionError("This repository path is not exposed")
        return requested

    def _git(self, *arguments: str) -> str:
        """Run one short read-only Git query and return stripped output."""
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def _cached(self, key: str, lifetime: float) -> Any | None:
        """Return a fresh cached value or ``None`` when it should be rebuilt."""
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < lifetime:
                return cached[1]
        return None

    def _store_cache(self, key: str, value: Any) -> None:
        """Store a computed read model with a monotonic timestamp."""
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), value)

    def _health(
        self,
        catalog: dict[str, Any],
        inventory: dict[str, Any],
        system: dict[str, Any],
        git: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Translate operational conditions into actionable status cards."""
        checks: list[dict[str, str]] = []
        if catalog["stale_running"]:
            checks.append(
                _health_item(
                    "warning",
                    "Stale logger runs",
                    f"{catalog['stale_running']} catalog runs are still marked running without recent events.",
                )
            )
        if catalog["missing_paths"]:
            checks.append(
                _health_item(
                    "warning",
                    "Orphaned catalog paths",
                    f"{catalog['missing_paths']} run rows point to directories that no longer exist.",
                )
            )
        if system["memory_percent"] >= 90:
            checks.append(
                _health_item(
                    "danger",
                    "Memory pressure",
                    f"System memory usage is {system['memory_percent']:.1f}%.",
                )
            )
        if system["disk_percent"] >= 90:
            checks.append(
                _health_item(
                    "danger",
                    "Disk pressure",
                    f"Repository disk usage is {system['disk_percent']:.1f}%.",
                )
            )
        if git["dirty"]:
            checks.append(
                _health_item(
                    "info",
                    "Working tree has changes",
                    f"{git['change_count']} tracked or untracked paths differ from HEAD.",
                )
            )
        missing_models = _missing_enabled_models(self.repository_root)
        if missing_models:
            checks.append(
                _health_item(
                    "warning",
                    "Configured models are missing",
                    ", ".join(missing_models),
                )
            )
        unregistered = len(inventory["entrypoints"]) - inventory["registered_entrypoints"]
        if unregistered:
            checks.append(
                _health_item(
                    "info",
                    "Discovered entry points",
                    f"{unregistered} low-level or replaced entry points are visible in Repository discovery.",
                )
            )
        if not checks:
            checks.append(_health_item("ok", "Repository healthy", "No immediate issues detected."))
        return checks


def _empty_catalog_summary() -> dict[str, Any]:
    """Return the stable catalog shape before the first recorded run."""
    return {
        "total_runs": 0,
        "statuses": {},
        "games": {},
        "missing_paths": 0,
        "stale_running": 0,
        "catalog_bytes": 0,
    }


def _game_status_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Return game status totals when the adapter table exists."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='games'"
    ).fetchone()
    if not exists:
        return {}
    rows = connection.execute(
        "SELECT status, COUNT(*) AS count FROM games GROUP BY status"
    ).fetchall()
    return {row["status"]: row["count"] for row in rows}


def _list_games(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Return bounded game rows for one run when recording is available."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='games'"
    ).fetchone()
    if not exists:
        return []
    rows = connection.execute(
        """
        SELECT game_id, status, victory, bosses_defeated, turns,
               started_at, ended_at, path, summary_json
        FROM games WHERE run_id = ? ORDER BY started_at DESC LIMIT 500
        """,
        (run_id,),
    ).fetchall()
    games = []
    for row in rows:
        game = dict(row)
        game["summary"] = _load_json(game.pop("summary_json"))
        games.append(game)
    return games


def _event_payload(row: sqlite3.Row) -> dict[str, Any]:
    """Deserialize one metric or telemetry database row."""
    return {
        "sequence": row["sequence"],
        "timestamp": row["timestamp"],
        "step": row["step"] if "step" in row.keys() else None,
        "payload": _load_json(row["payload_json"]),
    }


def _timeline_event(row: sqlite3.Row) -> dict[str, Any]:
    """Deserialize a generic event row for run history display."""
    return {
        "sequence": row["sequence"],
        "kind": row["kind"],
        "timestamp": row["timestamp"],
        "step": row["step"],
        "payload": _load_json(row["payload_json"]),
    }


def _metric_series(metrics: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert numeric metric payloads into bounded browser chart series."""
    series: dict[str, list[dict[str, Any]]] = {}
    for event in metrics:
        for name, value in event["payload"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            points = series.setdefault(name, [])
            points.append(
                {
                    "step": event["step"],
                    "timestamp": event["timestamp"],
                    "value": value,
                }
            )
    return {name: points[-300:] for name, points in series.items()}


def _walk_files(run_path: Path, repository_root: Path, limit: int) -> list[dict[str, Any]]:
    """List bounded run files without following paths outside the repository."""
    if not run_path.is_dir() or not _is_within(run_path.resolve(), repository_root.resolve()):
        return []
    files: list[dict[str, Any]] = []
    for path in run_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            {
                "name": path.name,
                "path": path.relative_to(repository_root).as_posix(),
                "run_relative_path": path.relative_to(run_path).as_posix(),
                "size": stat.st_size,
                "mime_type": mimetypes.guess_type(path.name)[0],
            }
        )
        if len(files) >= limit:
            break
    return files


def _read_text_tail(path: Path, max_bytes: int = 96_000) -> str:
    """Read a bounded tail from an existing UTF-8-compatible log file."""
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        return stream.read().decode("utf-8", errors="replace")


def _has_main_guard(path: Path) -> bool:
    """Detect a top-level ``__main__`` guard with Python's parser."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
            return True
    return False


def _directory_size(root: Path) -> tuple[int, int]:
    """Return regular-file count and bytes beneath a directory."""
    if not root.exists():
        return 0, 0
    count = 0
    total = 0
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(current_root) / filename).stat().st_size
                count += 1
            except OSError:
                continue
    return count, total


def _load_json(value: str | None) -> Any:
    """Deserialize catalog JSON while tolerating legacy invalid or empty values."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {"raw": value}


def _read_json_file(path: Path) -> Any:
    """Read one optional JSON file while tolerating legacy serialization errors."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_jsonl(path: Path, limit: int) -> tuple[list[Any], bool]:
    """Read a bounded replay stream and report whether more rows were omitted."""
    if not path.is_file():
        return [], False
    events: list[Any] = []
    truncated = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if len(events) >= limit:
                truncated = True
                break
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events, truncated


def _is_stale(timestamp: str | None, threshold_seconds: int = 900) -> bool:
    """Return whether a run has emitted no recent durable event."""
    if not timestamp:
        return True
    try:
        observed = datetime.fromisoformat(timestamp)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - observed).total_seconds() > threshold_seconds


def _pid_alive(pid: Any) -> bool:
    """Return whether a telemetry PID currently names a live process."""
    try:
        return pid is not None and psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return False


def _effective_run_state(
    logger_status: str,
    path_exists: bool,
    stale: bool,
    process_alive: bool,
) -> str:
    """Combine logger status with external path and process evidence."""
    if not path_exists:
        return "orphaned"
    if logger_status != "running":
        return logger_status
    if process_alive:
        return "process-alive"
    if stale:
        return "stale"
    return "logger-running"


def _duration_seconds(started_at: str, ended_at: str | None) -> float:
    """Calculate run wall time with current time for unfinished runs."""
    try:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at) if ended_at else datetime.now(timezone.utc)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=timezone.utc)
        return max(0.0, (ended - started).total_seconds())
    except ValueError:
        return 0.0


def _previewable(path: Path, mime_type: str | None) -> bool:
    """Return whether the browser has a supported safe preview route."""
    if path.is_dir():
        return False
    if path.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
        return True
    return bool(mime_type and mime_type.startswith("image/"))


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by another path."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _health_item(level: str, title: str, message: str) -> dict[str, str]:
    """Build one stable health payload."""
    return {"level": level, "title": title, "message": message}


def _missing_enabled_models(repository_root: Path) -> list[str]:
    """Return missing learned-model paths enabled by the main report config."""
    config_path = repository_root / "config.yaml"
    if not config_path.exists():
        return []
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    agents = config.get("experimental_report", {}).get("agents", {})
    missing: list[str] = []
    for name, key in (("ppo", "model_path"), ("alphazero", "checkpoint_path")):
        settings = agents.get(name, {})
        path = settings.get("kwargs", {}).get(key)
        if settings.get("enabled") and path and not (repository_root / path).exists():
            missing.append(f"{name}: {path}")
    return missing
