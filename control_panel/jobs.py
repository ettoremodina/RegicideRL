"""Durable subprocess supervision for allowlisted control-panel jobs."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil

from .models import CommandSpec

logger = logging.getLogger(__name__)
ACTIVE_STATUSES = {"running", "detached", "stopping"}
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass
class JobRecord:
    """Persist the panel-owned lifecycle state of one child process."""

    job_id: str
    command_id: str
    title: str
    status: str
    argv: list[str]
    parameters: dict[str, Any]
    log_path: str
    started_at: str
    pid: int | None = None
    process_created_at: float | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible job representation."""
        return asdict(self)


class JobManager:
    """Launch, reconcile, monitor, and stop repository commands safely."""

    def __init__(
        self,
        repository_root: Path,
        state_root: Path,
        commands: Mapping[str, CommandSpec],
        python_executable: Path,
        max_concurrent: int = 4,
    ):
        self.repository_root = repository_root.resolve()
        self.state_root = state_root.resolve()
        self.commands = dict(commands)
        self.python_executable = python_executable.resolve()
        self.max_concurrent = max_concurrent
        self.jobs_root = self.state_root / "jobs"
        self.registry_path = self.state_root / "jobs.json"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs = self._load_jobs()
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._reconcile_loaded_jobs()

    def start(
        self,
        command_id: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and launch an allowlisted command as an isolated process."""
        with self._lock:
            self._refresh_locked()
            if self.active_count() >= self.max_concurrent:
                raise RuntimeError(
                    f"At most {self.max_concurrent} panel jobs may run concurrently"
                )
            try:
                command = self.commands[command_id]
            except KeyError as error:
                raise KeyError(f"Unknown command: {command_id}") from error
            argv = command.build_argv(
                self.python_executable,
                parameters,
                self.repository_root,
            )
            job_id = _new_job_id()
            job_dir = self.jobs_root / job_id
            job_dir.mkdir(parents=True)
            argv, environment = self._snapshot_environment(job_dir, argv, job_id)
            log_path = job_dir / "output.log"
            _write_log_header(log_path, command, argv)
            stream = log_path.open("a", encoding="utf-8", buffering=1)
            process = self._spawn(argv, environment, stream)
            record = JobRecord(
                job_id=job_id,
                command_id=command_id,
                title=command.title,
                status="running",
                argv=argv,
                parameters=dict(parameters),
                log_path=str(log_path.relative_to(self.repository_root)),
                started_at=_utc_now(),
                pid=process.pid,
                process_created_at=_process_created_at(process.pid),
            )
            self._jobs[job_id] = record
            self._processes[job_id] = process
            self._save_locked()
            watcher = threading.Thread(
                target=self._watch_process,
                args=(job_id, process, stream),
                name=f"control-panel-job-{job_id}",
                daemon=True,
            )
            watcher.start()
            logger.info("Started control-panel job %s: %s", job_id, command.title)
            return self._decorate(record)

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return newest jobs after reconciling their process state."""
        with self._lock:
            self._refresh_locked()
            records = sorted(
                self._jobs.values(),
                key=lambda item: item.started_at,
                reverse=True,
            )[:limit]
            return [self._decorate(record) for record in records]

    def get(self, job_id: str) -> dict[str, Any]:
        """Return one reconciled job or raise when its ID is unknown."""
        with self._lock:
            self._refresh_locked()
            try:
                return self._decorate(self._jobs[job_id])
            except KeyError as error:
                raise KeyError(f"Unknown job: {job_id}") from error

    def stop(self, job_id: str, force: bool = False) -> dict[str, Any]:
        """Interrupt a job and require an explicit second request for force-kill."""
        with self._lock:
            self._refresh_locked()
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job: {job_id}")
            if record.status not in ACTIVE_STATUSES:
                return self._decorate(record)
            process = self._matching_process(record)
            if process is None:
                record.status = "interrupted"
                record.ended_at = _utc_now()
                record.message = "The recorded process is no longer alive."
                self._save_locked()
                return self._decorate(record)
            record.status = "stopping"
            record.message = "Interrupt requested."
            self._save_locked()
        self._interrupt_process(job_id, process)
        if _wait_for_exit(process, 2.5):
            return self.get(job_id)
        if not force:
            with self._lock:
                record = self._jobs[job_id]
                record.status = "running" if job_id in self._processes else "detached"
                record.message = "Still running; force-stop requires confirmation."
                self._save_locked()
                decorated = self._decorate(record)
                decorated["requires_force"] = True
                return decorated
        self._terminate_tree(process)
        with self._lock:
            record = self._jobs[job_id]
            record.status = "stopped"
            record.ended_at = record.ended_at or _utc_now()
            record.message = "Process tree force-stopped by the user."
            self._save_locked()
            return self._decorate(record)

    def tail(self, job_id: str, max_bytes: int = 96_000) -> dict[str, Any]:
        """Return a bounded ANSI-free tail of a job's captured output."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown job: {job_id}")
            path = self.repository_root / record.log_path
            decorated = self._decorate(record)
        content = _read_tail(path, max_bytes)
        return {**decorated, "log": ANSI_ESCAPE.sub("", content)}

    def active_count(self) -> int:
        """Return the number of live or unresolved child processes."""
        return sum(record.status in ACTIVE_STATUSES for record in self._jobs.values())

    def _spawn(
        self,
        argv: list[str],
        environment: dict[str, str],
        stream: Any,
    ) -> subprocess.Popen[Any]:
        """Start a child without a shell or inherited interactive input."""
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        return subprocess.Popen(
            argv,
            cwd=self.repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )

    def _snapshot_environment(
        self,
        job_dir: Path,
        argv: list[str],
        job_id: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Freeze source configs so later edits cannot alter an active job."""
        snapshots = job_dir / "config-snapshots"
        snapshots.mkdir()
        replacements: dict[str, str] = {}
        for filename in ("config.yaml", "config_test.yaml", "logger_config.yaml"):
            source = self.repository_root / filename
            if not source.exists():
                continue
            destination = snapshots / filename
            shutil.copy2(source, destination)
            replacements[filename] = str(destination)
        rewritten = [replacements.get(argument, argument) for argument in argv]
        environment = os.environ.copy()
        environment.update(
            {
                "ML_LOGGER_ARTIFACTS_DIR": str(self.repository_root / "artifacts"),
                "ML_LOGGER_CONFIG": replacements.get(
                    "logger_config.yaml",
                    str(self.repository_root / "logger_config.yaml"),
                ),
                "PYTHONUNBUFFERED": "1",
                "REGICIDE_CONTROL_PANEL_JOB_ID": job_id,
            }
        )
        return rewritten, environment

    def _watch_process(
        self,
        job_id: str,
        process: subprocess.Popen[Any],
        stream: Any,
    ) -> None:
        """Finalize a tracked record when its child exits."""
        exit_code = process.wait()
        stream.close()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.status != "stopped":
                record.status = "completed" if exit_code == 0 else "failed"
                if record.status == "failed" and record.message.startswith("Interrupt"):
                    record.status = "stopped"
            record.exit_code = exit_code
            record.ended_at = _utc_now()
            self._processes.pop(job_id, None)
            self._save_locked()
        logger.info("Control-panel job %s exited with code %s", job_id, exit_code)

    def _interrupt_process(self, job_id: str, process: psutil.Process) -> None:
        """Send the closest platform equivalent of an interactive interrupt."""
        tracked = self._processes.get(job_id)
        try:
            if tracked is not None and os.name == "nt":
                tracked.send_signal(signal.CTRL_BREAK_EVENT)
            elif tracked is not None:
                os.killpg(os.getpgid(tracked.pid), signal.SIGINT)
            else:
                process.terminate()
        except (OSError, psutil.Error, subprocess.SubprocessError):
            logger.exception("Could not interrupt job %s cleanly", job_id)

    def _terminate_tree(self, process: psutil.Process) -> None:
        """Terminate then kill the verified child process tree."""
        try:
            descendants = process.children(recursive=True)
            for child in descendants:
                child.terminate()
            process.terminate()
            _, alive = psutil.wait_procs([*descendants, process], timeout=3)
            for remaining in alive:
                remaining.kill()
            psutil.wait_procs(alive, timeout=2)
        except psutil.NoSuchProcess:
            return

    def _matching_process(self, record: JobRecord) -> psutil.Process | None:
        """Return a PID only when its creation timestamp still matches."""
        if record.pid is None:
            return None
        try:
            process = psutil.Process(record.pid)
            if record.process_created_at is None:
                return process
            if abs(process.create_time() - record.process_created_at) > 0.01:
                return None
            return process
        except psutil.Error:
            return None

    def _decorate(self, record: JobRecord) -> dict[str, Any]:
        """Add current elapsed time and process telemetry to a job record."""
        payload = record.as_dict()
        process = self._matching_process(record)
        payload["active"] = record.status in ACTIVE_STATUSES and process is not None
        payload["elapsed_seconds"] = _elapsed_seconds(record)
        if process is not None:
            try:
                payload["process"] = {
                    "cpu_percent": process.cpu_percent(interval=None),
                    "memory_bytes": process.memory_info().rss,
                    "threads": process.num_threads(),
                }
            except psutil.Error:
                payload["process"] = None
        else:
            payload["process"] = None
        return payload

    def _refresh_locked(self) -> None:
        """Reconcile tracked and detached jobs without rewriting logger state."""
        changed = False
        for job_id, record in self._jobs.items():
            if record.status not in ACTIVE_STATUSES:
                continue
            tracked = self._processes.get(job_id)
            if tracked is not None:
                code = tracked.poll()
                if code is None:
                    continue
                record.exit_code = code
                record.ended_at = record.ended_at or _utc_now()
                record.status = "completed" if code == 0 else "failed"
                self._processes.pop(job_id, None)
                changed = True
                continue
            if self._matching_process(record) is not None:
                if record.status != "detached":
                    record.status = "detached"
                    record.message = "Process survived a previous panel session."
                    changed = True
                continue
            record.status = "interrupted"
            record.ended_at = record.ended_at or _utc_now()
            record.message = "Process ended while the panel was not attached."
            changed = True
        if changed:
            self._save_locked()

    def _reconcile_loaded_jobs(self) -> None:
        """Mark persisted live jobs as detached or interrupted on startup."""
        with self._lock:
            self._refresh_locked()

    def _load_jobs(self) -> dict[str, JobRecord]:
        """Load the durable registry, tolerating a missing first-run file."""
        if not self.registry_path.exists():
            return {}
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return {
                item["job_id"]: JobRecord(**item)
                for item in payload.get("jobs", [])
            }
        except (OSError, ValueError, TypeError):
            logger.exception("Could not load %s", self.registry_path)
            return {}

    def _save_locked(self) -> None:
        """Atomically persist all job records while the manager lock is held."""
        self.state_root.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".json.tmp")
        payload = {
            "version": 1,
            "jobs": [record.as_dict() for record in self._jobs.values()],
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.registry_path)


def _new_job_id() -> str:
    """Return a sortable collision-resistant panel job identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"panel-{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    """Return the current UTC time in ISO 8601 form."""
    return datetime.now(timezone.utc).isoformat()


def _process_created_at(pid: int) -> float | None:
    """Capture a process creation timestamp for PID-reuse protection."""
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _write_log_header(path: Path, command: CommandSpec, argv: list[str]) -> None:
    """Create a readable provenance header before child output begins."""
    path.write_text(
        "\n".join(
            (
                f"Control Panel job: {command.title}",
                f"Started: {_utc_now()}",
                f"Source: {command.source}",
                f"Command: {subprocess.list2cmdline(argv)}",
                "-" * 80,
                "",
            )
        ),
        encoding="utf-8",
    )


def _read_tail(path: Path, max_bytes: int) -> str:
    """Read only the newest bytes from a potentially large output log."""
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        content = stream.read()
    return content.decode("utf-8", errors="replace")


def _wait_for_exit(process: psutil.Process, timeout: float) -> bool:
    """Return whether a process exits within the requested short timeout."""
    try:
        process.wait(timeout=timeout)
        return True
    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
        return not process.is_running()


def _elapsed_seconds(record: JobRecord) -> float:
    """Calculate elapsed wall time for active and completed records."""
    started = datetime.fromisoformat(record.started_at)
    ended = (
        datetime.fromisoformat(record.ended_at)
        if record.ended_at
        else datetime.now(timezone.utc)
    )
    return max(0.0, (ended - started).total_seconds())
