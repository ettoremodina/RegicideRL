"""Application startup, single-instance handling, and browser launch."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import psutil

from ml_logger import get_logger

from .server import ControlPanelApplication, ControlPanelServer

logger = get_logger(__name__)
DEFAULT_PORT = 8765


def run_control_panel(
    repository_root: str | Path | None = None,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> str:
    """Start or focus the repository's local-only browser control panel."""
    root = Path(repository_root or Path(__file__).parents[1]).resolve()
    state_root = root / "artifacts" / "control_panel"
    state_root.mkdir(parents=True, exist_ok=True)
    _configure_panel_logging(state_root)
    state_path = state_root / "server.json"
    existing_url = _existing_server_url(state_path)
    if existing_url:
        if open_browser:
            webbrowser.open(existing_url, new=1)
        logger.info("Focused existing control panel at %s", existing_url)
        return existing_url
    python_executable = _child_python_executable()
    application = ControlPanelApplication(
        repository_root=root,
        state_root=state_root,
        python_executable=python_executable,
    )
    server = _bind_server(application, port)
    host, bound_port = server.server_address[:2]
    url = f"http://{host}:{bound_port}/"
    _write_server_state(state_path, url)
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url, new=1)).start()
    logger.info("Regicide control panel started at %s", url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logger.info("Control panel interrupted")
    finally:
        server.server_close()
        _remove_owned_state(state_path)
        logger.info("Control panel stopped; child jobs continue independently")
    return url


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse the optional developer CLI while keeping double-click defaults."""
    parser = argparse.ArgumentParser(description="Regicide browser control panel")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    options = parser.parse_args(arguments)
    run_control_panel(port=options.port, open_browser=not options.no_browser)


def _configure_panel_logging(state_root: Path) -> None:
    """Use the logger facade without creating or polluting experiment runs."""
    package_logger = logging.getLogger("control_panel")
    if any(getattr(handler, "_control_panel_owned", False) for handler in package_logger.handlers):
        return
    handler = logging.handlers.RotatingFileHandler(
        state_root / "panel.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    setattr(handler, "_control_panel_owned", True)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    )
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(handler)
    package_logger.propagate = False


def _bind_server(
    application: ControlPanelApplication,
    requested_port: int,
) -> ControlPanelServer:
    """Bind the preferred localhost port and fall back to an ephemeral port."""
    try:
        return ControlPanelServer(("127.0.0.1", requested_port), application)
    except OSError:
        if requested_port == 0:
            raise
        logger.warning("Port %s is busy; selecting a free local port", requested_port)
        return ControlPanelServer(("127.0.0.1", 0), application)


def _child_python_executable() -> Path:
    """Select the available interpreter with the broadest repository capability."""
    executable = Path(sys.executable).resolve()
    current_console = (
        executable.with_name("python.exe")
        if executable.name.lower() == "pythonw.exe"
        else executable
    )
    candidates = [current_console]
    discovered = shutil.which("python")
    if discovered:
        candidates.append(Path(discovered).resolve())
    repository_python = Path(__file__).parents[1] / "venv" / "Scripts" / "python.exe"
    if repository_python.exists():
        candidates.append(repository_python.resolve())
    unique = list(dict.fromkeys(candidates))
    scored = [(candidate, _interpreter_score(candidate)) for candidate in unique]
    usable = [item for item in scored if item[1] >= 2]
    if not usable:
        return current_console
    return max(usable, key=lambda item: item[1])[0]


def _interpreter_score(executable: Path) -> int:
    """Count repository capabilities without importing heavy packages in the panel."""
    modules = (
        "yaml",
        "psutil",
        "pygame",
        "stable_baselines3",
        "sb3_contrib",
        "pandas",
        "scipy",
        "optuna",
        "pdoc",
        "pytest",
    )
    script = (
        "import importlib.util; "
        f"print(sum(importlib.util.find_spec(name) is not None for name in {modules!r}))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _existing_server_url(state_path: Path) -> str | None:
    """Return a prior live instance URL after PID creation-time verification."""
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        process = psutil.Process(int(state["pid"]))
        if abs(process.create_time() - float(state["process_created_at"])) > 0.01:
            return None
        if not process.is_running():
            return None
        return str(state["url"])
    except (OSError, ValueError, KeyError, psutil.Error):
        return None


def _write_server_state(state_path: Path, url: str) -> None:
    """Atomically persist enough identity to focus the running singleton."""
    process = psutil.Process(os.getpid())
    payload = {
        "pid": process.pid,
        "process_created_at": process.create_time(),
        "url": url,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(state_path)


def _remove_owned_state(state_path: Path) -> None:
    """Remove only a singleton file that still belongs to this process."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("pid", -1)) == os.getpid():
            state_path.unlink()
    except (OSError, ValueError):
        return


if __name__ == "__main__":
    main()
