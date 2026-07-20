"""Verify control-panel command safety, config writes, jobs, and HTTP access."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_panel.configuration import ConfigurationService
from control_panel.jobs import JobManager
from control_panel.models import CommandSpec, ParameterSpec
from control_panel.registry import command_map
from control_panel.repository import RepositoryService
from control_panel.server import ControlPanelApplication, ControlPanelServer


def test_command_registry_builds_list_argv_and_rejects_unknown_parameters(tmp_path):
    """Keep browser values in list-form argv with a closed parameter schema."""
    (tmp_path / "artifacts").mkdir()
    command = command_map()["benchmark"]

    argv = command.build_argv(
        Path(sys.executable),
        {"mode": "normal", "games": 12, "steps": 30, "jobs": 2},
        tmp_path,
    )

    assert argv[0] == str(Path(sys.executable))
    assert argv[1:] == [
        "benchmark.py",
        "--mode",
        "normal",
        "--games",
        "12",
        "--steps",
        "30",
        "--jobs",
        "2",
    ]
    with pytest.raises(ValueError, match="Unsupported parameters"):
        command.build_argv(Path(sys.executable), {"shell": "whoami"}, tmp_path)


def test_artifact_output_parameter_rejects_repository_escape(tmp_path):
    """Prevent output forms from writing outside the canonical artifact tree."""
    parameter = ParameterSpec(
        "output",
        "Output",
        "--out",
        kind="path",
        path_mode="artifact_output",
    )

    with pytest.raises(ValueError, match="outside the repository"):
        parameter.normalize(str(tmp_path.parent / "escape.npz"), tmp_path)
    with pytest.raises(ValueError, match="inside artifacts"):
        parameter.normalize("models/escape.npz", tmp_path)
    normalized = parameter.normalize("artifacts/datasets/safe.npz", tmp_path)
    assert normalized.replace("\\", "/") == "artifacts/datasets/safe.npz"


def test_configuration_save_is_validated_atomic_and_backed_up(tmp_path):
    """Save only valid YAML while retaining the exact previous source text."""
    original = _main_config("cpu")
    (tmp_path / "config.yaml").write_text(original, encoding="utf-8")
    state_root = tmp_path / "artifacts" / "control_panel"
    service = ConfigurationService(tmp_path, state_root)
    current = service.read("main")
    proposed = _main_config("cuda")

    preview = service.preview("main", proposed, current["sha256"])
    result = service.save("main", proposed, current["sha256"])

    assert preview["valid"] is True
    assert "device: cpu" in preview["diff"]
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == proposed
    assert result["backup"]
    backup = tmp_path / result["backup"]
    assert backup.read_text(encoding="utf-8") == original
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        service.preview("main", proposed, current["sha256"])


def test_logger_configuration_validation_checks_regex_and_intervals(tmp_path):
    """Catch logger values that the existing partial loader does not validate."""
    (tmp_path / "logger_config.yaml").write_text(
        """
dashboard:
  mode: off
  refresh_rate: 0
telemetry:
  sample_interval_sec: -1
highlights:
  - pattern: "["
""",
        encoding="utf-8",
    )
    service = ConfigurationService(tmp_path, tmp_path / "artifacts" / "panel")
    current = service.read("logger")

    preview = service.preview("logger", current["text"], current["sha256"])

    assert preview["valid"] is False
    assert any("refresh_rate" in error for error in preview["errors"])
    assert any("sample_interval" in error for error in preview["errors"])
    assert any("pattern is invalid" in error for error in preview["errors"])


def test_job_manager_captures_output_and_persists_completion(tmp_path):
    """Run one isolated child and retain its lifecycle and captured output."""
    command = CommandSpec(
        "probe",
        "Probe",
        "Quality",
        "Emit one line and exit.",
        ("-c", "print('panel-ready')"),
        creates_run=False,
    )
    manager = JobManager(
        tmp_path,
        tmp_path / "artifacts" / "control_panel",
        {"probe": command},
        Path(sys.executable),
    )

    started = manager.start("probe", {})
    completed = _wait_for_job(manager, started["job_id"])
    output = manager.tail(started["job_id"])
    reloaded = JobManager(
        tmp_path,
        tmp_path / "artifacts" / "control_panel",
        {"probe": command},
        Path(sys.executable),
    ).get(started["job_id"])

    assert completed["status"] == "completed"
    assert completed["exit_code"] == 0
    assert "panel-ready" in output["log"]
    assert reloaded["status"] == "completed"


def test_http_api_requires_ephemeral_header_token(tmp_path):
    """Keep repository data and mutations inaccessible to tokenless requests."""
    application = ControlPanelApplication(
        tmp_path,
        tmp_path / "artifacts" / "control_panel",
        Path(sys.executable),
        token="test-token",
    )
    server = ControlPanelServer(("127.0.0.1", 0), application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/bootstrap"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=3)
        request = Request(url, headers={"X-Control-Token": "test-token"})
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert error.value.code == 403
    assert payload["project"]["name"] == "Regicide AI"


def test_recorded_game_detail_reads_bounded_replay_files(tmp_path):
    """Replace the catalog replay CLI with a safe structured read model."""
    artifacts = tmp_path / "artifacts"
    game_dir = artifacts / "runs" / "date" / "run" / "games" / "game-1"
    game_dir.mkdir(parents=True)
    (game_dir / "initial_state.json").write_text('{"turn": 0}', encoding="utf-8")
    (game_dir / "events.jsonl").write_text(
        '{"sequence": 1, "action": {"kind": "play"}}\n',
        encoding="utf-8",
    )
    with sqlite3.connect(artifacts / "catalog.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY, run_id TEXT, status TEXT,
                victory INTEGER, bosses_defeated INTEGER, turns INTEGER,
                started_at TEXT, ended_at TEXT, path TEXT, summary_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "game-1",
                "run-1",
                "completed",
                1,
                12,
                9,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:01:00+00:00",
                str(game_dir.relative_to(tmp_path)),
                '{"victory": true}',
            ),
        )
    service = RepositoryService(tmp_path, set())

    game = service.game_detail("game-1")

    assert game["initial_state"] == {"turn": 0}
    assert game["events"][0]["action"]["kind"] == "play"
    assert game["summary"]["victory"] is True


def _wait_for_job(manager: JobManager, job_id: str) -> dict:
    """Poll one short test child without depending on timing-sensitive sleeps."""
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record["status"] not in {"running", "detached", "stopping"}:
            return record
        time.sleep(0.05)
    raise AssertionError("Control-panel test job did not finish")


def _main_config(device: str) -> str:
    """Return a minimal complete main configuration for write tests."""
    return f"""
env:
  num_players: 1
ppo:
  device: {device}
training:
  total_timesteps: 1
tuning:
  n_trials: 1
alphazero:
  device: {device}
experimental_report:
  protocol: {{games_per_agent: 1}}
  agents: {{}}
"""
