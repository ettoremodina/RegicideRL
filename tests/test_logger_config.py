from pathlib import Path

from integrations.regicide_logging import GameRecorder
from ml_logger import configure_logging, start_run
from ml_logger.configs.config_loader import load_config
from ml_logger.views.selection import matches_metric
from solvers.env import RegicideEnv


def test_project_config_disables_game_recording_for_benchmark(tmp_path):
    benchmark_settings = load_config(run_type="benchmark")
    game_settings = load_config(run_type="game")
    context = start_run("benchmark", root_dir=tmp_path)
    recorder = GameRecorder(context)

    assert _recording(benchmark_settings)["enabled"] is False
    assert _recording(game_settings)["enabled"] is True
    assert recorder.enabled is False
    context.complete()
    configure_logging()


def test_project_dashboard_includes_live_alphazero_metrics():
    filters = load_config(run_type="alphazero")["dashboard"]["metrics"]

    assert matches_metric("self_play/games_completed", filters)
    assert matches_metric("train/total_loss", filters)
    assert matches_metric("eval/win_rate", filters)


def test_disabled_optional_artifacts_are_not_written(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
logging:
  console: false
  file: false
saving:
  enabled: false
games:
  enabled: false
""",
    )
    context = start_run(
        "configured-test",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    )
    recorder = GameRecorder(context)
    environment = RegicideEnv(num_players=1, recorder=recorder)
    environment.reset(seed=11)
    context.log_metrics(1, {"value": 2})
    result_path = context.save_result("disabled.json", {"saved": False})
    context.complete()

    assert recorder.enabled is False
    assert not (context.run_dir / "logs" / "run.log").exists()
    assert not (context.run_dir / "metrics" / "metrics.jsonl").exists()
    assert not result_path.exists()
    assert not (context.run_dir / "games").exists()
    assert (context.run_dir / "manifest.json").exists()
    configure_logging()


def test_recording_level_is_loaded_from_config(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
logging:
  console: false
games:
  enabled: true
  recording_level: summary
""",
    )
    context = start_run(
        "summary-test",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    )
    recorder = GameRecorder(context)

    assert recorder.enabled is True
    assert recorder.recording_level == "summary"
    configure_logging()


def _write_config(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "logger_config.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def _recording(settings):
    return settings["integrations"]["regicide"]["recording"]
