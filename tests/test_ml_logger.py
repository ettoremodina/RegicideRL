import json
from ml_logger import (
    GameRecorder,
    RunCatalog,
    configure_logging,
    get_logger,
    start_run,
)
from solvers.env import RegicideEnv


def test_run_context_saves_manifest_log_metrics_and_result(tmp_path):
    context = start_run("test", root_dir=tmp_path, config={"seed": 42})
    get_logger("test.module").info("persisted message")
    context.log_metrics(1, {"win_rate": 0.5})
    result_path = context.save_result("summary.json", {"ok": True})
    context.complete({"games": 2})

    manifest = json.loads(
        (context.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert "persisted message" in (
        context.run_dir / "logs" / "run.log"
    ).read_text(encoding="utf-8")
    assert result_path.exists()
    assert (context.run_dir / "metrics" / "metrics.jsonl").exists()

    catalog = RunCatalog(tmp_path / "catalog.sqlite")
    assert catalog.get_run(context.run_id)["status"] == "completed"


def test_environment_records_a_replayable_game(tmp_path):
    context = start_run("game-test", root_dir=tmp_path)
    recorder = GameRecorder(context, recording_level="full")
    environment = RegicideEnv(num_players=1, recorder=recorder)
    observation, _ = environment.reset(seed=7)

    for _ in range(500):
        valid_actions = [
            index
            for index, is_valid in enumerate(observation["action_mask"])
            if is_valid
        ]
        if not valid_actions:
            break
        observation, _, terminated, truncated, _ = environment.step(valid_actions[0])
        if terminated or truncated:
            break

    games = context.catalog.list_games(context.run_id)
    assert len(games) == 1
    assert games[0]["status"] == "completed"
    game_dir = context.run_dir / "games" / games[0]["game_id"]
    assert (game_dir / "initial_state.json").exists()
    assert (game_dir / "events.jsonl").exists()
    assert (game_dir / "summary.json").exists()
    configure_logging()
