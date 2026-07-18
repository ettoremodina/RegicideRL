import ast
from pathlib import Path

from agents.random_agent import RandomAgent
from ml_logger import start_run
from scripts.migrate_artifacts import migrate_legacy_artifacts
from solvers.parallel import ParallelSimulator

PROJECT_ROOT = Path(__file__).parents[1]
EXCLUDED_PARTS = {".agents", "archive", "artifacts", "venv", ".git"}


def test_application_code_does_not_use_print_or_direct_stdout():
    violations = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_print_call(node) or _is_direct_stream_access(node):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert violations == []


def _is_print_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    )


def _is_direct_stream_access(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr in {"stdout", "stderr"}
    )


def test_single_worker_simulation_records_each_game(tmp_path):
    context = start_run("parallel-test", root_dir=tmp_path)
    simulator = ParallelSimulator(n_jobs=1, run_context=context)
    try:
        metrics = simulator.run_eval(RandomAgent, {"name": "random"}, 3)
    finally:
        simulator.close()
    context.complete()

    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert len(context.catalog.list_games(context.run_id)) == 3


def test_legacy_migration_preserves_outputs_and_promotes_models(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "metrics.json").write_text("[]", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "agent.zip").write_bytes(b"model")

    moved = migrate_legacy_artifacts(tmp_path)

    assert len(moved) == 2
    assert (tmp_path / "artifacts" / "promoted_models" / "agent.zip").exists()
    legacy_metrics = list(
        (tmp_path / "artifacts" / "legacy").rglob("runs/metrics.json")
    )
    assert len(legacy_metrics) == 1
