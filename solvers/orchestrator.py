"""Run PPO training followed by analysis under canonical artifact storage."""

import argparse
import subprocess
import sys
from pathlib import Path

from ml_logger import RunCatalog, get_logger, start_run
from solvers.analysis.run_analysis import run_analysis_pipeline
from solvers.config import load_config

logger = get_logger(__name__)


def run_experiment(config_path="config.yaml"):
    config = load_config(config_path)
    context = start_run(
        "experiment",
        name="ppo-training-and-analysis",
        config=config,
    )
    catalog = RunCatalog(context.root_dir / "catalog.sqlite")
    known_runs = {row["run_id"] for row in catalog.list_runs(100_000)}
    try:
        logger.info("Starting PPO training phase")
        subprocess.run(
            [sys.executable, "-m", "solvers.train_rl", "--config", config_path],
            check=True,
        )
        training_run = _find_new_training_run(catalog, known_runs)
        model_path = _find_model(training_run)
        logger.info("Starting analysis phase for %s", model_path)
        success = run_analysis_pipeline(
            model_path=str(model_path),
            num_games=100,
            out_dir=context.run_dir / "analysis",
            run_context=context,
        )
        if not success:
            raise RuntimeError("Policy analysis failed")
        context.complete(
            {
                "training_run_id": training_run["run_id"],
                "model_path": str(model_path),
            }
        )
        logger.info("Experiment completed in %s", context.run_dir)
        return context
    except Exception as error:
        context.fail(error)
        logger.exception("Experiment failed")
        raise


def _find_new_training_run(catalog, known_runs):
    candidates = [
        row
        for row in catalog.list_runs(100_000)
        if row["run_id"] not in known_runs and row["run_type"] == "ppo-training"
    ]
    if not candidates:
        raise RuntimeError("Training completed without registering a PPO run")
    return candidates[0]


def _find_model(training_run):
    models = sorted(Path(training_run["path"]).glob("models/*.zip"))
    if not models:
        raise FileNotFoundError(
            f"No trained model found in run {training_run['run_id']}"
        )
    return models[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PPO experiment pipeline")
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    run_experiment(arguments.config)
