"""Run PPO training followed by analysis under canonical artifact storage."""

import argparse
import subprocess
import sys
from pathlib import Path

from ml_logger import RunCatalog, get_logger, run_scope
from solvers.analysis.run_analysis import run_analysis_pipeline
from solvers.config import load_config

logger = get_logger(__name__)


def run_experiment(config_path="config.yaml"):
    """Train PPO in a child process and analyze the newly registered model.

    Args:
        config_path: Solver YAML shared by training and analysis.

    Returns:
        Parent experiment context linking the training run and analysis output.

    Raises:
        RuntimeError: If training registers no run or policy analysis fails.
        FileNotFoundError: If the training run contains no model.
    """
    config = load_config(config_path)
    with run_scope(
        "experiment",
        name="ppo-training-and-analysis",
        config=config,
    ) as context:
        catalog = RunCatalog(context.root_dir / "catalog.sqlite")
        known_runs = {row["run_id"] for row in catalog.list_runs(100_000)}
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
        context.log_summary(
            {
                "training_run_id": training_run["run_id"],
                "model_path": str(model_path),
            }
        )
        logger.info("Experiment completed in %s", context.run_dir)
        return context


def _find_new_training_run(catalog, known_runs):
    """Locate the PPO run registered after the parent experiment started."""
    candidates = [
        row
        for row in catalog.list_runs(100_000)
        if row["run_id"] not in known_runs and row["run_type"] == "ppo-training"
    ]
    if not candidates:
        raise RuntimeError("Training completed without registering a PPO run")
    return candidates[0]


def _find_model(training_run):
    """Return the last model archive stored by a completed training run."""
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
