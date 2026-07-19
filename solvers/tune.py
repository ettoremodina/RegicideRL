"""Optuna hyperparameter tuning with canonical artifacts and game recording."""

import argparse
from pathlib import Path

import numpy as np
import optuna
from sb3_contrib.ppo_mask import MaskablePPO

from integrations.regicide_logging import GameRecorder
from ml_logger import RunContext, get_logger, run_scope
from solvers.analysis.probe import probe_policy
from solvers.architecture import RegicideFeatureExtractor
from solvers.callbacks import EpisodeLoggerCallback
from solvers.config import load_config
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

logger = get_logger(__name__)


def objective(trial, config, context: RunContext, recorder):
    """Train and score one Optuna hyperparameter trial.

    Failed training trials return zero so a single numerical failure does not
    terminate the complete study.
    """
    hyperparameters = _suggest_hyperparameters(trial, config)
    trial_dir = context.run_dir / "checkpoints" / f"trial-{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    environment = NumericObsWrapper(
        RegicideEnv(
            num_players=config["env"].get("num_players", 1),
            recorder=recorder,
        )
    )
    model = _build_model(
        environment,
        config,
        hyperparameters,
        trial_dir,
        tensorboard_enabled=context.saving_enabled("metrics"),
    )
    pretrained_path = config["ppo"].get("pretrained_model_path")
    if pretrained_path and Path(pretrained_path + ".zip").exists():
        model.set_parameters(pretrained_path)
    try:
        model.learn(
            total_timesteps=config["tuning"].get("timesteps_per_trial", 100000),
            callback=EpisodeLoggerCallback(),
            progress_bar=False,
        )
    except Exception:
        logger.exception("Trial %d failed during training", trial.number)
        return 0.0
    model_path = trial_dir / "model"
    model.save(model_path)
    probe_results = probe_policy(
        str(model_path) + ".zip",
        num_games=50,
        recorder=recorder,
    )
    score = _score_probe(probe_results)
    context.log_metrics(trial.number, {"trial_score": score, **hyperparameters})
    return score


def _suggest_hyperparameters(trial, config):
    """Sample the PPO search space while preserving configured layer widths."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.00001, 0.05, log=True),
        "net_arch": config["ppo"].get("net_arch", [256, 256]),
    }


def _build_model(
    environment,
    config,
    hyperparameters,
    trial_dir,
    tensorboard_enabled,
):
    """Create a trial-specific MaskablePPO model and TensorBoard destination."""
    architecture = hyperparameters.pop("net_arch")
    policy_kwargs = {
        "features_extractor_class": RegicideFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": architecture, "vf": architecture},
    }
    model = MaskablePPO(
        "MultiInputPolicy",
        environment,
        device=config["ppo"]["device"],
        verbose=0,
        tensorboard_log=(
            str(trial_dir / "tensorboard") if tensorboard_enabled else None
        ),
        policy_kwargs=policy_kwargs,
        **hyperparameters,
    )
    hyperparameters["net_arch"] = architecture
    return model


def _score_probe(probe_results):
    """Score progress while penalizing conspicuous resource-wasting actions."""
    if not probe_results:
        return 0.0
    bosses = probe_results.get("bosses_killed", [])
    mean_bosses = float(np.mean(bosses)) if bosses else 0.0
    scenarios = probe_results.get("scenarios", {})
    penalty = 0.1 * (
        scenarios.get("over_defense", 0)
        + scenarios.get("wasted_jester", 0)
    )
    return mean_bosses - penalty


def run_tuner(config_path="config.yaml"):
    """Execute the configured Optuna study and persist its best trial."""
    config = load_config(config_path)
    with run_scope("hyperparameter-tuning", config=config) as context:
        recorder = GameRecorder(context)
        number_of_trials = config["tuning"].get("n_trials", 20)
        logger.info(
            "Starting Optuna optimization with %d trials",
            number_of_trials,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, config, context, recorder),
            n_trials=number_of_trials,
        )
        result = {
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
        }
        output = context.save_result("tuning.json", result)
        context.log_summary({"result": str(output), **result})
        logger.info(
            "Tuning complete: best trial=%d score=%.4f params=%s",
            study.best_trial.number,
            study.best_value,
            study.best_params,
        )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune MaskablePPO on Regicide")
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    run_tuner(arguments.config)
