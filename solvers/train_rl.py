import os
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from sb3_contrib.ppo_mask import MaskablePPO
from ml_logger import GameRecorder, get_logger, start_run
from solvers.config import load_config
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
from solvers.callbacks import EpisodeLoggerCallback

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Regicide")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config["env"]
    ppo_cfg = config["ppo"]
    train_cfg = config["training"]
    context = start_run("ppo-training", config=config)
    recorder = GameRecorder(context)

    logger.info("Initializing environment")
    # Apply environment configuration (for future expansion)
    raw_env = RegicideEnv(
        num_players=env_cfg.get("num_players", 1),
        recorder=recorder,
    )
    env = NumericObsWrapper(raw_env)
    
    checkpoint_dir = context.run_dir / "checkpoints"
    tensorboard_dir = context.run_dir / "metrics" / "tensorboard"
    tensorboard_log = None
    if context.saving_enabled("metrics"):
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(tensorboard_dir)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint_freq"],
        save_path=str(checkpoint_dir),
        name_prefix='rl_model'
    )
    
    logger_callback = EpisodeLoggerCallback()
    callbacks = CallbackList([checkpoint_callback, logger_callback])
    
    from solvers.architecture import RegicideFeatureExtractor
    
    logger.info("Initializing MaskablePPO agent on %s", ppo_cfg["device"].upper())
    
    # Custom architecture kwargs
    policy_kwargs = dict(
        features_extractor_class=RegicideFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=ppo_cfg.get("net_arch", [128, 128]), vf=ppo_cfg.get("net_arch", [128, 128]))
    )
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=ppo_cfg["device"],
        verbose=0,
        tensorboard_log=tensorboard_log,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        policy_kwargs=policy_kwargs
    )
    
    pretrained_path = ppo_cfg.get("pretrained_model_path", None)
    if pretrained_path and os.path.exists(pretrained_path + ".zip"):
        logger.info("Loading pretrained weights from %s", pretrained_path)
        model.set_parameters(pretrained_path)
    
    logger.info(
        "Starting training for %s timesteps",
        f"{train_cfg['total_timesteps']:,}",
    )
    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=False,
        )
        final_model_path = context.run_dir / "models" / train_cfg["model_name"]
        logger.info("Saving final model to %s", final_model_path)
        model.save(final_model_path)
        context.complete({"final_model": str(final_model_path) + ".zip"})
        logger.info("Training complete")
    except Exception as error:
        context.fail(error)
        logger.exception("Training failed")
        raise
    
if __name__ == "__main__":
    main()
