import os
import argparse
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from sb3_contrib.ppo_mask import MaskablePPO
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
from solvers.callbacks import EpisodeLoggerCallback

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Regicide")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config["env"]
    ppo_cfg = config["ppo"]
    train_cfg = config["training"]

    print("Initializing environment...")
    # Apply environment configuration (for future expansion)
    raw_env = RegicideEnv(num_players=env_cfg.get("num_players", 1))
    env = NumericObsWrapper(raw_env)
    
    os.makedirs(train_cfg["save_dir"], exist_ok=True)
    os.makedirs(train_cfg["log_dir"], exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint_freq"],
        save_path=f'./{train_cfg["save_dir"]}/logs/',
        name_prefix='rl_model'
    )
    
    logger_callback = EpisodeLoggerCallback()
    callbacks = CallbackList([checkpoint_callback, logger_callback])
    
    from solvers.architecture import RegicideFeatureExtractor
    
    print(f"Initializing MaskablePPO agent on {ppo_cfg['device'].upper()}...")
    
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
        verbose=1,
        tensorboard_log=train_cfg["log_dir"],
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
        print(f"Loading pretrained weights from {pretrained_path}...")
        model.set_parameters(pretrained_path)
    
    print(f"Starting training for {train_cfg['total_timesteps']:,} timesteps...")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )
    
    final_model_path = os.path.join(train_cfg["save_dir"], train_cfg["model_name"])
    print(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)
    print("Training Complete!")
    
if __name__ == "__main__":
    main()
