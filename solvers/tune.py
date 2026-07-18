import os
import optuna
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from solvers.env import RegicideEnv
from solvers.config import load_config
from solvers.wrappers import NumericObsWrapper
from solvers.callbacks import EpisodeLoggerCallback
from solvers.analysis.probe import probe_policy

def objective(trial, config):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.05, log=True)
    net_arch = config["ppo"].get("net_arch", [256, 256])
    
    
    # 2. Setup Environment
    raw_env = RegicideEnv(num_players=config["env"].get("num_players", 1))
    env = NumericObsWrapper(raw_env)
    
    # 3. Initialize Model
    # Use a temporary log directory for the trial
    trial_log_dir = os.path.join(config["training"]["log_dir"], f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    
    from solvers.architecture import RegicideFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=RegicideFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=net_arch, vf=net_arch)
    )
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=config["ppo"]["device"],
        verbose=0, # Keep it quiet during tuning
        tensorboard_log=trial_log_dir,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs
    )
    
    pretrained_path = config["ppo"].get("pretrained_model_path", None)
    if pretrained_path and os.path.exists(pretrained_path + ".zip"):
        # Load weights, matching architecture is required
        model.set_parameters(pretrained_path)
    
    # 4. Train
    timesteps = config["tuning"].get("timesteps_per_trial", 100000)
    logger_callback = EpisodeLoggerCallback()
    
    try:
        model.learn(total_timesteps=timesteps, callback=logger_callback)
    except Exception as e:
        print(f"Trial {trial.number} failed during training: {e}")
        return 0.0 # Return poor score if it crashes
        
    # 5. Evaluate (using the prober instead of standard eval)
    # Save temp model for probing
    temp_model_path = os.path.join(trial_log_dir, "temp_model")
    model.save(temp_model_path)
    
    probe_results = probe_policy(temp_model_path + ".zip", num_games=50)
    
    if not probe_results:
        return 0.0
        
    # Our objective is to maximize the mean number of bosses killed
    bosses = probe_results.get('bosses_killed', [])
    mean_bosses = float(np.mean(bosses)) if bosses else 0.0
    
    # Custom score: bosses killed minus a penalty for bad habits
    scenarios = probe_results.get('scenarios', {})
    over_defense = scenarios.get('over_defense', 0)
    wasted_jester = scenarios.get('wasted_jester', 0)
    
    penalty = (over_defense * 0.1) + (wasted_jester * 0.1)
    
    return mean_bosses - penalty

def run_tuner(config_path="config.yaml"):
    config = load_config(config_path)
    n_trials = config["tuning"].get("n_trials", 20)
    
    print(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)
    
    print("\n--- Tuning Complete ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score (Mean Bosses Killed): {study.best_value}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Ask the user if they want to save these to config.yaml
    print("\nUpdate config.yaml with these parameters for your full training run!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune MaskablePPO on Regicide")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_tuner(args.config)
