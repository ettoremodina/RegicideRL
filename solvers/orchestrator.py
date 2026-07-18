import os
import shutil
import argparse
from datetime import datetime
import subprocess

from solvers.config import load_config
from solvers.analysis.run_analysis import run_analysis_pipeline
from solvers.analysis.tb_extractor import find_latest_run

def run_experiment(config_path="config.yaml"):
    # 1. Create a timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", f"run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Starting new experiment: {experiment_dir}")
    
    # 2. Backup config
    shutil.copy(config_path, os.path.join(experiment_dir, "config.yaml"))
    
    config = load_config(config_path)
    model_name = config["training"]["model_name"]
    save_dir = config["training"]["save_dir"]
    model_path = os.path.join(save_dir, f"{model_name}.zip")

    # 3. Run training
    print("\n--- Phase 1: Training ---")
    try:
        # Run training as a subprocess to keep memory clean or just import and run
        subprocess.run(["python", "-m", "solvers.train_rl", "--config", config_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return
        
    # 4. Run Analysis
    print("\n--- Phase 2: Analysis ---")
    latest_tb = find_latest_run(config["training"]["log_dir"])
    run_analysis_pipeline(
        model_path=model_path,
        num_games=100, # Hardcoded evaluation amount or can be in config
        logdir=latest_tb,
        out_dir=experiment_dir
    )
    
    print(f"\nExperiment finished successfully. All artifacts saved to: {experiment_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an end-to-end RL experiment pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_experiment(args.config)
