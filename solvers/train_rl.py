import os
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.ppo_mask import MaskablePPO
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Regicide")
    parser.add_argument("--timesteps", type=int, default=5000000, help="Total timesteps to train (default: 5,000,000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (default: cpu)")
    parser.add_argument("--name", type=str, default="ppo_regicide_v2", help="Model save name")
    args = parser.parse_args()

    print("Initializing environment...")
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    # Define models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs/rl_logs", exist_ok=True)
    
    # Save a checkpoint every 500,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path='./models/logs/',
        name_prefix='rl_model'
    )
    
    print(f"Initializing MaskablePPO agent on {args.device.upper()}...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=args.device,
        verbose=1,
        tensorboard_log="runs/rl_logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    print(f"Starting training for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("Saving final model...")
    model.save(f"models/{args.name}")
    print(f"Done! Model saved to models/{args.name}.zip")
    
if __name__ == "__main__":
    main()
