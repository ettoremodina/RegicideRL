import os
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.ppo_mask import MaskablePPO
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

def main():
    print("Initializing environment...")
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    # Define models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs/rl_logs", exist_ok=True)
    
    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/logs/',
        name_prefix='rl_model'
    )
    
    print("Initializing MaskablePPO agent...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="runs/rl_logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    print("Starting training for 50,000 timesteps...")
    # Train for a modest amount of steps to verify it works
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("Saving final model...")
    model.save("models/ppo_regicide_v1")
    print("Done! Model saved to models/ppo_regicide_v1.zip")
    
if __name__ == "__main__":
    main()
