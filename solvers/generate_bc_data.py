import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
from solvers.agents.heuristic_agent import HeuristicAgent

def generate_data(num_games=1000, save_path="bc_data.npz"):
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    agent = HeuristicAgent(name="teacher")
    
    observations_list = []
    actions_list = []
    
    print(f"Generating BC data for {num_games} games...")
    
    for _ in tqdm(range(num_games)):
        obs, info = env.reset()
        done = False
        
        while not done:
            raw_obs = raw_env._get_obs()
            action_mask = agent.select_action(raw_obs, env=raw_env)
            
            if action_mask is None:
                break
                
            # Convert binary list mask to integer index (0-255)
            action_int = sum(val * (1 << i) for i, val in enumerate(action_mask))
            
            observations_list.append({
                'hand_values': obs['hand_values'].copy(),
                'hand_suits': obs['hand_suits'].copy(),
                'enemy_stats': obs['enemy_stats'].copy(),
                'flags': obs['flags'].copy(),
                'action_mask': obs['action_mask'].copy()
            })
            actions_list.append(action_int)
            
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            
    num_samples = len(actions_list)
    print(f"Generated {num_samples} state-action pairs.")
    
    hand_values_arr = np.stack([o['hand_values'] for o in observations_list])
    hand_suits_arr = np.stack([o['hand_suits'] for o in observations_list])
    enemy_stats_arr = np.stack([o['enemy_stats'] for o in observations_list])
    flags_arr = np.stack([o['flags'] for o in observations_list])
    action_masks_arr = np.stack([o['action_mask'] for o in observations_list])
    actions_arr = np.array(actions_list, dtype=np.int64)
    
    np.savez_compressed(
        save_path,
        hand_values=hand_values_arr,
        hand_suits=hand_suits_arr,
        enemy_stats=enemy_stats_arr,
        flags=flags_arr,
        action_masks=action_masks_arr,
        actions=actions_arr
    )
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000, help="Number of games to simulate")
    parser.add_argument("--out", type=str, default="bc_data.npz", help="Output file")
    args = parser.parse_args()
    
    generate_data(num_games=args.games, save_path=args.out)
