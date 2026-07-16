import numpy as np
from collections import defaultdict
import time
from game.regicide import Game
from game.action_handler import ActionHandler
from solvers.env import RegicideEnv

def calculate_theoretical_bounds():
    print("======================================================================")
    print("  THEORETICAL STATE SPACE DIMENSIONALITY")
    print("======================================================================")
    print("1. Maximum Hand Size: 8 cards")
    print("2. Deck Size: 52 cards (including 2 Jesters)")
    print("3. Possible 8-card hands: ~7.52 x 10^8 (Combinations of 52 choose 8)")
    print("4. Action Mask Size: 256 (2^8 possible binary masks for 8 cards)")
    print("5. Maximum Theoretical Actions per Turn: 256 (if all combinations were valid)")
    print("6. Castle Deck (Enemies): 12 cards (4 Jacks, 4 Queens, 4 Kings)")
    print("7. Max Game Depth: ~100-150 turns (every card drawn and played)")
    print("======================================================================\n")

def analyze_empirical_search_space(num_games=1000, seed=42):
    print(f"======================================================================")
    print(f"  EMPIRICAL SEARCH SPACE ANALYSIS ({num_games} Simulated Games)")
    print(f"======================================================================")
    
    env = RegicideEnv(num_players=1)
    
    # Metrics
    game_lengths = []
    attack_branching_factors = []
    defense_branching_factors = []
    actions_per_hand_size = defaultdict(list)
    
    start_time = time.time()
    
    for i in range(num_games):
        obs, _ = env.reset(seed=seed + i if seed else None)
        done = False
        steps = 0
        
        while not done:
            steps += 1
            valid_actions = obs['valid_actions']
            num_actions = len(valid_actions)
            hand_size = len(obs['hand'])
            
            actions_per_hand_size[hand_size].append(num_actions)
            
            if obs['defense_phase']:
                defense_branching_factors.append(num_actions)
            else:
                attack_branching_factors.append(num_actions)
            
            if num_actions == 0:
                print(f"Warning: 0 valid actions at step {steps}. Ending game.")
                break
                
            # Choose a random valid action to explore typical game trajectories
            action_idx = np.random.randint(num_actions)
            action = valid_actions[action_idx]
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        game_lengths.append(steps)
        
    end_time = time.time()
    
    # Display Results
    print(f"Simulation Time: {end_time - start_time:.2f} seconds\n")
    
    print("--- 1. Game Depth (Tree Depth) ---")
    print(f"Average Depth: {np.mean(game_lengths):.1f} turns")
    print(f"Median Depth:  {np.median(game_lengths):.1f} turns")
    print(f"Max Depth:     {np.max(game_lengths)} turns")
    print(f"Min Depth:     {np.min(game_lengths)} turns\n")
    
    print("--- 2. Branching Factor (Action Space) ---")
    all_branching = attack_branching_factors + defense_branching_factors
    print(f"Overall Avg Branching Factor: {np.mean(all_branching):.2f}")
    print(f"Overall Max Branching Factor: {np.max(all_branching)}")
    print(f"Attack Avg Branching Factor:  {np.mean(attack_branching_factors):.2f}")
    print(f"Defense Avg Branching Factor: {np.mean(defense_branching_factors):.2f}\n")
    
    print("--- 3. Branching Factor by Hand Size ---")
    print(f"{'Hand Size':<12} | {'Avg Actions':<12} | {'Max Actions':<12} | {'Samples'}")
    print("-" * 55)
    for size in sorted(actions_per_hand_size.keys(), reverse=True):
        actions = actions_per_hand_size[size]
        print(f"{size:<12} | {np.mean(actions):<12.2f} | {np.max(actions):<12} | {len(actions)}")
        
    print("\n--- Summary of Search Space Complexity ---")
    b_avg = np.mean(all_branching)
    d_avg = np.mean(game_lengths)
    print(f"Estimated Average Tree Size (b^d): {b_avg:.2f}^{d_avg:.1f} ≈ 10^{d_avg * np.log10(b_avg):.1f} nodes")
    print("Note: Regicide's difficulty lies in sparse rewards and long horizons (depth),")
    print("rather than a massive branching factor like Go or Chess.")

if __name__ == '__main__':
    calculate_theoretical_bounds()
    analyze_empirical_search_space(num_games=1000)
