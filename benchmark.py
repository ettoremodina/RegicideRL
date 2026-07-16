import time
import random
import argparse
from game.regicide import Game
from game.action_handler import ActionHandler
from solvers.parallel import ParallelSimulator
from solvers.agents.random_agent import RandomAgent
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

def simulate_normal(num_games=1000):
    print(f"\n--- Normal (Single-Thread) Benchmark ---")
    start_time = time.time()
    
    handler = ActionHandler(max_hand_size=8)
    victories = 0
    total_turns = 0
    enemies_defeated = 0
    
    for i in range(num_games):
        game = Game(num_players=1)
        required_defense = 0
        
        while not game.game_over:
            current = game.current_player
            hand = game.get_player_hand(current)
            
            # Simple AI: randomly choose a valid action
            if required_defense > 0:
                actions = handler.get_all_possible_actions(hand, "defense", {'enemy_attack': required_defense})
                if not actions:
                    # Auto defeat if cannot defend
                    game.game_over = True
                    break
                action = random.choice(actions)
                indices = handler.mask_to_card_indices(action, len(hand))
                res = game.defend_with_card_indices(indices)
                required_defense = 0
            else:
                actions = handler.get_all_possible_actions(hand, "attack", game.get_game_state())
                if not actions:
                    game.game_over = True
                    break
                action = random.choice(actions)
                
                is_solo_jester = (len(action) == 9 and action[8] == 1)
                
                if is_solo_jester:
                    res = game.use_solo_jester("step1")
                else:
                    indices = handler.mask_to_card_indices(action, len(hand))
                    if handler.is_yield_action(action):
                        res = game.yield_turn()
                    else:
                        res = game.play_card(indices)
                
                required_defense = res.get("defense_required", 0)
                
                # Handle Jester choice (solo mode defaults back to player 1)
                if res.get("phase") == "next_player_choice":
                    game.choose_next_player(0)
                    
            total_turns += 1
            
        if game.victory:
            victories += 1
        
        # 12 enemies total
        enemies_left = len(game.castle_deck) + (1 if game.current_enemy and not game.victory else 0)
        enemies_defeated += (12 - enemies_left)

    elapsed = time.time() - start_time
    fps = num_games / elapsed
    
    print(f"Games played: {num_games}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Speed: {fps:.2f} games/second")
    print(f"Avg turns per game: {total_turns / num_games:.1f}")
    print(f"Win rate: {victories / num_games * 100:.2f}%")
    print(f"Avg enemies defeated: {enemies_defeated / num_games:.2f} / 12")
    return fps

def simulate_parallel(num_games=1000, jobs=None):
    print(f"\n--- Parallel Benchmark (Jobs: {jobs or 'Max'}) ---")
    simulator = ParallelSimulator(n_jobs=jobs)
    
    metrics = simulator.run_eval(
        agent_cls=RandomAgent, 
        agent_kwargs={"name": "Random"}, 
        total_games=num_games
    )
    
    fps = metrics['games_per_second']
    print(f"Games played: {num_games}")
    print(f"Time taken: {metrics['total_time']:.2f} seconds")
    print(f"Speed: {fps:.2f} games/second")
    print(f"Avg turns per game: {metrics['avg_turns']:.1f}")
    print(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
    print(f"Avg enemies defeated: {metrics['avg_enemies_defeated']:.2f} / 12")
    return fps

def simulate_training(device, steps=10000):
    print(f"\n--- Training Benchmark ({device.upper()}) ---")
    import torch
    from sb3_contrib.ppo_mask import MaskablePPO
    
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=0,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
    )
    
    start_time = time.time()
    model.learn(total_timesteps=steps)
    end_time = time.time()
    
    elapsed = end_time - start_time
    fps = steps / elapsed
    
    print(f"Device: {device.upper()}")
    print(f"Steps: {steps}")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print(f"Speed: {fps:.2f} steps/second")
    return fps

def simulate_env(num_games=1000):
    print(f"\n--- Env Benchmark ---")
    start_time = time.time()
    
    env = RegicideEnv(num_players=1)
    victories = 0
    total_turns = 0
    
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            # Random agent logic using action_mask
            action_mask = obs['action_mask']
            # Find valid actions
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            total_turns += 1
            
        if env.game.victory:
            victories += 1

    elapsed = time.time() - start_time
    fps = num_games / elapsed
    
    print(f"Games played: {num_games}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Speed: {fps:.2f} games/second")
    print(f"Avg turns per game: {total_turns / num_games:.1f}")
    print(f"Win rate: {victories / num_games * 100:.2f}%")
    return fps

def main():
    import torch
    parser = argparse.ArgumentParser(description="Regicide Benchmarking Utility")
    parser.add_argument("--mode", type=str, choices=["all", "normal", "env", "parallel", "cpu", "gpu"], default="all", 
                        help="Which benchmark to run (default: all)")
    parser.add_argument("--games", type=int, default=1000, 
                        help="Number of games to simulate for normal/env/parallel (default: 1000)")
    parser.add_argument("--steps", type=int, default=10000, 
                        help="Number of training steps to simulate for cpu/gpu (default: 10000)")
    parser.add_argument("--jobs", type=int, default=None, 
                        help="Number of workers for parallel benchmark (default: max cores)")
    
    args = parser.parse_args()
    
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["normal", "env", "parallel", "cpu"]
        if torch.cuda.is_available():
            modes_to_run.append("gpu")
        else:
            print("\nCUDA is not available. Skipping GPU benchmark in 'all' mode.")
    else:
        modes_to_run = [args.mode]
        
    results = {}
    
    if "normal" in modes_to_run:
        results["Normal   (Games/sec)"] = simulate_normal(args.games)
        
    if "env" in modes_to_run:
        results["Env      (Games/sec)"] = simulate_env(args.games)
        
    if "parallel" in modes_to_run:
        results["Parallel (Games/sec)"] = simulate_parallel(args.games, args.jobs)
        
    if "cpu" in modes_to_run:
        results["Train CPU (Steps/sec)"] = simulate_training("cpu", args.steps)
        
    if "gpu" in modes_to_run:
        if not torch.cuda.is_available() and args.mode == "gpu":
            print("\nCUDA is not available on this machine. Cannot benchmark GPU.")
        else:
            results["Train GPU (Steps/sec)"] = simulate_training("cuda", args.steps)
            
    if len(results) > 1:
        print("\n=== Summary ===")
        for name, fps in results.items():
            print(f"{name}: {fps:.2f}")

if __name__ == "__main__":
    main()
