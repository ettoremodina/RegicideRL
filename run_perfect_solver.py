import argparse
import time
from solvers.env import RegicideEnv
from solvers.perfect_solver import PerfectSolver

def main():
    parser = argparse.ArgumentParser(description="Run the Perfect Information Solver on specific seeds.")
    parser.add_argument("--seed", type=int, default=40, help="Starting RNG seed for deck generation")
    parser.add_argument("--games", type=int, default=1, help="Number of games to solve sequentially")
    args = parser.parse_args()
    
    print(f"======================================================================")
    print(f"  REGICIDE PERFECT INFORMATION SOLVER")
    print(f"======================================================================")
    print(f"Starting Seed: {args.seed} | Games to solve: {args.games}")
    print("Warning: Solving full 52-card games can take significant time per game.")
    print("This solver uses DFS + Transposition Tables and assumes Deterministic Hearts.")
    
    wins = 0
    losses = 0
    
    for i in range(args.games):
        current_seed = args.seed + i
        print(f"\n--- Game {i+1}/{args.games} (Seed: {current_seed}) ---")
        
        # Initialize environment
        env = RegicideEnv(num_players=1)
        env.reset(seed=current_seed)
        
        # Enable deterministic mode for the solver
        env.game.deterministic_hearts = True
        
        start_time = time.time()
        
        try:
            from tqdm import tqdm
            with tqdm(desc=f"Seed {current_seed}", unit="nodes", smoothing=0.1) as pbar:
                solver = PerfectSolver(verbose=False, callback_freq=10000)
                def update_pbar(n):
                    pbar.update(n)
                    pbar.set_postfix(max_bosses=f"{solver.max_bosses_defeated}/12")
                solver.callback = update_pbar
                winning_actions = solver.solve(env)
        except ImportError:
            solver = PerfectSolver(verbose=False)
            winning_actions = solver.solve(env)
        except KeyboardInterrupt:
            print("\nSearch interrupted by user.")
            break
            
        elapsed = time.time() - start_time
        
        if winning_actions is not None:
            wins += 1
            print(f"[+] WINNABLE SEED FOUND! (Took {elapsed:.2f}s, {solver.nodes_evaluated} nodes)")
            print(f"    Required {len(winning_actions)} actions to win.")
            if args.games == 1:
                print("\nWinning Sequence:")
                env.reset(seed=current_seed)
                env.game.deterministic_hearts = True
                
                for step, act_idx in enumerate(winning_actions):
                    obs = env._get_obs()
                    mask = next(m for idx, m in zip(
                        [sum(val * (1 << j) for j, val in enumerate(m)) for m in obs['valid_actions']],
                        obs['valid_actions']
                    ) if idx == act_idx)
                    
                    cards_played = [env.game.players[0][j] for j, val in enumerate(mask) if val]
                    phase = "DEFENSE" if obs['defense_phase'] else "ATTACK"
                    action_str = ", ".join(str(c) for c in cards_played) if cards_played else "YIELD"
                    
                    print(f"{step+1:3d}. [{phase}] {action_str}")
                    env.step(mask)
                    
                print("\nVerification successful: sequence leads to Victory.")
        else:
            losses += 1
            print(f"[-] DOOMED SEED. (Took {elapsed:.2f}s, {solver.nodes_evaluated} nodes)")
            print("    There is mathematically no way to win this game.")
            print(f"    Maximum bosses defeated during search: {solver.max_bosses_defeated}/12")
            print(f"    Sequence to reach this state ({len(solver.best_sequence)} actions):")
            
            if args.games == 1 and solver.best_sequence:
                env.reset(seed=current_seed)
                env.game.deterministic_hearts = True
                
                for step, act_idx in enumerate(solver.best_sequence):
                    obs = env._get_obs()
                    mask = next(m for idx, m in zip(
                        [sum(val * (1 << j) for j, val in enumerate(m)) for m in obs['valid_actions']],
                        obs['valid_actions']
                    ) if idx == act_idx)
                    
                    cards_played = [env.game.players[0][j] for j, val in enumerate(mask) if val]
                    phase = "DEFENSE" if obs['defense_phase'] else "ATTACK"
                    action_str = ", ".join(str(c) for c in cards_played) if cards_played else "YIELD"
                    
                    print(f"{step+1:3d}. [{phase}] {action_str}")
                    env.step(mask)
                
                print(f"\nVerification successful: sequence reaches {solver.max_bosses_defeated} bosses defeated.")
            
        print(f"Score so far: {wins} Wins / {losses} Losses (Win Rate: {(wins/(wins+losses))*100:.1f}%)")

if __name__ == '__main__':
    main()
