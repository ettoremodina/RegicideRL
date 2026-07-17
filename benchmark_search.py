"""
Benchmark script for search-based agents (Phase 1 & 2).

Compares win rates and performance across:
  - Random baseline
  - Heuristic baseline  
  - PIMC (Phase 1)
  - ISMCTS (Phase 2)

Usage:
    python benchmark_search.py                           # defaults
    python benchmark_search.py --games 200               # more games
    python benchmark_search.py --agents pimc ismcts       # only specific agents
    python benchmark_search.py --determinizations 100     # more PIMC samples
    python benchmark_search.py --iterations 2000          # more ISMCTS iterations
"""

import time
import argparse
import sys
from solvers.env import RegicideEnv
from solvers.agents.random_agent import RandomAgent
from solvers.agents.heuristic_agent import HeuristicAgent
from solvers.agents.pimc_agent import PIMCAgent
from solvers.agents.ismcts_agent import ISMCTSAgent


def run_agent(agent, num_games, label):
    """Run a single agent for num_games and return metrics.

    Args:
        agent: An agent instance with select_action(obs, env=env).
        num_games: Number of games to play.
        label: Display name for logging.

    Returns:
        Dict with win_rate, avg_enemies_defeated, avg_time_per_game, total_time.
    """
    env = RegicideEnv(num_players=1)
    victories = 0
    total_enemies = 0
    
    start = time.time()

    for game_idx in range(num_games):
        if hasattr(agent, 'reset'):
            agent.reset()
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs, env=env)
            if action is None:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if env.game.victory:
            victories += 1

        enemies_left = len(env.game.castle_deck) + (
            1 if env.game.current_enemy and not env.game.victory else 0
        )
        total_enemies += 12 - enemies_left

        # Progress indicator for slow agents
        if (game_idx + 1) % max(1, num_games // 10) == 0:
            elapsed = time.time() - start
            print(f"  [{label}] {game_idx + 1}/{num_games} games "
                  f"({elapsed:.1f}s elapsed, "
                  f"win rate so far: {victories / (game_idx + 1) * 100:.1f}%)")

    total_time = time.time() - start

    return {
        'win_rate': victories / num_games,
        'wins': victories,
        'avg_enemies_defeated': total_enemies / num_games,
        'avg_time_per_game': total_time / num_games,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search agents for Regicide"
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games per agent (default: 100)"
    )
    parser.add_argument(
        "--agents", nargs="+",
        choices=["random", "heuristic", "pimc", "ismcts"],
        default=["random", "heuristic", "pimc", "ismcts"],
        help="Which agents to benchmark (default: all)"
    )
    parser.add_argument(
        "--determinizations", type=int, default=50,
        help="Number of PIMC determinizations per action (default: 50)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of ISMCTS iterations per decision (default: 1000)"
    )
    parser.add_argument(
        "--exploration", type=float, default=1.414,
        help="ISMCTS exploration constant C (default: 1.414)"
    )
    args = parser.parse_args()

    agents_to_run = []
    if "random" in args.agents:
        agents_to_run.append(("Random", RandomAgent(name="Random")))
    if "heuristic" in args.agents:
        agents_to_run.append(("Heuristic", HeuristicAgent(name="Heuristic")))
    if "pimc" in args.agents:
        agents_to_run.append((
            f"PIMC (d={args.determinizations})",
            PIMCAgent(n_determinizations=args.determinizations, name="PIMC")
        ))
    if "ismcts" in args.agents:
        agents_to_run.append((
            f"ISMCTS (i={args.iterations}, C={args.exploration})",
            ISMCTSAgent(
                n_iterations=args.iterations,
                exploration_constant=args.exploration,
                name="ISMCTS"
            )
        ))

    print(f"\n{'='*70}")
    print(f"  Regicide Search Agent Benchmark — {args.games} games each")
    print(f"{'='*70}\n")

    results = {}
    for label, agent in agents_to_run:
        print(f"--- Running: {label} ---")
        metrics = run_agent(agent, args.games, label)
        results[label] = metrics
        print(f"  Win rate: {metrics['win_rate'] * 100:.1f}% "
              f"({metrics['wins']}/{args.games})")
        print(f"  Avg enemies defeated: {metrics['avg_enemies_defeated']:.2f}/12")
        print(f"  Avg time/game: {metrics['avg_time_per_game']:.3f}s")
        print(f"  Total time: {metrics['total_time']:.1f}s\n")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent':<35} {'Win%':>7} {'Enemies':>9} {'Time/Game':>10}")
    print(f"{'-'*35} {'-'*7} {'-'*9} {'-'*10}")
    for label, m in results.items():
        print(f"{label:<35} {m['win_rate']*100:>6.1f}% "
              f"{m['avg_enemies_defeated']:>8.2f} "
              f"{m['avg_time_per_game']:>9.3f}s")
    print()


if __name__ == "__main__":
    main()
