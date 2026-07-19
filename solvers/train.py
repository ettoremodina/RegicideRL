"""CLI for agent evaluation and AlphaZero training."""

import argparse
from dataclasses import asdict, fields

from ml_logger import RunLogger, start_run
from .parallel import ParallelSimulator
from .metrics import plot_metrics

DEFAULT_EPISODES = 10
DEFAULT_GAMES_PER_EPISODE = 1000


def build_parser():
    """Create the unified solver command-line parser."""
    parser = argparse.ArgumentParser(description="Regicide Solver Training & Evaluation")
    parser.add_argument("--agent", type=str, default="random", help="Agent name, or 'alphazero' to train AlphaZero")
    parser.add_argument("--episodes", type=int, help=f"Number of evaluation loops (default: {DEFAULT_EPISODES})")
    parser.add_argument(
        "--games_per_episode", "--games-per-episode",
        type=int,
        help=f"Games per evaluation loop (default: {DEFAULT_GAMES_PER_EPISODE})",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of workers")
    parser.add_argument("--iterations", type=int, default=1000, help="ISMCTS iterations per decision")
    parser.add_argument("--exploration", type=float, default=10.0, help="ISMCTS exploration constant")
    parser.add_argument("--config", default="config.yaml", help="Configuration used for AlphaZero")
    parser.add_argument("--az-iterations", type=int, help="Override AlphaZero training iterations")
    parser.add_argument("--sims", type=int, help="Override AlphaZero MCTS simulations per move")
    parser.add_argument("--eval-games", type=int, help="Override AlphaZero evaluation games")
    parser.add_argument(
        "--heuristic-warmup-iterations",
        type=int,
        help="Override iterations using heuristic AlphaZero search priors",
    )
    parser.add_argument(
        "--heuristic-prior-weight",
        type=float,
        help="Override heuristic weight in warm-up search priors",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override AlphaZero device")
    parser.add_argument("--resume", help="AlphaZero checkpoint path without the .pt suffix")
    return parser


def main():
    """Dispatch to AlphaZero training or repeated agent evaluation."""
    args = build_parser().parse_args()
    if args.agent == "alphazero":
        train_alphazero(args)
        return
    run_evaluation(args)


def run_evaluation(args):
    """Evaluate a configured agent and persist metrics for every episode.

    Args:
        args: Parsed CLI namespace produced by :func:`build_parser`.
    """
    context = start_run(
        "evaluation",
        name=f"{args.agent}-evaluation",
        config=vars(args),
    )
    logger = RunLogger(context=context, run_name="solver.evaluation")
    logger.log(f"Starting run with arguments: {vars(args)}")
    agent_cls = get_agent_class(args.agent)
    logger.log(f"Using Agent: {agent_cls.__name__}")
    simulator = ParallelSimulator(n_jobs=args.jobs, run_context=context)
    logger.log(f"Initialized ParallelSimulator with {simulator.n_jobs} workers")
    episodes = args.episodes or DEFAULT_EPISODES
    games_per_episode = args.games_per_episode or DEFAULT_GAMES_PER_EPISODE
    try:
        for episode in range(1, episodes + 1):
            logger.log(f"--- Episode {episode}/{episodes} ---")
            metrics = simulator.run_eval(
                agent_cls=agent_cls,
                agent_kwargs=get_agent_kwargs(args),
                total_games=games_per_episode,
            )
            logger.log("Eval Results:")
            logger.log(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            logger.log(f"  Avg Enemies Defeated: {metrics['avg_enemies_defeated']:.2f} / 12")
            logger.log(f"  Avg Turns: {metrics['avg_turns']:.1f}")
            logger.log(f"  Speed: {metrics['games_per_second']:.2f} games/sec")
            logger.log_metrics(step=episode, metrics_dict=metrics)
    except Exception as error:
        context.fail(error)
        raise
    finally:
        simulator.close()

    logger.log("Run completed. Generating plots...")
    plot_metrics(logger.get_run_dir())
    logger.log(f"Plots saved in {logger.get_run_dir()}")
    context.complete({"episodes": episodes, "games_per_episode": games_per_episode})


def get_agent_class(name):
    """Resolve a supported CLI agent name to its implementation class.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    from agents.heuristic_agent import HeuristicAgent
    from agents.ismcts_agent import ISMCTSAgent
    from agents.pimc_agent import PIMCAgent
    from agents.ppo_agent import PPOAgent
    from agents.random_agent import RandomAgent

    agents = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        "ppo": PPOAgent,
        "ismcts": ISMCTSAgent,
        "pimc": PIMCAgent,
    }
    try:
        return agents[name]
    except KeyError as error:
        raise ValueError(f"Unknown agent: {name}") from error


def get_agent_kwargs(args):
    """Build agent-specific constructor arguments from CLI options."""
    kwargs = {"name": args.agent}
    if args.agent == "ismcts":
        kwargs["n_iterations"] = args.iterations
        kwargs["exploration_constant"] = args.exploration
    return kwargs


def train_alphazero(args):
    """Run AlphaZero training with YAML defaults and explicit CLI overrides."""
    from solvers.alphazero.config import AlphaZeroConfig
    from solvers.alphazero.orchestrator import AlphaZeroOrchestrator
    from solvers.config import load_config

    config_data = load_config(args.config).get("alphazero", {})
    valid_fields = {field.name for field in fields(AlphaZeroConfig)}
    config_values = {
        key: value for key, value in config_data.items() if key in valid_fields
    }
    overrides = {
        "max_iterations": args.az_iterations,
        "games_per_iteration": args.games_per_episode,
        "n_simulations": args.sims,
        "eval_games": args.eval_games,
        "heuristic_warmup_iterations": args.heuristic_warmup_iterations,
        "heuristic_prior_weight": args.heuristic_prior_weight,
        "device": args.device,
    }
    for name, value in overrides.items():
        if value is not None:
            config_values[name] = value
    config = AlphaZeroConfig(**config_values)
    context = start_run(
        "alphazero",
        name="alphazero-training",
        config=asdict(config),
    )
    try:
        AlphaZeroOrchestrator(
            config,
            resume_path=args.resume,
            run_context=context,
        ).run()
        context.complete()
    except Exception as error:
        context.fail(error)
        raise


if __name__ == "__main__":
    main()
