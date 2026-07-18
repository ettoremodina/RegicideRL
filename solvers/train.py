import argparse
from .logger import RunLogger
from .parallel import ParallelSimulator
from .metrics import plot_metrics

def main():
    parser = argparse.ArgumentParser(description="Regicide Solver Training & Evaluation")
    parser.add_argument("--agent", type=str, default="random", help="Agent name (e.g., 'random')")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training/eval loops")
    parser.add_argument("--games_per_episode", type=int, default=1000, help="Games to simulate per loop")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers (default: CPU cores - 1)")
    parser.add_argument("--iterations", type=int, default=1000, help="ISMCTS iterations per decision (default 1000)")
    parser.add_argument("--exploration", type=float, default=10.0, help="ISMCTS exploration constant C (default 10.0)")
    
    args = parser.parse_args()
    
    logger = RunLogger()
    logger.log(f"Starting run with arguments: {vars(args)}")
    
    # Dynamically load the agent class based on name
    def get_agent_class(name):
        from agents.random_agent import RandomAgent
        from agents.heuristic_agent import HeuristicAgent
        from agents.ppo_agent import PPOAgent
        from agents.ismcts_agent import ISMCTSAgent
        from agents.pimc_agent import PIMCAgent
        
        if name == 'random':
            return RandomAgent
        elif name == 'heuristic':
            return HeuristicAgent
        elif name == 'ppo':
            return PPOAgent
        elif name == 'ismcts':
            return ISMCTSAgent
        elif name == 'pimc':
            return PIMCAgent
        else:
            raise ValueError(f"Unknown agent: {name}")

    agent_cls = get_agent_class(args.agent)
        
    logger.log(f"Using Agent: {agent_cls.__name__}")
    
    simulator = ParallelSimulator(n_jobs=args.jobs)
    logger.log(f"Initialized ParallelSimulator with {simulator.n_jobs} workers")
    
    for episode in range(1, args.episodes + 1):
        logger.log(f"--- Episode {episode}/{args.episodes} ---")
        
        # Build agent kwargs dynamically
        agent_kwargs = {"name": args.agent}
        if args.agent == 'ismcts':
            agent_kwargs["n_iterations"] = args.iterations
            agent_kwargs["exploration_constant"] = args.exploration
        
        metrics = simulator.run_eval(
            agent_cls=agent_cls, 
            agent_kwargs=agent_kwargs, 
            total_games=args.games_per_episode
        )
        
        logger.log(f"Eval Results:")
        logger.log(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.log(f"  Avg Enemies Defeated: {metrics['avg_enemies_defeated']:.2f} / 12")
        logger.log(f"  Avg Turns: {metrics['avg_turns']:.1f}")
        logger.log(f"  Speed: {metrics['games_per_second']:.2f} games/sec")
        
        # Save metrics
        logger.log_metrics(step=episode, metrics_dict=metrics)
        
    logger.log("Run completed. Generating plots...")
    plot_metrics(logger.get_run_dir())
    logger.log(f"Plots saved in {logger.get_run_dir()}")
    
if __name__ == "__main__":
    main()
