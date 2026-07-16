"""
Entry point for AlphaZero training.

Usage:
    python train_alphazero.py                     # Default config
    python train_alphazero.py --sims 100          # More MCTS sims
    python train_alphazero.py --resume path/ckpt  # Resume from checkpoint
    python train_alphazero.py --device cuda        # Use GPU
"""

import argparse
import logging
import sys

from solvers.alphazero.config import AlphaZeroConfig
from solvers.alphazero.orchestrator import AlphaZeroOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Train a Regicide agent via AlphaZero Expert Iteration."
    )
    parser.add_argument(
        "--sims", type=int, default=50,
        help="MCTS simulations per move during self-play (default: 50)",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Self-play games per iteration (default: 100)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Total training iterations (default: 100)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for network inference and training (default: cpu)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (without .pt extension)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=50,
        help="Evaluation games per iteration (default: 50)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Silence the game engine logs during training
    logging.getLogger("regicide").setLevel(logging.WARNING)

    config = AlphaZeroConfig(
        n_simulations=args.sims,
        games_per_iteration=args.games,
        max_iterations=args.iterations,
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        eval_games=args.eval_games,
    )

    orchestrator = AlphaZeroOrchestrator(config, resume_path=args.resume)
    orchestrator.run()


if __name__ == "__main__":
    main()
