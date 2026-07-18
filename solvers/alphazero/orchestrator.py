"""
AlphaZero training orchestrator.

Runs the synchronous Expert Iteration loop:
    1. Self-play  →  generate training data
    2. Train      →  update the neural network
    3. Evaluate   →  benchmark the current agent
    4. Log + Save →  metrics and checkpoints
    5. Repeat
"""

import os
import time
import logging


from solvers.alphazero.config import AlphaZeroConfig
from solvers.alphazero.trainer import AlphaZeroTrainer
from solvers.alphazero.replay_buffer import ReplayBuffer
from solvers.alphazero.self_play import generate_self_play_data
from solvers.alphazero.eval import evaluate_network
from solvers.logger import RunLogger

logger = logging.getLogger("alphazero.orchestrator")


class AlphaZeroOrchestrator:
    """Main training loop controller.

    Args:
        config: AlphaZeroConfig with all hyperparameters.
        resume_path: Optional checkpoint path to resume from.
    """

    def __init__(self, config: AlphaZeroConfig, resume_path: str = None):
        self.config = config
        self.trainer = AlphaZeroTrainer(config)
        self.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
        self.run_logger = RunLogger(
            run_name="alphazero",
            base_dir=config.save_dir,
        )

        if resume_path:
            self.trainer.load_checkpoint(resume_path)
            logger.info(f"Resumed from {resume_path}")

    def run(self):
        """Execute the full training loop."""
        logger.info("=" * 60)
        logger.info("AlphaZero Training Loop Starting")
        logger.info(f"  Simulations/move: {self.config.n_simulations}")
        logger.info(f"  Games/iteration:  {self.config.games_per_iteration}")
        logger.info(f"  Max iterations:   {self.config.max_iterations}")
        logger.info(f"  Device:           {self.config.device}")
        logger.info("=" * 60)

        device = self.trainer.device

        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            logger.info(f"\n{'='*40} Iteration {iteration} {'='*40}")

            # --- 1. Self-Play ---
            sp_start = time.time()
            logger.info("Phase 1: Self-Play")
            game_data, sp_stats = generate_self_play_data(
                self.trainer.network, self.config, device
            )
            self.replay_buffer.add_game(game_data)
            sp_time = time.time() - sp_start
            logger.info(
                f"  Generated {sp_stats['total_samples']} samples from "
                f"{sp_stats['total_games']} games in {sp_time:.1f}s"
            )
            logger.info(
                f"  Avg enemies: {sp_stats['avg_enemies_defeated']:.2f}/12  "
                f"Win rate: {sp_stats['win_rate']:.1%}"
            )

            # --- 2. Training ---
            if len(self.replay_buffer) < self.config.min_buffer_size:
                logger.info(
                    f"  Buffer too small ({len(self.replay_buffer)}/"
                    f"{self.config.min_buffer_size}), skipping training."
                )
                continue

            tr_start = time.time()
            logger.info("Phase 2: Training")
            losses = self.trainer.train_on_buffer(self.replay_buffer)
            tr_time = time.time() - tr_start
            logger.info(
                f"  {losses['n_batches']} batches in {tr_time:.1f}s  "
                f"PolicyL={losses['policy_loss']:.4f}  "
                f"ValueL={losses['value_loss']:.4f}  "
                f"Total={losses['total_loss']:.4f}"
            )

            # --- 3. Evaluation ---
            ev_start = time.time()
            logger.info("Phase 3: Evaluation")
            eval_stats = evaluate_network(
                self.trainer.network, self.config, device
            )
            ev_time = time.time() - ev_start
            logger.info(
                f"  Eval ({self.config.eval_games} games, {ev_time:.1f}s): "
                f"Avg enemies: {eval_stats['avg_enemies_defeated']:.2f}/12  "
                f"Win rate: {eval_stats['win_rate']:.1%}"
            )

            # --- 4. Logging ---
            iter_time = time.time() - iter_start
            metrics = {
                "iteration": iteration,
                "buffer_size": len(self.replay_buffer),
                "self_play_time": sp_time,
                "train_time": tr_time,
                "eval_time": ev_time,
                "total_time": iter_time,
                **{f"sp_{k}": v for k, v in sp_stats.items()},
                **{f"train_{k}": v for k, v in losses.items()},
                **{f"eval_{k}": v for k, v in eval_stats.items()},
            }
            self.run_logger.log_metrics(iteration, metrics)
            logger.info(f"  Total iteration time: {iter_time:.1f}s")

            # --- 5. Checkpoint ---
            if iteration % self.config.checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    self.run_logger.get_run_dir(),
                    "models",
                    f"{self.config.checkpoint_name}_iter{iteration}",
                )
                self.trainer.save_checkpoint(ckpt_path)

        # Final checkpoint
        final_path = os.path.join(
            self.run_logger.get_run_dir(),
            "models",
            f"{self.config.checkpoint_name}_final",
        )
        self.trainer.save_checkpoint(final_path)
        logger.info("\nTraining complete!")
