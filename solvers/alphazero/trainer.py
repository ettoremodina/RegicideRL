"""
Network trainer for AlphaZero.

Handles the gradient updates, loss computation, and checkpoint
management.  The loss function follows the standard AlphaZero
formulation:

    L = (z - v)^2  -  π^T log(p)  +  c ||θ||^2

Where:
    z = game outcome (value target from self-play)
    v = network's value prediction
    π = MCTS visit-count distribution (policy target)
    p = network's policy output
    c = L2 regularization (handled by AdamW weight_decay)
"""

from dataclasses import asdict

import torch
import torch.nn.functional as F

from ml_logger import get_logger
from solvers.alphazero.checkpoints import checkpoint_file
from solvers.alphazero.network import RegicideNet
from solvers.alphazero.config import AlphaZeroConfig

logger = get_logger(__name__)


class AlphaZeroTrainer:
    """Owns the network, optimizer, and training step logic.

    Args:
        config: AlphaZeroConfig with training hyperparameters.
    """

    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.network = RegicideNet(
            state_dim=config.state_dim,
            action_dim=config.action_space_size,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.train_step_count = 0
        self.training_iteration = 0

    def train_on_buffer(self, replay_buffer):
        """Run multiple epochs of training on the replay buffer.

        Args:
            replay_buffer: A ReplayBuffer instance.

        Returns:
            avg_losses: Dict with average policy_loss, value_loss, total_loss
                over all mini-batches in this call.
        """
        self.network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss_sum = 0.0
        n_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            # Number of batches per epoch ≈ buffer_size / batch_size
            n_steps = max(1, len(replay_buffer) // self.config.batch_size)

            for _ in range(n_steps):
                states, policies, values = replay_buffer.sample_batch(
                    self.config.batch_size
                )
                losses = self._train_step(states, policies, values)
                total_policy_loss += losses["policy_loss"]
                total_value_loss += losses["value_loss"]
                total_loss_sum += losses["total_loss"]
                n_batches += 1

        avg = {
            "policy_loss": total_policy_loss / max(1, n_batches),
            "value_loss": total_value_loss / max(1, n_batches),
            "total_loss": total_loss_sum / max(1, n_batches),
            "n_batches": n_batches,
        }
        return avg

    def _train_step(self, states, policies, values):
        """One gradient update.

        Args:
            states: np.ndarray ``(B, state_dim)``.
            policies: np.ndarray ``(B, action_dim)`` — target distributions.
            values: np.ndarray ``(B, 1)`` — target values.

        Returns:
            Dict with ``policy_loss``, ``value_loss``, ``total_loss``.
        """
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        policies_t = torch.tensor(policies, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Forward pass
        policy_logits, value_pred = self.network(states_t)

        # --- Value loss: MSE ---
        value_loss = F.mse_loss(value_pred, values_t)

        # --- Policy loss: Cross-entropy ---
        # We use the target distribution π directly (not argmax labels).
        # CE(π, p) = -Σ π_a * log(p_a)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policies_t * log_probs).sum(dim=-1).mean()

        # --- Combined loss ---
        total_loss = (
            self.config.value_loss_weight * value_loss + policy_loss
        )

        # Backward + update
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.train_step_count += 1

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save_checkpoint(self, path):
        """Save model + optimizer state.

        Args:
            path: File path with an optional ``.pt`` extension.
        """
        destination = checkpoint_file(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
                "training_iteration": self.training_iteration,
                "config": asdict(self.config),
            },
            destination,
        )
        logger.info("Checkpoint saved to %s", destination)

    def load_checkpoint(self, path):
        """Load model + optimizer state.

        Args:
            path: File path with an optional ``.pt`` extension.
        """
        source = checkpoint_file(path)
        checkpoint = torch.load(source, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_count = checkpoint.get("train_step_count", 0)
        self.training_iteration = checkpoint.get("training_iteration", 0)
        logger.info(
            "Checkpoint loaded from %s (iteration %d, step %d)",
            source,
            self.training_iteration,
            self.train_step_count,
        )
