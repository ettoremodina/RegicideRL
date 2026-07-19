"""
Hyperparameter configuration for the AlphaZero training loop.

All tuneable knobs live in a single dataclass so they can be easily
serialized to / deserialized from YAML and logged alongside each run.
"""

from dataclasses import dataclass

from game.action_space import GLOBAL_ACTION_SPACE_SIZE
from solvers.alphazero.featurizer import STATE_DIM


@dataclass
class AlphaZeroConfig:
    """All hyperparameters for the AlphaZero Expert Iteration loop."""

    # --- Action space ---
    action_space_size: int = GLOBAL_ACTION_SPACE_SIZE

    # --- State featurizer ---
    state_dim: int = STATE_DIM

    # --- Network ---
    hidden_dim: int = 256  # Width of hidden layers in the shared trunk
    num_hidden_layers: int = 2  # Number of hidden layers in the shared trunk
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization
    value_loss_weight: float = 1.0  # Scaling factor for value loss vs policy loss

    # --- MCTS ---
    n_simulations: int = 50  # MCTS simulations per move during self-play
    c_puct: float = 1.5  # PUCT exploration constant
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha at root
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise at root

    # --- Temperature schedule ---
    # τ=1.0 for the first `temp_threshold` moves, then τ→0 (argmax).
    temp_threshold: int = 15  # Number of moves with τ=1.0

    # --- Self-play ---
    games_per_iteration: int = 100  # Self-play games per training iteration
    eval_games: int = 50  # Games to run for evaluation after each iteration
    heuristic_warmup_iterations: int = 10  # Mix heuristic priors first
    heuristic_prior_weight: float = 0.75

    # --- Replay buffer ---
    buffer_size: int = 50_000  # Max samples in the replay buffer
    batch_size: int = 256  # Mini-batch size for training
    min_buffer_size: int = 512  # Minimum samples before training starts

    # --- Training ---
    epochs_per_iteration: int = 10  # Training epochs per iteration
    max_iterations: int = 100  # Total outer-loop iterations
    checkpoint_freq: int = 5  # Save checkpoint every N iterations

    # --- Artifact naming ---
    checkpoint_name: str = "alphazero_latest"

    # --- Device ---
    device: str = "cpu"  # "cpu" or "cuda"

    def __post_init__(self):
        """Reject configurations that would produce invalid searches."""
        if self.action_space_size != GLOBAL_ACTION_SPACE_SIZE:
            raise ValueError(
                f"action_space_size must be {GLOBAL_ACTION_SPACE_SIZE}"
            )
        if self.state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}")
        positive_fields = (
            "hidden_dim",
            "num_hidden_layers",
            "n_simulations",
            "games_per_iteration",
            "eval_games",
            "buffer_size",
            "batch_size",
            "min_buffer_size",
            "epochs_per_iteration",
            "max_iterations",
            "checkpoint_freq",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError("min_buffer_size cannot exceed buffer_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.c_puct < 0:
            raise ValueError("c_puct cannot be negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be greater than zero")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be between zero and one")
        if self.temp_threshold < 0:
            raise ValueError("temp_threshold cannot be negative")
        if self.heuristic_warmup_iterations < 0:
            raise ValueError("heuristic_warmup_iterations cannot be negative")
        if not 0 <= self.heuristic_prior_weight <= 1:
            raise ValueError("heuristic_prior_weight must be between zero and one")
