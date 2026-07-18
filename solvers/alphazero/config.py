"""
Hyperparameter configuration for the AlphaZero training loop.

All tuneable knobs live in a single dataclass so they can be easily
serialized to / deserialized from YAML and logged alongside each run.
"""

from dataclasses import dataclass

from game.action_space import GLOBAL_ACTION_SPACE_SIZE


@dataclass
class AlphaZeroConfig:
    """All hyperparameters for the AlphaZero Expert Iteration loop."""

    # --- Action space ---
    action_space_size: int = GLOBAL_ACTION_SPACE_SIZE

    # --- State featurizer ---
    state_dim: int = 56  # Flat feature vector size (see featurizer.py)

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

    # --- Replay buffer ---
    buffer_size: int = 50_000  # Max samples in the replay buffer
    batch_size: int = 256  # Mini-batch size for training
    min_buffer_size: int = 512  # Minimum samples before training starts

    # --- Training ---
    epochs_per_iteration: int = 10  # Training epochs per iteration
    max_iterations: int = 100  # Total outer-loop iterations
    checkpoint_freq: int = 5  # Save checkpoint every N iterations

    # --- Paths ---
    save_dir: str = "runs/alphazero"
    checkpoint_name: str = "alphazero_latest"

    # --- Device ---
    device: str = "cpu"  # "cpu" or "cuda"
