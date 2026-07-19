"""
AlphaZero agent — plugs a trained network + MCTS into the BaseAgent interface.

Loads a checkpoint and uses MCTS with PUCT for action selection, making it
compatible with the existing benchmark / evaluation infrastructure.
"""

import numpy as np
import torch

from agents.base_agent import BaseAgent
from ml_logger import get_logger
from solvers.alphazero.config import AlphaZeroConfig
from solvers.alphazero.network import RegicideNet
from solvers.alphazero.mcts import run_mcts

logger = get_logger(__name__)


class AlphaZeroAgent(BaseAgent):
    """Deployable agent using a trained AlphaZero network + MCTS.

    Args:
        checkpoint_path: Path to a saved checkpoint (without ``.pt``).
        n_simulations: MCTS simulations per decision (overrides config).
        device: ``"cpu"`` or ``"cuda"``.
        name: Agent name for logging.
    """

    def __init__(
        self,
        checkpoint_path: str,
        n_simulations: int = 50,
        device: str = "cpu",
        name: str = "AlphaZeroAgent",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = AlphaZeroConfig(
            n_simulations=n_simulations,
            device=device,
        )
        self.device = torch.device(device)

        # Build and load network
        self.network = RegicideNet(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_space_size,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path + ".pt", map_location=self.device
        )
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        logger.info(f"Loaded checkpoint from {checkpoint_path}.pt")

    def select_action(self, obs, env=None):
        """Select an action via MCTS + trained network.

        Args:
            obs: Observation dict from RegicideEnv.
            env: The live RegicideEnv instance (required).

        Returns:
            Selected global action identifier, or ``None`` if no action is legal.

        Raises:
            ValueError: If ``env`` is omitted.
        """
        if env is None:
            raise ValueError("AlphaZeroAgent requires the env object.")

        import numpy as np
        action_mask = obs["action_mask"]
        valid_actions = np.nonzero(action_mask)[0].tolist()
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Run MCTS (greedy — no temperature)
        policy, _ = run_mcts(env, self.network, self.config, self.device)
        action_id = int(np.argmax(policy))

        return action_id
