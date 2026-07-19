"""
Evaluation utilities for AlphaZero.

Runs the trained network (with greedy MCTS) over multiple games to
measure current strength.  Used by the orchestrator after each training
iteration.
"""

import numpy as np

from ml_logger import get_logger
from solvers.env import RegicideEnv
from solvers.alphazero.mcts import run_mcts
from solvers.alphazero.outcomes import enemies_defeated

logger = get_logger(__name__)


def evaluate_network(network, config, device, n_games=None, recorder=None):
    """Play games using greedy MCTS (τ→0) and return performance stats.

    Args:
        network: RegicideNet in eval mode.
        config: AlphaZeroConfig.
        device: torch device.
        n_games: Override for ``config.eval_games``.

    Returns:
        Dict with ``avg_enemies_defeated``, ``win_rate``, ``avg_moves``.
    """
    network.eval()
    n_games = config.eval_games if n_games is None else n_games
    if n_games <= 0:
        raise ValueError("n_games must be greater than zero")
    total_enemies = 0
    victories = 0
    total_moves = 0

    for _ in range(n_games):
        env = RegicideEnv(num_players=1, recorder=recorder)
        obs, _ = env.reset()
        moves = 0

        while not env.game.game_over:
            action_mask_obs = obs["action_mask"]
            valid_actions = np.nonzero(action_mask_obs)[0].tolist()
            if not valid_actions:
                break

            if len(valid_actions) == 1:
                obs, _, terminated, truncated, _ = env.step(valid_actions[0])
                moves += 1
                continue

            policy, _ = run_mcts(env, network, config, device)
            # Greedy: pick action with highest visit count
            action_id = int(np.argmax(policy))
            
            obs, _, terminated, truncated, _ = env.step(action_id)
            moves += 1

        total_enemies += enemies_defeated(env.game)
        victories += int(env.game.victory)
        total_moves += moves

    return {
        "avg_enemies_defeated": total_enemies / max(1, n_games),
        "win_rate": victories / max(1, n_games),
        "avg_moves": total_moves / max(1, n_games),
    }
