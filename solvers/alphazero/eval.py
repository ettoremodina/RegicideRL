"""
Evaluation utilities for AlphaZero.

Runs the trained network (with greedy MCTS) over multiple games to
measure current strength.  Used by the orchestrator after each training
iteration.
"""

import logging
import numpy as np

from solvers.env import RegicideEnv
from solvers.alphazero.mcts import run_mcts
from solvers.alphazero.featurizer import encode_state

logger = logging.getLogger("alphazero.eval")


def evaluate_network(network, config, device, n_games=None):
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
    n_games = n_games or config.eval_games
    total_enemies = 0
    victories = 0
    total_moves = 0

    for _ in range(n_games):
        env = RegicideEnv(num_players=1)
        obs, _ = env.reset()
        moves = 0

        while not env.game.game_over:
            valid_actions = obs["valid_actions"]
            if not valid_actions:
                break

            if len(valid_actions) == 1:
                obs, _, terminated, truncated, _ = env.step(valid_actions[0])
                moves += 1
                continue

            policy, _ = run_mcts(env, network, config, device)
            # Greedy: pick action with highest visit count
            action_id = int(np.argmax(policy))

            hand = env.game.get_player_hand(env.game.current_player)
            try:
                card_indices = env.handler.global_action_to_hand_indices(
                    action_id, hand
                )
                action_mask = env.handler.cards_to_mask(card_indices)
                obs, _, terminated, truncated, _ = env.step(action_mask)
            except ValueError:
                # Fallback
                obs, _, terminated, truncated, _ = env.step(valid_actions[0])
            moves += 1

        enemies_left = len(env.game.castle_deck) + (
            1 if env.game.current_enemy and not env.game.victory else 0
        )
        total_enemies += 12 - enemies_left
        victories += int(env.game.victory)
        total_moves += moves

    return {
        "avg_enemies_defeated": total_enemies / max(1, n_games),
        "win_rate": victories / max(1, n_games),
        "avg_moves": total_moves / max(1, n_games),
    }
