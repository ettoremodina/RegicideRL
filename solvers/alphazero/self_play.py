"""
Self-play data generation for AlphaZero.

Plays full games using MCTS + the neural network, recording
``(state, policy, outcome)`` at every decision point.  The outcome
(progress-based value) is only known at game end and is applied
retroactively to all states in the game.
"""

import logging

import numpy as np

from solvers.env import RegicideEnv
from solvers.alphazero.featurizer import encode_state
from solvers.alphazero.mcts import run_mcts

logger = logging.getLogger("alphazero.self_play")


def run_self_play_game(network, config, device):
    """Play one full game via MCTS+Network and return training data.

    Args:
        network: A RegicideNet in eval mode.
        config: AlphaZeroConfig.
        device: torch device.

    Returns:
        game_data: List of ``(state, policy, value)`` tuples where
            ``value`` is the progress-based terminal value applied to
            every state in the game.
        game_info: Dict with game-level stats (enemies_defeated, victory).
    """
    env = RegicideEnv(num_players=1)
    obs, _ = env.reset()

    history = []  # [(state_vector, policy_vector), ...]
    move_count = 0

    while not env.game.game_over:
        action_mask_obs = obs["action_mask"]
        valid_actions = np.nonzero(action_mask_obs)[0].tolist()
        if not valid_actions:
            break

        # Single valid action — skip MCTS
        if len(valid_actions) == 1:
            obs, _, terminated, truncated, _ = env.step(valid_actions[0])
            move_count += 1
            continue

        # Run MCTS from current position
        network.eval()
        policy, _ = run_mcts(env, network, config, device)

        # Record the state *before* acting
        state_vec = encode_state(env)

        # Temperature-based action selection
        if move_count < config.temp_threshold:
            # τ = 1.0 — sample proportionally to visit counts
            action_id = _sample_from_policy(policy)
        else:
            # τ → 0 — pick the most-visited action
            action_id = int(np.argmax(policy))

        history.append((state_vec, policy))

        # Execute the action
        obs, _, terminated, truncated, _ = env.step(action_id)
        move_count += 1

    # --- Compute terminal value (progress-based) ---
    enemies_left = len(env.game.castle_deck) + (
        1 if env.game.current_enemy and not env.game.victory else 0
    )
    enemies_defeated = 12 - enemies_left
    progress = enemies_defeated / 12.0

    if env.game.victory:
        terminal_value = 1.0
    else:
        terminal_value = progress * 2.0 - 1.0

    # Apply the terminal value to all recorded states
    game_data = [
        (state, policy, terminal_value) for state, policy in history
    ]

    game_info = {
        "enemies_defeated": enemies_defeated,
        "victory": env.game.victory,
        "moves": move_count,
        "samples": len(game_data),
    }

    return game_data, game_info


def generate_self_play_data(network, config, device):
    """Run multiple self-play games and aggregate results.

    Args:
        network: RegicideNet.
        config: AlphaZeroConfig.
        device: torch device.

    Returns:
        all_data: List of ``(state, policy, value)`` tuples.
        stats: Dict with aggregated statistics.
    """
    all_data = []
    total_enemies = 0
    victories = 0

    for game_i in range(config.games_per_iteration):
        game_data, game_info = run_self_play_game(network, config, device)
        all_data.extend(game_data)
        total_enemies += game_info["enemies_defeated"]
        victories += int(game_info["victory"])

        if (game_i + 1) % 10 == 0:
            logger.info(
                f"  Self-play game {game_i + 1}/{config.games_per_iteration} "
                f"— defeated {game_info['enemies_defeated']}/12"
            )

    stats = {
        "total_games": config.games_per_iteration,
        "total_samples": len(all_data),
        "avg_enemies_defeated": total_enemies / max(1, config.games_per_iteration),
        "win_rate": victories / max(1, config.games_per_iteration),
    }
    return all_data, stats


def _sample_from_policy(policy):
    """Sample an action index from the policy distribution.

    Handles edge cases where the policy may be all zeros (shouldn't
    happen, but defensive coding).
    """
    total = policy.sum()
    if total <= 0:
        # Uniform over non-zero entries
        nonzero = np.where(policy > 0)[0]
        if len(nonzero) == 0:
            return 0
        return int(np.random.choice(nonzero))
    probs = policy / total
    return int(np.random.choice(len(probs), p=probs))
