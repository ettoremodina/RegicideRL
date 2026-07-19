"""
Self-play data generation for AlphaZero.

Plays full games using MCTS + the neural network, recording
``(state, policy, outcome)`` at every decision point.  The outcome
(progress-based value) is only known at game end and is applied
retroactively to all states in the game.
"""

import numpy as np

from ml_logger import get_logger
from solvers.env import RegicideEnv
from solvers.alphazero.featurizer import encode_state
from solvers.alphazero.mcts import run_mcts
from solvers.alphazero.outcomes import enemies_defeated, terminal_value

logger = get_logger(__name__)


def run_self_play_game(
    network,
    config,
    device,
    recorder=None,
    use_heuristic_guidance=False,
):
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
    env = RegicideEnv(num_players=1, recorder=recorder)
    obs, _ = env.reset()
    history, move_count = _play_episode(
        env,
        obs,
        network,
        config,
        device,
        use_heuristic_guidance,
    )

    defeated_count = enemies_defeated(env.game)
    outcome = terminal_value(env.game)
    game_data = [
        (state, policy, outcome) for state, policy in history
    ]
    game_info = {
        "enemies_defeated": defeated_count,
        "victory": env.game.victory,
        "moves": move_count,
        "samples": len(game_data),
    }
    return game_data, game_info


def _play_episode(
    env,
    observation,
    network,
    config,
    device,
    use_heuristic_guidance,
):
    """Play until termination and retain non-forced policy targets."""
    history = []
    move_count = 0
    while not env.game.game_over:
        action_mask_obs = observation["action_mask"]
        valid_actions = np.nonzero(action_mask_obs)[0].tolist()
        if not valid_actions:
            break

        if len(valid_actions) == 1:
            observation, _, _, _, _ = env.step(valid_actions[0])
            move_count += 1
            continue

        network.eval()
        policy, _ = run_mcts(
            env,
            network,
            config,
            device,
            add_exploration_noise=True,
            use_heuristic_guidance=use_heuristic_guidance,
        )
        state_vec = encode_state(env)

        if move_count < config.temp_threshold:
            action_id = _sample_from_policy(policy)
        else:
            action_id = int(np.argmax(policy))

        history.append((state_vec, policy))
        observation, _, _, _, _ = env.step(action_id)
        move_count += 1
    return history, move_count


def generate_self_play_data(
    network,
    config,
    device,
    recorder=None,
    use_heuristic_guidance=False,
):
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
        game_data, game_info = run_self_play_game(
            network,
            config,
            device,
            recorder=recorder,
            use_heuristic_guidance=use_heuristic_guidance,
        )
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
