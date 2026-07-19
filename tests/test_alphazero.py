"""Focused tests for neural-guided ISMCTS and AlphaZero data contracts."""

import numpy as np
import pytest
import torch

from agents.alphazero_agent import AlphaZeroAgent
from game.action_space import SOLO_JESTER_ACTION_ID
from solvers.alphazero.config import AlphaZeroConfig
from solvers.alphazero.featurizer import (
    global_index_to_action_mask,
    information_state_key,
)
from solvers.alphazero.mcts import run_mcts
from solvers.alphazero.trainer import AlphaZeroTrainer
from solvers.env import RegicideEnv


class FixedPolicyNetwork:
    """Network stub assigning all prior mass to one preferred legal action."""

    def __init__(self, preferred_action):
        self.preferred_action = preferred_action

    def predict(self, state, action_mask):
        mask = action_mask.detach().cpu().numpy().astype(bool)
        priors = np.zeros(mask.shape[-1], dtype=np.float32)
        legal_actions = np.flatnonzero(mask)
        selected = (
            self.preferred_action
            if mask[self.preferred_action]
            else int(legal_actions[0])
        )
        priors[selected] = 1.0
        return priors, 0.0


@pytest.mark.parametrize("simulation_count", [0, -1])
def test_config_rejects_non_positive_simulation_budget(simulation_count):
    with pytest.raises(ValueError, match="n_simulations"):
        AlphaZeroConfig(n_simulations=simulation_count)


def test_config_rejects_negative_heuristic_warmup():
    with pytest.raises(ValueError, match="heuristic_warmup_iterations"):
        AlphaZeroConfig(heuristic_warmup_iterations=-1)


@pytest.mark.parametrize("prior_weight", [-0.1, 1.1])
def test_config_rejects_invalid_heuristic_prior_weight(prior_weight):
    with pytest.raises(ValueError, match="heuristic_prior_weight"):
        AlphaZeroConfig(heuristic_prior_weight=prior_weight)


def test_solo_jester_round_trip_uses_sentinel_bit():
    env = RegicideEnv(num_players=1)
    env.reset(seed=5)
    hand = env.game.get_player_hand(env.game.current_player)

    mask = global_index_to_action_mask(
        SOLO_JESTER_ACTION_ID,
        hand,
        env.handler,
    )

    assert mask == [0] * 8 + [1]


def test_information_state_key_distinguishes_public_discard_composition():
    first_env = RegicideEnv(num_players=1)
    first_env.reset(seed=7)
    second_env = first_env.clone()
    first_env.game.discard_pile.append(first_env.game.players[0].pop())
    second_env.game.discard_pile.append(second_env.game.players[0].pop(0))

    assert information_state_key(first_env) != information_state_key(second_env)


def test_root_noise_changes_the_first_self_play_selection(monkeypatch):
    env = RegicideEnv(num_players=1)
    observation, _ = env.reset(seed=11)
    legal_actions = np.flatnonzero(observation["action_mask"]).tolist()
    noise_target = legal_actions[0]
    network_preference = legal_actions[-1]
    network = FixedPolicyNetwork(network_preference)
    config = AlphaZeroConfig(
        n_simulations=1,
        dirichlet_epsilon=1.0,
    )

    def one_hot_noise(alpha):
        noise = np.zeros(len(alpha), dtype=np.float64)
        noise[0] = 1.0
        return noise

    monkeypatch.setattr(np.random, "dirichlet", one_hot_noise)
    policy, _ = run_mcts(
        env,
        network,
        config,
        torch.device("cpu"),
        add_exploration_noise=True,
    )

    assert int(np.argmax(policy)) == noise_target


def test_evaluation_search_does_not_add_root_noise(monkeypatch):
    env = RegicideEnv(num_players=1)
    observation, _ = env.reset(seed=13)
    legal_actions = np.flatnonzero(observation["action_mask"]).tolist()
    network_preference = legal_actions[-1]
    network = FixedPolicyNetwork(network_preference)
    config = AlphaZeroConfig(
        n_simulations=1,
        dirichlet_epsilon=1.0,
    )

    def fail_if_called(_):
        raise AssertionError("evaluation search must not add Dirichlet noise")

    monkeypatch.setattr(np.random, "dirichlet", fail_if_called)
    policy, _ = run_mcts(
        env,
        network,
        config,
        torch.device("cpu"),
    )

    assert int(np.argmax(policy)) == network_preference


def test_warmup_search_mixes_heuristic_action_priors(monkeypatch):
    env = RegicideEnv(num_players=1)
    observation, _ = env.reset(seed=17)
    legal_actions = np.flatnonzero(observation["action_mask"]).tolist()
    heuristic_preference = legal_actions[0]
    network_preference = legal_actions[-1]
    network = FixedPolicyNetwork(network_preference)
    config = AlphaZeroConfig(
        n_simulations=1,
        heuristic_prior_weight=1.0,
    )

    def fixed_scores(_agent, obs, env=None):
        valid_actions = np.flatnonzero(obs["action_mask"]).tolist()
        return {
            action_id: float(action_id == heuristic_preference)
            for action_id in valid_actions
        }

    monkeypatch.setattr(
        "agents.heuristic_agent.HeuristicAgent.score_actions",
        fixed_scores,
    )
    policy, _ = run_mcts(
        env,
        network,
        config,
        torch.device("cpu"),
        use_heuristic_guidance=True,
    )

    assert int(np.argmax(policy)) == heuristic_preference


def test_checkpoint_preserves_network_architecture_and_suffix(tmp_path):
    config = AlphaZeroConfig(
        hidden_dim=32,
        num_hidden_layers=1,
        n_simulations=2,
    )
    trainer = AlphaZeroTrainer(config)
    checkpoint_path = tmp_path / "agent.pt"

    trainer.save_checkpoint(checkpoint_path)
    agent = AlphaZeroAgent(
        checkpoint_path=str(checkpoint_path),
        n_simulations=3,
    )

    assert checkpoint_path.exists()
    assert not (tmp_path / "agent.pt.pt").exists()
    assert agent.config.hidden_dim == 32
    assert agent.config.num_hidden_layers == 1
    assert agent.config.n_simulations == 3


def test_checkpoint_preserves_training_iteration(tmp_path):
    trainer = AlphaZeroTrainer(
        AlphaZeroConfig(hidden_dim=16, num_hidden_layers=1)
    )
    trainer.training_iteration = 7
    checkpoint_path = tmp_path / "resume"
    trainer.save_checkpoint(checkpoint_path)

    restored_trainer = AlphaZeroTrainer(trainer.config)
    restored_trainer.load_checkpoint(checkpoint_path)

    assert restored_trainer.training_iteration == 7
