import random

from game.action_space import SOLO_JESTER_ACTION_ID
from solvers.env import RegicideEnv


def test_attack_mask_never_offers_an_illegal_yield():
    env = RegicideEnv(num_players=1)
    observation, _ = env.reset(seed=7)

    assert env.game.can_yield() is False
    assert observation["action_mask"][0] == 0


def test_random_masked_actions_are_valid():
    random.seed(11)

    for episode_seed in range(25):
        env = RegicideEnv(num_players=1)
        observation, _ = env.reset(seed=episode_seed)
        done = False

        while not done:
            valid_actions = [
                action for action, valid in enumerate(observation["action_mask"])
                if valid
            ]
            action = random.choice(valid_actions)
            observation, _, terminated, truncated, info = env.step(action)

            assert info.get("success", True), info.get("message")
            done = terminated or truncated


def test_impossible_defense_only_offers_solo_jester():
    env = RegicideEnv(num_players=1)
    observation, _ = env.reset(seed=3)
    env.required_defense = sum(
        card.get_discard_value() for card in env.game.get_current_player_hand()
    ) + 1

    observation = env._get_obs()
    valid_actions = [
        action for action, valid in enumerate(observation["action_mask"])
        if valid
    ]

    assert valid_actions == [SOLO_JESTER_ACTION_ID]
