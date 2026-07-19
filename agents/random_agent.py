"""Uniform-random baseline over the currently legal global actions."""

import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Choose uniformly from the valid actions supplied by the environment."""

    def __init__(self, name="RandomAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def select_action(self, obs, env=None):
        """Sample one legal action without inspecting game-state features.

        Args:
            obs: Observation containing the binary ``action_mask``.
            env: Unused compatibility argument.

        Returns:
            Selected global action identifier, or ``None`` for an empty mask.
        """
        import numpy as np
        action_mask = obs['action_mask']
        valid_indices = np.nonzero(action_mask)[0]
        if len(valid_indices) == 0:
            return None # The engine will handle this as a game over
        return int(random.choice(valid_indices))
