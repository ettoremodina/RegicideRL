import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    A baseline agent that chooses a random valid action.
    """
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
        
    def select_action(self, obs, env=None):
        import numpy as np
        action_mask = obs['action_mask']
        valid_indices = np.nonzero(action_mask)[0]
        if len(valid_indices) == 0:
            return None # The engine will handle this as a game over
        return int(random.choice(valid_indices))
