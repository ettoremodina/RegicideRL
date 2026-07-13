import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    A baseline agent that chooses a random valid action.
    """
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
        
    def select_action(self, obs, env=None):
        valid_actions = obs['valid_actions']
        if not valid_actions:
            return None # The engine will handle this as a game over
        return random.choice(valid_actions)
