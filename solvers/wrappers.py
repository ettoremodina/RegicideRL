"""Observation adapters used by neural Regicide policies."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from game.action_space import GLOBAL_ACTION_SPACE_SIZE, MAX_HAND_SIZE

class NumericObsWrapper(gym.ObservationWrapper):
    """
    Converts the raw Regicide game state into strict NumPy arrays 
    suitable for Deep RL Neural Networks (e.g., Stable-Baselines3).
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Max hand size is 8. Card values 0-13, Suits 0-4
        self.observation_space = spaces.Dict({
            'hand_values': spaces.Box(
                low=0, high=13, shape=(MAX_HAND_SIZE,), dtype=np.int8
            ),
            'hand_suits': spaces.Box(
                low=0, high=4, shape=(MAX_HAND_SIZE,), dtype=np.int8
            ),
            'enemy_stats': spaces.Box(low=0, high=100, shape=(3,), dtype=np.int8), # health, attack, suit
            'flags': spaces.Box(low=0, high=50, shape=(2,), dtype=np.int8), # defense_phase (0/1), required_defense
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(GLOBAL_ACTION_SPACE_SIZE,),
                dtype=np.int8,
            )
        })
        
        self._last_action_mask = None
        
    def observation(self, obs):
        """Encode cards, enemy state, flags, and mask as fixed NumPy arrays.

        Empty hand slots and absent suits are encoded as zero. Enemy health is
        remaining health rather than starting health.
        """
        hand = obs['hand']
        
        # Parse Hand
        hand_values = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        hand_suits = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        
        suit_map = {"♥": 1, "♦": 2, "♣": 3, "♠": 4}
        for i, card in enumerate(hand):
            if i < MAX_HAND_SIZE:
                hand_values[i] = card.value
                hand_suits[i] = suit_map.get(card.suit.value, 0)
                
        # Parse Enemy
        enemy_stats = np.zeros(3, dtype=np.int8)
        enemy_obj = self.env.unwrapped.game.current_enemy
        if enemy_obj is not None:
            enemy_stats[0] = max(0, enemy_obj.health - enemy_obj.damage_taken)
            enemy_stats[1] = enemy_obj.attack
            enemy_stats[2] = suit_map.get(enemy_obj.card.suit.value, 0)
            
        # Parse Flags
        flags = np.zeros(2, dtype=np.int8)
        flags[0] = 1 if obs['defense_phase'] else 0
        flags[1] = obs['required_defense']
        
        self._last_action_mask = obs['action_mask']
        
        return {
            'hand_values': hand_values,
            'hand_suits': hand_suits,
            'enemy_stats': enemy_stats,
            'flags': flags,
            'action_mask': obs['action_mask']
        }
        
    def action_masks(self):
        """
        Required by sb3-contrib MaskablePPO.
        Returns the mask of the most recent observation.
        """
        if self._last_action_mask is None:
            # If called before the first step/reset, fetch it
            obs = self.env.unwrapped._get_obs()
            return obs['action_mask']
        return self._last_action_mask
