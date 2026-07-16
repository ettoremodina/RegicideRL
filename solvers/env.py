import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game.regicide import Game
from game.action_handler import ActionHandler

class RegicideEnv(gym.Env):
    """
    Gymnasium wrapper for the Regicide game engine.
    Abstracts away the attack/defense phases and Jester player-choice,
    so the agent just receives a state and a list of valid actions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=1):
        super().__init__()
        self.num_players = num_players
        # Assuming maximum hand size is 8
        self.handler = ActionHandler(max_hand_size=8)
        self.game = None
        self.required_defense = 0
        
        # Action space: 286 global attack actions + 256 hand-relative defense actions + 1 Jester action
        self.action_space = spaces.Discrete(543)
        
        # Observation space: 
        # For now, we only formally define the action_mask for Gym algorithms.
        # The raw game state and hand are passed as dict elements for custom featurizers.
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(543,), dtype=np.int8)
        })
    
    def clone(self) -> 'RegicideEnv':
        """Create a fast clone of the environment for search simulation.
        
        Clones the inner Game state and the wrapper's required_defense,
        so search agents can fork a full env state without affecting
        the real game.
        
        Returns:
            A new RegicideEnv with identical but independent state.
        """
        new_env = object.__new__(RegicideEnv)
        new_env.num_players = self.num_players
        new_env.handler = self.handler  # ActionHandler is stateless, safe to share
        new_env.required_defense = self.required_defense
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        new_env.game = self.game.clone()
        return new_env
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # If the game engine supports seeding, apply it here
            import random
            random.seed(seed)
            
        self.game = Game(num_players=self.num_players)
        self.required_defense = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        state = self.game.get_raw_state()
        current = self.game.current_player
        hand = self.game.get_player_hand(current)
        
        if self.required_defense > 0:
            actions = self.handler.get_all_possible_actions(hand, "defense", {'enemy_attack': self.required_defense})
        else:
            actions = self.handler.get_all_possible_actions(hand, "attack", state)
            
        # Create global action mask for Gymnasium
        phase = "defense" if self.required_defense > 0 else "attack"
        action_mask = np.array(self.handler.get_global_action_mask(hand, phase, state, valid_local_masks=actions), dtype=np.int8)
            
        return {
            'game_state': state,
            'hand': hand,
            'current_player': current,
            'valid_actions': actions, # Kept for backward compatibility with HeuristicAgent
            'action_mask': action_mask,
            'defense_phase': self.required_defense > 0,
            'required_defense': self.required_defense
        }
        
    def step(self, action):
        """
        Takes an action index (0-541) OR a list mask (backward compatibility).
        Returns: next_obs, reward, terminated, truncated, info
        """
        if self.game.game_over:
            return self._get_obs(), 0.0, True, False, {}
            
        hand = self.game.get_player_hand(self.game.current_player)
        
        # Convert action integer to mask if necessary
        is_solo_jester = False
        if isinstance(action, (int, np.integer)):
            if self.action_space.n == 543:
                # Decode 543-dimensional global action ID
                indices = self.handler.global_action_to_hand_indices(int(action), hand)
                is_solo_jester = (indices == [-1])
                # determine if it's a yield by checking if it's attack phase and action == 0
                is_yield = (self.required_defense == 0 and action == 0)
            else:
                # Fallback for old 256 space just in case
                action_mask = [(action >> i) & 1 for i in range(self.handler.max_hand_size)]
                indices = self.handler.mask_to_card_indices(action_mask, len(hand))
                is_yield = self.handler.is_yield_action(action_mask)
        else:
            action_mask = action
            is_solo_jester = (len(action_mask) == 9 and action_mask[8] == 1)
            if is_solo_jester:
                indices = [-1]
            else:
                indices = self.handler.mask_to_card_indices(action_mask, len(hand))
            is_yield = self.handler.is_yield_action(action_mask)
        
        if is_solo_jester:
            phase_str = "step4" if self.required_defense > 0 else "step1"
            res = self.game.use_solo_jester(phase_str)
            if not res.get('success', False):
                return self._get_obs(), -1.0, True, False, res
            # Standard transition reward
            self.required_defense = res.get("defense_required", self.required_defense)
            return self._get_obs(), 0.0, self.game.game_over, False, res
            
        if self.required_defense > 0:
            res = self.game.defend_with_card_indices(indices)
            self.required_defense = 0
            
            # If defense failed, game_over will be True
            if self.game.game_over:
                # Agent died, huge negative reward
                return self._get_obs(), -1.0, True, False, res
            
            reward = 0.0 # Standard transition reward
        else:
            if is_yield:
                res = self.game.yield_turn()
            else:
                res = self.game.play_card(indices)
                
            if not res.get('success', False):
                # Invalid action taken - terminate to prevent infinite loops
                return self._get_obs(), -1.0, True, False, res
                
            self.required_defense = res.get("defense_required", 0)
            
            # If it's a multi-player game and Jester was played
            if res.get("phase") == "next_player_choice":
                # For now, default to player 0 in solo. In multi, we would need to pass this to the agent.
                self.game.choose_next_player(0)
                
            # Basic reward shaping: 
            # +1 for winning, -1 for losing, small reward for killing an enemy
            reward = 0.0
            if res.get("phase") == "enemy_defeated":
                reward = 0.1
                
        terminated = self.game.game_over
        truncated = False
        
        if terminated:
            if self.game.victory:
                reward = 1.0
            else:
                reward = -1.0
                
        return self._get_obs(), reward, terminated, truncated, res
