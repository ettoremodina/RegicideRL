"""Gymnasium adapter exposing Regicide through one masked global action space."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import secrets

from game.action_space import GLOBAL_ACTION_SPACE_SIZE, MAX_HAND_SIZE
from game.regicide import Game
from game.action_handler import ActionHandler
from integrations.regicide_logging import GameRecorder, serialize_game

class RegicideEnv(gym.Env):
    """
    Gymnasium wrapper for the Regicide game engine.
    Abstracts away the attack/defense phases and Jester player-choice,
    so the agent just receives a state and a list of valid actions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=1, recorder: GameRecorder | None = None):
        super().__init__()
        self.num_players = num_players
        self.recorder = recorder
        self.handler = ActionHandler(max_hand_size=MAX_HAND_SIZE)
        self.game = None
        self.required_defense = 0
        
        self.action_space = spaces.Discrete(GLOBAL_ACTION_SPACE_SIZE)
        
        # Observation space: 
        # For now, we only formally define the action_mask for Gym algorithms.
        # The raw game state and hand are passed as dict elements for custom featurizers.
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(GLOBAL_ACTION_SPACE_SIZE,),
                dtype=np.int8,
            )
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
        new_env.recorder = None
        new_env.handler = self.handler  # ActionHandler is stateless, safe to share
        new_env.required_defense = self.required_defense
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        new_env.game = self.game.clone()
        return new_env
        
    def reset(self, seed=None, options=None):
        """Start a new game and initialize optional persistent recording.

        Args:
            seed: Random seed. A seed is generated when recording is enabled so
                every persisted game can be reproduced.
            options: Reserved by the Gymnasium reset contract.

        Returns:
            Initial observation and an empty info dictionary.
        """
        if seed is None and self.recorder and self.recorder.enabled:
            seed = secrets.randbits(63)
        super().reset(seed=seed)
        if self.recorder and self.recorder.active:
            self.recorder.abort("environment_reset")
        if seed is not None:
            # If the game engine supports seeding, apply it here
            import random
            random.seed(seed)
            
        self.game = Game(num_players=self.num_players)
        self.required_defense = 0
        if self.recorder and self.recorder.enabled:
            self.recorder.begin_game(
                self.game,
                seed=seed,
                metadata={"source": "RegicideEnv"},
            )
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        """Build the mixed raw observation and its legal global-action mask."""
        current = self.game.current_player
        hand = self.game.get_player_hand(current)
        
        state_info = self.game.get_raw_state()
        if self.required_defense > 0:
            state_info["enemy_attack"] = self.required_defense
        
        # Create global action mask for Gymnasium
        phase = "defense" if self.required_defense > 0 else "attack"
        action_mask = self.handler.get_global_action_mask(hand, phase, state_info)
            
        return {
            'game_state': state_info,
            'hand': hand,
            'current_player': current,
            'action_mask': action_mask,
            'defense_phase': self.required_defense > 0,
            'required_defense': self.required_defense
        }
        
    def step(self, action):
        """Execute and persist one real environment transition."""
        recording_enabled = bool(self.recorder and self.recorder.enabled)
        state_before = serialize_game(self.game) if recording_enabled else None
        action_record = self._describe_action(action)
        transition = self._step_impl(action)
        observation, _, terminated, truncated, result = transition
        if recording_enabled:
            self.recorder.record_event(
                action_record,
                result,
                self.game,
                state_before=state_before,
            )
            if terminated or truncated or self.game.game_over:
                reason = result.get("message") if isinstance(result, dict) else None
                self.recorder.finish(self.game, reason=reason)
        return observation, transition[1], terminated, truncated, result

    def _step_impl(self, action):
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
            if self.action_space.n == GLOBAL_ACTION_SPACE_SIZE:
                # Decode the global action ID.
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
            if self._end_if_defense_is_impossible():
                res["message"] += " The new hand cannot defend against the attack."
                res["game_over"] = True
                return self._get_obs(), -1.0, True, False, res
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
            if self._end_if_defense_is_impossible():
                res["message"] += " No legal defense is available."
                res["game_over"] = True
            
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

    def _describe_action(self, action):
        """Convert a raw action into a stable, replay-oriented description."""
        hand = self.game.get_player_hand(self.game.current_player)
        description = {
            "action_id": int(action) if isinstance(action, (int, np.integer)) else None,
            "raw_action": action,
            "player": self.game.current_player,
            "phase": "defense" if self.required_defense > 0 else "attack",
        }
        try:
            if isinstance(action, (int, np.integer)):
                indices = self.handler.global_action_to_hand_indices(int(action), hand)
            else:
                indices = self.handler.mask_to_card_indices(action, len(hand))
            description["card_indices"] = indices
            description["cards"] = [
                str(hand[index]) for index in indices if 0 <= index < len(hand)
            ]
            description["kind"] = self._action_kind(indices)
        except (IndexError, TypeError, ValueError):
            description["kind"] = "invalid"
        return description

    def _action_kind(self, indices):
        """Classify decoded indices using the current attack or defense phase."""
        if indices == [-1]:
            return "solo_jester"
        if not indices and self.required_defense == 0:
            return "yield"
        return "defend" if self.required_defense > 0 else "play"

    def _end_if_defense_is_impossible(self):
        """End a forced-defense state when no card play or jester can save it."""
        if self.required_defense <= 0:
            return False
        if self.game.can_defend() or self.game.can_use_solo_jester():
            return False
        self.game.game_over = True
        return True
