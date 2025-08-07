"""
Elegant Regicide Gymnasium Environment
A clean, comprehensive implementation that handles all edge cases and supports the updated game mechanics.
Returns PyTorch tensors for seamless integration with neural networks.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union

from regicide import Game, Card, Suit
from action_handler import ActionHandler


class RegicideGymEnv(gym.Env):
    """
    Elegant Regicide environment with proper yield mechanics and comprehensive observation space.
    Supports both simple vector observations and structured card-aware observations.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(
        self, 
        num_players: int = 2, 
        max_hand_size: int = 8, 
        observation_mode: str = "card_aware",  # "vector" or "card_aware"
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Core configuration
        self.num_players = num_players
        self.max_hand_size = max_hand_size
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.card_vocab_size = 54  # 53 cards + 1 padding
        # Game components
        self.game = None
        self.action_handler = ActionHandler(max_hand_size)
        self.current_phase = "attack"  # "attack" or "defense"
        self.valid_actions = []
        
        # Statistics tracking
        self.bosses_killed_this_episode = 0
        self.episode_length = 0
        self.last_damage_dealt = 0
        
        # Action space - fixed size with masking
        self.max_actions = 30  # Sufficient for most scenarios
        self.action_space = spaces.Discrete(self.max_actions)
        
        # Observation space depends on mode
        if observation_mode == "card_aware":
            self._setup_card_aware_observation_space()
        else:
            self._setup_vector_observation_space()
        
        self.reset()
    
    def _setup_card_aware_observation_space(self):
        """Setup structured observation space for card-aware learning"""
        # Card indices: 0 = padding, 1-52 = regular cards, 53 = jester
        
        
        # Define spaces as a dict for structured observations
        self.observation_space = spaces.Dict({
            'hand_cards': spaces.Box(0, self.card_vocab_size-1, (self.max_hand_size,), dtype=np.int32),
            'hand_size': spaces.Box(0, self.max_hand_size, (), dtype=np.int32),
            'enemy_card': spaces.Box(0, self.card_vocab_size-1, (), dtype=np.int32),
            'game_state': spaces.Box(0, 1, (12,), dtype=np.float32),
            'discard_pile_cards': spaces.Box(0, 1, (self.card_vocab_size,), dtype=np.bool_),
            'action_mask': spaces.Box(0, 1, (self.max_actions,), dtype=np.bool_)
        })
    
    def _setup_vector_observation_space(self):
        """Setup flat vector observation space for simple learning"""
        # Hand: one-hot encoding (max_hand_size * card_vocab_size)
        # Enemy: card one-hot + stats (card_vocab_size + 6)
        # Game state: 12 values
        # Discard pile: bit tensor (card_vocab_size)
        # Action info: max_actions mask
        obs_size = (self.max_hand_size * self.card_vocab_size + 
                   self.card_vocab_size + 6 + 12 + self.card_vocab_size + self.max_actions)
        self.observation_space = spaces.Box(0, 1, (obs_size,), dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Union[Dict, torch.Tensor], Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize new game
        self.game = Game(self.num_players)
        self.current_phase = "attack"
        self.bosses_killed_this_episode = 0
        self.episode_length = 0
        self.last_damage_dealt = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Union[Dict, torch.Tensor], float, bool, bool, Dict]:
        """Execute one environment step"""
        # Early termination check
        if self.game.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Handle no valid actions (safety check)
        if len(self.valid_actions) == 0:
            return self._handle_no_valid_actions()
        
        # Validate action
        if action >= len(self.valid_actions):
            return self._handle_invalid_action(action)
        
        # Track state before action
        enemies_before = len(self.game.castle_deck)
        enemy_health_before = (self.game.current_enemy.health - 
                             self.game.current_enemy.damage_taken if self.game.current_enemy else 0)
        
        # Execute action
        success = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(success, enemies_before, enemy_health_before)
        
        # Update episode tracking
        self.episode_length += 1
        
        # Get updated observation
        observation = self._get_observation()
        terminated = self.game.game_over
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _execute_action(self, action: int) -> bool:
        """Execute the chosen action and handle phase transitions"""
        action_mask = self.valid_actions[action]
        card_indices = self.action_handler.mask_to_card_indices(
            action_mask, len(self.game.get_current_player_hand())  # Use sorted hand getter
        )
        
        if self.current_phase == "attack":
            return self._execute_attack_action(card_indices)
        else:  # defense phase
            return self._execute_defense_action(card_indices)
    
    def _execute_attack_action(self, card_indices: List[int]) -> bool:
        """Execute attack phase action with proper flow handling"""
        if not card_indices:
            # Yield action
            result = self.game.yield_turn()
            if not result['success']:
                return False
            
            # Handle defense if needed
            if result['defense_required'] > 0:
                self.current_phase = "defense"
            
            return True
        else:
            # Play cards
            result = self.game.play_card(card_indices)
            if not result['success']:
                return False
            
            # Track damage for rewards
            self.last_damage_dealt = result.get('enemy_damage', 0)
            
            # Handle different result phases
            if result['phase'] == 'next_player_choice':
                # Jester played - choose next player (for now, just go to next)
                next_player = (self.game.current_player + 1) % self.num_players
                self.game.choose_next_player(next_player)
            
            elif result['phase'] == 'defense_needed':
                self.current_phase = "defense"
            
            elif result['phase'] in ['enemy_defeated', 'turn_complete', 'victory']:
                self.current_phase = "attack"
                if result['phase'] == 'enemy_defeated':
                    self.bosses_killed_this_episode += 1
            
            return True
    
    def _execute_defense_action(self, card_indices: List[int]) -> bool:
        """Execute defense phase action"""
        if not card_indices:
            # Cannot defend with no cards - game over
            self.game.game_over = True
            return False
        
        result = self.game.defend_with_card_indices(card_indices)
        if result['success']:
            self.current_phase = "attack"
            return True
        else:
            # Defense failed - game over
            return False
    
    def _get_observation(self) -> Union[Dict, torch.Tensor]:
        """Get current observation based on observation mode"""
        if self.game.game_over:
            return self._get_terminal_observation()
        
        # Update valid actions
        self._update_valid_actions()
        
        if self.observation_mode == "card_aware":
            return self._get_card_aware_observation()
        else:
            return self._get_vector_observation()
    
    def _get_card_aware_observation(self) -> Dict[str, torch.Tensor]:
        """Get structured card-aware observation as PyTorch tensors"""
        current_hand = self.game.get_current_player_hand()  # Use sorted hand getter
        
        # Convert hand to card indices
        hand_indices = [self._card_to_index(card) for card in current_hand]
        hand_indices += [0] * (self.max_hand_size - len(hand_indices))  # Pad with 0
        hand_indices = hand_indices[:self.max_hand_size]  # Truncate if needed
        
        # Game state features (normalized)
        game_state = torch.tensor([
            self.game.current_enemy.health / 40.0,  # Enemy max health
            (self.game.current_enemy.health - self.game.current_enemy.damage_taken) / 40.0,  # Current health
            self.game.current_enemy.spade_protection / 20.0,  # Spade protection
            len(self.game.castle_deck) / 12.0,  # Enemies remaining
            len(self.game.tavern_deck) / 53.0,  # Tavern deck size
            len(current_hand) / self.max_hand_size,  # Hand fullness
            1.0 if self.current_phase == "attack" else 0.0,  # Phase
            1.0 if self.game.can_yield() else 0.0,  # Can yield
            self.game.current_player / self.num_players,  # Current player
            1.0 if self.game.jester_immunity_cancelled else 0.0,  # Jester immunity
            len(self.game.discard_pile) / 100.0,  # Discard pile size
            self.episode_length / 1000.0  # Episode progress
        ], dtype=torch.float32)
        
        # Action mask
        action_mask = torch.zeros(self.max_actions, dtype=torch.bool)
        action_mask[:len(self.valid_actions)] = True
        
        # Discard pile bit tensor
        discard_pile_bits = self._get_discard_pile_bit_tensor().float()
        
        # Get action card indices for each valid action
        action_card_indices = []
        for action_mask_item in self.valid_actions:
            card_indices = self.action_handler.mask_to_card_indices(action_mask_item, len(current_hand))
            if card_indices:
                # Convert card indices to embedding indices
                embedding_indices = [self._card_to_index(current_hand[idx]) for idx in card_indices]
            else:
                embedding_indices = []  # Yield action
            action_card_indices.append(embedding_indices)
        
        return {
            'hand_cards': torch.tensor(hand_indices, dtype=torch.long),
            'hand_size': torch.tensor(len(current_hand), dtype=torch.long),
            'enemy_card': torch.tensor(self._card_to_index(self.game.current_enemy.card), dtype=torch.long),
            'game_state': game_state,
            'discard_pile_cards': discard_pile_bits,
            'action_mask': action_mask,
            'num_valid_actions': torch.tensor(len(self.valid_actions), dtype=torch.long),
            'action_card_indices': action_card_indices
        }
    
    def _get_vector_observation(self) -> torch.Tensor:
        """Get flat vector observation as PyTorch tensor"""
        current_hand = self.game.get_current_player_hand()  # Use sorted hand getter
        
        # Hand encoding (one-hot)
        hand_encoding = torch.zeros(self.max_hand_size * self.card_vocab_size)
        for i, card in enumerate(current_hand[:self.max_hand_size]):
            card_idx = self._card_to_index(card)
            hand_encoding[i * self.card_vocab_size + card_idx] = 1.0
        
        # Enemy encoding (one-hot + stats)
        enemy_encoding = torch.zeros(self.card_vocab_size + 6)
        enemy_card_idx = self._card_to_index(self.game.current_enemy.card)
        enemy_encoding[enemy_card_idx] = 1.0
        enemy_encoding[self.card_vocab_size:] = torch.tensor([
            self.game.current_enemy.health / 40.0,
            (self.game.current_enemy.health - self.game.current_enemy.damage_taken) / 40.0,
            self.game.current_enemy.damage_taken / 40.0,
            self.game.current_enemy.attack / 20.0,
            self.game.current_enemy.spade_protection / 20.0,
            1.0 if self.game.current_enemy.is_defeated() else 0.0
        ])
        
        # Game state
        game_state = torch.tensor([
            self.game.current_player / self.num_players,
            1.0 if self.current_phase == "attack" else 0.0,
            len(self.game.tavern_deck) / 100.0,
            len(self.game.discard_pile) / 100.0,
            len(self.game.castle_deck) / 12.0,
            len(current_hand) / self.max_hand_size,
            1.0 if self.game.can_yield() else 0.0,
            1.0 if self.game.jester_immunity_cancelled else 0.0,
            1.0 if self.game.victory else 0.0,
            self.bosses_killed_this_episode / 12.0,
            self.episode_length / 1000.0,
            self.last_damage_dealt / 20.0
        ], dtype=torch.float32)
        
        # Action mask
        action_mask = torch.zeros(self.max_actions)
        action_mask[:len(self.valid_actions)] = 1.0
        
        # Discard pile bit tensor
        discard_pile_bits = self._get_discard_pile_bit_tensor().float()
        
        # Concatenate all parts
        observation = torch.cat([hand_encoding, enemy_encoding, game_state, discard_pile_bits, action_mask])
        return observation
    
    def _get_terminal_observation(self) -> Union[Dict, torch.Tensor]:
        """Get observation for terminal state"""
        if self.observation_mode == "card_aware":
            return {
                'hand_cards': torch.zeros(self.max_hand_size, dtype=torch.long),
                'hand_size': torch.tensor(0, dtype=torch.long),
                'enemy_card': torch.tensor(0, dtype=torch.long),
                'game_state': torch.zeros(12, dtype=torch.float32),
                'discard_pile_cards': torch.zeros(self.card_vocab_size, dtype=torch.bool),
                'action_mask': torch.zeros(self.max_actions, dtype=torch.bool),
                'num_valid_actions': torch.tensor(0, dtype=torch.long),
                'action_card_indices': []
            }
        else:
            obs_size = (self.max_hand_size * self.card_vocab_size + 
                       self.card_vocab_size + 6 + 12 + self.card_vocab_size + self.max_actions)
            return torch.zeros(obs_size, dtype=torch.float32)
    
    def _update_valid_actions(self):
        """Update valid actions for current state"""
        if self.game.game_over:
            self.valid_actions = []
            return
        
        current_hand = self.game.get_current_player_hand()  # Use sorted hand getter
        game_state = {
            'allow_yield': self.game.can_yield(),
            'enemy_attack':  self.game.current_enemy.get_effective_attack(),
            'current_shields': self.game.current_enemy.spade_protection
        }
        if self.current_phase == "attack":
            self.valid_actions = self.action_handler.get_all_possible_actions(
                current_hand, "attack", game_state
            )
        else:  # defense phase
            required_defense = self.game.current_enemy.get_effective_attack()
            self.valid_actions = self.action_handler.get_all_possible_actions(
                current_hand, "defense", game_state
            )
        
        # Limit to max_actions
        if len(self.valid_actions) > self.max_actions:
            self.valid_actions = self.valid_actions[:self.max_actions]
    
    def _card_to_index(self, card: Card) -> int:
        """Convert card to index for embedding lookup"""
        if card.value == 0:  # Jester
            return 53
        
        # Regular cards: suit * 13 + (value - 1)
        suit_idx = list(Suit).index(card.suit)
        return suit_idx * 13 + (card.value - 1) + 1  # +1 to reserve 0 for padding
    
    def _get_discard_pile_bit_tensor(self) -> torch.Tensor:
        """Create bit tensor representing which cards are in the discard pile"""
        discard_bits = torch.zeros(self.card_vocab_size, dtype=torch.bool)
        
        if hasattr(self.game, 'discard_pile') and self.game.discard_pile:
            for card in self.game.discard_pile:
                card_idx = self._card_to_index(card)
                discard_bits[card_idx] = True
        
        return discard_bits
    
    def _calculate_reward(self, success: bool, enemies_before: int, enemy_health_before: int) -> float:
        """Calculate enhanced reward signal focused on boss progression"""
        if self.game.game_over:
            if self.game.victory:
                return 50.0  # Higher victory bonus
            else:
                # Progressive death penalty based on how far we got
                bosses_killed = 12 - len(self.game.castle_deck)
                death_penalty = max(-20.0, -10.0 - (5 - bosses_killed) * 2.0)  # Less harsh for progress
                return death_penalty
        
        reward = 0.0
        
        # Boss progression rewards (ENHANCED)
        enemies_after = len(self.game.castle_deck)
        bosses_killed_this_step = enemies_before - enemies_after
        
        if bosses_killed_this_step > 0:
            # Progressive boss kill rewards (later bosses worth more)
            total_bosses_killed = 12 - enemies_after
            if total_bosses_killed <= 4:  # Jacks (easiest)
                reward += 10.0
            elif total_bosses_killed <= 8:  # Queens (medium)
                reward += 15.0
            elif total_bosses_killed <= 12:  # Kings (hardest)
                reward += 25.0
            
            # Bonus for reaching milestones
            if total_bosses_killed == 4:  # All Jacks defeated
                reward += 5.0
            elif total_bosses_killed == 8:  # All Queens defeated
                reward += 10.0
            elif total_bosses_killed == 12:  # All Kings defeated (victory)
                reward += 20.0
        
        # Enhanced damage rewards
        if self.last_damage_dealt > 0:
            enemy_health_after = (self.game.current_enemy.health - 
                                self.game.current_enemy.damage_taken)
            progress_ratio = self.last_damage_dealt / max(enemy_health_before, 1)
            
            # Scale damage reward by boss difficulty
            total_bosses_killed = 12 - len(self.game.castle_deck)
            difficulty_multiplier = 1.0 + (total_bosses_killed // 4) * 0.5  # 1.0, 1.5, 2.0
            
            base_damage_reward = min(progress_ratio * 2.0, 1.5)  # Reduced base
            reward += base_damage_reward * difficulty_multiplier
            
            # Bonus for significant damage (>= 50% of enemy health)
            if progress_ratio >= 0.5:
                reward += 1.0 * difficulty_multiplier
        
        # Progress maintenance reward (staying alive longer)
        total_bosses_killed = 12 - len(self.game.castle_deck)
        if total_bosses_killed > 0:
            reward += 0.1 * total_bosses_killed  # Small reward for maintaining progress
        
        # Efficiency rewards
        if success:
            reward += 0.3  # Slightly higher success reward
        else:
            # More forgiving failure penalty early in the game
            if total_bosses_killed == 0:
                reward -= 0.5  # Lighter penalty when just starting
            else:
                reward -= 1.0  # Normal penalty when making progress
        
        # Adaptive step penalty (less harsh early game)
        if total_bosses_killed == 0:
            reward -= 0.02  # Very light step penalty early
        else:
            reward -= 0.05  # Normal step penalty when progressing
        
        # Card efficiency bonus (reward for not hoarding cards)
        hand_size = len(self.game.players[self.game.current_player])
        if hand_size <= 3 and success:  # Playing with few cards successfully
            reward += 0.2
        
        return reward
    
    def _handle_no_valid_actions(self) -> Tuple[Union[Dict, torch.Tensor], float, bool, bool, Dict]:
        """Handle case with no valid actions"""
        self.game.game_over = True
        observation = self._get_observation()
        reward = -10.0  # Penalty for getting into impossible state
        info = self._get_info()
        return observation, reward, True, False, info
    
    def _handle_invalid_action(self, action: int) -> Tuple[Union[Dict, torch.Tensor], float, bool, bool, Dict]:
        """Handle invalid action selection"""
        observation = self._get_observation()
        reward = -0.5  # Small penalty for invalid action
        info = self._get_info()
        info['invalid_action'] = True
        return observation, reward, False, False, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary"""
        action_mask = torch.zeros(self.max_actions, dtype=torch.bool)
        action_mask[:len(self.valid_actions)] = True
        
        return {
            'valid_actions': len(self.valid_actions),
            'action_mask': action_mask,
            'current_player': self.game.current_player,
            'phase': self.current_phase,
            'game_over': self.game.game_over,
            'victory': getattr(self.game, 'victory', False),
            'enemies_remaining': len(self.game.castle_deck) if self.game.castle_deck else 0,
            'bosses_killed': self.bosses_killed_this_episode,
            'episode_length': self.episode_length,
            'can_yield': self.game.can_yield(),
            'last_damage': self.last_damage_dealt
        }
    
    def render(self):
        """Render current game state"""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_human(self):
        """Human-readable rendering"""
        if self.game is None or self.game.game_over:
            print("Game Over" + (" - Victory!" if getattr(self.game, 'victory', False) else " - Defeat"))
            return
        
        state = self.game.get_game_state()
        print(f"\n{'='*60}")
        print(f"🏰 REGICIDE - Episode Length: {self.episode_length}")
        print(f"🐉 Enemy: {state['current_enemy']}")
        print(f"⚔️ Phase: {self.current_phase.upper()}")
        print(f"👤 Player {state['current_player'] + 1}'s turn")
        print(f"🏆 Bosses killed: {self.bosses_killed_this_episode}")
        print(f"🎴 Hand: {', '.join(state['player_hands'][state['current_player']])}")
        print(f"🎯 Valid actions: {len(self.valid_actions)}")
        print(f"🔄 Can yield: {'Yes' if state['can_yield'] else 'No'}")
        print(f"{'='*60}")
    
    def _render_ansi(self) -> str:
        """ANSI string rendering"""
        if self.game is None:
            return "No game active"
        
        state = self.game.get_game_state()
        return (f"Enemy: {state['current_enemy']} | "
                f"Player {state['current_player'] + 1} | "
                f"Phase: {self.current_phase} | "
                f"Actions: {len(self.valid_actions)}")
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable meanings for each valid action"""
        meanings = []
        current_hand = self.game.get_current_player_hand() if self.game else []  # Use sorted hand getter
        
        for i, action_mask in enumerate(self.valid_actions):
            card_indices = self.action_handler.mask_to_card_indices(action_mask, len(current_hand))
            
            if not card_indices:
                meanings.append("Yield turn")
            else:
                cards = [current_hand[idx] for idx in card_indices]
                card_str = ", ".join(str(card) for card in cards)
                meanings.append(f"Play: {card_str}")
        
        return meanings


# Convenience function for easy environment creation
def make_regicide_env(
    num_players: int = 2,
    max_hand_size: int = 8,
    observation_mode: str = "card_aware",
    render_mode: Optional[str] = None
) -> RegicideGymEnv:
    """Create a Regicide environment with specified configuration"""
    return RegicideGymEnv(
        num_players=num_players,
        max_hand_size=max_hand_size,
        observation_mode=observation_mode,
        render_mode=render_mode
    )


# Test the environment
def test_environment():
    """Basic environment test"""
    print("🧪 Testing Regicide Gym Environment")
    print("="*50)
    
    # Test both observation modes
    for obs_mode in ["vector"]:
        print(f"\nTesting {obs_mode} observation mode:")
        
        env = make_regicide_env(
            num_players=4,
            max_hand_size=8,
            observation_mode=obs_mode,
            render_mode="human"
        )
        
        obs, info = env.reset()
        print(f"Observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"Observation keys: {obs.keys()}")
            print(f"Hand shape: {obs['hand_cards'].shape}")
            print(f"Game state shape: {obs['game_state'].shape}")
        else:
            print(f"Observation shape: {obs.shape}")
        
        print(f"Valid actions: {info['valid_actions']}")
        
        # Take a few random steps
        for step in range(3):
            if info['valid_actions'] > 0:
                action = np.random.randint(0, info['valid_actions'])
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step {step+1}: Action {action}, Reward {reward:.2f}, Done {terminated}")
                
                if terminated:
                    break
            else:
                break
        
        env.close()
        print("-" * 30)


if __name__ == "__main__":
    test_environment()
