"""
Rule-Based Semi-Random Policy for Regicide
Combines rule-based suggestions with neural network action-level corrections.

The policy applies all rules equally and accumulates their scores, then uses a neural 
network to learn context-dependent corrections for individual actions. This allows the 
policy to start with rule-based behavior and gradually learn improvements through training.

Game State Features (11 dimensions):
    Enemy features (0-3):
        0: enemy_max_health / 40.0
        1: enemy_current_health / 40.0  
        2: enemy_damage_ratio
        3: enemy_effective_attack / 20.0
    
    Deck features (4-6):
        4: tavern_deck_size
        5: enemies_remaining
        6: discard_pile_size
    
    Hand features (7):
        7: hand_fullness
    
    Context features (8-9):
        8: is_attack_phase (binary)
        9: can_yield (binary)
    
    Player context (10):
        10: current_player / num_players
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from game.regicide import Card, Suit

# Game State Index Definitions
# Enemy features
ENEMY_MAX_HEALTH = 0
ENEMY_CURRENT_HEALTH = 1
ENEMY_DAMAGE_RATIO = 2
ENEMY_EFFECTIVE_ATTACK = 3

# Deck features
TAVERN_DECK_SIZE = 4
ENEMIES_REMAINING = 5
DISCARD_PILE_SIZE = 6

# Hand features
HAND_FULLNESS = 7

# Context features
IS_ATTACK_PHASE = 8
CAN_YIELD = 9

# Player context
CURRENT_PLAYER = 10


class RuleBasedPolicy(nn.Module):
    """
    Semi-random policy that combines rule-based suggestions with neural network action corrections.
    
    Architecture:
    - 11 strategic rules each provide equal-weight suggestions for actions
    - Neural network learns context-dependent corrections for individual actions
    - Final action scores = sum(rule_scores) + neural_corrections
    - Compatible with existing training infrastructure
    """
    
    def __init__(self, max_hand_size: int = 8, max_actions: int = 20, 
                 game_state_dim: int = 11, hidden_dim: int = 64, temperature: float = 1.0):  # Updated to 11 features
        super(RuleBasedPolicy, self).__init__()
        
        self.max_hand_size = max_hand_size
        self.max_actions = max_actions
        self.game_state_dim = game_state_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature  # For exploration control

        self.card_embed_dim = 12  # Embedding size for cards (can be adjusted)
        
        # Define all rules
        self.rules = [
            self._rule_prioritize_clubs_high_damage,
            self._rule_prioritize_perfect_kill,
            self._rule_prioritize_diamonds_low_cards,
            self._rule_prioritize_hearts_low_tavern,
            self._rule_prioritize_perfect_defense,
            self._rule_prioritize_enemy_suit_defense,
            self._rule_prioritize_spades_expect_damage,
            self._rule_avoid_all_diamonds_at_once,
            self._rule_save_face_cards_for_attack,
            self._rule_maintain_hand_diversity,
            self._rule_consider_tavern_size_diamonds
        ]
        
        self.num_rules = len(self.rules)
        
        # Neural network for action score correction
        # Input: game state features + action representation
        # Output: single correction value per action (initialized to output 0)
        action_input_dim = game_state_dim + self.card_embed_dim  # game state + action cards representation
        self.action_correction_network = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Single correction value per action
        )
        
        # Card embedding for action representation
        self.card_embedding = nn.Embedding(54, self.card_embed_dim, padding_idx=0)
        
        # Initialize weights to start with zero corrections
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network to start with zero corrections"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize to very small values so initial corrections are near zero
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.padding_idx is not None:
                    torch.nn.init.constant_(m.weight[m.padding_idx], 0)

    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: combine equal rule weights with neural network action corrections
        
        Args:
            observation: Game observation dictionary
            
        Returns:
            Action logits for valid actions
        """
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 1 else 1
        device = observation['hand_cards'].device
        
        # Ensure batch dimension
        game_state = observation['game_state']
        if game_state.dim() == 1:
            game_state = game_state.unsqueeze(0)
        
        num_valid_actions = observation['num_valid_actions'].item()
        
        if num_valid_actions == 0:
            # No valid actions - return dummy logits
            return torch.full((batch_size, self.max_actions), -1e8, device=device)
        
        # Initialize action scores with zero base
        action_scores = torch.zeros(batch_size, num_valid_actions, device=device)
        
        # Apply all rules with equal weight and accumulate scores
        for rule_func in self.rules:
            rule_scores = rule_func(observation)  # [num_valid_actions] or None
            if rule_scores is not None:
                # Normalize and prepare rule scores
                rule_scores = self._normalize_rule_scores(rule_scores, num_valid_actions, batch_size, device)
                # Add rule scores equally weighted
                action_scores += rule_scores
        
        
        for i in range(num_valid_actions):
            # Get better action representation based on actual action cards
            action_embed = self._get_action_embedding(observation, i, device)
            
            # Combine state and action features
            combined_input = torch.cat([game_state, action_embed], dim=-1)
            
            # Get correction for this action
            correction = self.action_correction_network(combined_input).squeeze(-1)
            
            # Apply correction to action scores
            action_scores[:, i] += correction
        
        # Convert to logits and pad for invalid actions
        final_logits = action_scores
        
        # Pad with very negative values for invalid actions
        if num_valid_actions < self.max_actions:
            padding = torch.full((batch_size, self.max_actions - num_valid_actions), -1e8, device=device)
            final_logits = torch.cat([final_logits, padding], dim=1)
        
        return final_logits
    
    # ======================== RULE IMPLEMENTATIONS ========================
    
    def _rule_prioritize_clubs_high_damage(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize using actions with clubs when enemy damage is high"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
            
        enemy_effective_attack = observation['game_state'][0][ENEMY_EFFECTIVE_ATTACK].item()  # Normalized
        if enemy_effective_attack < 0.8:  # Not high damage
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == Suit.CLUBS for card in action_cards):
                scores[i] = 2.0  # Boost clubs actions
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_perfect_kill(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize actions which perfect kill the enemy"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None

        # Check if perfect kill is possible by looking at enemy health and hand cards
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        enemy_current_health = observation['game_state'][0][ENEMY_CURRENT_HEALTH].item() * 40.0  # Denormalize
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards:
                total_damage = sum(card.get_attack_value() for card in action_cards)
                # Check for clubs doubling
                if any(card.suit == Suit.CLUBS for card in action_cards):
                    total_damage *= 2
                
                if abs(total_damage - enemy_current_health) < 1.0:  # Perfect kill
                    scores[i] = 3.0  # Strong boost for perfect kills
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_diamonds_low_cards(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize actions with DIAMONDS when card count is low"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
            
        hand_fullness = observation['game_state'][0][HAND_FULLNESS].item()
        if hand_fullness > 0.4:  # Not low hand (more than 40% full)
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == Suit.DIAMONDS for card in action_cards):
                scores[i] = 2.0  # Boost diamond actions when hand is low
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_hearts_low_tavern(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize HEARTS when the tavern is low on cards"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
            
        tavern_deck_size = observation['game_state'][0][TAVERN_DECK_SIZE].item()
        if tavern_deck_size > 15:  # Not low tavern (more than 15 cards)
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == Suit.HEARTS for card in action_cards):
                scores[i] = 1.8  # Boost hearts when tavern is low
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_perfect_defense(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """During defense, prioritize perfect defense actions"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 1:  # Attack phase
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        enemy_effective_attack = observation['game_state'][0][ENEMY_EFFECTIVE_ATTACK].item() * 20.0  # Denormalize
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards:
                total_defense = sum(card.get_discard_value() for card in action_cards)
                if abs(total_defense - enemy_effective_attack) < 1.0:  # Perfect defense
                    scores[i] = 2.5  # Strong boost for perfect defense
                elif total_defense >= enemy_effective_attack:
                    scores[i] = 1.5  # Moderate boost for sufficient defense
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_enemy_suit_defense(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize using the suit of the current enemy to defend"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 1:  # Attack phase
            return None
        
        enemy_card_idx = observation['enemy_card'].item()
        enemy_suit = self._index_to_suit(enemy_card_idx)
        if enemy_suit is None:
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == enemy_suit for card in action_cards):
                scores[i] = 1.5  # Boost enemy suit for defense
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_prioritize_spades_expect_damage(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Prioritize spades when you expect to take damage"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
            
        enemy_effective_attack = observation['game_state'][0][ENEMY_EFFECTIVE_ATTACK].item()
        if enemy_effective_attack < 0.3:  # Low enemy attack
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == Suit.SPADES for card in action_cards):
                scores[i] = 1.6  # Boost spades when expecting damage
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_avoid_all_diamonds_at_once(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Avoid using all diamonds at once"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        diamond_cards_in_hand = sum(1 for card in hand_cards if card.suit == Suit.DIAMONDS)
        
        if diamond_cards_in_hand <= 1:
            return None  # Not relevant if we have 0-1 diamonds
        
        num_valid_actions = observation['num_valid_actions'].item()
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards:
                diamonds_used = sum(1 for card in action_cards if card.suit == Suit.DIAMONDS)
                if diamonds_used >= diamond_cards_in_hand:  # Using all diamonds
                    scores[i] = 0.3  # Penalize using all diamonds
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_save_face_cards_for_attack(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Save face cards for attack unless absolutely necessary"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 1:  # Attack phase
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.value >= 11 for card in action_cards):
                # Check if we can defend without face cards
                other_actions_can_defend = False
                for j in range(num_valid_actions):
                    if j != i:
                        other_cards = self._get_action_cards(observation, j, hand_cards)
                        if other_cards and not any(card.value >= 11 for card in other_cards):
                            other_actions_can_defend = True
                            break
                
                if other_actions_can_defend:
                    scores[i] = 0.5  # Penalize using face cards if alternatives exist
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_maintain_hand_diversity(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Don't empty all of one suit from hand"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        
        # Count cards per suit
        suit_counts = {suit: 0 for suit in Suit}
        for card in hand_cards:
            suit_counts[card.suit] += 1
        
        num_valid_actions = observation['num_valid_actions'].item()
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards:
                for suit in Suit:
                    cards_of_suit_used = sum(1 for card in action_cards if card.suit == suit)
                    if cards_of_suit_used > 0 and cards_of_suit_used == suit_counts[suit]:
                        scores[i] *= 0.7  # Penalize emptying a suit
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    def _rule_consider_tavern_size_diamonds(self, observation: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Consider tavern deck size when deciding on diamond usage"""
        if observation['game_state'][0][IS_ATTACK_PHASE].item() == 0:  # Not attack phase
            return None
        
        tavern_deck_size = observation['game_state'][0][TAVERN_DECK_SIZE].item()
        if tavern_deck_size > 25:  # Plenty of cards in tavern
            return None
        
        hand_cards = self._get_hand_cards_from_observation(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        scores = torch.ones(num_valid_actions)
        
        for i in range(num_valid_actions):
            action_cards = self._get_action_cards(observation, i, hand_cards)
            if action_cards and any(card.suit == Suit.DIAMONDS for card in action_cards):
                if tavern_deck_size < 10:  # Very low tavern
                    scores[i] = 0.6  # Penalize diamonds when tavern is very low
                else:
                    scores[i] = 0.8  # Slight penalty when tavern is low
        
        return F.softmax(scores, dim=0).unsqueeze(0)
    
    # ======================== HELPER METHODS ========================
    
    def _get_hand_cards_from_observation(self, observation: Dict[str, torch.Tensor]) -> List[Card]:
        """Convert hand card indices to Card objects"""
        hand_indices = observation['hand_cards']
        if hand_indices.dim() > 1:
            hand_indices = hand_indices.squeeze(0)
        
        hand_size = observation['hand_size'].item()
        cards = []
        
        for i in range(hand_size):
            card_idx = hand_indices[i].item()
            card = self._index_to_card(card_idx)
            if card:
                cards.append(card)
        
        return cards
    
    def _get_action_cards(self, observation: Dict[str, torch.Tensor], action_idx: int, hand_cards: List[Card]) -> List[Card]:
        """Get the cards involved in a specific action"""
        if action_idx >= len(observation['action_card_indices']):
            return []
        
        card_indices = observation['action_card_indices'][action_idx]
        if not card_indices:  # Yield action
            return []
        
        action_cards = []
        for idx in card_indices:
            if idx < len(hand_cards):
                action_cards.append(hand_cards[idx])
        
        return action_cards
    
    def _index_to_card(self, idx: int) -> Optional[Card]:
        """Convert card index to Card object"""
        if idx == 0:  # Padding
            return None
        elif idx == 53:  # Jester
            return Card(0, Suit.HEARTS)  # Jester represented as 0 of Hearts
        else:
            # Regular cards: suit * 13 + (value - 1) + 1
            idx -= 1  # Remove padding offset
            suit_idx = idx // 13
            value = (idx % 13) + 1
            
            if 0 <= suit_idx < 4:
                suits = list(Suit)
                return Card(value, suits[suit_idx])
        
        return None
    
    def _index_to_suit(self, idx: int) -> Optional[Suit]:
        """Convert card index to Suit"""
        card = self._index_to_card(idx)
        return card.suit if card else None
    
    def _get_action_embedding(self, observation: Dict[str, torch.Tensor], action_idx: int, device) -> torch.Tensor:
        """Get a meaningful embedding for an action based on the cards involved"""
        hand_cards = self._get_hand_cards_from_observation(observation)
        action_cards = self._get_action_cards(observation, action_idx, hand_cards)
        
        if not action_cards:  # Yield action or no cards
            # Use a special "empty action" embedding (index 0 - padding)
            return self.card_embedding(torch.tensor([0], device=device))
        
        # For multiple cards, average their embeddings
        card_indices = []
        for card in action_cards:
            # Convert card back to index for embedding lookup
            if card.value == 0:  # Jester
                card_indices.append(53)
            else:
                # Regular cards: suit * 13 + (value - 1) + 1
                suit_idx = list(Suit).index(card.suit)
                idx = suit_idx * 13 + (card.value - 1) + 1
                card_indices.append(idx)
        
        # Get embeddings and average them
        card_tensor = torch.tensor(card_indices, device=device)
        embeddings = self.card_embedding(card_tensor)  # [num_cards, embed_dim]
        
        # Average the embeddings to get a single action representation
        action_embed = embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
        
        return action_embed
    
    def _normalize_rule_scores(self, rule_scores, num_valid_actions: int, batch_size: int, device) -> torch.Tensor:
        """Normalize rule scores to consistent format and range"""
        # Ensure proper shape and device
        if isinstance(rule_scores, (list, np.ndarray)):
            rule_scores = torch.tensor(rule_scores, dtype=torch.float32, device=device)
        if rule_scores.dim() == 1:
            rule_scores = rule_scores.unsqueeze(0)
        
        # Pad or truncate to match valid actions
        if rule_scores.size(1) > num_valid_actions:
            rule_scores = rule_scores[:, :num_valid_actions]
        elif rule_scores.size(1) < num_valid_actions:
            padding = torch.zeros(batch_size, num_valid_actions - rule_scores.size(1), device=device)
            rule_scores = torch.cat([rule_scores, padding], dim=1)
        
        # Normalize to prevent any single rule from dominating
        # Convert from softmax probabilities back to logits for accumulation
        rule_scores = torch.log(rule_scores + 1e-8)  # Add epsilon to avoid log(0)
        
        return rule_scores
    
    # ======================== INTERFACE COMPATIBILITY ========================
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   action_mask: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """Get action and log probability from the policy (compatible with CardAwarePolicy)"""
        # Ensure we're in eval mode for consistent action selection
        was_training = self.training
        self.eval()
        
        # Ensure batch dimension
        if observation['hand_cards'].dim() == 1:
            for key, value in observation.items():
                if isinstance(value, torch.Tensor):
                    observation[key] = value.unsqueeze(0)
        
        # Forward pass
        logits = self.forward(observation)  # [1, max_actions]
        
        # Apply action mask (only consider valid actions)
        num_valid_actions = observation['num_valid_actions'].item()
        if num_valid_actions > 0:
            # Get probabilities using the same method as get_action_probabilities
            valid_logits = logits[:, :num_valid_actions] #/ self.temperature
            probs = F.softmax(valid_logits, dim=-1)
            
            # Choose the most probable action
    
            # Sample from valid actions
            action_dist = torch.distributions.Categorical(probs)
            # action = torch.argmax(probs, dim=-1)
            action = action_dist.sample()
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)))
            log_prob = action_dist.log_prob(action)

            # Restore training mode
            if was_training:
                self.train()
                
            return action.item(), log_prob.squeeze()
        else:
            # No valid actions (shouldn't happen)
            if was_training:
                self.train()
            return 0, torch.tensor(0.0)
    
    def get_action_probabilities(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get action probabilities for analysis (compatible with CardAwarePolicy)"""
        # Ensure we're in eval mode for consistent results
        was_training = self.training
        self.eval()
        
        logits = self.forward(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        if num_valid_actions > 0:
            # Only return probabilities for valid actions with temperature scaling
            valid_logits = logits[:, :num_valid_actions]# / self.temperature
            result = F.softmax(valid_logits, dim=-1)
        else:
            result = torch.zeros(1, 1)
        
        # Restore training mode
        if was_training:
            self.train()
            
        return result
    
    def analyze_decision(self, observation: Dict[str, torch.Tensor], 
                        card_names: Optional[List[str]] = None) -> Dict:
        """Analyze the decision-making process for debugging (compatible with CardAwarePolicy)"""
        # Ensure we're in eval mode for consistent results
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Ensure batch dimension
            if observation['hand_cards'].dim() == 1:
                for key, value in observation.items():
                    if isinstance(value, torch.Tensor):
                        observation[key] = value.unsqueeze(0)
        
            logits = self.forward(observation)
            probs = self.get_action_probabilities(observation)
            
            num_valid_actions = observation['num_valid_actions'].item()
            
            analysis = {
                'action_probabilities': probs.squeeze().tolist() if probs.numel() > 0 else [],
                'action_logits': logits[:, :num_valid_actions].squeeze().tolist() if num_valid_actions > 0 else [],
                'hand_size': observation['hand_size'].item(),
                'num_valid_actions': num_valid_actions,
                'game_state': observation['game_state'].squeeze().tolist(),
                'rule_names': [func.__name__ for func in self.rules],
                'policy_type': 'rule_based_with_corrections'
            }
            
            # Add card information if provided
            if card_names:
                analysis['hand_cards'] = card_names
        
        # Restore training mode
        if was_training:
            self.train()
            
        return analysis
                
