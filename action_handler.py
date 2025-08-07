"""
Action Handler for Regicide Game
Generates all possible valid actions for attack and defense phases
"""

from typing import List, Dict, Tuple, Optional
from regicide import Card, Game, Enemy
import itertools


class ActionHandler:
    """
    Handles action generation for Regicide players
    Returns action masks for all possible valid card combinations
    """
    
    def __init__(self, max_hand_size: int = 8):
        """
        Initialize the action handler
        
        Args:
            max_hand_size: Maximum number of cards a player can have
        """
        self.max_hand_size = max_hand_size
        # Create a temporary game instance to use its validation logic
        self._temp_game = Game(1)  # Single player for validation only
    
    def get_all_possible_actions(self, 
                               hand: List[Card], 
                               phase: str, 
                               game_state: Optional[Dict] = None) -> List[List[int]]:
        """
        Get all possible actions for the current phase
        
        Args:
            hand: Player's current hand
            phase: Current game phase ("attack" or "defense")
            game_state: Current game state (enemy info, shields, etc.)
            
        Returns:
            List of action masks, each mask is a binary list of length max_hand_size
        """
        if phase == "attack":
            allow_yield = game_state.get('allow_yield', True) if game_state else True
            return self._get_attack_actions(hand, allow_yield)
        elif phase == "defense":
            return self._get_defense_actions(hand, game_state)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def _get_attack_actions(self, hand: List[Card], allow_yield: bool = True) -> List[List[int]]:
        """
        Generate all valid attack combinations following Regicide rules
        Uses the game's built-in validation logic to ensure consistency.
        
        Args:
            hand: Player's current hand
            allow_yield: Whether yielding (empty action) is allowed
            
        Returns:
            List of action masks for valid attack combinations
        """
        valid_actions = []
        hand_size = len(hand)
        
        # Use set to avoid duplicates - track all actions including yield
        seen_actions = set()
        
        # Add yield action (empty mask) if allowed
        if allow_yield:
            yield_action = [0] * self.max_hand_size
            yield_tuple = tuple(yield_action)
            seen_actions.add(yield_tuple)
            valid_actions.append(yield_action)
        
        # Generate all possible combinations of cards (except empty set)
        for r in range(1, hand_size + 1):  # Include all cards (fixed range)
            for combo_indices in itertools.combinations(range(hand_size), r):
                combo_cards = [hand[i] for i in combo_indices]
                
                # Use the game's validation logic
                if self._temp_game._is_valid_combo(combo_cards):
                    # Create action mask
                    action_mask = [0] * self.max_hand_size
                    for idx in combo_indices:
                        if idx < self.max_hand_size:
                            action_mask[idx] = 1
                    
                    # Convert to tuple for set comparison to avoid duplicates
                    action_tuple = tuple(action_mask)
                    if action_tuple not in seen_actions:
                        seen_actions.add(action_tuple)
                        valid_actions.append(action_mask)
        
        return valid_actions
    
    def _get_defense_actions(self, 
                           hand: List[Card], 
                           game_state: Optional[Dict] = None) -> List[List[int]]:
        """
        Generate all minimal valid defense combinations
        
        A valid defense must have total discard value >= enemy attack - shields
        Returns only the minimum subsets of cards that can defend (no supersets)
        
        Args:
            hand: Player's current hand
            game_state: Dict containing 'enemy_attack' and 'current_shields'
            
        Returns:
            List of action masks for minimal valid defense combinations
        """
        if not game_state:
            # If no game state provided, return all possible combinations
            return self._get_all_combinations(hand)
        
        enemy_attack = game_state.get('enemy_attack', 0)
        current_shields = game_state.get('current_shields', 0)
        required_defense = max(0, enemy_attack - current_shields)
    
        hand_size = len(hand)
        
        # First, find all combinations that can defend
        all_valid_combos = []
        for r in range(0, hand_size + 1):
            for combo_indices in itertools.combinations(range(hand_size), r):
                combo_cards = [hand[i] for i in combo_indices]
                
                # Calculate total discard value
                total_defense = sum(card.get_discard_value() for card in combo_cards)
                
                if total_defense >= required_defense:
                    all_valid_combos.append(set(combo_indices))
        
        # Filter to keep only minimal combinations (no subset is also valid)
        minimal_combos = []
        for combo in all_valid_combos:
            is_minimal = True
            for other_combo in all_valid_combos:
                if other_combo != combo and other_combo.issubset(combo):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_combos.append(combo)
        # minimal_cobos = all_valid_combos

        # Convert to action masks
        valid_actions = []
        for combo_indices in minimal_combos:
            action_mask = [0] * self.max_hand_size
            for idx in combo_indices:
                if idx < self.max_hand_size:
                    action_mask[idx] = 1
            valid_actions.append(action_mask)
        
        return valid_actions
    
    def _get_all_combinations(self, hand: List[Card]) -> List[List[int]]:
        """
        Get all possible combinations of cards (for when game state is unknown)
        
        Args:
            hand: Player's current hand
            
        Returns:
            List of all possible action masks
        """
        valid_actions = []
        hand_size = len(hand)
        
        # Generate all possible combinations (including empty)
        for r in range(0, hand_size + 1):
            for combo_indices in itertools.combinations(range(hand_size), r):
                action_mask = [0] * self.max_hand_size
                for idx in combo_indices:
                    if idx < self.max_hand_size:
                        action_mask[idx] = 1
                valid_actions.append(action_mask)
        
        return valid_actions
    
    def _is_valid_attack_combo(self, cards: List[Card]) -> bool:
        """
        Check if a combination of cards is valid for attack
        This method is deprecated - use the game's built-in validation instead.
        Kept for backward compatibility.
        
        Args:
            cards: List of cards to check
            
        Returns:
            True if combination is valid, False otherwise
        """
        # Use the game's validation logic instead
        return self._temp_game._is_valid_combo(cards)

    def is_yield_action(self, action_mask: List[int]) -> bool:
        """
        Check if an action mask represents a yield action (all zeros)
        
        Args:
            action_mask: Binary mask to check
            
        Returns:
            True if action is yield (all zeros), False otherwise
        """
        return all(x == 0 for x in action_mask)

    
    def get_action_count(self, hand: List[Card], phase: str, game_state: Optional[Dict] = None) -> int:
        """
        Get the number of possible actions for the current state
        
        Args:
            hand: Player's current hand
            phase: Current game phase
            game_state: Current game state
            
        Returns:
            Number of possible actions
        """
        actions = self.get_all_possible_actions(hand, phase, game_state)
        return len(actions)
    
    def mask_to_card_indices(self, action_mask: List[int], hand_size: int) -> List[int]:
        """
        Convert action mask to list of card indices
        
        Args:
            action_mask: Binary mask indicating which cards to use
            hand_size: Current hand size
            
        Returns:
            List of card indices to use
        """
        indices = []
        for i in range(min(len(action_mask), hand_size)):
            if action_mask[i] == 1:
                indices.append(i)
        return indices
    
    def cards_to_mask(self, card_indices: List[int]) -> List[int]:
        """
        Convert list of card indices to action mask
        
        Args:
            card_indices: List of card indices
            
        Returns:
            Action mask of length max_hand_size
        """
        action_mask = [0] * self.max_hand_size
        for idx in card_indices:
            if idx < self.max_hand_size:
                action_mask[idx] = 1
        return action_mask


def main():
    """
    Example usage and testing
    """
    from regicide import Card, Suit
    
    # Create test hand
    test_hand = [
        Card(2, Suit.HEARTS),
        Card(2, Suit.DIAMONDS),
        Card(3, Suit.HEARTS),
        Card(3, Suit.DIAMONDS),
        Card(4, Suit.DIAMONDS),
        Card(5, Suit.CLUBS),
        Card(7, Suit.SPADES),
    ]
    
    # Create action handler
    handler = ActionHandler(max_hand_size=7)  # For 2-player game
    
    print("=== Testing Attack Actions ===")
    attack_actions = handler.get_all_possible_actions(test_hand, "attack")
    print(f"Number of valid attack combinations: {len(attack_actions)}")
    
    # Show first few actions
    for i, action in enumerate(attack_actions):
        indices = handler.mask_to_card_indices(action, len(test_hand))
        selected_cards = [test_hand[idx] for idx in indices]
        print(f"Action {i+1}: {action} -> Cards: {[str(card) for card in selected_cards]}")
    
    print("\n=== Testing Defense Actions ===")
    game_state = {
        'enemy_attack': 15,
        'current_shields': 0
    }
    defense_actions = handler.get_all_possible_actions(test_hand, "defense", game_state)
    print(f"Number of valid defense combinations: {len(defense_actions)}")
    
    # Show first few actions
    for i, action in enumerate(defense_actions):
        indices = handler.mask_to_card_indices(action, len(test_hand))
        selected_cards = [test_hand[idx] for idx in indices]
        total_defense = sum(card.get_discard_value() for card in selected_cards)
        print(f"Defense {i+1}: {action} -> Cards: {[str(card) for card in selected_cards]} (Defense: {total_defense})")


if __name__ == "__main__":
    main()
