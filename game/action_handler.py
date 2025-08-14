"""
Action Handler for Regicide Game
Generates all possible valid actions for attack and defense phases
"""

from typing import List, Dict, Tuple, Optional
from game.regicide import Card, Game, Enemy, Suit
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
            return self._get_attack_actions(hand, allow_yield, game_state)
        elif phase == "defense":
            return self._get_defense_actions(hand, game_state)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def _get_attack_actions(self, hand: List[Card], allow_yield: bool = True, game_state: Optional[Dict] = None) -> List[List[int]]:
        """
        Generate all valid attack combinations following Regicide rules
        Uses the game's built-in validation logic to ensure consistency.
        Optionally filters out actions that would make defense impossible.
        
        Args:
            hand: Player's current hand
            allow_yield: Whether yielding (empty action) is allowed
            game_state: Current game state for filtering impossible actions
            
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
        
        # Apply filtering if game state is provided
        if game_state and len(valid_actions) > 1:  # Don't filter if only one action available
            filtered_actions = self._filter_survivable_actions(hand, valid_actions, game_state)
            # Only use filtered actions if some remain (otherwise player is doomed anyway)
            if filtered_actions:
                valid_actions = filtered_actions
        
        return valid_actions
    
    def _filter_survivable_actions(self, hand: List[Card], actions: List[List[int]], game_state: Dict) -> List[List[int]]:
        """
        Filter out attack actions that would make defense impossible.
        
        Args:
            hand: Player's current hand
            actions: List of action masks to filter
            game_state: Current game state with enemy and tavern info
            
        Returns:
            List of action masks that allow for possible defense
        """
        if not game_state:
            return actions
            
        # Extract game state information
        enemy_attack = game_state.get('enemy_attack', 0)
        enemy_health = game_state.get('enemy_health', 0)
        enemy_damage_taken = game_state.get('enemy_damage_taken', 0)
        enemy_suit = game_state.get('enemy_suit')
        jester_immunity_cancelled = game_state.get('jester_immunity_cancelled', False)
        
        if enemy_attack == 0:
            return actions  # No need to defend if enemy doesn't attack
            
        survivable_actions = []
        
        for action_mask in actions:
            # Skip yield action (all zeros) - handle separately
            if self.is_yield_action(action_mask):
                survivable_actions.append(action_mask)
                continue
                
            # Get cards being played
            card_indices = self.mask_to_card_indices(action_mask, len(hand))
            cards_played = [hand[i] for i in card_indices]
            
            # Calculate attack damage
            total_attack = sum(card.get_attack_value() for card in cards_played)
            
            # Calculate total damage including clubs single doubling effect
            total_damage = total_attack
            has_clubs = any(card.suit.value == "♣" for card in cards_played if 
                          not self._is_immune_to_suit_power(card, enemy_suit, jester_immunity_cancelled))
            if has_clubs:
                total_damage += total_attack  # Clubs double the damage once per play
            
            # Check if this action would kill the enemy
            if enemy_damage_taken + total_damage >= enemy_health:
                # Action kills enemy - always include these
                survivable_actions.append(action_mask)
                continue
                
            # Check for Jester - these don't deal damage but cancel immunity
            if any(card.value == 0 for card in cards_played):
                # Jester actions don't trigger enemy attack, so they're safe
                survivable_actions.append(action_mask)
                continue
                
            # Calculate remaining hand after playing cards
            remaining_hand = [card for i, card in enumerate(hand) if i not in card_indices]
            
            # Check if diamonds are played (for potential card draws)
            has_diamonds = any(card.suit.value == "♦" for card in cards_played if 
                             not self._is_immune_to_suit_power(card, enemy_suit, jester_immunity_cancelled))
            
            # If diamonds are played and not immune, don't filter this action
            if has_diamonds:
                survivable_actions.append(action_mask)
                continue
            
            # Calculate spade protection from this attack
            spade_protection = 0
            has_spades = any(card.suit.value == "♠" for card in cards_played if
                             not self._is_immune_to_suit_power(card, enemy_suit, jester_immunity_cancelled))
            if has_spades:
                spade_protection += total_attack
            
            # Calculate effective enemy attack after spade protection
            effective_enemy_attack = max(0, enemy_attack - spade_protection)
            
            # Check if defense is possible with remaining hand
            if self._can_defend_with_hand(remaining_hand, effective_enemy_attack, 0):
                survivable_actions.append(action_mask)
        
        return survivable_actions
    
    def _is_immune_to_suit_power(self, card: Card, enemy_suit, jester_immunity_cancelled: bool) -> bool:
        """Check if a card's suit power is immune due to matching enemy suit"""
        if jester_immunity_cancelled:
            return False
        return card.suit.value == enemy_suit
    
    def _can_defend_with_hand(self, hand: List[Card], required_defense: int, extra_potential: int = 0) -> bool:
        """Check if a hand can defend against the required damage"""
        if required_defense <= 0:
            return True
            
        total_defense = sum(card.get_discard_value() for card in hand) + extra_potential
        return total_defense >= required_defense
    
    def _get_defense_actions(self, 
                           hand: List[Card], 
                           game_state: Optional[Dict] = None) -> List[List[int]]:
        """
        Generate all minimal valid defense combinations
        
        A valid defense must have total discard value >= enemy attack - shields
        Returns only the minimum subsets of cards that can defend (no supersets)
        
        Args:
            hand: Player's current hand
            game_state: Dict containing 'enemy_attack'
            
        Returns:
            List of action masks for minimal valid defense combinations
        """
        if not game_state:
            # If no game state provided, return all possible combinations
            return self._get_all_combinations(hand)
        
        required_defense = game_state.get('enemy_attack', 0)
    
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
    # Create test hand
    test_hand = [
        Card(6, Suit.CLUBS),
        Card(6, Suit.DIAMONDS),
        Card(8, Suit.CLUBS),
        Card(9, Suit.CLUBS),
        Card(10, Suit.CLUBS)
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
        'enemy_attack': 10,
        'current_shields': 8
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
