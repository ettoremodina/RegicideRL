"""
Action Handler for Regicide Game
Generates all possible valid actions for attack and defense phases
"""

from typing import List, Dict, Tuple, Optional
from game.regicide import Card, Game, Enemy, Suit
import itertools
import numpy as np


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
        self._global_attack_actions = self._generate_global_attack_actions()
        self._attack_action_to_id = {
            action["sorted_cards"]: i for i, action in enumerate(self._global_attack_actions)
        }
        
        # Precompute signatures for fast attack generation
        self._attack_action_signatures = []
        for action in self._global_attack_actions:
            counts = {}
            for card in action["cards"]:
                counts[card] = counts.get(card, 0) + 1
            self._attack_action_signatures.append(counts)
            
    def _generate_global_attack_actions(self) -> List[Dict]:
        """
        Generates all 286 possible unique attack actions in Regicide,
        independent of the player's current hand.
        """
        actions = []
        
        # 0. Yield
        actions.append({"type": "Yield", "cards": []})
        
        # Generate a standard deck for reference
        deck = []
        for suit in Suit:
            for val in range(1, 14):
                deck.append(Card(val, suit))
        jester = Card(0, Suit.HEARTS)
        
        # 1. Single Cards
        for card in deck: # All 52 standard cards
            actions.append({"type": "Single", "cards": [card]})
        actions.append({"type": "Single", "cards": [jester]}) # 1 Jester action
        
        # 2. Ace + Ace (Animal Companions)
        aces = [c for c in deck if c.value == 1]
        for combo in itertools.combinations(aces, 2):
            actions.append({"type": "Ace+Ace", "cards": list(combo)})
            
        # 3. Ace + Non-Jester
        non_jesters = [c for c in deck if c.value not in (0, 1)]
        for ace in aces:
            for other in non_jesters:
                actions.append({"type": "Ace+Other", "cards": [ace, other]})
                
        # 4. Same-value combos (sum <= 10)
        for val in range(2, 11):
            same_val_cards = [c for c in deck if c.value == val]
            for r in range(2, 5):
                if val * r <= 10:
                    for combo in itertools.combinations(same_val_cards, r):
                        actions.append({"type": "SameValue", "cards": list(combo)})
                        
        # Precompute sorted tuples for faster mask generation
        for action in actions:
            action["sorted_cards"] = tuple(sorted(action["cards"]))
            
        return actions

    def get_global_action_mask(self, hand: List[Card], phase: str, game_state: Optional[Dict] = None, valid_local_masks: Optional[List[List[int]]] = None) -> List[int]:
        """
        Get a 543-length binary array representing all valid actions globally.
        Indices 0-285: Attack Actions (Global combinations)
        Indices 286-541: Defense Actions (Hand-relative subsets)
        Index 542: Use Solo Jester
        """
        mask = np.zeros(543, dtype=np.int8)
        
        hand_size = len(hand)
        hand_counts = {}
        for c in hand:
            hand_counts[c] = hand_counts.get(c, 0) + 1
            
        if phase == "attack":
            allow_yield = game_state.get('can_yield', True) if game_state else True
            if allow_yield:
                mask[0] = 1
                
            enemy_attack = 0
            enemy_health = 0
            enemy_damage_taken = 0
            enemy_suit = None
            jester_cancelled = False
            
            if game_state:
                enemy_attack = game_state.get('enemy_attack', 0)
                enemy_health = game_state.get('enemy_health', 0)
                enemy_damage_taken = game_state.get('enemy_damage_taken', 0)
                enemy_suit = game_state.get('enemy_suit')
                jester_cancelled = game_state.get('jester_immunity_cancelled', False)
                
            hand_defense = sum(c.get_discard_value() for c in hand)
            
            has_survivable = False
            
            # Find all playable actions first
            playable_actions = []
            for i in range(1, len(self._global_attack_actions)):
                action_counts = self._attack_action_signatures[i]
                can_play = True
                for card, count in action_counts.items():
                    if hand_counts.get(card, 0) < count:
                        can_play = False
                        break
                if can_play:
                    playable_actions.append(i)
            
            if not game_state or enemy_attack == 0:
                for i in playable_actions:
                    mask[i] = 1
            else:
                for i in playable_actions:
                    cards = self._global_attack_actions[i]["cards"]
                    
                    has_jester = any(c.value == 0 for c in cards)
                    if has_jester:
                        mask[i] = 1
                        has_survivable = True
                        continue
                        
                    total_attack = sum(c.get_attack_value() for c in cards)
                    
                    has_clubs = False
                    has_diamonds = False
                    has_spades = False
                    
                    for c in cards:
                        immune = False if jester_cancelled else (c.suit.value == enemy_suit)
                        if not immune:
                            val = c.suit.value
                            if val == "♣": has_clubs = True
                            elif val == "♦": has_diamonds = True
                            elif val == "♠": has_spades = True
                            
                    if has_diamonds:
                        mask[i] = 1
                        has_survivable = True
                        continue
                        
                    total_damage = total_attack * 2 if has_clubs else total_attack
                    if enemy_damage_taken + total_damage >= enemy_health:
                        mask[i] = 1
                        has_survivable = True
                        continue
                        
                    spade_protection = total_attack if has_spades else 0
                    eff_enemy_attack = max(0, enemy_attack - spade_protection)
                    
                    played_defense = sum(c.get_discard_value() for c in cards)
                    remaining_defense = hand_defense - played_defense
                    
                    if remaining_defense >= eff_enemy_attack:
                        mask[i] = 1
                        has_survivable = True
                        
                # If no survivable actions (excluding yield), revert to all playable
                if not has_survivable:
                    for i in playable_actions:
                        mask[i] = 1
                        
        elif phase == "defense":
            offset = len(self._global_attack_actions)
            req = game_state.get('enemy_attack', 0) if game_state else 0
            hand_vals = np.array([c.get_discard_value() for c in hand])
            
            has_defense = False
            for b in range(1, 1 << hand_size):
                defense = 0
                for i in range(hand_size):
                    if b & (1 << i):
                        defense += hand_vals[i]
                        
                if defense >= req:
                    is_min = True
                    for i in range(hand_size):
                        if b & (1 << i):
                            if defense - hand_vals[i] >= req:
                                is_min = False
                                break
                    if is_min:
                        mask[offset + b] = 1
                        has_defense = True
                        
            if not has_defense:
                if hand_size > 0:
                    for b in range(1, 1 << hand_size):
                        mask[offset + b] = 1
                else:
                    mask[0] = 1

        if game_state and game_state.get('can_use_solo_jester', False):
            mask[542] = 1
            
        # Fallback: MaskablePPO fails drastically if the mask is all zeros
        if sum(mask) == 0:
            mask[0] = 1
            
        return mask.tolist()

    def get_all_possible_actions(self, hand: List[Card], phase: str, game_state: Optional[Dict] = None) -> List[List[int]]:
        """
        Get all possible actions for the current phase
        Backward compatibility wrapper that derives local masks from the global mask.
        """
        global_mask = self.get_global_action_mask(hand, phase, game_state)
        local_masks = []
        
        if phase == "attack":
            for i in range(len(self._global_attack_actions)):
                if global_mask[i]:
                    if i == 0:
                        local_masks.append([0] * self.max_hand_size)
                    else:
                        indices = self.global_action_to_hand_indices(i, hand)
                        local_masks.append(self.cards_to_mask(indices))
        elif phase == "defense":
            offset = len(self._global_attack_actions)
            for i in range(offset, offset + 256):
                if global_mask[i]:
                    val = i - offset
                    local_mask = [0] * self.max_hand_size
                    for j in range(self.max_hand_size):
                        if val & (1 << j):
                            local_mask[j] = 1
                    local_masks.append(local_mask)
                    
        if global_mask[542]:
            local_masks.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
            
        return local_masks

    def is_yield_action(self, action_mask: List[int]) -> bool:
        """Check if an action mask represents a yield action (all zeros)"""
        return all(x == 0 for x in action_mask)

    def mask_to_card_indices(self, action_mask: List[int], hand_size: int) -> List[int]:
        """Convert action mask to list of card indices"""
        indices = []
        for i in range(min(len(action_mask), hand_size)):
            if action_mask[i] == 1:
                indices.append(i)
        return indices
    
    def cards_to_mask(self, card_indices: List[int]) -> List[int]:
        """Convert list of card indices to action mask"""
        action_mask = [0] * self.max_hand_size
        for idx in card_indices:
            if idx < self.max_hand_size:
                action_mask[idx] = 1
        return action_mask

    def global_action_to_hand_indices(self, action_id: int, hand: List[Card]) -> List[int]:
        """
        Decodes a global action ID (0-541) into a list of hand indices to pass to the env.
        """
        if action_id == 542:
            return [-1]
            
        if action_id < 0 or action_id >= 543:
            raise ValueError(f"Invalid global action id: {action_id}")
            
        offset = len(self._global_attack_actions)
            
        if action_id < offset:
            if action_id == 0:
                return []
            global_action = self._global_attack_actions[action_id]
            cards_to_play = global_action["cards"]
            
            indices = []
            used = set()
            for card in cards_to_play:
                found = False
                for i, hand_card in enumerate(hand):
                    if i not in used and hand_card == card:
                        indices.append(i)
                        used.add(i)
                        found = True
                        break
                if not found:
                    hand_str = ", ".join(str(c) for c in hand)
                    req_str = ", ".join(str(c) for c in cards_to_play)
                    raise ValueError(f"Action requires {card} (full action: {req_str}) but it's not in hand or already used. Hand: [{hand_str}]")
            return indices
            
        else:
            # Defense action
            val = action_id - offset
            indices = []
            for i in range(self.max_hand_size):
                if val & (1 << i):
                    indices.append(i)
            # Ensure indices don't exceed current hand size
            return [idx for idx in indices if idx < len(hand)]


def main():
    pass

if __name__ == "__main__":
    main()
