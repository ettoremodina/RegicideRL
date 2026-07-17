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
            cards = action["cards"]
            action["has_jester"] = any(c.value == 0 for c in cards)
            action["total_attack"] = sum(c.get_attack_value() for c in cards)
            action["played_defense"] = sum(c.get_discard_value() for c in cards)
            
            suit_vals = set(c.suit.value for c in cards)
            action["has_clubs"] = "♣" in suit_vals
            action["has_diamonds"] = "♦" in suit_vals
            action["has_spades"] = "♠" in suit_vals
            action["card_suits"] = list(suit_vals)
            
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
            
            aces = []
            non_jesters = []
            same_val_groups = {}
            for card in hand:
                idx = self._attack_action_to_id.get((card,))
                if idx is not None:
                    playable_actions.append(idx)
                    
                if card.value == 1:
                    aces.append(card)
                elif card.value != 0:
                    non_jesters.append(card)
                if 2 <= card.value <= 10:
                    same_val_groups.setdefault(card.value, []).append(card)
                    
            if len(aces) >= 2:
                for c1, c2 in itertools.combinations(aces, 2):
                    idx = self._attack_action_to_id.get(tuple(sorted((c1, c2))))
                    if idx is not None:
                        playable_actions.append(idx)
                        
            if aces and non_jesters:
                for ace in aces:
                    for other in non_jesters:
                        idx = self._attack_action_to_id.get(tuple(sorted((ace, other))))
                        if idx is not None:
                            playable_actions.append(idx)
                            
            for val, cards in same_val_groups.items():
                n = len(cards)
                if n >= 2:
                    for r in range(2, n + 1):
                        if val * r <= 10:
                            for combo in itertools.combinations(cards, r):
                                idx = self._attack_action_to_id.get(tuple(sorted(combo)))
                                if idx is not None:
                                    playable_actions.append(idx)
            
            if not game_state or enemy_attack == 0:
                for i in playable_actions:
                    mask[i] = 1
            else:
                for i in playable_actions:
                    action_info = self._global_attack_actions[i]
                    cards = action_info["cards"]
                    
                    has_jester = action_info["has_jester"]
                    if has_jester:
                        mask[i] = 1
                        has_survivable = True
                        continue
                        
                    total_attack = action_info["total_attack"]
                    has_clubs = action_info["has_clubs"]
                    has_diamonds = action_info["has_diamonds"]
                    has_spades = action_info["has_spades"]
                    
                    if not jester_cancelled and enemy_suit in action_info["card_suits"]:
                        has_clubs = False
                        has_diamonds = False
                        has_spades = False
                        for c in cards:
                            if c.suit.value != enemy_suit:
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
                    
                    played_defense = action_info["played_defense"]
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
            hand_vals = [c.get_discard_value() for c in hand]
            
            sums = [0] * (1 << hand_size)
            for i in range(hand_size):
                val = hand_vals[i]
                bit = 1 << i
                for b in range(bit):
                    sums[b | bit] = sums[b] + val
            
            has_defense = False
            for b in range(1, 1 << hand_size):
                defense = sums[b]
                if defense >= req:
                    is_min = True
                    for i in range(hand_size):
                        if (b & (1 << i)) and (defense - hand_vals[i] >= req):
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
        Direct computation for fast local mask generation.
        """
        local_masks = []
        hand_size = len(hand)
        
        def make_mask(indices):
            m = [0] * self.max_hand_size
            for idx in indices:
                if idx < self.max_hand_size:
                    m[idx] = 1
            return m
            
        if phase == "attack":
            allow_yield = game_state.get('can_yield', True) if game_state else True
            if allow_yield:
                local_masks.append([0] * self.max_hand_size)
                
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
            
            playable_indices = []
            
            aces_idx = []
            non_jesters_idx = []
            same_val_groups = {}
            for i, card in enumerate(hand):
                playable_indices.append([i])
                if card.value == 1:
                    aces_idx.append(i)
                elif card.value != 0:
                    non_jesters_idx.append(i)
                if 2 <= card.value <= 10:
                    same_val_groups.setdefault(card.value, []).append(i)
                    
            if len(aces_idx) >= 2:
                for c1, c2 in itertools.combinations(aces_idx, 2):
                    playable_indices.append([c1, c2])
                    
            if aces_idx and non_jesters_idx:
                for ace_i in aces_idx:
                    for other_i in non_jesters_idx:
                        playable_indices.append([ace_i, other_i])
                        
            for val, cards_idx in same_val_groups.items():
                n = len(cards_idx)
                if n >= 2:
                    for r in range(2, n + 1):
                        if val * r <= 10:
                            for combo in itertools.combinations(cards_idx, r):
                                playable_indices.append(list(combo))
                                
            if not game_state or enemy_attack == 0:
                for indices in playable_indices:
                    local_masks.append(make_mask(indices))
            else:
                has_survivable = False
                survivable_masks = []
                for indices in playable_indices:
                    cards = [hand[i] for i in indices]
                    has_jester = any(c.value == 0 for c in cards)
                    if has_jester:
                        survivable_masks.append(make_mask(indices))
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
                        survivable_masks.append(make_mask(indices))
                        has_survivable = True
                        continue
                        
                    total_damage = total_attack * 2 if has_clubs else total_attack
                    if enemy_damage_taken + total_damage >= enemy_health:
                        survivable_masks.append(make_mask(indices))
                        has_survivable = True
                        continue
                        
                    spade_protection = total_attack if has_spades else 0
                    eff_enemy_attack = max(0, enemy_attack - spade_protection)
                    
                    played_defense = sum(c.get_discard_value() for c in cards)
                    remaining_defense = hand_defense - played_defense
                    
                    if remaining_defense >= eff_enemy_attack:
                        survivable_masks.append(make_mask(indices))
                        has_survivable = True
                        
                if has_survivable:
                    local_masks.extend(survivable_masks)
                else:
                    for indices in playable_indices:
                        local_masks.append(make_mask(indices))
                        
        elif phase == "defense":
            req = game_state.get('enemy_attack', 0) if game_state else 0
            hand_vals = [c.get_discard_value() for c in hand]
            
            sums = [0] * (1 << hand_size)
            for i in range(hand_size):
                val = hand_vals[i]
                bit = 1 << i
                for b in range(bit):
                    sums[b | bit] = sums[b] + val
            
            has_defense = False
            for b in range(1, 1 << hand_size):
                defense = sums[b]
                if defense >= req:
                    is_min = True
                    for i in range(hand_size):
                        if (b & (1 << i)) and (defense - hand_vals[i] >= req):
                            is_min = False
                            break
                    if is_min:
                        m = [0] * self.max_hand_size
                        for i in range(hand_size):
                            if b & (1 << i):
                                m[i] = 1
                        local_masks.append(m)
                        has_defense = True
                        
            if not has_defense:
                if hand_size > 0:
                    for b in range(1, 1 << hand_size):
                        m = [0] * self.max_hand_size
                        for i in range(hand_size):
                            if b & (1 << i):
                                m[i] = 1
                        local_masks.append(m)
                else:
                    local_masks.append([0] * self.max_hand_size)

        if game_state and game_state.get('can_use_solo_jester', False):
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
