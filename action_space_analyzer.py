import itertools
from math import comb
from enum import Enum

class Suit(Enum):
    HEARTS = '♥'
    DIAMONDS = '♦'
    CLUBS = '♣'
    SPADES = '♠'

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
    def __str__(self):
        val_str = {1: 'A', 11: 'J', 12: 'Q', 13: 'K', 0: 'Jester'}.get(self.value, str(self.value))
        return f"{val_str}{self.suit.value if self.value != 0 else ''}"
    def __repr__(self): return str(self)

def generate_global_attack_actions():
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
    deck.append(Card(0, Suit.HEARTS)) # Jester 1
    deck.append(Card(0, Suit.HEARTS)) # Jester 2 (for solo, though functionally identical)
    
    # We only care about unique semantic combinations. 
    # Jesters are identical in terms of gameplay when played.
    jester = Card(0, Suit.HEARTS)
    
    # 1. Single Cards
    for card in deck[:-1]: # All 52 standard cards
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
                    
    return actions

def main():
    print("=== Regicide Action Space Analysis ===")
    
    # 1. Attack Actions
    attack_actions = generate_global_attack_actions()
    print(f"Total Unique Attack Actions (Hand-Independent): {len(attack_actions)}")
    
    # Count by type
    types = {}
    for a in attack_actions:
        types[a["type"]] = types.get(a["type"], 0) + 1
    for k, v in types.items():
        print(f"  - {k}: {v}")
        
    # 2. Defense Actions
    # Any combination of cards from a 54 card deck is ~1.2 billion.
    # However, from an 8-card hand, there are exactly 2^8 = 256 subsets.
    print(f"\nTotal Unique Defense Actions (Hand-Relative): 256")
    
    print(f"\nProposed Hybrid Action Space Size: {len(attack_actions) + 256} (Dense & Enumerable)")
    
    print("\nHow this works for the RL Agent:")
    print("Action indices 0 to 285 correspond to exact card combinations (e.g., 'Play 2 of Hearts').")
    print("Action indices 286 to 541 correspond to hand-relative discard subsets (e.g., 'Discard cards at hand index 0 and 2').")
    
    print("\nImpact on existing agents:")
    print("- You do NOT need to delete the current 2^8 masking system.")
    print("- We can add `get_global_action_mask()` to ActionHandler that returns a 542-length array.")
    print("- This array maps directly to the network's output logits.")
    print("- We also add `global_action_to_hand_indices(action_id, hand)` to translate the chosen action back to the list of indices the env expects.")
    print("- Old agents (like ISMCTS) can still use `get_all_possible_actions()` which returns the old 256-masks.")

if __name__ == '__main__':
    main()
