"""
Clear Demonstration of Regicide Rule Fixes

This script tests the key rule corrections made to the game engine:

1. HEARTS HEALING: Cards healed by Hearts power go to the BOTTOM of the tavern deck
   (not the top), so they're drawn later, not immediately.

2. EXACT KILL PLACEMENT: When an enemy is defeated with exact damage, 
   the enemy card goes to the TOP of the tavern deck for immediate redraw.

3. SUIT POWER ORDERING: Hearts powers resolve before Diamonds powers.

4. CLUBS DOUBLING: Damage is doubled ONCE per play (not once per Club card).

5. ANIMAL COMPANION RULES: Aces can only be in 2-card combos or played alone.

Each test shows BEFORE and AFTER states with clear explanations.
"""
from regicide import Game, Card, Suit, Enemy
import random


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_deck_state(game, label="Current State"):
    """Show the current state of tavern deck and discard pile"""
    print(f"\n--- {label} ---")
    if game.tavern_deck:
        print(f"Tavern deck size: {len(game.tavern_deck)}")
        print(f"  TOP (drawn first):    {[str(c) for c in game.tavern_deck[-3:]]}")
        print(f"  BOTTOM (drawn last):  {[str(c) for c in game.tavern_deck[:3]]}")
    else:
        print("Tavern deck: EMPTY")
    
    if game.discard_pile:
        print(f"Discard pile size: {len(game.discard_pile)}")
        print(f"  Recent discards: {[str(c) for c in game.discard_pile[-3:]]}")
    else:
        print("Discard pile: EMPTY")


def test_hearts_healing():
    """Test that Hearts healing puts cards at the BOTTOM of tavern deck"""
    print_separator("TEST 1: HEARTS HEALING PLACEMENT")
    
    game = Game(num_players=2)
    
    # Setup: Put specific cards in discard pile
    specific_cards = [Card(2, Suit.HEARTS), Card(9, Suit.CLUBS), Card(7, Suit.DIAMONDS)]
    game.discard_pile = specific_cards.copy()
    
    print("SETUP: Put known cards in discard pile:")
    print(f"  Discard pile: {[str(c) for c in game.discard_pile]}")
    
    print_deck_state(game, "BEFORE Hearts Healing")
    
    # Apply Hearts healing for 2 cards
    print(f"\nAPPLYING: Hearts power heals 2 cards from discard")
    game._hearts_power(2)
    
    print_deck_state(game, "AFTER Hearts Healing")
    
    print("\nEXPECTED RESULT:")
    print("  - 2 cards should be removed from discard pile") 
    print("  - Those 2 cards should appear at the BOTTOM of tavern deck")
    print("  - When drawing, top cards are drawn first, so healed cards come last")


def test_exact_kill_placement():
    """Test that exact-kill enemies go to TOP of tavern deck"""
    print_separator("TEST 2: EXACT KILL ENEMY PLACEMENT")
    
    game = Game(num_players=1)
    
    # Setup enemy close to death
    original_enemy = game.current_enemy.card
    enemy_health = game.current_enemy.health
    print(f"SETUP: Enemy {original_enemy} has {enemy_health} HP")
    
    # Damage enemy to 1 HP remaining
    game.current_enemy.damage_taken = enemy_health - 1
    print(f"  Damaged enemy to {enemy_health - game.current_enemy.damage_taken} HP remaining")
    
    print_deck_state(game, "BEFORE Exact Kill")
    
    # Give player a card that will do exactly 1 damage
    ace_card = Card(1, Suit.HEARTS)  # Animal Companion = 1 attack
    game.players[0] = [ace_card]  # Replace hand with just this card
    
    print(f"\nEXECUTING: Player plays {ace_card} (1 damage) for exact kill")
    result = game.play_card([0])  # Play the ace
    
    print(f"  Result: {result['phase']} - {result['message']}")
    
    print_deck_state(game, "AFTER Exact Kill")
    
    if game.tavern_deck:
        top_card = game.tavern_deck[-1]  # Top of deck (last in list)
        print(f"\nEXPECTED RESULT:")
        print(f"  - Enemy {original_enemy} should be at TOP of tavern deck")
        print(f"  - Actual top card: {top_card}")
        print(f"  - Match? {'YES' if str(top_card) == str(original_enemy) else 'NO'}")


def test_suit_power_ordering():
    """Test that Hearts powers resolve before Diamonds powers"""
    print_separator("TEST 3: SUIT POWER ORDERING (Hearts before Diamonds)")
    
    game = Game(num_players=1)
    
    # Setup: Put cards in discard for Hearts to heal
    game.discard_pile = [Card(5, Suit.SPADES), Card(6, Suit.CLUBS)]
    original_hand_size = len(game.players[0])
    original_tavern_size = len(game.tavern_deck)
    original_discard_size = len(game.discard_pile)
    
    print("SETUP:")
    print(f"  Player hand size: {original_hand_size}")
    print(f"  Tavern deck size: {original_tavern_size}")
    print(f"  Discard pile size: {original_discard_size}")
    print(f"  Enemy: {game.current_enemy.card} (suit immunity affects these suits)")
    
    # Create a combo with Hearts and Diamonds both present
    # Use value 3 cards to stay under the limit of 10 total
    hearts_card = Card(3, Suit.HEARTS)
    diamonds_card = Card(3, Suit.DIAMONDS)
    
    # Replace player hand
    game.players[0] = [hearts_card, diamonds_card]
    
    print(f"\nEXECUTING: Player plays {hearts_card} + {diamonds_card}")
    print("  Expected: Hearts heals first, then Diamonds draws")
    
    result = game.play_card([0, 1])  # Play both cards
    
    final_hand_size = len(game.players[0])
    final_tavern_size = len(game.tavern_deck)
    final_discard_size = len(game.discard_pile)
    
    print(f"\nRESULTS:")
    print(f"  Hand size: {original_hand_size} → {final_hand_size}")
    print(f"  Tavern size: {original_tavern_size} → {final_tavern_size}")
    print(f"  Discard size: {original_discard_size} → {final_discard_size}")
    print(f"  Play result: {result['phase']}")
    
    print(f"\nEXPECTED BEHAVIOR:")
    print(f"  - Hearts should heal cards from discard to tavern bottom")
    print(f"  - Diamonds should then draw cards from tavern top to hand")
    print(f"  - Order matters for complex interactions!")


def test_clubs_doubling():
    """Test that Clubs damage is doubled only once per play"""
    print_separator("TEST 4: CLUBS DOUBLING (Once per play, not per card)")
    
    game = Game(num_players=1)
    
    # Setup enemy that won't be immune to Clubs
    game.current_enemy = Enemy(Card(11, Suit.HEARTS))  # Jack of Hearts
    original_damage = game.current_enemy.damage_taken
    
    print("SETUP:")
    print(f"  Enemy: {game.current_enemy.card}")
    print(f"  Enemy damage taken: {original_damage}")
    
    # Test 1: Single Club card
    print(f"\nTEST 4A: Single Club card")
    clubs_card = Card(4, Suit.CLUBS)
    game.players[0] = [clubs_card]
    base_attack = clubs_card.get_attack_value()
    
    print(f"  Playing: {clubs_card} (base attack: {base_attack})")
    game.play_card([0])
    
    damage_after_single = game.current_enemy.damage_taken
    damage_dealt = damage_after_single - original_damage
    
    print(f"  Damage dealt: {damage_dealt} (expected: {base_attack * 2} = base doubled)")
    
    # Test 2: Multiple Club cards  
    print(f"\nTEST 4B: Multiple Club cards")
    # Reset for clean test
    game.current_enemy.damage_taken = 0
    clubs1 = Card(3, Suit.CLUBS)
    clubs2 = Card(3, Suit.CLUBS)
    game.players[0] = [clubs1, clubs2]
    base_attack_total = clubs1.get_attack_value() + clubs2.get_attack_value()
    
    print(f"  Playing: {clubs1} + {clubs2} (total base attack: {base_attack_total})")
    game.play_card([0, 1])
    
    final_damage = game.current_enemy.damage_taken
    
    print(f"  Total damage dealt: {final_damage}")
    print(f"  Expected: {base_attack_total} (base) + {base_attack_total} (single doubling) = {base_attack_total * 2}")
    print(f"  NOT expected: {base_attack_total * 3} (if each Club doubled separately)")


def test_animal_companion_rules():
    """Test that Animal Companions (Aces) follow pairing rules"""
    print_separator("TEST 5: ANIMAL COMPANION (ACE) PAIRING RULES")
    
    game = Game(num_players=1)
    
    ace_hearts = Card(1, Suit.HEARTS)
    ace_spades = Card(1, Suit.SPADES)
    five_clubs = Card(5, Suit.CLUBS)
    three_diamonds = Card(3, Suit.DIAMONDS)
    
    print("TESTING VARIOUS ACE COMBINATIONS:")
    
    # Test valid combinations
    valid_combos = [
        ([ace_hearts], "Single Ace"),
        ([ace_hearts, ace_spades], "Ace + Ace"), 
        ([ace_hearts, five_clubs], "Ace + Non-Ace"),
    ]
    
    # Test invalid combinations  
    invalid_combos = [
        ([ace_hearts, five_clubs, three_diamonds], "Ace + Two Others"),
        ([ace_hearts, ace_spades, five_clubs], "Two Aces + One Other"),
    ]
    
    for combo, description in valid_combos:
        is_valid = game._is_valid_combo(combo)
        combo_str = " + ".join(str(c) for c in combo)
        print(f"  ✓ {description}: {combo_str} → {'VALID' if is_valid else 'INVALID (ERROR!)'}")
    
    for combo, description in invalid_combos:
        is_valid = game._is_valid_combo(combo)
        combo_str = " + ".join(str(c) for c in combo)
        print(f"  ✗ {description}: {combo_str} → {'INVALID' if not is_valid else 'VALID (ERROR!)'}")


def main():
    """Run all rule fix demonstrations"""
    print("REGICIDE RULE FIXES - COMPREHENSIVE DEMONSTRATION")
    print("This script verifies that the game engine correctly implements:")
    print("- Hearts healing placement, Exact kill placement, Suit power ordering")
    print("- Clubs doubling limits, Animal Companion pairing rules")
    
    test_hearts_healing()
    test_exact_kill_placement() 
    test_suit_power_ordering()
    test_clubs_doubling()
    test_animal_companion_rules()
    
    print_separator("ALL TESTS COMPLETED")
    print("Review the results above to verify rule fixes are working correctly!")


if __name__ == "__main__":
    main()
