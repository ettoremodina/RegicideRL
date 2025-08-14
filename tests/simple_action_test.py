"""
Simple Action Test Script
Tests the result of playing specific cards against a hardcoded enemy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.regicide import Game, Card, Suit, Enemy


def test_action():
    """Test a hardcoded action and print the results"""
    
    # Create a game instance
    game = Game(1)
    
    # Define hardcoded hand (you can modify these)
    hand = [  
        Card(3, Suit.DIAMONDS),   
        Card(3, Suit.CLUBS),   
        Card(3, Suit.SPADES),   
        Card(6, Suit.CLUBS),    
    
    ]
    action_indices = [0,1,2]  # Modify this to test different combinations
    
    # Set the hand for player 0
    game.players[0] = hand.copy()
    

    enemy_card = Card(11, Suit.HEARTS)  
    game.current_enemy = Enemy(enemy_card)
    
    # Define the action (indices of cards to play)
    # Example: play cards at indices [0, 2] (8 of Hearts + 10 of Clubs)
    
    print("=" * 50)
    print("SIMPLE ACTION TEST")
    print("=" * 50)
    
    # Print initial state
    print("\n--- INITIAL STATE ---")
    print(f"Player hand: {[str(card) for card in hand]}")
    print(f"Enemy: {game.current_enemy}")
    print(f"Action: Playing cards at indices {action_indices}")
    
    cards_to_play = [hand[i] for i in action_indices]
    print(f"Cards being played: {[str(card) for card in cards_to_play]}")
    
    # Calculate expected attack value
    total_attack = sum(card.get_attack_value() for card in cards_to_play)
    print(f"Base attack value: {total_attack}")
    
    # Check for suit powers
    suits_played = set(card.suit for card in cards_to_play)
    print(f"Suits played: {[suit.value for suit in suits_played]}")
    
    # Execute the action
    print("\n--- EXECUTING ACTION ---")
    result = game.play_card(action_indices)
    
    # Print results
    print("\n--- RESULTS ---")
    print(f"Success: {result['success']}")
    print(f"Phase: {result['phase']}")
    print(f"Message: {result['message']}")
    print(f"Enemy damage dealt: {result.get('enemy_damage', 0)}")
    print(f"Defense required: {result.get('defense_required', 0)}")
    
    # Print enemy state after action
    print(f"\n--- ENEMY STATE AFTER ACTION ---")
    if game.current_enemy:
        print(f"Enemy: {game.current_enemy}")
        print(f"Damage taken: {game.current_enemy.damage_taken}/{game.current_enemy.health}")
        print(f"Spade protection: {game.current_enemy.spade_protection}")
        print(f"Effective attack: {game.current_enemy.get_effective_attack()}")
        print(f"Is defeated: {game.current_enemy.is_defeated()}")
    else:
        print("Enemy defeated!")
    
    # Print remaining hand
    print(f"\n--- REMAINING HAND ---")
    print(f"Cards left: {[str(card) for card in game.players[0]]}")
    
    # Print tavern deck changes
    print(f"\n--- TAVERN DECK ---")
    print(f"Cards in tavern: {len(game.tavern_deck)}")
    
    print("\n" + "=" * 50)


def test_multiple_scenarios():
    """Test multiple predefined scenarios"""
    
    scenarios = [
        {
            "name": "Simple Hearts Attack",
            "hand": [Card(8, Suit.HEARTS), Card(5, Suit.DIAMONDS)],
            "enemy": Card(11, Suit.SPADES),  # Jack of Spades
            "action": [0]  # Play 8 of Hearts
        },
        {
            "name": "Clubs Double Damage",
            "hand": [Card(10, Suit.CLUBS), Card(3, Suit.SPADES)],
            "enemy": Card(11, Suit.HEARTS),  # Jack of Hearts
            "action": [0]  # Play 10 of Clubs
        },
        {
            "name": "Combo Attack",
            "hand": [Card(5, Suit.HEARTS), Card(5, Suit.DIAMONDS), Card(2, Suit.SPADES)],
            "enemy": Card(12, Suit.CLUBS),  # Queen of Clubs
            "action": [0, 1]  # Play both 5s
        },
        {
            "name": "Spades Protection",
            "hand": [Card(7, Suit.SPADES), Card(3, Suit.HEARTS)],
            "enemy": Card(11, Suit.DIAMONDS),  # Jack of Diamonds
            "action": [0]  # Play 7 of Spades
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create fresh game
        game = Game(1)
        game.players[0] = scenario['hand'].copy()
        game.current_enemy = Enemy(scenario['enemy'])
        
        print(f"Hand: {[str(card) for card in scenario['hand']]}")
        print(f"Enemy: {game.current_enemy}")
        print(f"Action: {scenario['action']}")
        
        cards_to_play = [scenario['hand'][i] for i in scenario['action']]
        print(f"Playing: {[str(card) for card in cards_to_play]}")
        
        # Execute
        result = game.play_card(scenario['action'])
        
        print(f"\nResult: {result['phase']} - {result['message']}")
        print(f"Damage dealt: {result.get('enemy_damage', 0)}")
        
        if game.current_enemy:
            print(f"Enemy after: {game.current_enemy}")
        else:
            print("Enemy defeated!")


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Single action test (modify code to change action)")
    print("2. Multiple predefined scenarios")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_action()
    elif choice == "2":
        test_multiple_scenarios()
    else:
        print("Invalid choice. Running single action test...")
        test_action()
