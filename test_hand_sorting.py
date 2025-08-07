#!/usr/bin/env python3
"""
Test script to verify that player hands are always sorted
"""

from regicide import Game, Card, Suit
from regicide_gym_env import make_regicide_env

def test_hand_sorting():
    """Test that hands are properly sorted"""
    print("🧪 Testing hand sorting functionality")
    print("=" * 50)
    
    # Test 1: Basic game initialization
    print("Test 1: Game initialization")
    game = Game(2)
    
    print("Player hands after initialization:")
    for i in range(2):
        hand = game.get_player_hand(i)
        print(f"  Player {i+1}: {[str(card) for card in hand]}")
        
        # Verify hand is sorted
        is_sorted = all(hand[i] <= hand[i+1] for i in range(len(hand)-1))
        print(f"  Sorted: {'✅' if is_sorted else '❌'}")
    
    print()
    
    # Test 2: Play some cards and check sorting
    print("Test 2: After playing cards")
    current_hand = game.get_current_player_hand()
    print(f"Current player hand before: {[str(card) for card in current_hand]}")
    
    # Try to play the first card
    if current_hand:
        result = game.play_card([0])  # Play first card
        print(f"Play result: {result['success']}")
        
        updated_hand = game.get_current_player_hand()
        print(f"Current player hand after: {[str(card) for card in updated_hand]}")
        
        # Verify still sorted
        is_sorted = all(updated_hand[i] <= updated_hand[i+1] for i in range(len(updated_hand)-1))
        print(f"Still sorted: {'✅' if is_sorted else '❌'}")
    
    print()
    
    # Test 3: Environment integration
    print("Test 3: Environment integration")
    env = make_regicide_env(num_players=2, max_hand_size=7)
    obs, info = env.reset()
    
    print("Hand cards from environment observation:")
    hand_indices = obs['hand_cards'].tolist()
    current_hand = env.game.get_current_player_hand()
    print(f"  Hand: {[str(card) for card in current_hand]}")
    print(f"  Indices: {hand_indices}")
    
    # Verify sorting
    is_sorted = all(current_hand[i] <= current_hand[i+1] for i in range(len(current_hand)-1))
    print(f"  Environment hand sorted: {'✅' if is_sorted else '❌'}")
    
    # Take a few actions
    print("\nTaking some actions:")
    for step in range(3):
        if info['valid_actions'] > 0:
            action = 0  # Always take first valid action
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_hand = env.game.get_current_player_hand()
            print(f"  Step {step+1}: {[str(card) for card in current_hand]}")
            
            is_sorted = all(current_hand[i] <= current_hand[i+1] for i in range(len(current_hand)-1))
            print(f"    Sorted: {'✅' if is_sorted else '❌'}")
            
            if terminated:
                break
        else:
            break
    
    print("\n✅ Hand sorting test completed!")

if __name__ == "__main__":
    test_hand_sorting()
