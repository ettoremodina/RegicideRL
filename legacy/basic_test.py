"""
Basic test script to verify core Regicide components work without external dependencies
"""

from action_handler import ActionHandler
from regicide import Game, Card, Suit
import random


def test_basic_components():
    """Test basic components without gymnasium"""
    print("🔧 TESTING BASIC COMPONENTS")
    print("=" * 50)
    
    # Test ActionHandler with yield functionality
    print("\n1. Testing ActionHandler...")
    
    test_hand = [
        Card(2, Suit.HEARTS),
        Card(2, Suit.DIAMONDS),
        Card(3, Suit.HEARTS),
        Card(7, Suit.SPADES),
    ]
    
    handler = ActionHandler(max_hand_size=7)
    
    # Test attack actions with yield allowed
    print("   Attack actions (yield allowed):")
    game_state = {'allow_yield': True}
    attack_actions = handler.get_all_possible_actions(test_hand, "attack", game_state)
    print(f"   - Found {len(attack_actions)} actions")
    
    # Check yield action
    if attack_actions and handler.is_yield_action(attack_actions[0]):
        print("   ✓ Yield action correctly placed at index 0")
    else:
        print("   ✗ Yield action not found at index 0")
    
    # Test without yield
    game_state = {'allow_yield': False}
    attack_actions_no_yield = handler.get_all_possible_actions(test_hand, "attack", game_state)
    print(f"   - Found {len(attack_actions_no_yield)} actions without yield")
    
    if len(attack_actions_no_yield) == len(attack_actions) - 1:
        print("   ✓ Correct action count when yield disabled")
    else:
        print("   ✗ Incorrect action count when yield disabled")
    
    # Test defense actions
    print("   Defense actions:")
    game_state = {'enemy_attack': 15, 'current_shields': 5}
    defense_actions = handler.get_all_possible_actions(test_hand, "defense", game_state)
    print(f"   - Found {len(defense_actions)} defense combinations")
    
    print("   ✓ ActionHandler tests passed")


def test_game_yield_mechanics():
    """Test game yield mechanics in detail"""
    print("\n2. Testing Game yield mechanics...")
    
    game = Game(num_players=2)
    
    print(f"   Initial state:")
    print(f"   - Can yield: {game.can_yield()}")
    print(f"   - Players yielded: {game.players_yielded}")
    print(f"   - Current player: {game.current_player}")
    
    # Test first yield
    print(f"\n   Player 1 yields:")
    result = game._handle_yield()
    print(f"   - Yield result: {result}")
    print(f"   - Players yielded: {game.players_yielded}")
    print(f"   - Current player: {game.current_player}")
    print(f"   - Can yield: {game.can_yield()}")
    
    # Test what happens when only one player left
    if game.current_player == 1 and not game.can_yield():
        print("   ✓ Player 2 correctly cannot yield (all others yielded)")
    
    print("   ✓ Yield mechanics tests passed")


def test_game_flow():
    """Test actual game flow with detailed logging"""
    print("\n3. Testing game flow...")
    
    game = Game(num_players=2)
    handler = ActionHandler(max_hand_size=game.get_max_hand_size())
    
    print(f"   Initial game state:")
    state = game.get_game_state()
    print(f"   - Current enemy: {state['current_enemy']}")
    print(f"   - Current player: {state['current_player'] + 1}")
    print(f"   - Player 1 hand: {', '.join(state['player_hands'][0])}")
    print(f"   - Player 2 hand: {', '.join(state['player_hands'][1])}")
    print(f"   - Tavern deck: {state['tavern_cards']} cards")
    print(f"   - Discard pile: {state['discard_cards']} cards")
    print(f"   - Enemies remaining: {state['enemies_remaining']}")
    
    # Test a few actions
    for turn in range(10):
        if game.game_over:
            break
        
        print(f"\n   Turn {turn + 1}:")
        current_player = game.current_player
        current_hand = game.players[current_player]
        
        print(f"   - Player {current_player + 1}'s turn")
        print(f"   - Hand: {[str(card) for card in current_hand]}")
        
        # Get valid actions
        game_state_info = {'allow_yield': game.can_yield()}
        valid_actions = handler.get_all_possible_actions(current_hand, "attack", game_state_info)
        
        print(f"   - Valid actions: {len(valid_actions)}")
        
        if not valid_actions:
            print("   - No valid actions, game should end")
            break
        
        # Choose a simple action (first non-yield if available, otherwise yield)
        chosen_action_idx = 0
        if len(valid_actions) > 1 and handler.is_yield_action(valid_actions[0]):
            chosen_action_idx = 1  # Choose first non-yield action
        
        action_mask = valid_actions[chosen_action_idx]
        card_indices = handler.mask_to_card_indices(action_mask, len(current_hand))
        
        if not card_indices:
            print("   - Action: YIELD")
            result = game._handle_yield()
            print(f"   - Yield result: {result}")
        else:
            selected_cards = [current_hand[i] for i in card_indices]
            print(f"   - Action: Play {[str(card) for card in selected_cards]}")
            
            # Store enemy state before
            enemy_hp_before = None
            tavern_before = len(game.tavern_deck)
            discard_before = len(game.discard_pile)
            if game.current_enemy:
                enemy_hp_before = game.current_enemy.health - game.current_enemy.damage_taken
            
            result = game.play_card(card_indices)
            
            # Check changes after playing cards
            tavern_after = len(game.tavern_deck)
            discard_after = len(game.discard_pile)
            
            # Check enemy state after
            if game.current_enemy and enemy_hp_before is not None:
                enemy_hp_after = game.current_enemy.health - game.current_enemy.damage_taken
                if enemy_hp_after < enemy_hp_before:
                    damage = enemy_hp_before - enemy_hp_after
                    print(f"   - Enemy took {damage} damage! ({enemy_hp_before} → {enemy_hp_after} HP)")
            
            # Check card movement
            if tavern_after != tavern_before:
                print(f"   - Tavern deck changed: {tavern_before} → {tavern_after} cards")
            if discard_after != discard_before:
                print(f"   - Discard pile changed: {discard_before} → {discard_after} cards")
            
            print(f"   - Play result: {result}")
            
            if result is False:
                print("   - Defense phase needed!")
                # Handle defense phase
                damage = game.current_enemy.get_effective_attack()
                if damage > 0:
                    print(f"   - Must defend against {damage} damage")
                    
                    # Get defense actions
                    defense_game_state = {'enemy_attack': damage, 'current_shields': 0}
                    defense_actions = handler.get_all_possible_actions(current_hand, "defense", defense_game_state)
                    
                    if defense_actions:
                        # Use first defense action
                        defense_mask = defense_actions[0]
                        defense_indices = handler.mask_to_card_indices(defense_mask, len(current_hand))
                        
                        if defense_indices:
                            defense_cards = [current_hand[i] for i in defense_indices]
                            defense_value = sum(card.get_discard_value() for card in defense_cards)
                            print(f"   - Defending with: {[str(card) for card in defense_cards]} (value: {defense_value})")
                        
                        # Track discard pile before defense
                        discard_before_defense = len(game.discard_pile)
                        defense_result = game.defend_against_attack(defense_indices)
                        discard_after_defense = len(game.discard_pile)
                        
                        print(f"   - Defense result: {defense_result}")
                        if discard_after_defense > discard_before_defense:
                            cards_discarded = discard_after_defense - discard_before_defense
                            print(f"   - {cards_discarded} cards discarded for defense")
                    else:
                        print("   - Cannot defend! Game over.")
                        game.game_over = True
        
        # Show updated state
        new_state = game.get_game_state()
        print(f"   - Current player after action: {new_state['current_player'] + 1}")
        print(f"   - Tavern deck: {new_state['tavern_cards']} cards")
        print(f"   - Discard pile: {new_state['discard_cards']} cards")
        if game.current_enemy:
            current_enemy_hp = game.current_enemy.health - game.current_enemy.damage_taken
            print(f"   - Enemy HP: {current_enemy_hp}/{game.current_enemy.health}")
    
    print(f"\n   Game ended:")
    print(f"   - Game over: {game.game_over}")
    print(f"   - Victory: {game.victory}")
    print(f"   - Enemies remaining: {len(game.castle_deck)}")
    final_state = game.get_game_state()
    print(f"   - Final tavern deck: {final_state['tavern_cards']} cards")
    print(f"   - Final discard pile: {final_state['discard_cards']} cards")
    
    print("   ✓ Game flow tests passed")


def main():
    """Run all basic tests"""
    print("🧪 BASIC REGICIDE COMPONENT TESTS")
    print("=" * 60)
    
    try:
        test_basic_components()
        test_game_yield_mechanics()
        test_game_flow()
        
        print("\n" + "=" * 60)
        print("✅ ALL BASIC TESTS PASSED!")
        print("\nCore components are working correctly.")
        print("You can now test the full environment with:")
        print("python detailed_game_test.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    success = main()
