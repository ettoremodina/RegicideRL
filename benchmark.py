import time
import random
from game.regicide import Game
from game.action_handler import ActionHandler

def simulate_games(num_games=1000):
    print(f"Starting simulation of {num_games} games...")
    start_time = time.time()
    
    handler = ActionHandler(max_hand_size=8)
    victories = 0
    total_turns = 0
    enemies_defeated = 0
    
    for i in range(num_games):
        game = Game(num_players=1)
        required_defense = 0
        
        while not game.game_over:
            current = game.current_player
            hand = game.get_player_hand(current)
            
            # Simple AI: randomly choose a valid action
            if required_defense > 0:
                actions = handler.get_all_possible_actions(hand, "defense", {'enemy_attack': required_defense})
                if not actions:
                    # Auto defeat if cannot defend
                    game.game_over = True
                    break
                action = random.choice(actions)
                indices = handler.mask_to_card_indices(action, len(hand))
                res = game.defend_with_card_indices(indices)
                required_defense = 0
            else:
                actions = handler.get_all_possible_actions(hand, "attack", game.get_game_state())
                if not actions:
                    game.game_over = True
                    break
                action = random.choice(actions)
                indices = handler.mask_to_card_indices(action, len(hand))
                
                if handler.is_yield_action(action):
                    res = game.yield_turn()
                else:
                    res = game.play_card(indices)
                
                required_defense = res.get("defense_required", 0)
                
                # Handle Jester choice (solo mode defaults back to player 1)
                if res.get("phase") == "next_player_choice":
                    game.choose_next_player(0)
                    
            total_turns += 1
            
        if game.victory:
            victories += 1
        
        # 12 enemies total
        enemies_left = len(game.castle_deck) + (1 if game.current_enemy and not game.victory else 0)
        enemies_defeated += (12 - enemies_left)

    elapsed = time.time() - start_time
    
    print("=== Benchmark Results ===")
    print(f"Games played: {num_games}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Speed: {num_games / elapsed:.2f} games/second")
    print(f"Avg turns per game: {total_turns / num_games:.1f}")
    print(f"Win rate: {victories / num_games * 100:.2f}%")
    print(f"Avg enemies defeated: {enemies_defeated / num_games:.2f} / 12")
    
if __name__ == "__main__":
    simulate_games(100000)
