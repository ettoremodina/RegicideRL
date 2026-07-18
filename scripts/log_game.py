import random
from game.regicide import Game
from game.action_handler import ActionHandler
from game.action_space import MAX_HAND_SIZE

from ml_logger import DashboardLogger

def play_one_game_with_logs():
    # Configure the dashboard logger
    logger = DashboardLogger()
    logger.start()
    
    try:
        logger.info("=== Starting 1 Simulated Game ===")
        
        game = Game(num_players=1)
        handler = ActionHandler(max_hand_size=MAX_HAND_SIZE)
        required_defense = 0
        
        while not game.game_over:
            current = game.current_player
            hand = game.get_player_hand(current)
            
            if required_defense > 0:
                actions = handler.get_all_possible_actions(hand, "defense", {'enemy_attack': required_defense})
                if not actions:
                    logger.info(f"Player {current + 1} cannot defend against {required_defense} damage. Game Over.")
                    game.game_over = True
                    break
                action = random.choice(actions)
                indices = handler.mask_to_card_indices(action, len(hand))
                res = game.defend_with_card_indices(indices)
                required_defense = 0
            else:
                actions = handler.get_all_possible_actions(hand, "attack", game.get_game_state())
                if not actions:
                    logger.info(f"Player {current + 1} has no valid actions. Game Over.")
                    game.game_over = True
                    break
                action = random.choice(actions)
                is_solo_jester = len(action) > MAX_HAND_SIZE and action[MAX_HAND_SIZE] == 1
                indices = handler.mask_to_card_indices(action, len(hand))

                if is_solo_jester:
                    res = game.use_solo_jester("step1")
                elif handler.is_yield_action(action):
                    res = game.yield_turn()
                else:
                    res = game.play_card(indices)
                
                required_defense = res.get("defense_required", 0)
                
                if res.get("phase") == "next_player_choice":
                    game.choose_next_player(0)
                    
        if game.victory:
            logger.info("\n=== Victory! ===")
        else:
            logger.info("\n=== Defeat! ===")
            
        logger.log_game_run({
            "victory": game.victory,
            "bosses_defeated": 12 - len(game.castle_deck),
        })
    finally:
        logger.stop()

if __name__ == "__main__":
    play_one_game_with_logs()
