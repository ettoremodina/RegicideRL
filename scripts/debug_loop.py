import sys
sys.path.append('.')
from game.regicide import Game
from game.action_handler import ActionHandler
from ml_logger import DashboardLogger
import random

logger = DashboardLogger()
logger.start()

try:
    handler = ActionHandler()
    for game_num in range(100):
        game = Game(num_players=1)
        res = {}
        required_defense = 0
        turns = 0
    
        while not game.game_over:
            turns += 1
            current = game.current_player
            hand = game.get_player_hand(current)
            
            if turns > 1000:
                logger.error(f'Game {game_num} stuck! Phase: {res.get("phase")}, ReqDefense: {required_defense}, Hand: {hand}')
                sys.exit(1)
            
        if required_defense > 0:
            actions = handler.get_all_possible_actions(hand, 'defense', {'enemy_attack': required_defense})
            if not actions:
                game.game_over = True
                break
            action = random.choice(actions)
            indices = handler.mask_to_card_indices(action, len(hand))
            res = game.defend_with_card_indices(indices)
            required_defense = res.get('defense_required', 0)
        else:
            actions = handler.get_all_possible_actions(hand, 'attack', game.get_game_state())
            if not actions:
                game.game_over = True
                break
            action = random.choice(actions)
            indices = handler.mask_to_card_indices(action, len(hand))
            
            is_solo_jester = (len(action) == 9 and action[8] == 1)
            if is_solo_jester:
                res = game.use_solo_jester("step1")
            else:
                if handler.is_yield_action(action):
                    res = game.yield_turn()
                else:
                    res = game.play_card(indices)
            required_defense = res.get('defense_required', 0)
            
            if res.get("phase") == "next_player_choice":
                res = game.choose_next_player(1)
    logger.info('Finished without getting stuck!')
finally:
    logger.stop()
