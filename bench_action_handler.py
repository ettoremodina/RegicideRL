import time
from game.action_handler import ActionHandler
from game.regicide import Card, Suit

handler = ActionHandler()
hand = [
    Card(2, Suit.HEARTS),
    Card(2, Suit.SPADES),
    Card(1, Suit.CLUBS),
    Card(5, Suit.DIAMONDS),
    Card(13, Suit.HEARTS),
    Card(10, Suit.SPADES),
    Card(1, Suit.HEARTS)
]

game_state = {
    'enemy_attack': 10,
    'enemy_health': 15,
    'enemy_damage_taken': 5,
    'enemy_suit': Suit.CLUBS,
    'jester_immunity_cancelled': False,
    'can_yield': True,
    'can_use_solo_jester': False
}

start = time.time()
for _ in range(10000):
    handler.get_global_action_mask(hand, "attack", game_state)
end = time.time()
print(f"Attack: {10000 / (end - start):.2f} calls/sec")

start = time.time()
for _ in range(10000):
    handler.get_global_action_mask(hand, "defense", game_state)
end = time.time()
print(f"Defense: {10000 / (end - start):.2f} calls/sec")
