import pytest
from game.regicide import Game, Card, Suit, Enemy
from solvers.env import RegicideEnv
from solvers.perfect_solver import PerfectSolver

def test_perfect_solver_finds_win():
    # Setup a mini-game
    env = RegicideEnv(num_players=1)
    env.reset()
    game = env.game
    game.deterministic_hearts = True
    
    # Custom state
    game.players[0] = [Card(10, Suit.HEARTS), Card(10, Suit.SPADES)]
    game.tavern_deck = []
    game.castle_deck = []
    game.current_enemy = Enemy(Card(11, Suit.CLUBS)) # Jack of Clubs (20 HP, 10 ATK)
    game.jester_immunity_cancelled = False
    game.attack_cards_buffer = []
    
    # Path to win:
    # 1. Play 10 Spades. Damage = 10. Enemy HP = 10. Enemy ATK = 10 - 10 = 0. No defense needed.
    # 2. Play 10 Hearts. Damage = 10. Enemy HP = 0. Defeated. Victory!
    
    solver = PerfectSolver(verbose=True)
    winning_actions = solver.solve(env)
    
    assert winning_actions is not None
    assert len(winning_actions) == 2
    
    # Verify the actions actually lead to a win
    for act_idx in winning_actions:
        obs, reward, term, trunc, info = env.step(act_idx)
        
    assert env.game.victory

def test_perfect_solver_finds_loss():
    # Setup an unwinnable mini-game
    env = RegicideEnv(num_players=1)
    env.reset()
    game = env.game
    game.deterministic_hearts = True
    
    # Custom state
    game.players[0] = [Card(2, Suit.HEARTS)]
    game.tavern_deck = []
    game.castle_deck = []
    game.current_enemy = Enemy(Card(11, Suit.CLUBS)) # Jack of Clubs (20 HP, 10 ATK)
    game.jester_immunity_cancelled = False
    game.attack_cards_buffer = []
    
    # Only 2 damage, cannot win.
    solver = PerfectSolver(verbose=True)
    winning_actions = solver.solve(env)
    
    assert winning_actions is None
