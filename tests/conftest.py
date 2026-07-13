"""
Shared pytest fixtures for Regicide game tests.
"""

import pytest
import random
from game.regicide import Game, Card, Suit, Enemy


@pytest.fixture
def solo_game():
    """A fresh 1-player game with fixed seed for reproducibility."""
    random.seed(42)
    return Game(1)


@pytest.fixture
def two_player_game():
    """A fresh 2-player game with fixed seed."""
    random.seed(42)
    return Game(2)


@pytest.fixture
def three_player_game():
    """A fresh 3-player game with fixed seed."""
    random.seed(42)
    return Game(3)


@pytest.fixture
def four_player_game():
    """A fresh 4-player game with fixed seed."""
    random.seed(42)
    return Game(4)


def make_card(value: int, suit: Suit) -> Card:
    """Helper to create a card."""
    return Card(value, suit)


def make_hand(*specs) -> list:
    """Create a hand from (value, suit) tuples.
    
    Example: make_hand((5, Suit.HEARTS), (3, Suit.CLUBS))
    """
    return [Card(v, s) for v, s in specs]


def setup_game_with_hand(num_players: int, player_idx: int, hand: list, 
                          enemy_value: int = 11, enemy_suit: Suit = Suit.HEARTS) -> Game:
    """Create a game and inject a specific hand and enemy for testing.
    
    Args:
        num_players: Number of players
        player_idx: Which player gets the injected hand
        hand: List of Card objects
        enemy_value: Enemy card value (11=Jack, 12=Queen, 13=King)
        enemy_suit: Enemy suit
    
    Returns:
        Configured Game instance
    """
    random.seed(42)
    game = Game(num_players)
    game.players[player_idx] = hand.copy()
    game.current_player = player_idx
    game.current_enemy = Enemy(Card(enemy_value, enemy_suit))
    return game
