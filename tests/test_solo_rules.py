"""
Tests for Solo Jester rules.

Solo play uses different Jester mechanics than multiplayer:
- 2 Jester tokens set aside (not in deck)
- Activation: discard hand, refill to 8 cards (not counted as diamond drawing)
- Does NOT cancel enemy immunity
- Can be used at start of Step 1 or start of Step 4
- Victory tiers: Gold (0 used), Silver (1 used), Bronze (2 used)
"""

import pytest
import random
from game.regicide import Game, Card, Suit, Enemy
from tests.conftest import setup_game_with_hand


class TestSoloJesterSetup:
    """Test solo jester initial state."""

    def test_solo_has_two_jesters(self):
        """Solo game starts with 2 jester tokens."""
        random.seed(42)
        game = Game(1)
        assert game.solo_jesters_remaining == 2
        assert game.solo_jesters_used == 0

    def test_multiplayer_no_solo_jesters(self):
        """Multiplayer games have no solo jester tokens."""
        for n in [2, 3, 4]:
            random.seed(42)
            game = Game(n)
            assert game.solo_jesters_remaining == 0

    def test_no_jesters_in_solo_tavern_deck(self):
        """Solo tavern deck should contain no Jester cards (value 0)."""
        random.seed(42)
        game = Game(1)
        all_cards = game.tavern_deck + game.players[0]
        jester_cards = [c for c in all_cards if c.value == 0]
        assert len(jester_cards) == 0


class TestSoloJesterUsage:
    """Test using solo jester tokens."""

    def test_use_jester_step1(self):
        """Using solo jester at step1: discards hand, refills to 8."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        # Ensure tavern has enough cards
        game.tavern_deck = [Card(v, Suit.SPADES) for v in range(2, 11)] * 2
        
        old_hand = game.players[0].copy()
        result = game.use_solo_jester('step1')
        
        assert result['success']
        assert game.solo_jesters_remaining == 1
        assert game.solo_jesters_used == 1
        assert len(game.players[0]) == 8
        # Old hand cards should be in discard
        for card in old_hand:
            assert card in game.discard_pile

    def test_use_jester_step4(self):
        """Using solo jester at step4 also works."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        game.tavern_deck = [Card(v, Suit.SPADES) for v in range(2, 11)] * 2
        
        result = game.use_solo_jester('step4')
        
        assert result['success']
        assert len(game.players[0]) == 8

    def test_use_both_jesters(self):
        """Can use both jester tokens in one game."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        game.tavern_deck = [Card(v, Suit.SPADES) for v in range(2, 11)] * 4
        
        result1 = game.use_solo_jester('step1')
        assert result1['success']
        assert game.solo_jesters_remaining == 1
        
        result2 = game.use_solo_jester('step1')
        assert result2['success']
        assert game.solo_jesters_remaining == 0
        assert game.solo_jesters_used == 2

    def test_no_jesters_remaining(self):
        """Cannot use jester when none remain."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        game.solo_jesters_remaining = 0
        
        result = game.use_solo_jester('step1')
        
        assert not result['success']

    def test_invalid_timing(self):
        """Invalid timing string is rejected."""
        random.seed(42)
        game = Game(1)
        
        result = game.use_solo_jester('step2')
        
        assert not result['success']
        assert 'Invalid timing' in result['message']

    def test_multiplayer_cannot_use_solo_jester(self):
        """Multiplayer games cannot use solo jester tokens."""
        random.seed(42)
        game = Game(2)
        
        assert not game.can_use_solo_jester()
        result = game.use_solo_jester('step1')
        assert not result['success']

    def test_jester_does_not_cancel_immunity(self):
        """Solo jester does NOT cancel enemy immunity (unlike multiplayer jester)."""
        game = setup_game_with_hand(1, 0,
            [Card(8, Suit.CLUBS)],
            enemy_value=11, enemy_suit=Suit.CLUBS)
        game.tavern_deck = [Card(v, Suit.SPADES) for v in range(2, 11)] * 2
        
        game.use_solo_jester('step1')
        
        # Immunity should still be in effect
        assert not game.jester_immunity_cancelled

    def test_jester_partial_refill(self):
        """If tavern has fewer than 8 cards, hand gets fewer cards."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        game.tavern_deck = [Card(4, Suit.SPADES), Card(5, Suit.SPADES)]  # Only 2 cards
        
        result = game.use_solo_jester('step1')
        
        assert result['success']
        assert len(game.players[0]) == 2  # Only 2 cards available


class TestVictoryTiers:
    """Test solo victory tier calculation."""

    def test_gold_victory(self):
        """0 jesters used = gold victory."""
        random.seed(42)
        game = Game(1)
        game.victory = True
        game.solo_jesters_used = 0
        assert game.get_victory_tier() == 'gold'

    def test_silver_victory(self):
        """1 jester used = silver victory."""
        random.seed(42)
        game = Game(1)
        game.victory = True
        game.solo_jesters_used = 1
        assert game.get_victory_tier() == 'silver'

    def test_bronze_victory(self):
        """2 jesters used = bronze victory."""
        random.seed(42)
        game = Game(1)
        game.victory = True
        game.solo_jesters_used = 2
        assert game.get_victory_tier() == 'bronze'

    def test_no_victory_returns_none(self):
        """No tier if game not won."""
        random.seed(42)
        game = Game(1)
        game.victory = False
        assert game.get_victory_tier() is None

    def test_multiplayer_victory_tier(self):
        """Multiplayer victory has no tiers."""
        random.seed(42)
        game = Game(2)
        game.victory = True
        assert game.get_victory_tier() == 'victory'

    def test_victory_tier_in_game_state(self):
        """Victory tier is included in game state."""
        random.seed(42)
        game = Game(1)
        game.victory = True
        game.solo_jesters_used = 0
        state = game.get_game_state()
        assert state['victory_tier'] == 'gold'


class TestSoloJesterGameState:
    """Test that game state includes solo jester information."""

    def test_game_state_includes_jester_info(self):
        """Game state dict should contain solo jester fields."""
        random.seed(42)
        game = Game(1)
        state = game.get_game_state()
        
        assert 'solo_jesters_remaining' in state
        assert 'solo_jesters_used' in state
        assert 'can_use_solo_jester' in state
        assert state['solo_jesters_remaining'] == 2
        assert state['solo_jesters_used'] == 0
        assert state['can_use_solo_jester'] is True

    def test_game_state_after_jester_use(self):
        """Game state updates after using a solo jester."""
        game = setup_game_with_hand(1, 0,
            [Card(2, Suit.HEARTS)],
            enemy_value=11, enemy_suit=Suit.DIAMONDS)
        game.tavern_deck = [Card(v, Suit.SPADES) for v in range(2, 11)] * 2
        
        game.use_solo_jester('step1')
        state = game.get_game_state()
        
        assert state['solo_jesters_remaining'] == 1
        assert state['solo_jesters_used'] == 1
