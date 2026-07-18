"""
Tests for ActionHandler — attack action generation and defense action generation.
"""

import pytest
from game.regicide import Game, Card, Suit
from game.action_handler import ActionHandler
from game.action_space import (
    DEFENSE_ACTION_OFFSET,
    GLOBAL_ACTION_SPACE_SIZE,
)


@pytest.fixture
def handler():
    """ActionHandler with max hand size 8."""
    return ActionHandler(max_hand_size=8)


class TestAttackActionGeneration:
    """Test attack action mask generation."""

    def test_single_cards_always_valid(self, handler):
        """Each card in hand should produce a single-card action."""
        hand = [Card(3, Suit.HEARTS), Card(5, Suit.CLUBS), Card(8, Suit.DIAMONDS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # Should have at least 3 single-card actions
        single_card_actions = [a for a in actions if sum(a) == 1]
        assert len(single_card_actions) == 3

    def test_yield_included_when_allowed(self, handler):
        """Yield action (all zeros) included when allow_yield is True."""
        hand = [Card(5, Suit.HEARTS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'can_yield': True})
        
        yield_action = [0] * 8
        assert yield_action in actions

    def test_yield_excluded_when_disallowed(self, handler):
        """Yield action NOT included when allow_yield is False."""
        hand = [Card(5, Suit.HEARTS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'can_yield': False})
        
        yield_action = [0] * 8
        assert yield_action not in actions

    def test_same_value_combos_generated(self, handler):
        """Pairs of same value ≤ 10 total should be in actions."""
        hand = [Card(3, Suit.HEARTS), Card(3, Suit.CLUBS), Card(3, Suit.SPADES)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # Should include pairs and triple (3+3+3=9 ≤ 10)
        multi_card_actions = [a for a in actions if sum(a) > 1]
        assert len(multi_card_actions) > 0

    def test_invalid_combos_excluded(self, handler):
        """Pairs of different values should not be in actions."""
        hand = [Card(3, Suit.HEARTS), Card(5, Suit.CLUBS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # No multi-card actions should exist (3+5 is invalid combo)
        multi_card_actions = [a for a in actions if sum(a) > 1]
        assert len(multi_card_actions) == 0

    def test_ace_combos_generated(self, handler):
        """Ace + regular card combos should be in actions."""
        hand = [Card(1, Suit.HEARTS), Card(8, Suit.DIAMONDS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # Should include A♥ + 8♦ combo
        two_card_actions = [a for a in actions if sum(a) == 2]
        assert len(two_card_actions) == 1

    def test_ace_jester_combo_excluded(self, handler):
        """Ace + Jester combo should NOT be in actions (rule fix)."""
        hand = [Card(1, Suit.HEARTS), Card(0, Suit.HEARTS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # Should NOT include A♥ + Jester combo
        two_card_actions = [a for a in actions if sum(a) == 2]
        assert len(two_card_actions) == 0

    def test_over_10_combo_excluded(self, handler):
        """Same value pairs totaling > 10 should not be in actions."""
        hand = [Card(6, Suit.HEARTS), Card(6, Suit.CLUBS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'allow_yield': False})
        
        # Only single-card actions, no pair (6+6=12 > 10)
        multi_card_actions = [a for a in actions if sum(a) > 1]
        assert len(multi_card_actions) == 0

    def test_no_duplicate_actions(self, handler):
        """No duplicate action masks should be generated."""
        hand = [Card(2, Suit.HEARTS), Card(2, Suit.CLUBS), Card(2, Suit.SPADES),
                Card(5, Suit.DIAMONDS), Card(8, Suit.HEARTS)]
        actions = handler.get_all_possible_actions(hand, "attack", {'can_yield': True})
        
        action_tuples = [tuple(a) for a in actions]
        assert len(action_tuples) == len(set(action_tuples))

    def test_empty_hand(self, handler):
        """Empty hand with yield allowed should only have yield action."""
        hand = []
        actions = handler.get_all_possible_actions(hand, "attack", {'can_yield': True})
        
        assert len(actions) == 1  # Only yield
        assert handler.is_yield_action(actions[0])


class TestDefenseActionGeneration:
    """Test defense action mask generation."""

    def test_minimal_defense_combinations(self, handler):
        """Only minimal defense sets should be returned."""
        hand = [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS), Card(5, Suit.SPADES),
                Card(10, Suit.DIAMONDS)]
        game_state = {'enemy_attack': 5}
        actions = handler.get_all_possible_actions(hand, "defense", game_state)
        
        # The 5♠ alone suffices, and 2+3 also suffices
        # 10♦ alone also suffices
        # These are the MINIMAL sets
        for action in actions:
            indices = handler.mask_to_card_indices(action, len(hand))
            cards = [hand[i] for i in indices]
            total = sum(c.get_discard_value() for c in cards)
            assert total >= 5  # All defenses meet threshold

    def test_no_superset_defense(self, handler):
        """No defense combo should be a superset of another valid combo."""
        hand = [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS), Card(5, Suit.SPADES),
                Card(10, Suit.DIAMONDS)]
        game_state = {'enemy_attack': 5}
        actions = handler.get_all_possible_actions(hand, "defense", game_state)
        
        # Convert to index sets
        index_sets = []
        for action in actions:
            indices = set(handler.mask_to_card_indices(action, len(hand)))
            index_sets.append(indices)
        
        # No set should be a strict superset of another
        for i, s1 in enumerate(index_sets):
            for j, s2 in enumerate(index_sets):
                if i != j:
                    assert not s2.issubset(s1) or s1 == s2

    def test_zero_enemy_attack(self, handler):
        """Zero enemy attack should allow empty defense."""
        hand = [Card(5, Suit.HEARTS)]
        game_state = {'enemy_attack': 0}
        actions = handler.get_all_possible_actions(hand, "defense", game_state)
        
        # Empty set (no cards needed) should be valid
        has_empty = any(sum(a) == 0 for a in actions)
        assert has_empty

    def test_impossible_defense(self, handler):
        """When total hand value < enemy attack, no valid defenses exist."""
        hand = [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS)]  # Total = 5
        game_state = {'enemy_attack': 10}
        actions = handler.get_all_possible_actions(hand, "defense", game_state)
        
        assert len(actions) == 0


class TestActionHandlerUtilities:
    """Test utility methods."""

    def test_mask_to_card_indices(self, handler):
        """Convert mask to indices correctly."""
        mask = [1, 0, 1, 0, 0, 0, 0, 0]
        indices = handler.mask_to_card_indices(mask, 4)
        assert indices == [0, 2]

    def test_cards_to_mask(self, handler):
        """Convert indices to mask correctly."""
        mask = handler.cards_to_mask([0, 2])
        assert mask == [1, 0, 1, 0, 0, 0, 0, 0]

    def test_is_yield_action(self, handler):
        """Yield detection works."""
        assert handler.is_yield_action([0, 0, 0, 0, 0, 0, 0, 0])
        assert not handler.is_yield_action([1, 0, 0, 0, 0, 0, 0, 0])

    def test_action_count(self, handler):
        """get_action_count returns correct number."""
        hand = [Card(5, Suit.HEARTS), Card(3, Suit.CLUBS)]
        count = handler.get_action_count(hand, "attack", {'can_yield': True})
        # Yield + 5♥ + 3♣ = 3 actions (no valid combos since different values)
        assert count == 3

class TestGlobalActionSpace:
    """Test the global 543-dimensional action space."""

    def test_get_global_action_mask_attack(self, handler):
        """Check attack mask is correctly mapped."""
        hand = [Card(2, Suit.HEARTS)]
        mask = handler.get_global_action_mask(hand, "attack", {'can_yield': False})
        
        assert len(mask) == GLOBAL_ACTION_SPACE_SIZE
        
        # Only 1 action (Play 2 of Hearts) should be 1
        # No yield because can_yield is False
        assert sum(mask) == 1
        
        # Find the active index
        active_idx = mask.index(1)
        
        # Assert the active index is in the attack range
        assert 0 <= active_idx < DEFENSE_ACTION_OFFSET

    def test_get_global_action_mask_defense(self, handler):
        """Check defense mask is correctly mapped."""
        hand = [Card(10, Suit.HEARTS)]
        game_state = {'enemy_attack': 5}
        mask = handler.get_global_action_mask(hand, "defense", game_state)
        
        assert len(mask) == GLOBAL_ACTION_SPACE_SIZE
        
        # 10 is enough to defend against 5. 
        # Only the minimal set (the single 10) should be available.
        assert sum(mask) == 1
        
        active_idx = mask.index(1)
        
        # Assert the active index is in the defense range
        assert DEFENSE_ACTION_OFFSET <= active_idx < GLOBAL_ACTION_SPACE_SIZE - 1
        
        # It should correspond to masking index 0 (which is 1)
        assert active_idx == DEFENSE_ACTION_OFFSET + 1

    def test_global_action_to_hand_indices(self, handler):
        """Decode global action to local indices."""
        hand = [Card(2, Suit.HEARTS), Card(3, Suit.CLUBS), Card(2, Suit.SPADES)]
        
        # Let's find the global action ID for 'Play 2 of Hearts + 2 of Spades'
        global_id = -1
        for i, a in enumerate(handler._global_attack_actions):
            if a["type"] == "SameValue" and len(a["cards"]) == 2:
                suits = {c.suit for c in a["cards"]}
                if a["cards"][0].value == 2 and Suit.HEARTS in suits and Suit.SPADES in suits:
                    global_id = i
                    break
        
        assert global_id != -1
        
        indices = handler.global_action_to_hand_indices(global_id, hand)
        # Should map to indices 0 (2 of Hearts) and 2 (2 of Spades)
        assert sorted(indices) == [0, 2]
