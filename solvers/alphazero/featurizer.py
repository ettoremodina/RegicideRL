"""
State featurizer for AlphaZero.

Converts a RegicideEnv into a flat numeric feature vector suitable for
the neural network. Operates directly on ``env.game`` for speed — this
runs inside the MCTS loop thousands of times per decision, so we avoid
Gymnasium wrappers and string allocations.

Also provides bidirectional helpers between the 542-dimensional global
action index and the 8-element hand-relative action mask used internally
by the ISMCTS tree.
"""

import numpy as np
from game.regicide import Suit

# Canonical suit ordering used for one-hot encoding.
_SUIT_INDEX = {
    Suit.HEARTS: 0,
    Suit.DIAMONDS: 1,
    Suit.CLUBS: 2,
    Suit.SPADES: 3,
}


def encode_state(env) -> np.ndarray:
    """Encode a RegicideEnv snapshot into a flat float32 feature vector.

    The vector is constructed purely from *public* information visible to
    the current player (hand, current enemy, counts of hidden zones, phase
    flags).  Hidden deck orderings are deliberately excluded.

    Returns:
        np.ndarray of shape ``(state_dim,)`` — currently 56 floats.
    """
    game = env.game
    hand = game.get_player_hand(game.current_player)
    enemy = game.current_enemy

    features = []

    # --- Hand cards (8 slots × 5 features = 40) ---
    for i in range(8):
        if i < len(hand):
            card = hand[i]
            # Normalized card value (0-13 → 0.0-1.0)
            features.append(card.value / 13.0)
            # One-hot suit (4 dims)
            suit_oh = [0.0] * 4
            suit_oh[_SUIT_INDEX[card.suit]] = 1.0
            features.extend(suit_oh)
        else:
            # Empty slot
            features.append(0.0)
            features.extend([0.0] * 4)

    # --- Enemy features (7) ---
    if enemy is not None:
        hp_remaining = max(0, enemy.health - enemy.damage_taken)
        features.append(hp_remaining / 40.0)          # normalized HP
        features.append(enemy.attack / 20.0)           # normalized attack
        # Enemy suit one-hot (4 dims)
        suit_oh = [0.0] * 4
        suit_oh[_SUIT_INDEX[enemy.card.suit]] = 1.0
        features.extend(suit_oh)
        features.append(enemy.spade_protection / 20.0)  # normalized spade prot
    else:
        features.extend([0.0] * 7)

    # --- Global flags (9) ---
    features.append(1.0 if game.jester_immunity_cancelled else 0.0)
    features.append(1.0 if env.required_defense > 0 else 0.0)
    features.append(env.required_defense / 20.0)
    features.append(len(game.tavern_deck) / 40.0)
    features.append(len(game.discard_pile) / 52.0)
    features.append(len(game.castle_deck) / 12.0)
    features.append(len(hand) / 8.0)
    features.append(1.0 if game.can_yield() else 0.0)
    features.append(game.solo_jesters_remaining / 2.0)

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Action-index helpers
# ---------------------------------------------------------------------------

def action_mask_to_global_index(action_mask, hand, handler, is_defense):
    """Convert a hand-relative action mask to a global action index (0-541).

    During MCTS we work with hand-relative masks (the ``valid_actions``
    list). This converts one such mask into the global index that the
    neural network's policy head uses.

    Args:
        action_mask: list[int] of length 8 — the hand-relative binary mask.
        hand: The current player's hand (list of Card).
        handler: An ActionHandler instance.
        is_defense: Whether we are in the defense phase.

    Returns:
        int in [0, 541].
    """
    if is_defense:
        # Defense actions use the bitmask encoding in indices 286-541.
        val = 0
        for i, bit in enumerate(action_mask):
            if bit:
                val += (1 << i)
        return 286 + val
    else:
        # Attack actions: match the played cards against the global table.
        if handler.is_yield_action(action_mask):
            return 0  # Yield is always index 0

        indices = handler.mask_to_card_indices(action_mask, len(hand))
        cards_played = sorted([hand[i] for i in indices])
        cards_tuple = tuple(cards_played)

        for i, global_action in enumerate(handler._global_attack_actions):
            if tuple(sorted(global_action["cards"])) == cards_tuple:
                return i

        # Fallback — should never happen if action was valid.
        raise ValueError(
            f"Could not map action mask {action_mask} to a global index. "
            f"Cards: {[str(c) for c in cards_played]}"
        )


def global_index_to_action_mask(global_index, hand, handler):
    """Convert a global action index (0-541) back to a hand-relative mask.

    This is the inverse of ``action_mask_to_global_index``.  Used when the
    network picks an action and we need to execute it in the environment.

    Args:
        global_index: int in [0, 541].
        hand: The current player's hand (list of Card).
        handler: An ActionHandler instance.

    Returns:
        list[int] of length ``handler.max_hand_size``.
    """
    card_indices = handler.global_action_to_hand_indices(global_index, hand)
    return handler.cards_to_mask(card_indices)
