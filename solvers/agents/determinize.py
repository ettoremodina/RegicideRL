"""
Determinization utilities for imperfect-information search in Regicide.

Both PIMC and ISMCTS need to "determinize" the hidden game state — i.e., sample
a concrete ordering of the unseen cards that is consistent with the player's
current observations. This module provides a shared implementation.

In solo Regicide the hidden information is:
  1. The order of the tavern deck (player sees their hand but not the deck).
  2. The order of the remaining castle deck (only the current enemy is revealed).

The determinization shuffles these unknowns while keeping all observed state
(hand, current enemy, discard pile, damage counters) fixed.
"""

import random


def determinize_env(env_clone):
    """Shuffle the hidden decks of a *cloned* environment in-place.

    IMPORTANT: Only call this on env clones — it mutates the game state.
    The caller is responsible for cloning the env before calling this.

    The procedure:
      1. Shuffle the tavern deck (the player cannot see its order).
      2. Shuffle the castle deck (the player only knows the *current* enemy;
         the remaining enemies' order within each tier is unknown, but the
         tier ordering Jacks→Queens→Kings is fixed by game rules).

    Args:
        env_clone: A cloned RegicideEnv whose game state will be mutated.
    """
    game = env_clone.game

    # --- Tavern deck ---
    # The player can see their own hand and the discard pile, but not the
    # tavern deck order.  Shuffle it freely.
    random.shuffle(game.tavern_deck)

    # --- Castle deck ---
    # The remaining castle enemies are separated into tiers (J, Q, K) and
    # each tier is internally shuffled. The tier order is preserved:
    # all remaining Jacks come first, then Queens, then Kings.
    if game.castle_deck:
        jacks = [c for c in game.castle_deck if c.value == 11]
        queens = [c for c in game.castle_deck if c.value == 12]
        kings = [c for c in game.castle_deck if c.value == 13]
        random.shuffle(jacks)
        random.shuffle(queens)
        random.shuffle(kings)
        game.castle_deck = jacks + queens + kings
