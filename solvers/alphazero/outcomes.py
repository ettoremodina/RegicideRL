"""Shared outcome calculations for AlphaZero search and training."""


def enemies_defeated(game) -> int:
    """Return the number of defeated castle enemies."""
    enemies_left = len(game.castle_deck) + (
        1 if game.current_enemy and not game.victory else 0
    )
    return 12 - enemies_left


def terminal_value(game) -> float:
    """Map terminal progress to the shaped value target in [-1, 1]."""
    if game.victory:
        return 1.0
    progress = enemies_defeated(game) / 12.0
    return progress * 2.0 - 1.0
