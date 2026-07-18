"""JSON serialization helpers for run artifacts and Regicide game state."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common project values into JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def serialize_card(card: Any) -> dict[str, Any]:
    """Serialize a card without depending on the game module."""
    return {
        "value": card.value,
        "suit": card.suit.name,
        "label": str(card),
    }


def serialize_enemy(enemy: Any) -> dict[str, Any] | None:
    """Serialize the current enemy."""
    if enemy is None:
        return None
    return {
        "card": serialize_card(enemy.card),
        "health": enemy.health,
        "attack": enemy.attack,
        "damage_taken": enemy.damage_taken,
        "spade_protection": enemy.spade_protection,
    }


def serialize_game(game: Any, include_hidden: bool = True) -> dict[str, Any]:
    """Capture a stable, replay-oriented snapshot of a game."""
    state = {
        "num_players": game.num_players,
        "current_player": game.current_player,
        "players": [
            [serialize_card(card) for card in hand]
            for hand in game.players
        ],
        "current_enemy": serialize_enemy(game.current_enemy),
        "discard_pile": [serialize_card(card) for card in game.discard_pile],
        "attack_cards_buffer": [
            serialize_card(card) for card in game.attack_cards_buffer
        ],
        "game_over": game.game_over,
        "victory": game.victory,
        "victory_tier": game.get_victory_tier(),
        "jester_immunity_cancelled": game.jester_immunity_cancelled,
        "solo_jesters_remaining": game.solo_jesters_remaining,
        "solo_jesters_used": game.solo_jesters_used,
        "players_yielded_this_round": list(game.players_yielded_this_round),
        "last_active_player": game.last_active_player,
    }
    if include_hidden:
        state["tavern_deck"] = [
            serialize_card(card) for card in game.tavern_deck
        ]
        state["castle_deck"] = [
            serialize_card(card) for card in game.castle_deck
        ]
    else:
        state["tavern_cards"] = len(game.tavern_deck)
        state["enemies_remaining"] = len(game.castle_deck)
    return state
