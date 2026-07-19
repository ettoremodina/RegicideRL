"""Regicide-specific recording adapter for the generic ``ml_logger`` core."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ml_logger.serialization import json_safe
from ml_logger.storage import RunContext, utc_now

RecordingLevel = Literal["summary", "actions", "full"]
VALID_RECORDING_LEVELS = {"summary", "actions", "full"}


@dataclass(frozen=True)
class GameCatalogRecord:
    """One mutable game lifecycle snapshot written to the adapter catalog."""

    game_id: str
    run_id: str
    status: str
    started_at: str
    path: Path
    summary: dict[str, Any]


class GameCatalog:
    """Regicide-owned SQLite index sharing the generic catalog database."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def from_context(cls, context: RunContext) -> "GameCatalog":
        """Open the game index stored beside a run catalog."""
        return cls(context.catalog.database_path)

    def upsert(self, record: GameCatalogRecord) -> None:
        """Insert a running game or refresh its final summary."""
        payload = (
            json.dumps(json_safe(record.summary), ensure_ascii=False)
            if record.summary
            else None
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO games (
                    game_id, run_id, status, victory, bosses_defeated,
                    turns, started_at, ended_at, path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    status=excluded.status,
                    victory=excluded.victory,
                    bosses_defeated=excluded.bosses_defeated,
                    turns=excluded.turns,
                    ended_at=excluded.ended_at,
                    summary_json=excluded.summary_json
                """,
                _catalog_values(record, payload),
            )

    def list_games(self, run_id: str) -> list[dict[str, Any]]:
        """Return games belonging to a run in start-time order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM games WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        """Return one game row by identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        return dict(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        """Create the adapter-owned table without altering generic schemas."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    victory INTEGER,
                    bosses_defeated INTEGER,
                    turns INTEGER,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    path TEXT NOT NULL,
                    summary_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_games_run_id ON games(run_id);
                """
            )


class GameRecorder:
    """Record replayable Regicide games below a generic parent run."""

    def __init__(
        self,
        context: RunContext,
        recording_level: RecordingLevel | None = None,
    ):
        settings = _recording_settings(context.settings)
        effective_level = recording_level or settings.get("level", "actions")
        if effective_level not in VALID_RECORDING_LEVELS:
            raise ValueError(f"Unsupported recording level: {effective_level}")
        self.context = context
        self.catalog = GameCatalog.from_context(context)
        self.enabled = bool(
            context.settings.get("saving", {}).get("enabled", True)
            and settings.get("enabled", True)
        )
        self.recording_level = effective_level
        self.game_id: str | None = None
        self.game_dir: Path | None = None
        self.started_at: str | None = None
        self.event_sequence = 0
        self.turns = 0

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict[str, str],
        recording_level: RecordingLevel | None = None,
    ) -> "GameRecorder":
        """Attach a worker-local recorder without owning shared JSONL streams."""
        context = RunContext.attach(
            descriptor["run_id"],
            descriptor["run_dir"],
            descriptor["root_dir"],
        )
        return cls(context, recording_level)

    @property
    def active(self) -> bool:
        """Return whether a game has started and not yet finalized."""
        return self.game_id is not None

    def begin_game(
        self,
        game: Any,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start one game artifact and index its initial state."""
        if not self.enabled:
            return ""
        if self.active:
            self.abort("new_game_started")
        self._initialize_game()
        initial = {
            "schema_version": 1,
            "game_id": self.game_id,
            "run_id": self.context.run_id,
            "started_at": self.started_at,
            "seed": seed,
            "recording_level": self.recording_level,
            "metadata": json_safe(metadata or {}),
            "state": serialize_game(game),
        }
        self._write_json("initial_state.json", initial)
        self._upsert_catalog("running")
        return self.game_id or ""

    def record_event(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
        game: Any,
        state_before: dict[str, Any] | None = None,
    ) -> None:
        """Append one action transition at the selected recording level."""
        if not self.enabled:
            return
        if not self.active:
            raise RuntimeError("No active game recording")
        self.event_sequence += 1
        self.turns += 1
        if self.recording_level == "summary":
            return
        event = {
            "schema_version": 1,
            "game_id": self.game_id,
            "sequence": self.event_sequence,
            "timestamp": utc_now(),
            "action": json_safe(action),
            "result": json_safe(result),
        }
        if self.recording_level == "full":
            event["state_before"] = state_before
            event["state_after"] = serialize_game(game)
        self._append_jsonl("events.jsonl", event)

    def finish(
        self,
        game: Any,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize a completed game and return its persisted summary."""
        if not self.enabled:
            return {}
        if not self.active:
            raise RuntimeError("No active game recording")
        summary = self._build_summary(game, reason, metadata)
        self._write_json("summary.json", summary)
        self._upsert_catalog("completed", summary)
        self._reset_active()
        return summary

    def abort(self, reason: str) -> None:
        """Finalize an active game with interrupted status."""
        if not self.active:
            return
        summary = {
            "schema_version": 1,
            "game_id": self.game_id,
            "run_id": self.context.run_id,
            "status": "interrupted",
            "started_at": self.started_at,
            "ended_at": utc_now(),
            "turns": self.turns,
            "reason": reason,
        }
        self._write_json("summary.json", summary)
        self._upsert_catalog("interrupted", summary)
        self._reset_active()

    def _initialize_game(self) -> None:
        self.game_id = f"game-{uuid.uuid4().hex}"
        self.game_dir = self.context.run_dir / "games" / self.game_id
        self.game_dir.mkdir(parents=True)
        self.started_at = utc_now()
        self.event_sequence = 0
        self.turns = 0

    def _build_summary(
        self,
        game: Any,
        reason: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the final domain summary from a completed game state."""
        enemies_left = len(game.castle_deck)
        if game.current_enemy is not None and not game.victory:
            enemies_left += 1
        return {
            "schema_version": 1,
            "game_id": self.game_id,
            "run_id": self.context.run_id,
            "status": "completed",
            "started_at": self.started_at,
            "ended_at": utc_now(),
            "victory": bool(game.victory),
            "victory_tier": game.get_victory_tier(),
            "bosses_defeated": 12 - enemies_left,
            "turns": self.turns,
            "reason": reason,
            "metadata": json_safe(metadata or {}),
            "final_state": serialize_game(game),
        }

    def _upsert_catalog(
        self,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Translate recorder state into one game catalog record."""
        self.catalog.upsert(
            GameCatalogRecord(
                game_id=self.game_id or "",
                run_id=self.context.run_id,
                status=status,
                started_at=self.started_at or utc_now(),
                path=self._require_game_dir(),
                summary=summary or {},
            )
        )

    def _write_json(self, filename: str, data: dict[str, Any]) -> None:
        path = self._require_game_dir() / filename
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(json_safe(data), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _append_jsonl(self, filename: str, data: dict[str, Any]) -> None:
        path = self._require_game_dir() / filename
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(data), ensure_ascii=False) + "\n")

    def _require_game_dir(self) -> Path:
        """Return the active game directory or reject invalid lifecycle use."""
        if self.game_dir is None:
            raise RuntimeError("No active game directory")
        return self.game_dir

    def _reset_active(self) -> None:
        self.game_id = None
        self.game_dir = None
        self.started_at = None


def serialize_card(card: Any) -> dict[str, Any]:
    """Serialize a Regicide card."""
    return {"value": card.value, "suit": card.suit.name, "label": str(card)}


def serialize_enemy(enemy: Any) -> dict[str, Any] | None:
    """Serialize the current Regicide enemy."""
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
    """Capture a stable replay-oriented Regicide game snapshot."""
    state = _visible_game_state(game)
    if include_hidden:
        state["tavern_deck"] = [serialize_card(card) for card in game.tavern_deck]
        state["castle_deck"] = [serialize_card(card) for card in game.castle_deck]
    else:
        state["tavern_cards"] = len(game.tavern_deck)
        state["enemies_remaining"] = len(game.castle_deck)
    return state


def _visible_game_state(game: Any) -> dict[str, Any]:
    """Serialize the game state visible regardless of hidden-card policy."""
    return {
        "num_players": game.num_players,
        "current_player": game.current_player,
        "players": [
            [serialize_card(card) for card in hand] for hand in game.players
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


def _recording_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve current adapter settings with legacy-key compatibility."""
    integration = (
        settings.get("integrations", {})
        .get("regicide", {})
        .get("recording")
    )
    if integration is not None:
        return integration
    legacy = settings.get("games", {})
    return {
        "enabled": legacy.get("enabled", True),
        "level": legacy.get("recording_level", "actions"),
    }


def _catalog_values(
    record: GameCatalogRecord,
    payload: str | None,
) -> tuple[Any, ...]:
    """Flatten a game record into the SQLite statement value order."""
    summary = record.summary
    return (
        record.game_id,
        record.run_id,
        record.status,
        summary.get("victory"),
        summary.get("bosses_defeated"),
        summary.get("turns"),
        record.started_at,
        summary.get("ended_at"),
        str(record.path),
        payload,
    )
