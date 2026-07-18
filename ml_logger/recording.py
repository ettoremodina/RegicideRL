"""Persistent recording of real Regicide games."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

from .serialization import json_safe, serialize_game
from .storage import RunContext, utc_now

RecordingLevel = Literal["summary", "actions", "full"]
VALID_RECORDING_LEVELS = {"summary", "actions", "full"}


class GameRecorder:
    """Record one real game at a time under a parent run."""

    def __init__(
        self,
        context: RunContext,
        recording_level: RecordingLevel | None = None,
    ):
        effective_level = recording_level or context.game_recording_level
        if effective_level not in VALID_RECORDING_LEVELS:
            raise ValueError(f"Unsupported recording level: {effective_level}")
        self.context = context
        self.enabled = context.game_recording_enabled
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
        context = RunContext.attach(
            descriptor["run_id"],
            descriptor["run_dir"],
            descriptor["root_dir"],
        )
        return cls(context, recording_level)

    @property
    def active(self) -> bool:
        return self.game_id is not None

    def begin_game(
        self,
        game: Any,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.enabled:
            return ""
        if self.active:
            self.abort("new_game_started")
        self.game_id = f"game-{uuid.uuid4().hex}"
        self.game_dir = self.context.run_dir / "games" / self.game_id
        self.game_dir.mkdir(parents=True)
        self.started_at = utc_now()
        self.event_sequence = 0
        self.turns = 0
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
        self.context.catalog.upsert_game(
            self.game_id,
            self.context.run_id,
            "running",
            self.started_at,
            self.game_dir,
        )
        return self.game_id

    def record_event(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
        game: Any,
        state_before: dict[str, Any] | None = None,
    ) -> None:
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
        if not self.enabled:
            return {}
        if not self.active:
            raise RuntimeError("No active game recording")
        enemies_left = len(game.castle_deck)
        if game.current_enemy is not None and not game.victory:
            enemies_left += 1
        summary = {
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
        self._write_json("summary.json", summary)
        self._catalog_summary(summary)
        self._reset_active()
        return summary

    def abort(self, reason: str) -> None:
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
        self._catalog_summary(summary)
        self._reset_active()

    def _catalog_summary(self, summary: dict[str, Any]) -> None:
        self.context.catalog.upsert_game(
            summary["game_id"],
            self.context.run_id,
            summary["status"],
            summary["started_at"],
            self.game_dir,
            summary,
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
        if self.game_dir is None:
            raise RuntimeError("No active game directory")
        return self.game_dir

    def _reset_active(self) -> None:
        self.game_id = None
        self.game_dir = None
        self.started_at = None
