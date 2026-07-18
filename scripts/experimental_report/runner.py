"""Run configured Regicide agents under a paired, reproducible protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_logger import RunContext, get_logger, start_run
from solvers.env import RegicideEnv

from .configuration import (
    apply_protocol_overrides,
    load_report_config,
    select_agents,
    snapshot_report_config,
)
from .registry import build_agent

logger = get_logger(__name__)

GAME_COLUMNS = (
    "agent",
    "label",
    "family",
    "seed",
    "status",
    "victory",
    "bosses_defeated",
    "turns",
    "duration_seconds",
    "mean_decision_ms",
    "p95_decision_ms",
    "decisions",
    "defense_decisions",
    "yield_actions",
    "invalid_actions",
    "reward",
)


def run_comparison(
    config_path: str | Path = "config.yaml",
    requested_agents: list[str] | None = None,
    games: int | None = None,
    base_seed: int | None = None,
    run_context: RunContext | None = None,
) -> tuple[pd.DataFrame, RunContext]:
    """Evaluate agents on identical seeds and persist one row per game."""
    report_config = apply_protocol_overrides(
        load_report_config(config_path),
        games=games,
        base_seed=base_seed,
    )
    agents = select_agents(report_config, requested_agents)
    owns_context = run_context is None
    context = run_context or start_run(
        "experimental-comparison",
        name="agent-comparison",
        config=report_config,
    )
    try:
        if owns_context:
            effective_config = {**report_config, "agents": agents}
            snapshot_report_config(config_path, effective_config, context.run_dir)
        output_path = context.run_dir / "datasets" / "games.csv"
        rows = _load_checkpoint(context.run_dir)
        if rows:
            logger.info("Recovered %d completed games from checkpoint", len(rows))
            _save_checkpoint(rows, output_path)
        rows = _evaluate_agents(
            agents,
            report_config["protocol"],
            context,
            rows,
            output_path,
        )
        results = pd.DataFrame(rows)
        if owns_context:
            context.complete({"games_csv": str(output_path)})
        return results, context
    except Exception as error:
        if owns_context:
            context.fail(error)
        raise


def _evaluate_agents(
    agents: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    context: RunContext,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    seeds = _paired_seeds(protocol)
    completed = {(row["agent"], int(row["seed"])) for row in rows}
    for agent_key, specification in agents.items():
        pending_seeds = [
            seed for seed in seeds if (agent_key, seed) not in completed
        ]
        if not pending_seeds:
            logger.info("Skipping %s; all games are already complete", agent_key)
            continue
        logger.info("Evaluating %s on %d games", agent_key, len(seeds))
        agent = build_agent(specification)
        for game_index, seed in enumerate(seeds):
            if (agent_key, seed) in completed:
                continue
            row = _play_game(agent_key, specification, agent, seed, protocol)
            rows.append(row)
            context.log_metrics(len(rows), row)
            _save_checkpoint(rows, output_path)
            completed.add((agent_key, seed))
            logger.info(
                "%s game %d/%d: win=%s, bosses=%d, time=%.3fs",
                specification["label"],
                game_index + 1,
                len(seeds),
                row["victory"],
                row["bosses_defeated"],
                row["duration_seconds"],
            )
    return rows


def _load_checkpoint(run_dir: Path) -> list[dict[str, Any]]:
    dataset_path = run_dir / "datasets" / "games.csv"
    if dataset_path.exists() and dataset_path.stat().st_size:
        games = pd.read_csv(dataset_path)
        return _deduplicate_rows(games.to_dict(orient="records"))
    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if all(column in payload for column in GAME_COLUMNS):
                rows.append({column: payload[column] for column in GAME_COLUMNS})
    return _deduplicate_rows(rows)


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        unique[(row["agent"], int(row["seed"]))] = row
    return list(unique.values())


def _save_checkpoint(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=GAME_COLUMNS).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def _play_game(
    agent_key: str,
    specification: dict[str, Any],
    agent: Any,
    seed: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    _seed_everything(seed)
    environment = RegicideEnv(num_players=1)
    observation, _ = environment.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    counters = _new_counters()
    started_at = time.perf_counter()
    status = _run_episode(agent, environment, observation, counters, protocol)
    duration = time.perf_counter() - started_at
    bosses_defeated = _count_bosses_defeated(environment)
    return _game_row(
        agent_key,
        specification,
        seed,
        environment,
        counters,
        status,
        duration,
        bosses_defeated,
    )


def _run_episode(
    agent: Any,
    environment: RegicideEnv,
    observation: dict[str, Any],
    counters: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    limit = int(protocol["max_decisions_per_game"])
    for turn in range(limit):
        if hasattr(agent, "set_context"):
            agent.set_context(1, 1, turn + 1)
        decision_started = time.perf_counter()
        action = agent.select_action(observation, env=environment)
        counters["decision_times"].append(time.perf_counter() - decision_started)
        if action is None:
            return "no_action"
        _count_action(observation, int(action), counters)
        observation, reward, terminated, truncated, info = environment.step(action)
        counters["reward"] += float(reward)
        counters["invalid_actions"] += int(not info.get("success", True))
        if terminated or truncated:
            return "completed"
    return "decision_limit"


def _new_counters() -> dict[str, Any]:
    return {
        "decisions": 0,
        "defense_decisions": 0,
        "yield_actions": 0,
        "invalid_actions": 0,
        "reward": 0.0,
        "decision_times": [],
    }


def _count_action(
    observation: dict[str, Any],
    action: int,
    counters: dict[str, Any],
) -> None:
    counters["decisions"] += 1
    counters["defense_decisions"] += int(observation["defense_phase"])
    counters["yield_actions"] += int(action == 0 and not observation["defense_phase"])


def _game_row(
    agent_key: str,
    specification: dict[str, Any],
    seed: int,
    environment: RegicideEnv,
    counters: dict[str, Any],
    status: str,
    duration: float,
    bosses_defeated: int,
) -> dict[str, Any]:
    decision_times = np.asarray(counters.pop("decision_times"), dtype=float)
    return {
        "agent": agent_key,
        "label": specification["label"],
        "family": specification.get("family", "Unspecified"),
        "seed": seed,
        "status": status,
        "victory": bool(environment.game.victory),
        "bosses_defeated": bosses_defeated,
        "turns": counters["decisions"],
        "duration_seconds": duration,
        "mean_decision_ms": _milliseconds(decision_times.mean()),
        "p95_decision_ms": _milliseconds(_safe_percentile(decision_times, 95)),
        **counters,
    }


def _milliseconds(value: float) -> float:
    return float(value * 1000.0) if np.isfinite(value) else 0.0


def _safe_percentile(values: np.ndarray, percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _count_bosses_defeated(environment: RegicideEnv) -> int:
    enemies_left = len(environment.game.castle_deck)
    if environment.game.current_enemy is not None and not environment.game.victory:
        enemies_left += 1
    return 12 - enemies_left


def _paired_seeds(protocol: dict[str, Any]) -> list[int]:
    base_seed = int(protocol["base_seed"])
    games = int(protocol["games_per_agent"])
    return [base_seed + offset for offset in range(games)]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--games", type=int)
    parser.add_argument("--base-seed", type=int)
    return parser


def main() -> None:
    """CLI entry point for raw experiment execution."""
    arguments = _build_parser().parse_args()
    run_comparison(
        config_path=arguments.config,
        requested_agents=arguments.agents,
        games=arguments.games,
        base_seed=arguments.base_seed,
    )


if __name__ == "__main__":
    main()
