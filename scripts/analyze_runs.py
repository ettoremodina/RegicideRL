"""Aggregate persisted game summaries from the canonical catalog."""

import argparse
from pathlib import Path

from ml_logger import RunCatalog, get_logger, start_run

logger = get_logger(__name__)


def analyze_runs(artifacts_dir="artifacts", run_id=None):
    """Return aggregate statistics for one run or the entire catalog."""
    catalog = RunCatalog(Path(artifacts_dir) / "catalog.sqlite")
    runs = [catalog.get_run(run_id)] if run_id else catalog.list_runs(100_000)
    games = [
        game
        for run in runs
        if run
        for game in catalog.list_games(run["run_id"])
    ]
    completed = [game for game in games if game["status"] == "completed"]
    total_games = len(completed)
    victories = sum(bool(game["victory"]) for game in completed)
    result = {
        "run_id": run_id,
        "total_games": total_games,
        "victories": victories,
        "win_rate": victories / total_games if total_games else 0.0,
        "avg_bosses_defeated": _average(completed, "bosses_defeated"),
        "avg_turns": _average(completed, "turns"),
    }
    logger.info(
        "Analyzed %d games: win rate %.2f%%, average bosses %.2f",
        total_games,
        result["win_rate"] * 100,
        result["avg_bosses_defeated"],
    )
    return result


def _average(rows, field):
    if not rows:
        return 0.0
    return sum(row[field] or 0 for row in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded Regicide games")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    context = start_run(
        "run-analysis",
        config=vars(args),
        root_dir=args.artifacts_dir,
    )
    try:
        result = analyze_runs(args.artifacts_dir, args.run_id)
        output = context.save_result("game_summary.json", result)
        context.complete({"result": str(output)})
    except Exception as error:
        context.fail(error)
        logger.exception("Run analysis failed")
        raise


if __name__ == "__main__":
    main()
