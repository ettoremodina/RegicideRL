"""Inspect canonical runs and recorded games without direct database access."""

import argparse
import json
from pathlib import Path

from ml_logger import RunCatalog, configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Inspect runs, games, and replay events through the catalog CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging()
    catalog = RunCatalog(Path(args.artifacts_dir) / "catalog.sqlite")
    commands = {
        "list": lambda: _list_runs(catalog, args.limit),
        "show": lambda: _show_run(catalog, args.run_id),
        "games": lambda: _list_games(catalog, args.run_id),
        "replay": lambda: _replay_game(catalog, args.game_id),
    }
    commands[args.command]()


def _build_parser():
    """Create subcommands for catalog and replay inspection."""
    parser = argparse.ArgumentParser(description="Inspect Regicide artifacts")
    parser.add_argument("--artifacts-dir", default="artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("run_id")
    games_parser = subparsers.add_parser("games")
    games_parser.add_argument("run_id")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("game_id")
    return parser


def _list_runs(catalog, limit):
    """Log a compact table of the newest runs."""
    rows = catalog.list_runs(limit)
    if not rows:
        logger.info("No recorded runs")
        return
    for row in rows:
        logger.info(
            "%s | %-12s | %-10s | %s",
            row["run_id"],
            row["run_type"],
            row["status"],
            row["path"],
        )


def _show_run(catalog, run_id):
    row = catalog.get_run(run_id)
    if not row:
        logger.error("Unknown run: %s", run_id)
        return
    logger.info("%s", json.dumps(json.loads(row["manifest_json"]), indent=2))


def _list_games(catalog, run_id):
    """Log recorded game summaries belonging to one run."""
    games = catalog.list_games(run_id)
    if not games:
        logger.info("No games recorded for %s", run_id)
        return
    for game in games:
        logger.info(
            "%s | %-11s | victory=%s | bosses=%s | turns=%s",
            game["game_id"],
            game["status"],
            game["victory"],
            game["bosses_defeated"],
            game["turns"],
        )


def _replay_game(catalog, game_id):
    """Stream a recorded action history in sequence order."""
    game = catalog.get_game(game_id)
    if not game:
        logger.error("Unknown game: %s", game_id)
        return
    events_path = Path(game["path"]) / "events.jsonl"
    if not events_path.exists():
        logger.warning("Game %s was recorded at summary level", game_id)
        return
    with events_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            action = event["action"]
            logger.info(
                "%04d | player=%s | phase=%s | %s %s",
                event["sequence"],
                action.get("player"),
                action.get("phase"),
                action.get("kind"),
                ", ".join(action.get("cards", [])),
            )


if __name__ == "__main__":
    main()
