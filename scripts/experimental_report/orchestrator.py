"""Run agents and generate the complete experimental report in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_logger import get_logger, start_run

from .analysis import analyze_experiment
from .configuration import (
    apply_protocol_overrides,
    load_report_config,
    select_agents,
    snapshot_report_config,
)
from .runner import run_comparison

logger = get_logger(__name__)


def run_pipeline(
    config_path: str | Path = "config.yaml",
    requested_agents: list[str] | None = None,
    games: int | None = None,
    base_seed: int | None = None,
):
    """Execute simulations followed by all report-generation stages."""
    report_config = apply_protocol_overrides(
        load_report_config(config_path),
        games=games,
        base_seed=base_seed,
    )
    selected_agents = select_agents(report_config, requested_agents)
    report_config["agents"] = selected_agents
    context = start_run(
        "experimental-report",
        name="agent-comparison-report",
        config=report_config,
    )
    try:
        snapshot_report_config(config_path, report_config, context.run_dir)
        run_comparison(
            config_path=config_path,
            requested_agents=list(selected_agents),
            games=int(report_config["protocol"]["games_per_agent"]),
            base_seed=int(report_config["protocol"]["base_seed"]),
            run_context=context,
        )
        outputs = analyze_experiment(
            context.run_dir,
            config_path=config_path,
            report_config=report_config,
        )
        context.complete({key: str(value) for key, value in outputs.items()})
        logger.info("Experimental pipeline completed in %s", context.run_dir)
        return context
    except Exception as error:
        context.fail(error)
        logger.exception("Experimental pipeline failed")
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--games", type=int)
    parser.add_argument("--base-seed", type=int)
    return parser


def main() -> None:
    """CLI entry point for the complete report pipeline."""
    arguments = _build_parser().parse_args()
    run_pipeline(
        config_path=arguments.config,
        requested_agents=arguments.agents,
        games=arguments.games,
        base_seed=arguments.base_seed,
    )


if __name__ == "__main__":
    main()
