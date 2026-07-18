"""Run agents and generate the complete experimental report in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_logger import RunContext, configure_logging, get_logger, start_run

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
    resume_run: str | Path | None = None,
    jobs: int = 1,
):
    """Execute simulations followed by all report-generation stages."""
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    context, report_config, effective_config_path, is_resuming = _prepare_run(
        config_path,
        requested_agents,
        games,
        base_seed,
        resume_run,
    )
    report_config["protocol"]["parallel_jobs"] = jobs
    context.manifest["config"] = report_config
    try:
        if not is_resuming:
            snapshot_report_config(config_path, report_config, context.run_dir)
        run_comparison(
            config_path=effective_config_path,
            requested_agents=list(report_config["agents"]),
            games=int(report_config["protocol"]["games_per_agent"]),
            base_seed=int(report_config["protocol"]["base_seed"]),
            run_context=context,
            jobs=jobs,
        )
        outputs = analyze_experiment(
            context.run_dir,
            config_path=effective_config_path,
            report_config=report_config,
        )
        context.complete({key: str(value) for key, value in outputs.items()})
        logger.info("Experimental pipeline completed in %s", context.run_dir)
        return context
    except Exception as error:
        context.fail(error)
        logger.exception("Experimental pipeline failed")
        raise


def _prepare_run(
    config_path: str | Path,
    requested_agents: list[str] | None,
    games: int | None,
    base_seed: int | None,
    resume_run: str | Path | None,
):
    if resume_run is not None:
        if requested_agents or games is not None or base_seed is not None:
            raise ValueError(
                "--resume-run cannot be combined with --agents, --games, "
                "or --base-seed"
            )
        return _resume_existing_run(Path(resume_run))
    report_config = apply_protocol_overrides(
        load_report_config(config_path),
        games=games,
        base_seed=base_seed,
    )
    report_config["agents"] = select_agents(report_config, requested_agents)
    context = start_run(
        "experimental-report",
        name="agent-comparison-report",
        config=report_config,
    )
    return context, report_config, Path(config_path), False


def _resume_existing_run(run_dir: Path):
    snapshot_path = run_dir / "experimental_report_config.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run configuration: {snapshot_path}")
    report_config = load_report_config(snapshot_path)
    root_dir = run_dir.parents[2]
    context = RunContext.attach(run_dir.name, run_dir, root_dir)
    previous_result = context.manifest.pop("result", None)
    if previous_result:
        context.manifest.setdefault("metadata", {})["previous_result"] = previous_result
    context.manifest["status"] = "running"
    context.manifest["ended_at"] = None
    context.manifest["config"] = report_config
    context.catalog.upsert_run(context.manifest, context.run_dir)
    configure_logging(context)
    logger.info("Resuming experimental run %s", context.run_id)
    return context, report_config, snapshot_path, True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--games", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--resume-run")
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> None:
    """CLI entry point for the complete report pipeline."""
    arguments = _build_parser().parse_args()
    run_pipeline(
        config_path=arguments.config,
        requested_agents=arguments.agents,
        games=arguments.games,
        base_seed=arguments.base_seed,
        resume_run=arguments.resume_run,
        jobs=arguments.jobs,
    )


if __name__ == "__main__":
    main()
