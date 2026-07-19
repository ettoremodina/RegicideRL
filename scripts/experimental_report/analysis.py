"""Create statistics, tables, plots, and a Markdown report from raw games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_logger import get_logger

from .configuration import load_report_config
from .plots import create_plots
from .report import generate_report, write_tables
from .statistics import compare_pairs, summarize_results

logger = get_logger(__name__)


def analyze_experiment(
    run_dir: str | Path,
    config_path: str | Path | None = None,
    report_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Analyze an existing experimental run without repeating simulations."""
    run_path = Path(run_dir)
    games_path = run_path / "datasets" / "games.csv"
    games = pd.read_csv(games_path)
    effective_config = _resolve_config(run_path, config_path, report_config)
    protocol = effective_config["protocol"]
    summary = summarize_results(
        games,
        confidence_level=float(protocol["confidence_level"]),
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        random_seed=int(protocol["base_seed"]),
    )
    pairwise = compare_pairs(
        games,
        confidence_level=float(protocol["confidence_level"]),
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        random_seed=int(protocol["base_seed"]) + 1,
    )
    output_dir = run_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = write_tables(summary, pairwise, output_dir)
    plot_paths = create_plots(games, summary, output_dir)
    report_path = generate_report(
        games,
        summary,
        pairwise,
        effective_config,
        output_dir,
    )
    _write_statistics_json(summary, pairwise, output_dir)
    logger.info("Experimental report generated at %s", report_path)
    return {
        "report": report_path,
        "summary": table_paths[0],
        "pairwise": table_paths[1],
        "dashboard": plot_paths[-1],
    }


def _write_statistics_json(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist full aggregate and pairwise records for downstream consumers."""
    payload = {
        "summary": summary.to_dict(orient="records"),
        "pairwise_tests": pairwise.to_dict(orient="records"),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve_config(
    run_path: Path,
    config_path: str | Path | None,
    report_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve explicit config data, a requested file, or the run snapshot."""
    if report_config is not None:
        return report_config
    if config_path is not None:
        return load_report_config(config_path)
    snapshot = run_path / "experimental_report_config.yaml"
    return load_report_config(snapshot if snapshot.exists() else "config.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing datasets/games.csv")
    parser.add_argument("--config")
    return parser


def main() -> None:
    """CLI entry point for report regeneration."""
    arguments = _build_parser().parse_args()
    analyze_experiment(arguments.run_dir, arguments.config)


if __name__ == "__main__":
    main()
