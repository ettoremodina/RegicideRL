"""Markdown report generation for Regicide agent comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    report_config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Render a self-contained English experimental report."""
    destination = Path(output_dir)
    report_path = destination / "experimental_report.md"
    protocol = report_config["protocol"]
    content = "\n\n".join(
        [
            "# Regicide — experimental report",
            "",
            _protocol_section(games, protocol),
            _algorithm_section(games, report_config["agents"]),
            _results_section(summary),
            _statistics_section(pairwise, protocol),
            _visualization_section(),
            _interpretation_section(summary, pairwise, protocol),
            _limitations_section(),
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def write_tables(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Persist complete CSV tables and compact Markdown/LaTeX tables."""
    destination = Path(output_dir)
    summary_csv = destination / "summary.csv"
    pairwise_csv = destination / "pairwise_tests.csv"
    summary.to_csv(summary_csv, index=False)
    pairwise.to_csv(pairwise_csv, index=False)
    compact_summary = _compact_summary(summary)
    compact_pairs = _compact_pairwise(pairwise)
    markdown_path = destination / "tables.md"
    markdown_path.write_text(
        "## Aggregate results\n\n"
        + _as_markdown(compact_summary)
        + "\n\n## Paired statistical comparisons\n\n"
        + _as_markdown(compact_pairs),
        encoding="utf-8",
    )
    latex_path = destination / "tables.tex"
    latex_path.write_text(
        compact_summary.to_latex(index=False, float_format="%.3f")
        + "\n\n"
        + compact_pairs.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )
    return [summary_csv, pairwise_csv, markdown_path, latex_path]


def _protocol_section(games: pd.DataFrame, protocol: dict[str, Any]) -> str:
    """Describe pairing, limits, confidence settings, and parallel timing."""
    agents = games["agent"].nunique()
    seeds = games["seed"].nunique()
    parallel_jobs = int(protocol.get("parallel_jobs", 1))
    parallel_note = _parallel_timing_note(parallel_jobs)
    return f"""## Protocol

The comparison uses the same {seeds} seeds for each of the {agents} agents
({len(games)} games in total). Pairing by seed reduces variation caused by the
initial card order. Each game is limited to
{protocol["max_decisions_per_game"]} decisions. Timings and metrics are measured
per game; raw runs are stored in `datasets/games.csv`.

Confidence level: {protocol["confidence_level"]:.0%}; bootstrap samples:
{protocol["bootstrap_samples"]}. Parallel workers: {parallel_jobs}.
{parallel_note}"""


def _parallel_timing_note(parallel_jobs: int) -> str:
    if parallel_jobs == 1:
        return ""
    return (
        "Games for agents still awaiting evaluation were distributed across "
        f"{parallel_jobs} processes. Per-game timings are affected by CPU "
        "contention and are not directly comparable with agents run sequentially."
    )


def _algorithm_section(
    games: pd.DataFrame,
    agent_specs: dict[str, dict[str, Any]],
) -> str:
    """Describe only agents represented in the completed game dataset."""
    descriptions = []
    for agent_key in games["agent"].drop_duplicates():
        specification = agent_specs[agent_key]
        kwargs = specification.get("kwargs", {})
        parameters = ", ".join(
            f"`{key}={value}`" for key, value in kwargs.items() if key != "name"
        )
        suffix = f" Parameters: {parameters}." if parameters else ""
        descriptions.append(
            f"- **{specification['label']}** "
            f"({specification.get('family', 'N/A')}): "
            f"{specification['description']}{suffix}"
        )
    return "## Compared algorithms\n\n" + "\n".join(descriptions)


def _results_section(summary: pd.DataFrame) -> str:
    table = _as_markdown(_compact_summary(summary))
    return f"""## Results

{table}

Complete tables, including standard deviations, medians, and confidence
intervals for all metrics, are available in `summary.csv` and `tables.tex`."""


def _statistics_section(
    pairwise: pd.DataFrame,
    protocol: dict[str, Any],
) -> str:
    """Render paired-test results and explain the inferential methods."""
    confidence = protocol["confidence_level"]
    if pairwise.empty:
        table = "Only one agent was evaluated, so paired tests do not apply."
    else:
        table = _as_markdown(_compact_pairwise(pairwise))
    return f"""## Statistical analysis

{table}

Win rates use Wilson intervals. Continuous metrics use {confidence:.0%}
bootstrap intervals. Win comparisons use the exact McNemar test; comparisons
of bosses defeated use the paired Wilcoxon test. P-values are corrected with
Holm's method separately for each test family. `Cohen dz` quantifies the paired
effect on bosses defeated; probability of superiority assigns half weight to
ties."""


def _visualization_section() -> str:
    return """## Plots

![Comprehensive dashboard](comprehensive_dashboard.png)

![Win rate](win_rate.png)

![Bosses defeated](bosses_defeated.png)

![Execution time](execution_time.png)

![Quality-cost trade-off](quality_cost_tradeoff.png)"""


def _interpretation_section(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    protocol: dict[str, Any],
) -> str:
    """Summarize observed leaders without overstating sample rankings."""
    best_win = summary.sort_values("win_rate", ascending=False).iloc[0]
    best_progress = summary.sort_values("bosses_mean", ascending=False).iloc[0]
    fastest = summary.sort_values("duration_seconds_mean").iloc[0]
    alpha = 1.0 - float(protocol["confidence_level"])
    significant = _significant_comparisons(pairwise, alpha)
    return f"""## Interpretation

- Best observed win rate: **{best_win['label']}**
  ({best_win['win_rate']:.1%}).
- Greatest mean progress: **{best_progress['label']}**
  ({best_progress['bosses_mean']:.2f} bosses out of 12).
- Shortest mean time per game: **{fastest['label']}**
  ({fastest['duration_seconds_mean']:.4f} s).
- Significant comparisons after Holm correction: {significant}.

These rankings describe the sample. A claim that one method is superior should
consider confidence intervals, p-values, effect size, and computational cost
together."""


def _limitations_section() -> str:
    return """## Limitations and reproducibility

The protocol compares solo-mode play on the same machine. Timings should not be
compared directly with runs collected on different hardware or under different
system loads. A shared seed gives each agent the same initial state, but
stochastic algorithms may consume random numbers differently during play.
Neural policies disabled in the configuration are excluded until a valid
checkpoint is provided.

The effective configuration and per-game data are saved with the report so
tables and plots can be regenerated without repeating the simulations."""


def _compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Select and rename publication-facing aggregate columns."""
    compact = summary[
        [
            "label",
            "n_games",
            "win_rate",
            "win_ci_low",
            "win_ci_high",
            "bosses_mean",
            "bosses_ci_low",
            "bosses_ci_high",
            "duration_seconds_mean",
            "decision_ms_mean",
        ]
    ].copy()
    compact.columns = [
        "Agent",
        "N",
        "Win rate",
        "Win CI lower",
        "Win CI upper",
        "Mean bosses",
        "Boss CI lower",
        "Boss CI upper",
        "Mean time (s)",
        "Mean decision (ms)",
    ]
    return compact.round(4)


def _compact_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Select and rename publication-facing paired-comparison columns."""
    if pairwise.empty:
        return pairwise
    compact = pairwise[
        [
            "agent_a",
            "agent_b",
            "n_pairs",
            "win_rate_difference",
            "win_p_holm",
            "bosses_mean_difference",
            "bosses_p_holm",
            "bosses_cohen_dz",
            "median_duration_ratio",
        ]
    ].copy()
    compact.columns = [
        "Agent A",
        "Agent B",
        "N",
        "Δ win rate",
        "p Holm (win)",
        "Δ bosses",
        "p Holm (bosses)",
        "Cohen dz",
        "Time ratio A/B",
    ]
    return compact.round(4)


def _as_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "Not applicable."
    headers = [str(column) for column in table.columns]
    rows = [[_markdown_cell(value) for value in row] for row in table.itertuples(False)]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *data_lines])


def _markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _significant_comparisons(pairwise: pd.DataFrame, alpha: float) -> str:
    if pairwise.empty:
        return "not applicable"
    win_count = int((pairwise["win_p_holm"] < alpha).sum())
    boss_count = int((pairwise["bosses_p_holm"] < alpha).sum())
    return f"{win_count} for wins and {boss_count} for bosses (α={alpha:.3f})"
