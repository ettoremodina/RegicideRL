"""Publication-ready plots for the experimental comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = "colorblind"


def create_plots(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate individual figures and a comprehensive dashboard."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    paths = [
        _plot_win_rate(summary, destination),
        _plot_bosses(games, destination),
        _plot_execution_time(games, destination),
        _plot_tradeoff(summary, destination),
        _plot_dashboard(games, summary, destination),
    ]
    return paths


def _plot_win_rate(summary: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(summary))
    rates = summary["win_rate"].to_numpy() * 100.0
    errors = _win_rate_errors(summary, rates)
    colors = sns.color_palette(PALETTE, len(summary))
    axis.bar(positions, rates, color=colors, alpha=0.9)
    axis.errorbar(positions, rates, yerr=errors, fmt="none", color="black", capsize=5)
    axis.set(
        title="Percentuale di vittorie con intervallo di confidenza",
        ylabel="Vittorie (%)",
        xlabel="Agente",
        xticks=positions,
        xticklabels=summary["label"],
        ylim=(0, 100),
    )
    return _save(figure, output_dir / "win_rate.png")


def _plot_bosses(games: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    order = games.groupby("label", sort=False)["bosses_defeated"].mean()
    order = order.sort_values(ascending=False).index
    sns.violinplot(
        data=games,
        x="label",
        y="bosses_defeated",
        order=order,
        hue="label",
        palette=PALETTE,
        inner="box",
        cut=0,
        legend=False,
        ax=axis,
    )
    axis.set(
        title="Distribuzione dei boss sconfitti",
        xlabel="Agente",
        ylabel="Boss sconfitti",
        ylim=(-0.5, 12.5),
    )
    return _save(figure, output_dir / "bosses_defeated.png")


def _plot_execution_time(games: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    order = games.groupby("label", sort=False)["duration_seconds"].mean()
    order = order.sort_values().index
    sns.boxplot(
        data=games,
        x="label",
        y="duration_seconds",
        order=order,
        hue="label",
        palette=PALETTE,
        legend=False,
        showfliers=False,
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set(
        title="Tempo di esecuzione per partita",
        xlabel="Agente",
        ylabel="Secondi (scala logaritmica)",
    )
    return _save(figure, output_dir / "execution_time.png")


def _plot_tradeoff(summary: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 6))
    sizes = 100.0 + summary["win_rate"].to_numpy() * 500.0
    sns.scatterplot(
        data=summary,
        x="duration_seconds_mean",
        y="bosses_mean",
        hue="family",
        size=sizes,
        sizes=(100, 600),
        ax=axis,
    )
    for row in summary.itertuples():
        axis.annotate(
            row.label,
            (row.duration_seconds_mean, row.bosses_mean),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    _use_observed_log_ticks(axis, summary["duration_seconds_mean"])
    axis.set(
        title="Trade-off qualità–costo computazionale",
        xlabel="Tempo medio per partita (secondi, scala log)",
        ylabel="Boss sconfitti in media",
        ylim=(-0.5, 12.5),
    )
    return _save(figure, output_dir / "quality_cost_tradeoff.png")


def _plot_dashboard(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    label_order = summary["label"].tolist()
    _dashboard_win_rate(axes[0, 0], summary)
    sns.boxplot(
        data=games,
        x="label",
        y="bosses_defeated",
        order=label_order,
        ax=axes[0, 1],
    )
    axes[0, 1].set(title="Boss sconfitti", xlabel="", ylabel="Boss")
    sns.boxplot(
        data=games,
        x="label",
        y="duration_seconds",
        order=label_order,
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Tempo per partita", xlabel="", ylabel="Secondi (log)")
    sns.scatterplot(
        data=summary,
        x="duration_seconds_mean",
        y="win_rate",
        hue="family",
        s=180,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xscale("log")
    _use_observed_log_ticks(axes[1, 1], summary["duration_seconds_mean"])
    axes[1, 1].set(
        title="Vittorie rispetto al costo",
        xlabel="Secondi medi (log)",
        ylabel="Percentuale di vittorie",
    )
    axes[1, 1].yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0)
    )
    for row in summary.itertuples():
        axes[1, 1].annotate(
            row.label,
            (row.duration_seconds_mean, row.win_rate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
        )
    figure.suptitle("Regicide — confronto sperimentale degli agenti", fontsize=20)
    figure.tight_layout()
    return _save(figure, output_dir / "comprehensive_dashboard.png")


def _dashboard_win_rate(axis: plt.Axes, summary: pd.DataFrame) -> None:
    positions = np.arange(len(summary))
    rates = summary["win_rate"].to_numpy() * 100.0
    errors = _win_rate_errors(summary, rates)
    axis.bar(positions, rates)
    axis.errorbar(positions, rates, yerr=errors, fmt="none", color="black", capsize=5)
    axis.set(
        title="Percentuale di vittorie",
        ylabel="Vittorie (%)",
        xlabel="",
        xticks=positions,
        xticklabels=summary["label"],
        ylim=(0, 100),
    )


def _use_observed_log_ticks(axis: plt.Axes, values: pd.Series) -> None:
    ticks = np.sort(values.astype(float).unique())
    axis.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))


def _win_rate_errors(summary: pd.DataFrame, rates: np.ndarray) -> np.ndarray:
    lower = rates - summary["win_ci_low"].to_numpy() * 100.0
    upper = summary["win_ci_high"].to_numpy() * 100.0 - rates
    return np.maximum(0.0, np.vstack((lower, upper)))


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path
