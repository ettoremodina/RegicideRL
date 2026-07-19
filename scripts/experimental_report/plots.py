"""Publication-ready plots for the experimental comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BACKGROUND = "#F7F6F2"
INK = "#17212B"
MUTED = "#66717D"
GRID = "#D9DDE1"
OUTLINE = "#BFC6CD"
AGENT_COLORS = {
    "random": "#6B7280",
    "heuristic": "#D97706",
    "pimc": "#0F766E",
    "ismcts": "#2563EB",
    "ppo": "#7C3AED",
    "alphazero": "#E11D48",
}
FALLBACK_COLORS = ("#0891B2", "#BE123C", "#65A30D", "#9333EA", "#C2410C")
FAMILY_MARKERS = {
    "Baseline": "o",
    "Rule-based": "s",
    "Search": "D",
    "Reinforcement learning": "^",
    "Neural tree search": "P",
}


def create_plots(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate individual figures and a comprehensive dashboard."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _configure_theme()
    return [
        _plot_win_rate(summary, destination),
        _plot_bosses(games, summary, destination),
        _plot_execution_time(games, summary, destination),
        _plot_tradeoff(summary, destination),
        _plot_dashboard(games, summary, destination),
    ]


def _plot_win_rate(summary: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = _new_figure((9.6, 5.8))
    _draw_win_rate(axis, summary)
    return _save(figure, output_dir / "win_rate.png")


def _plot_bosses(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axis = _new_figure((9.6, 5.8))
    _draw_bosses(axis, games, summary)
    return _save(figure, output_dir / "bosses_defeated.png")


def _plot_execution_time(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axis = _new_figure((9.6, 5.8))
    _draw_execution_time(axis, games, summary)
    return _save(figure, output_dir / "execution_time.png")


def _plot_tradeoff(summary: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = _new_figure((9.6, 6.2))
    _draw_tradeoff(axis, summary)
    return _save(figure, output_dir / "quality_cost_tradeoff.png")


def _plot_dashboard(
    games: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Combine the four core comparisons into one publication overview."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10.5),
        facecolor=BACKGROUND,
    )
    _draw_win_rate(axes[0, 0], summary, compact=True)
    _draw_bosses(axes[0, 1], games, summary, compact=True)
    _draw_execution_time(axes[1, 0], games, summary, compact=True)
    _draw_tradeoff(axes[1, 1], summary, compact=True)
    game_counts = summary["n_games"].astype(int)
    count_text = (
        f"{game_counts.iloc[0]} games per agent"
        if game_counts.nunique() == 1
        else f"{game_counts.min()}–{game_counts.max()} games per agent"
    )
    figure.suptitle(
        "Regicide · experimental agent comparison",
        x=0.04,
        y=0.975,
        ha="left",
        color=INK,
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.925,
        f"{count_text} · error bars show 95% confidence intervals",
        color=MUTED,
        fontsize=11,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.08,
        top=0.84,
        wspace=0.3,
        hspace=0.52,
    )
    return _save(figure, output_dir / "comprehensive_dashboard.png")


def _draw_win_rate(
    axis: plt.Axes,
    summary: pd.DataFrame,
    compact: bool = False,
) -> None:
    """Draw ordered win estimates with Wilson confidence intervals."""
    ordered = summary.sort_values(
        ["win_rate", "bosses_mean"],
        ascending=False,
    ).reset_index(drop=True)
    colors = _color_map(ordered["agent"])
    positions = np.arange(len(ordered))
    rates = ordered["win_rate"].to_numpy(dtype=float) * 100.0
    lows = ordered["win_ci_low"].to_numpy(dtype=float) * 100.0
    highs = ordered["win_ci_high"].to_numpy(dtype=float) * 100.0
    x_errors = np.vstack((rates - lows, highs - rates))

    _draw_win_rate_marks(axis, ordered, rates, x_errors, colors, compact)
    upper_limit = _percentage_axis_limit(highs)
    _configure_win_rate_axis(axis, ordered, positions, upper_limit, compact)
    for position, rate in enumerate(rates):
        _annotate_interval_value(
            axis,
            position,
            low=lows[position],
            high=highs[position],
            value=f"{rate:.0f}%",
            axis_limit=upper_limit,
        )


def _draw_win_rate_marks(
    axis: plt.Axes,
    ordered: pd.DataFrame,
    rates: np.ndarray,
    x_errors: np.ndarray,
    colors: dict[str, str],
    compact: bool,
) -> None:
    """Draw one colored confidence interval and point per agent."""
    for position, row in enumerate(ordered.itertuples()):
        color = colors[str(row.agent)]
        axis.errorbar(
            rates[position],
            position,
            xerr=x_errors[:, position : position + 1],
            fmt="none",
            color=color,
            elinewidth=2.0,
            capsize=4,
            capthick=1.5,
            zorder=2,
        )
        axis.scatter(
            rates[position],
            position,
            s=95 if compact else 120,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=1.5,
            clip_on=False,
            zorder=3,
        )


def _configure_win_rate_axis(
    axis: plt.Axes,
    ordered: pd.DataFrame,
    positions: np.ndarray,
    upper_limit: float,
    compact: bool,
) -> None:
    """Apply percentage scaling, labels, ordering, and shared styling."""
    axis.set_xlim(0.0, upper_limit)
    axis.set_yticks(positions, ordered["label"])
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100))
    axis.xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    )
    axis.set_xlabel("Games won")
    axis.set_ylabel("")
    _style_axis(
        axis,
        "Win rate",
        "Observed estimate and 95% Wilson interval",
        grid_axis="x",
        compact=compact,
    )


def _draw_bosses(
    axis: plt.Axes,
    games: pd.DataFrame,
    summary: pd.DataFrame,
    compact: bool = False,
) -> None:
    """Draw per-game progress distributions with annotated agent means."""
    ordered = summary.sort_values("bosses_mean", ascending=False).reset_index(drop=True)
    order = ordered["label"].tolist()
    colors = _label_color_map(ordered)
    _draw_categorical_distribution(
        axis,
        games,
        "bosses_defeated",
        order,
        colors,
        compact,
    )
    _annotate_boss_means(axis, ordered, colors, compact)
    _configure_boss_axis(axis, compact)


def _annotate_boss_means(
    axis: plt.Axes,
    ordered: pd.DataFrame,
    colors: dict[str, str],
    compact: bool,
) -> None:
    """Overlay and label mean bosses on categorical distributions."""
    for position, row in enumerate(ordered.itertuples()):
        color = colors[str(row.label)]
        axis.scatter(
            row.bosses_mean,
            position,
            marker="D",
            s=52 if compact else 64,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=1.4,
            zorder=4,
        )
        axis.annotate(
            f"{row.bosses_mean:.1f}",
            (row.bosses_mean, position),
            xytext=(7, -12),
            textcoords="offset points",
            color=INK,
            fontsize=9 if compact else 10,
            fontweight="bold",
            zorder=5,
        )


def _configure_boss_axis(axis: plt.Axes, compact: bool) -> None:
    """Configure the bounded 0–12 progress axis and victory marker."""
    axis.axvline(12, color=OUTLINE, linewidth=1.2, linestyle=(0, (3, 3)), zorder=0)
    axis.set_xlim(-0.25, 12.55)
    axis.set_xticks(np.arange(0, 13, 2))
    axis.set_xlabel("Bosses defeated")
    axis.set_ylabel("")
    _style_axis(
        axis,
        "Game progress",
        "Dots = games · box = median and IQR · diamond = mean",
        grid_axis="x",
        compact=compact,
    )
    axis.text(
        12,
        1.01,
        "victory",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8 if compact else 9,
    )


def _draw_execution_time(
    axis: plt.Axes,
    games: pd.DataFrame,
    summary: pd.DataFrame,
    compact: bool = False,
) -> None:
    """Draw positive per-game duration distributions on a logarithmic scale."""
    ordered = summary.sort_values("duration_seconds_median").reset_index(drop=True)
    order = ordered["label"].tolist()
    colors = _label_color_map(ordered)
    positive_games = games.loc[games["duration_seconds"] > 0]

    _draw_categorical_distribution(
        axis,
        positive_games,
        "duration_seconds",
        order,
        colors,
        compact,
    )
    _annotate_duration_medians(axis, ordered, colors, compact)
    _configure_duration_axis(axis, compact)


def _annotate_duration_medians(
    axis: plt.Axes,
    ordered: pd.DataFrame,
    colors: dict[str, str],
    compact: bool,
) -> None:
    """Overlay and format median duration for each agent."""
    for position, row in enumerate(ordered.itertuples()):
        color = colors[str(row.label)]
        axis.scatter(
            row.duration_seconds_median,
            position,
            marker="D",
            s=46 if compact else 58,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=1.3,
            zorder=4,
        )
        axis.annotate(
            _format_duration(row.duration_seconds_median),
            (row.duration_seconds_median, position),
            xytext=(7, -12),
            textcoords="offset points",
            color=INK,
            fontsize=9 if compact else 10,
            fontweight="bold",
            zorder=5,
        )


def _configure_duration_axis(axis: plt.Axes, compact: bool) -> None:
    axis.set_xscale("log")
    _format_duration_axis(axis)
    axis.set_xlabel("Time per game · log scale")
    axis.set_ylabel("")
    _style_axis(
        axis,
        "Computational cost",
        "Time distribution · diamond = median",
        grid_axis="x",
        compact=compact,
    )


def _draw_categorical_distribution(
    axis: plt.Axes,
    games: pd.DataFrame,
    metric: str,
    order: list[str],
    colors: dict[str, str],
    compact: bool,
) -> None:
    """Layer jittered game observations beneath robust box summaries."""
    sns.stripplot(
        data=games,
        x=metric,
        y="label",
        order=order,
        hue="label",
        palette=colors,
        jitter=0.23,
        size=2.1 if compact else 2.7,
        alpha=0.22,
        linewidth=0,
        legend=False,
        ax=axis,
        zorder=1,
    )
    sns.boxplot(
        data=games,
        x=metric,
        y="label",
        order=order,
        hue="label",
        palette=colors,
        width=0.38 if compact else 0.42,
        saturation=0.85,
        showfliers=False,
        linewidth=1.25,
        legend=False,
        boxprops={"alpha": 0.64},
        medianprops={"color": INK, "linewidth": 2.0},
        whiskerprops={"color": INK, "linewidth": 1.2},
        capprops={"color": INK, "linewidth": 1.2},
        ax=axis,
        zorder=2,
    )


def _draw_tradeoff(
    axis: plt.Axes,
    summary: pd.DataFrame,
    compact: bool = False,
) -> None:
    """Plot mean progress against computational cost and Pareto efficiency."""
    ordered = summary.sort_values("duration_seconds_mean").reset_index(drop=True)
    colors = _color_map(ordered["agent"])
    midpoint = float(
        np.sqrt(
            ordered["duration_seconds_mean"].min()
            * ordered["duration_seconds_mean"].max()
        )
    )

    for row in ordered.itertuples():
        color = colors[str(row.agent)]
        _draw_tradeoff_point(axis, row, color, midpoint, compact)

    efficient = _pareto_frontier(ordered)
    if len(efficient) > 1:
        axis.plot(
            efficient["duration_seconds_mean"],
            efficient["bosses_mean"],
            color=OUTLINE,
            linewidth=1.4,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
    _configure_tradeoff_axis(axis, compact)


def _draw_tradeoff_point(
    axis: plt.Axes,
    row,
    color: str,
    midpoint: float,
    compact: bool,
) -> None:
    """Draw one agent with two-dimensional uncertainty and a readable label."""
    x_error = np.array(
        [
            [max(0.0, row.duration_seconds_mean - row.duration_seconds_ci_low)],
            [max(0.0, row.duration_seconds_ci_high - row.duration_seconds_mean)],
        ]
    )
    y_error = np.array(
        [
            [max(0.0, row.bosses_mean - row.bosses_ci_low)],
            [max(0.0, row.bosses_ci_high - row.bosses_mean)],
        ]
    )
    axis.errorbar(
        row.duration_seconds_mean,
        row.bosses_mean,
        xerr=x_error,
        yerr=y_error,
        fmt="none",
        color=color,
        alpha=0.55,
        elinewidth=1.5,
        capsize=3,
        zorder=2,
    )
    axis.scatter(
        row.duration_seconds_mean,
        row.bosses_mean,
        marker=FAMILY_MARKERS.get(str(row.family), "o"),
        s=115 if compact else 145,
        color=color,
        edgecolor=BACKGROUND,
        linewidth=1.6,
        zorder=3,
    )
    align_right = row.duration_seconds_mean > midpoint
    axis.annotate(
        f"{row.label} · {row.win_rate:.0%}",
        (row.duration_seconds_mean, row.bosses_mean),
        xytext=(-8 if align_right else 8, 7),
        textcoords="offset points",
        ha="right" if align_right else "left",
        va="bottom",
        color=INK,
        fontsize=9 if compact else 10,
        fontweight="bold",
        zorder=4,
    )


def _configure_tradeoff_axis(axis: plt.Axes, compact: bool) -> None:
    """Configure logarithmic cost and bounded progress axes."""
    axis.set_xscale("log")
    axis.set_ylim(-0.35, 12.55)
    axis.set_yticks(np.arange(0, 13, 2))
    _format_duration_axis(axis)
    axis.set_xlabel("Mean time per game · log scale")
    axis.set_ylabel("Mean bosses defeated")
    _style_axis(
        axis,
        "Quality versus cost",
        "Higher and farther left is better · labels show win rate",
        grid_axis="both",
        compact=compact,
    )


def _configure_theme() -> None:
    """Set the shared color, font, grid, and sizing defaults."""
    sns.set_theme(
        style="ticks",
        context="notebook",
        font="DejaVu Sans",
        rc={
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": OUTLINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )


def _new_figure(size: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=size, facecolor=BACKGROUND, layout="constrained")


def _style_axis(
    axis: plt.Axes,
    title: str,
    subtitle: str,
    grid_axis: str,
    compact: bool,
) -> None:
    """Apply consistent titles, subtitle, grid, spines, and ticks."""
    axis.set_facecolor(BACKGROUND)
    axis.set_title(
        title,
        loc="left",
        pad=30 if compact else 34,
        fontsize=14 if compact else 17,
        fontweight="bold",
    )
    axis.text(
        0,
        1.015,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.5 if compact else 10,
    )
    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(OUTLINE)
    axis.spines["bottom"].set_color(OUTLINE)
    axis.tick_params(axis="both", length=0, pad=7)


def _color_map(agent_names: Iterable[object]) -> dict[str, str]:
    names = [str(name) for name in agent_names]
    fallback = iter(FALLBACK_COLORS)
    colors: dict[str, str] = {}
    for name in names:
        colors[name] = AGENT_COLORS.get(name.lower()) or next(
            fallback,
            "#475569",
        )
    return colors


def _label_color_map(summary: pd.DataFrame) -> dict[str, str]:
    agent_colors = _color_map(summary["agent"])
    return {
        str(row.label): agent_colors[str(row.agent)]
        for row in summary.itertuples()
    }


def _percentage_axis_limit(highs: np.ndarray) -> float:
    """Choose a readable percentage limit while leaving annotation space."""
    observed = float(np.nanmax(highs)) if highs.size else 0.0
    padded = max(10.0, observed * 1.28)
    for limit in (10.0, 20.0, 25.0, 50.0, 75.0, 100.0):
        if padded <= limit:
            return limit
    return 100.0


def _annotate_interval_value(
    axis: plt.Axes,
    position: int,
    low: float,
    high: float,
    value: str,
    axis_limit: float,
) -> None:
    """Place a value label on the side of an interval with available space."""
    if high < axis_limit * 0.82:
        x_value = high
        offset = (7, 0)
        alignment = "left"
    else:
        x_value = low
        offset = (-7, 0)
        alignment = "right"
    axis.annotate(
        value,
        (x_value, position),
        xytext=offset,
        textcoords="offset points",
        ha=alignment,
        va="center",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )


def _format_duration_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
    axis.xaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10, subs=(2, 5), numticks=16)
    )
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda value, _: _format_duration(value) if value > 0 else ""
        )
    )
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _format_duration(seconds: float) -> str:
    """Format seconds using microseconds, milliseconds, seconds, or minutes."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.1f} min"


def _pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """Return agents that improve quality as mean cost increases."""
    efficient_rows = []
    best_quality = -np.inf
    for _, row in summary.sort_values("duration_seconds_mean").iterrows():
        if float(row["bosses_mean"]) > best_quality:
            efficient_rows.append(row)
            best_quality = float(row["bosses_mean"])
    return pd.DataFrame(efficient_rows)


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return path
