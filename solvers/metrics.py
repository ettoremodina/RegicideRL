"""Plot persisted evaluation metrics from a canonical run directory."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from ml_logger import get_logger

logger = get_logger(__name__)


def plot_metrics(run_dir):
    """Read metrics JSONL and save a summary plot under ``analysis``."""
    run_path = Path(run_dir)
    metrics_file = run_path / "metrics" / "metrics.jsonl"
    if not metrics_file.exists():
        logger.warning("No metrics found at %s", metrics_file)
        return None
    data = _read_jsonl(metrics_file)
    if not data:
        logger.warning("Metrics file is empty: %s", metrics_file)
        return None
    figure = _build_figure(data)
    plot_file = run_path / "analysis" / "metrics.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved metrics plot to %s", plot_file)
    return plot_file


def _read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_figure(data):
    steps = [row.get("step", index) for index, row in enumerate(data)]
    win_rates = [row.get("win_rate", 0) * 100 for row in data]
    average_enemies = [row.get("avg_enemies_defeated", 0) for row in data]
    average_turns = [row.get("avg_turns", 0) for row in data]
    speeds = [row.get("games_per_second", 0) for row in data]
    turns_distribution = data[-1].get("turns_distribution", [])
    enemies_distribution = data[-1].get("enemies_distribution", [])

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    figure.suptitle("Regicide Training Progress", fontsize=16)
    _plot_line(axes[0, 0], steps, win_rates, "Win Rate", "Win Rate (%)", "g")
    _plot_line(
        axes[0, 1], steps, average_turns, "Average Episode Length", "Turns", "orange"
    )
    _plot_line(
        axes[0, 2], steps, average_enemies, "Average Bosses Killed", "Bosses", "b"
    )
    axes[0, 2].set_ylim(0, 12.5)
    _plot_line(
        axes[1, 0], steps, speeds, "Simulation Speed", "Games / Second", "purple"
    )
    _plot_histogram(axes[1, 1], turns_distribution, "Game Length", "Turns")
    _plot_histogram(axes[1, 2], enemies_distribution, "Boss Kills", "Bosses")
    figure.tight_layout()
    return figure


def _plot_line(axis, x_values, y_values, title, ylabel, color):
    axis.plot(x_values, y_values, color=color, marker="o")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _plot_histogram(axis, values, title, xlabel):
    if values:
        axis.hist(values, bins=20, alpha=0.7, edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Frequency")
    axis.grid(True, alpha=0.3)
