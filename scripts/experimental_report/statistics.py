"""Statistical summaries and paired tests for agent-comparison results."""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SUMMARY_METRICS = {
    "bosses_defeated": "bosses",
    "turns": "turns",
    "duration_seconds": "duration_seconds",
    "mean_decision_ms": "decision_ms",
    "reward": "reward",
}


def summarize_results(
    games: pd.DataFrame,
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build one descriptive-statistics row per agent."""
    _validate_games(games)
    rng = np.random.default_rng(random_seed)
    rows = [
        _summarize_agent(group, confidence_level, bootstrap_samples, rng)
        for _, group in games.groupby("agent", sort=False)
    ]
    return pd.DataFrame(rows).sort_values(
        ["win_rate", "bosses_mean"],
        ascending=False,
    )


def compare_pairs(
    games: pd.DataFrame,
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run paired tests between every pair of agents sharing the same seeds."""
    _validate_paired_seeds(games)
    rng = np.random.default_rng(random_seed)
    agents = games["agent"].drop_duplicates().tolist()
    rows = [
        _compare_agent_pair(games, first, second, confidence_level, bootstrap_samples, rng)
        for first, second in itertools.combinations(agents, 2)
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["win_p_holm"] = _holm_adjust(result["win_p_value"].to_numpy())
    result["bosses_p_holm"] = _holm_adjust(result["bosses_p_value"].to_numpy())
    return result


def _summarize_agent(
    group: pd.DataFrame,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    victories = group["victory"].astype(int).to_numpy()
    win_low, win_high = _wilson_interval(victories.sum(), len(victories), confidence_level)
    row: dict[str, float | int | str] = {
        "agent": group["agent"].iloc[0],
        "label": group["label"].iloc[0],
        "family": group["family"].iloc[0],
        "n_games": len(group),
        "completed_games": int((group["status"] == "completed").sum()),
        "wins": int(victories.sum()),
        "win_rate": float(victories.mean()),
        "win_ci_low": win_low,
        "win_ci_high": win_high,
        "total_time_seconds": float(group["duration_seconds"].sum()),
        "games_per_second": _safe_ratio(len(group), group["duration_seconds"].sum()),
        "yield_rate": _safe_ratio(group["yield_actions"].sum(), group["decisions"].sum()),
        "invalid_action_rate": _safe_ratio(
            group["invalid_actions"].sum(),
            group["decisions"].sum(),
        ),
    }
    for column, prefix in SUMMARY_METRICS.items():
        values = group[column].astype(float).to_numpy()
        low, high = _bootstrap_interval(
            values,
            np.mean,
            confidence_level,
            bootstrap_samples,
            rng,
        )
        row.update(
            {
                f"{prefix}_mean": float(np.mean(values)),
                f"{prefix}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                f"{prefix}_median": float(np.median(values)),
                f"{prefix}_ci_low": low,
                f"{prefix}_ci_high": high,
            }
        )
    return row


def _compare_agent_pair(
    games: pd.DataFrame,
    first: str,
    second: str,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    paired = _paired_rows(games, first, second)
    win_difference = paired["victory_a"].astype(float) - paired["victory_b"].astype(float)
    boss_difference = paired["bosses_defeated_a"] - paired["bosses_defeated_b"]
    duration_ratio = paired["duration_seconds_a"] / paired["duration_seconds_b"].clip(
        lower=np.finfo(float).eps
    )
    win_low, win_high = _bootstrap_interval(
        win_difference.to_numpy(),
        np.mean,
        confidence_level,
        bootstrap_samples,
        rng,
    )
    boss_low, boss_high = _bootstrap_interval(
        boss_difference.to_numpy(),
        np.mean,
        confidence_level,
        bootstrap_samples,
        rng,
    )
    boss_statistic, boss_p_value = _wilcoxon_test(boss_difference.to_numpy())
    return {
        "agent_a": first,
        "agent_b": second,
        "n_pairs": len(paired),
        "win_rate_difference": float(win_difference.mean()),
        "win_diff_ci_low": win_low,
        "win_diff_ci_high": win_high,
        "win_p_value": _mcnemar_exact(paired["victory_a"], paired["victory_b"]),
        "bosses_mean_difference": float(boss_difference.mean()),
        "bosses_diff_ci_low": boss_low,
        "bosses_diff_ci_high": boss_high,
        "bosses_wilcoxon_statistic": boss_statistic,
        "bosses_p_value": boss_p_value,
        "bosses_cohen_dz": _cohen_dz(boss_difference.to_numpy()),
        "bosses_probability_superiority": _probability_superiority(
            boss_difference.to_numpy()
        ),
        "median_duration_ratio": float(np.median(duration_ratio)),
    }


def _paired_rows(games: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    columns = ["seed", "victory", "bosses_defeated", "duration_seconds"]
    left = games.loc[games["agent"] == first, columns]
    right = games.loc[games["agent"] == second, columns]
    return left.merge(right, on="seed", suffixes=("_a", "_b"), validate="one_to_one")


def _bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    confidence_level: float,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    indices = rng.integers(0, values.size, size=(samples, values.size))
    estimates = np.apply_along_axis(statistic, 1, values[indices])
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def _wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float,
) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (proportion + z_score**2 / (2.0 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_score**2 / (4.0 * trials**2)
        )
        / denominator
    )
    lower = min(proportion, max(0.0, center - margin))
    upper = max(proportion, min(1.0, center + margin))
    return lower, upper


def _mcnemar_exact(first: pd.Series, second: pd.Series) -> float:
    first_values = first.astype(bool).to_numpy()
    second_values = second.astype(bool).to_numpy()
    first_only = int(np.sum(first_values & ~second_values))
    second_only = int(np.sum(~first_values & second_values))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def _wilcoxon_test(differences: np.ndarray) -> tuple[float, float]:
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = wilcoxon(differences, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def _cohen_dz(differences: np.ndarray) -> float:
    if differences.size < 2:
        return 0.0
    standard_deviation = float(np.std(differences, ddof=1))
    if standard_deviation == 0.0:
        return 0.0
    return float(np.mean(differences) / standard_deviation)


def _probability_superiority(differences: np.ndarray) -> float:
    if differences.size == 0:
        return 0.5
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_games(games: pd.DataFrame) -> None:
    required = {
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
        "reward",
        "yield_actions",
        "invalid_actions",
        "decisions",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing experimental columns: {', '.join(sorted(missing))}")
    if games.empty:
        raise ValueError("The experimental dataset is empty")


def _validate_paired_seeds(games: pd.DataFrame) -> None:
    _validate_games(games)
    seed_sets = [
        set(group["seed"].tolist())
        for _, group in games.groupby("agent", sort=False)
    ]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Pairwise tests require the same seed set for every agent")
    if games.duplicated(["agent", "seed"]).any():
        raise ValueError("Each agent/seed pair must occur exactly once")
