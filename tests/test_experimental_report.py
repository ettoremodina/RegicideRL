"""Tests for the reproducible experimental-report pipeline."""

from __future__ import annotations

import pandas as pd

from scripts.experimental_report.analysis import analyze_experiment
from scripts.experimental_report.statistics import compare_pairs, summarize_results


def test_statistical_summary_and_paired_comparison():
    games = _synthetic_games()

    summary = summarize_results(games, 0.95, 200, 42)
    pairwise = compare_pairs(games, 0.95, 200, 43)

    assert set(summary["agent"]) == {"baseline", "search"}
    search = summary.loc[summary["agent"] == "search"].iloc[0]
    assert search["win_rate"] == 5 / 6
    assert search["bosses_mean"] > 8
    assert len(pairwise) == 1
    assert pairwise.iloc[0]["n_pairs"] == 6
    assert 0.0 <= pairwise.iloc[0]["win_p_holm"] <= 1.0
    assert pairwise.iloc[0]["bosses_mean_difference"] < 0


def test_zero_win_rate_interval_never_exceeds_observed_rate():
    games = _synthetic_games()
    games["victory"] = False

    summary = summarize_results(games, 0.95, 200, 42)

    assert (summary["win_rate"] == 0.0).all()
    assert (summary["win_ci_low"] == 0.0).all()
    assert (summary["win_ci_high"] >= summary["win_rate"]).all()


def test_analysis_generates_report_tables_and_plots(tmp_path):
    run_dir = tmp_path / "run"
    datasets_dir = run_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    _synthetic_games().to_csv(datasets_dir / "games.csv", index=False)

    outputs = analyze_experiment(
        run_dir,
        report_config=_report_config(),
    )

    assert all(path.exists() for path in outputs.values())
    report = outputs["report"].read_text(encoding="utf-8")
    assert "Analisi statistiche" in report
    assert "comprehensive_dashboard.png" in report
    assert (run_dir / "analysis" / "tables.tex").exists()
    assert (run_dir / "analysis" / "statistics.json").exists()


def _synthetic_games() -> pd.DataFrame:
    rows = []
    baseline_wins = [False, False, True, False, True, False]
    search_wins = [True, True, True, False, True, True]
    for agent, label, wins, offset in (
        ("baseline", "Baseline", baseline_wins, 0),
        ("search", "Search", search_wins, 3),
    ):
        for index, victory in enumerate(wins):
            decisions = 20 + index
            rows.append(
                {
                    "agent": agent,
                    "label": label,
                    "family": "Test",
                    "seed": 100 + index,
                    "status": "completed",
                    "victory": victory,
                    "bosses_defeated": min(12, 5 + index + offset),
                    "turns": decisions,
                    "duration_seconds": 0.1 + index * 0.01 + offset * 0.02,
                    "mean_decision_ms": 2.0 + offset,
                    "p95_decision_ms": 3.0 + offset,
                    "decisions": decisions,
                    "defense_decisions": 5,
                    "yield_actions": 1,
                    "invalid_actions": 0,
                    "reward": float(victory),
                }
            )
    return pd.DataFrame(rows)


def _report_config():
    return {
        "protocol": {
            "games_per_agent": 6,
            "base_seed": 100,
            "max_decisions_per_game": 100,
            "confidence_level": 0.95,
            "bootstrap_samples": 200,
        },
        "agents": {
            "baseline": {
                "label": "Baseline",
                "family": "Test",
                "description": "Baseline sintetica.",
                "kwargs": {},
            },
            "search": {
                "label": "Search",
                "family": "Test",
                "description": "Ricerca sintetica.",
                "kwargs": {"budget": 10},
            },
        },
    }
