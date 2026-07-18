"""Tests for the reproducible experimental-report pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agents.ismcts_agent import ISMCTSAgent, ISMCTSNode
from agents.pimc_agent import PIMCAgent
from scripts.experimental_report.analysis import analyze_experiment
from scripts.experimental_report.runner import (
    EvaluationStore,
    GAME_COLUMNS,
    _evaluate_agent_in_parallel,
    _load_checkpoint,
    _normalize_jobs,
    _save_checkpoint,
)
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


def test_ismcts_final_choice_excludes_stale_illegal_children():
    root = ISMCTSNode()
    legal_child = ISMCTSNode(action=7, parent=root)
    legal_child.visit_count = 3
    stale_child = ISMCTSNode(action=99, parent=root)
    stale_child.visit_count = 1000
    root.children = {7: legal_child, 99: stale_child}

    selected = ISMCTSAgent._best_root_action(root, valid_actions=[7])

    assert selected == 7


def test_ismcts_forced_action_discards_unadvanced_tree():
    agent = ISMCTSAgent(n_iterations=1)
    agent.root = ISMCTSNode()
    action_mask = np.zeros(543, dtype=np.int8)
    action_mask[12] = 1

    selected = agent.select_action({"action_mask": action_mask}, env=object())

    assert selected == 12
    assert agent.root is None


def test_pimc_distributes_total_rollout_budget_across_actions():
    agent = PIMCAgent(rollout_budget=3000)

    allocations = agent._allocate_rollouts(action_count=7)

    assert sum(allocations) == 3000
    assert max(allocations) - min(allocations) <= 1


def test_pimc_rejects_budget_smaller_than_legal_action_count():
    agent = PIMCAgent(rollout_budget=2)

    with pytest.raises(ValueError, match="at least"):
        agent._allocate_rollouts(action_count=3)


def test_checkpoint_recovers_game_rows_from_metrics(tmp_path):
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    rows = _synthetic_games().to_dict(orient="records")
    with (metrics_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for step, row in enumerate(rows, start=1):
            stream.write(json.dumps({"step": step, **row}) + "\n")

    recovered = _load_checkpoint(run_dir)
    output_path = run_dir / "datasets" / "games.csv"
    _save_checkpoint(recovered, output_path)

    assert len(recovered) == len(rows)
    assert list(pd.read_csv(output_path).columns) == list(GAME_COLUMNS)


def test_parallel_workers_return_results_to_parent_checkpoint(tmp_path):
    from ml_logger import RunContext

    context = RunContext.create(
        "test-experimental-parallel",
        root_dir=tmp_path / "artifacts",
    )
    output_path = context.run_dir / "datasets" / "games.csv"
    store = EvaluationStore(context, [], output_path)
    specification = {
        "class_path": "agents.random_agent.RandomAgent",
        "label": "Random",
        "family": "Baseline",
        "description": "Test",
        "kwargs": {"name": "Random"},
    }
    protocol = {"max_decisions_per_game": 100}

    _evaluate_agent_in_parallel(
        "random",
        specification,
        [10, 11],
        protocol,
        store,
        jobs=2,
    )

    assert len(store.rows) == 2
    assert output_path.exists()
    assert {int(row["seed"]) for row in store.rows} == {10, 11}


def test_parallel_job_count_must_be_positive():
    assert _normalize_jobs(4) == 4
    with pytest.raises(ValueError, match="at least 1"):
        _normalize_jobs(0)


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
    assert "Statistical analysis" in report
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
                "description": "Synthetic baseline.",
                "kwargs": {},
            },
            "search": {
                "label": "Search",
                "family": "Test",
                "description": "Synthetic search agent.",
                "kwargs": {"budget": 10},
            },
        },
    }
