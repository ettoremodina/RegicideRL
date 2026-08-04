"""Verify that an ISMCTS trace is a faithful, replayable record of the search."""

from __future__ import annotations

import json
import random

from agents.ismcts_agent import ISMCTSAgent
from agents.ismcts_trace import ISMCTSTracer
from scripts.trace_ismcts import TRACE_PLACEHOLDER, render_page
from solvers.env import RegicideEnv


def _traced_decision(seed: int = 7, iterations: int = 60):
    """Run one traced decision on a fresh game and return the trace and root."""
    random.seed(seed)
    env = RegicideEnv(num_players=1)
    obs, _ = env.reset(seed=seed)
    tracer = ISMCTSTracer(max_iterations=iterations, max_depth=6)
    agent = ISMCTSAgent(n_iterations=iterations, exploration_constant=1.414, tracer=tracer)
    action = agent.select_action(obs, env=env)
    return tracer.to_dict(), agent.root, action


def _replay(trace: dict) -> list[dict]:
    """Rebuild per-node counters from the recorded increments, as the viewer does."""
    stats = [{"v": 0, "a": 0, "r": 0.0} for _ in trace["nodes"]]
    for iteration in trace["iterations"]:
        for step in iteration["steps"]:
            for node_id in step["available"]:
                stats[node_id]["a"] += 1
            if step["expanded"]:
                stats[step["child"]]["a"] = 1
        for node_id in iteration["path"]:
            stats[node_id]["v"] += 1
            stats[node_id]["r"] += iteration["reward"]
    return stats


def test_replayed_counters_match_the_search_tree():
    """The trace stores increments only; replaying them must restore the tree."""
    trace, root, _ = _traced_decision()
    stats = _replay(trace)
    assert trace["decision"]["root_children"], "the root should have been expanded"
    for child in trace["decision"]["root_children"]:
        replayed = stats[child["id"]]
        assert replayed["v"] == child["visits"]
        assert replayed["a"] == child["availability"]


def test_tracing_does_not_change_the_decision():
    """Attaching a tracer must leave the search itself untouched."""
    traced, _, traced_action = _traced_decision(seed=11)

    random.seed(11)
    env = RegicideEnv(num_players=1)
    obs, _ = env.reset(seed=11)
    plain = ISMCTSAgent(n_iterations=60, exploration_constant=1.414)
    assert plain.select_action(obs, env=env) == traced_action
    assert traced["meta"]["recorded_iterations"] == 60


def test_trace_structure_is_serializable_and_consistent():
    """Every recorded reference must point at a node that exists in the trace."""
    trace, _, best = _traced_decision()
    node_ids = {node["id"] for node in trace["nodes"]}
    assert trace["nodes"][0]["parent"] is None
    assert trace["decision"]["best_action"] == best
    for iteration in trace["iterations"]:
        assert iteration["path"][0] == 0
        assert set(iteration["path"]) <= node_ids
        for step in iteration["steps"]:
            assert step["child"] in node_ids
            assert set(step["available"]) <= node_ids
            assert step["legal"] >= step["untried"]
    json.dumps(trace)  # must not raise


def test_render_page_inlines_the_trace(tmp_path):
    """The viewer template is filled with data and escaped for inline use."""
    template = tmp_path / "template.html"
    template.write_text(f"<script>const T = {TRACE_PLACEHOLDER};</script>", encoding="utf-8")
    output = tmp_path / "out" / "page.html"
    render_page({"meta": {"note": "</script>"}, "nodes": []}, template, output)

    page = output.read_text(encoding="utf-8")
    assert TRACE_PLACEHOLDER not in page
    assert "</script>" not in page.replace("</script>;", "").replace("</script>", "", 1)
    assert '"note":"<\\/script>"' in page
