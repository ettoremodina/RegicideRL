"""Record one real ISMCTS decision and render it as a self-contained page.

The script plays a seeded game with the heuristic agent up to a realistic
mid-game position, hands a single decision to :class:`ISMCTSAgent` with a
tracer attached, then writes:

  * a JSON trace, reusable by any other viewer or analysis;
  * an HTML page with the trace inlined, so it opens from the file system
    without a server.

Every parameter comes from the ``ismcts_trace`` section of the project config.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.heuristic_agent import HeuristicAgent
from agents.ismcts_agent import ISMCTSAgent
from agents.ismcts_trace import ISMCTSTracer
from ml_logger import get_logger
from solvers.env import RegicideEnv

logger = get_logger(__name__)

TRACE_PLACEHOLDER = "/*__ISMCTS_TRACE__*/null"


def load_trace_config(config_path: Path) -> Dict[str, Any]:
    """Load the ``ismcts_trace`` section of the project configuration."""
    with config_path.open("r", encoding="utf-8") as stream:
        project_config = yaml.safe_load(stream) or {}
    section = project_config.get("ismcts_trace")
    if not isinstance(section, dict):
        raise ValueError("Missing 'ismcts_trace' section in the configuration")
    return section


def find_decision(env: RegicideEnv, config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int]:
    """Advance the game with the heuristic agent until a good decision appears.

    A "good" decision is a real branching point: it comes after the configured
    warm-up, offers enough legal actions to be interesting and few enough to be
    drawable.

    Args:
        env: A freshly reset environment, advanced in place.
        config: The ``ismcts_trace`` configuration section.

    Returns:
        The observation to search from and the number of decisions played
        before it, or ``(None, played)`` when the game ended first.
    """
    warmup = HeuristicAgent(name="TraceWarmup")
    obs = env._get_obs()
    played = 0
    for _ in range(config["max_warmup_scan"]):
        if env.game.game_over:
            return None, played
        legal = int(np.count_nonzero(obs['action_mask']))
        if (played >= config["warmup_decisions"]
                and config["min_legal_actions"] <= legal <= config["max_legal_actions"]):
            return obs, played
        action = warmup.select_action(obs, env=env)
        if action is None:
            return None, played
        obs, _, terminated, truncated, _ = env.step(action)
        played += 1
        if terminated or truncated:
            return None, played
    return None, played


def record_decision(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Play to a mid-game position and trace one ISMCTS decision there."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    env = RegicideEnv(num_players=1)
    env.reset(seed=seed)

    obs, warmup_played = find_decision(env, config)
    if obs is None:
        raise RuntimeError(
            f"No decision matching the configured filters was found (seed {seed}, "
            f"{warmup_played} decisions played). Try another seed or widen "
            "min_legal_actions/max_legal_actions."
        )

    record_config = config["record"]
    tracer = ISMCTSTracer(
        max_iterations=record_config["max_iterations"],
        tavern_preview=record_config["tavern_preview"],
        max_depth=record_config["max_depth"],
    )
    search_config = config["search"]
    agent = ISMCTSAgent(
        n_iterations=search_config["n_iterations"],
        exploration_constant=search_config["exploration_constant"],
        name="ISMCTS",
        tracer=tracer,
    )
    # A fresh tree keeps every node in the trace: a retained root would carry
    # children created before recording started.
    agent.reset()

    logger.info(
        "Tracing an ISMCTS decision after %d warm-up decisions (seed %d, %d iterations)",
        warmup_played, seed, search_config["n_iterations"],
    )
    agent.select_action(obs, env=env)

    trace = tracer.to_dict()
    trace["meta"].update({
        "seed": seed,
        "warmup_decisions": warmup_played,
        "warmup_agent": "Heuristic",
        "rollout_agent": "Heuristic",
    })
    return trace


def render_page(trace: Dict[str, Any], template_path: Path, output_path: Path) -> None:
    """Inline the trace into the viewer template and write the page."""
    template = template_path.read_text(encoding="utf-8")
    if TRACE_PLACEHOLDER not in template:
        raise ValueError(f"Template {template_path} has no trace placeholder")
    payload = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    # A closing script tag inside the data would end the script element early.
    payload = payload.replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(TRACE_PLACEHOLDER, payload), encoding="utf-8")


def main() -> int:
    """Entry point: record a trace, save the JSON and render the page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Project configuration file")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed")
    parser.add_argument("--iterations", type=int, default=None, help="Override the ISMCTS budget")
    parser.add_argument("--exploration", type=float, default=None,
                        help="Override the UCB exploration constant C")
    parser.add_argument("--json-only", action="store_true", help="Skip rendering the HTML page")
    args = parser.parse_args()

    config = load_trace_config(PROJECT_ROOT / args.config)
    if args.iterations is not None:
        config["search"]["n_iterations"] = args.iterations
        config["record"]["max_iterations"] = min(
            config["record"]["max_iterations"], args.iterations
        )
    if args.exploration is not None:
        config["search"]["exploration_constant"] = args.exploration
    seed = args.seed if args.seed is not None else config["seed"]

    trace = record_decision(config, seed)

    output = config["output"]
    json_path = PROJECT_ROOT / output["trace_json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("Trace written to %s (%.1f KB)", json_path, json_path.stat().st_size / 1024)

    if not args.json_only:
        page_path = PROJECT_ROOT / output["page"]
        render_page(trace, PROJECT_ROOT / output["template"], page_path)
        logger.info("Viewer written to %s", page_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
