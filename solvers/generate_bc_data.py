"""Generate a behavioral-cloning dataset and record its source games."""

import argparse
from pathlib import Path

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from ml_logger import GameRecorder, get_logger, start_run
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper

logger = get_logger(__name__)


def generate_data(num_games, save_path, recorder=None):
    raw_environment = RegicideEnv(num_players=1, recorder=recorder)
    environment = NumericObsWrapper(raw_environment)
    agent = HeuristicAgent(name="teacher")
    observations = []
    actions = []
    logger.info("Generating BC data from %d games", num_games)
    progress_interval = max(1, num_games // 10)
    for game_index in range(num_games):
        observation, _ = environment.reset()
        done = False
        while not done:
            action = agent.select_action(raw_environment._get_obs(), env=raw_environment)
            if action is None:
                if recorder and recorder.active:
                    recorder.finish(
                        raw_environment.game,
                        reason="teacher_returned_no_action",
                    )
                break
            observations.append(_copy_observation(observation))
            actions.append(action)
            observation, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
        if (game_index + 1) % progress_interval == 0:
            logger.info("Generated %d/%d games", game_index + 1, num_games)
    if not actions:
        raise RuntimeError("The teacher generated no state-action pairs")
    arrays = _stack_dataset(observations, actions)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    logger.info("Saved %d state-action pairs to %s", len(actions), output_path)
    return {"games": num_games, "samples": len(actions), "path": str(output_path)}


def _copy_observation(observation):
    return {
        "hand_values": observation["hand_values"].copy(),
        "hand_suits": observation["hand_suits"].copy(),
        "enemy_stats": observation["enemy_stats"].copy(),
        "flags": observation["flags"].copy(),
        "action_mask": observation["action_mask"].copy(),
    }


def _stack_dataset(observations, actions):
    return {
        "hand_values": np.stack([item["hand_values"] for item in observations]),
        "hand_suits": np.stack([item["hand_suits"] for item in observations]),
        "enemy_stats": np.stack([item["enemy_stats"] for item in observations]),
        "flags": np.stack([item["flags"] for item in observations]),
        "action_masks": np.stack([item["action_mask"] for item in observations]),
        "actions": np.array(actions, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--out")
    args = parser.parse_args()
    context = start_run("bc-data-generation", config=vars(args))
    recorder = GameRecorder(context)
    output = args.out or context.run_dir / "datasets" / "bc_data.npz"
    try:
        result = generate_data(args.games, output, recorder)
        context.save_result("dataset.json", result)
        context.complete(result)
    except Exception as error:
        context.fail(error)
        logger.exception("BC dataset generation failed")
        raise


if __name__ == "__main__":
    main()
