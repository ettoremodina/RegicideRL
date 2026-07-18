"""Play and persist one random Regicide game."""

import random

from ml_logger import GameRecorder, get_logger, start_run
from solvers.env import RegicideEnv

logger = get_logger(__name__)


def play_one_game_with_logs(seed=None):
    context = start_run(
        "game",
        name="random-simulation",
        config={"seed": seed, "agent": "random"},
    )
    recorder = GameRecorder(context)
    environment = RegicideEnv(num_players=1, recorder=recorder)
    observation, _ = environment.reset(seed=seed)
    try:
        while not environment.game.game_over:
            valid_actions = [
                action_id
                for action_id, valid in enumerate(observation["action_mask"])
                if valid
            ]
            if not valid_actions:
                environment.game.game_over = True
                if recorder.active:
                    recorder.finish(environment.game, reason="no_valid_actions")
                break
            action = random.choice(valid_actions)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                break
        summary = {
            "victory": environment.game.victory,
            "victory_tier": environment.game.get_victory_tier(),
        }
        result = context.save_result("game_result.json", summary)
        context.complete({"result": str(result), **summary})
        logger.info("Game completed: victory=%s", environment.game.victory)
        return summary
    except Exception as error:
        if recorder.active:
            recorder.abort("exception")
        context.fail(error)
        logger.exception("Game simulation failed")
        raise


if __name__ == "__main__":
    play_one_game_with_logs()
