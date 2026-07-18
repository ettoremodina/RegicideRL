from collections import Counter

from ml_logger import get_logger, start_run
from game.action_handler import ActionHandler
from game.action_space import (
    ATTACK_ACTION_COUNT,
    DEFENSE_ACTION_COUNT,
    DEFENSE_ACTION_OFFSET,
    GLOBAL_ACTION_SPACE_SIZE,
    SOLO_JESTER_ACTION_ID,
)

logger = get_logger(__name__)


def main() -> None:
    """Display action-space dimensions and attack-action categories."""
    handler = ActionHandler()
    action_types = Counter(
        action["type"] for action in handler._global_attack_actions
    )

    context = start_run("action-space-analysis")
    result = {
        "attack_actions": ATTACK_ACTION_COUNT,
        "attack_action_types": dict(action_types),
        "defense_actions": DEFENSE_ACTION_COUNT,
        "global_action_space_size": GLOBAL_ACTION_SPACE_SIZE,
        "solo_jester_action_id": SOLO_JESTER_ACTION_ID,
    }
    logger.info("Total attack actions: %d", ATTACK_ACTION_COUNT)
    for action_type, count in action_types.items():
        logger.info("Action type %s: %d", action_type, count)

    logger.info("Hand-relative defense actions: %d", DEFENSE_ACTION_COUNT)
    logger.info("Global action space size: %d", GLOBAL_ACTION_SPACE_SIZE)
    logger.info(
        "Attack IDs: 0-%d; defense IDs: %d-%d; solo Jester ID: %d",
        ATTACK_ACTION_COUNT - 1,
        DEFENSE_ACTION_OFFSET,
        SOLO_JESTER_ACTION_ID - 1,
        SOLO_JESTER_ACTION_ID,
    )
    output = context.save_result("action_space.json", result)
    context.complete({"result": str(output)})

if __name__ == '__main__':
    main()
