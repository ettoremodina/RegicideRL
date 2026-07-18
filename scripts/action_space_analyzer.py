from collections import Counter

from game.action_handler import ActionHandler
from game.action_space import (
    ATTACK_ACTION_COUNT,
    DEFENSE_ACTION_COUNT,
    DEFENSE_ACTION_OFFSET,
    GLOBAL_ACTION_SPACE_SIZE,
    SOLO_JESTER_ACTION_ID,
)

def main() -> None:
    """Display action-space dimensions and attack-action categories."""
    handler = ActionHandler()
    action_types = Counter(
        action["type"] for action in handler._global_attack_actions
    )

    print("=== Regicide Action Space Analysis ===")

    print(f"Total attack actions: {ATTACK_ACTION_COUNT}")
    for action_type, count in action_types.items():
        print(f"  - {action_type}: {count}")

    print(f"Hand-relative defense actions: {DEFENSE_ACTION_COUNT}")
    print(f"Global action space size: {GLOBAL_ACTION_SPACE_SIZE}")
    print(
        "Attack IDs: 0-{}; defense IDs: {}-{}; solo Jester ID: {}".format(
            ATTACK_ACTION_COUNT - 1,
            DEFENSE_ACTION_OFFSET,
            SOLO_JESTER_ACTION_ID - 1,
            SOLO_JESTER_ACTION_ID,
        )
    )

if __name__ == '__main__':
    main()
