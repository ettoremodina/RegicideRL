"""Common action-selection interface implemented by every Regicide agent."""


class BaseAgent:
    """Define the minimal interface shared by evaluation and solver workflows.

    Args:
        name: Human-readable identifier used in logs and reports.
    """

    def __init__(self, name="BaseAgent", **kwargs):
        self.name = name
        
    def select_action(self, obs, env=None):
        """Choose a global action identifier for the current observation.

        Args:
            obs: Observation dictionary returned by ``RegicideEnv``.
            env: Live environment when an implementation needs cloning or
                direct access to the game state.

        Returns:
            Global action identifier, or ``None`` when no action is legal.

        Raises:
            NotImplementedError: Always; subclasses define the policy.
        """
        raise NotImplementedError("Agents must implement select_action")
