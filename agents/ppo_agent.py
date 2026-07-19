"""Adapter that exposes a trained MaskablePPO policy as a Regicide agent."""

from sb3_contrib.ppo_mask import MaskablePPO
from agents.base_agent import BaseAgent
from solvers.wrappers import NumericObsWrapper

class PPOAgent(BaseAgent):
    """Load a trained MaskablePPO model and make deterministic decisions."""

    def __init__(self, name="ppo_regicide_v1", model_path="models/ppo_regicide_v1.zip", **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = MaskablePPO.load(model_path)
        
        # We need the numeric wrapper to format observations for the model,
        # but without a real env attached. We'll use a dummy wrapper.
        # However, the easiest way is to wrap the env that is passed in select_action.
        self.wrapper = None
        
    def select_action(self, obs, env=None):
        """Encode a raw observation and query the masked deterministic policy.

        Args:
            obs: Raw observation returned by ``RegicideEnv``.
            env: Live environment wrapped lazily for numeric conversion.

        Returns:
            Global action identifier predicted by the policy.

        Raises:
            ValueError: If ``env`` is omitted.
        """
        if env is None:
            raise ValueError("PPOAgent requires the env object to be passed in to access wrapper logic.")
            
        if self.wrapper is None or self.wrapper.env != env:
            self.wrapper = NumericObsWrapper(env)
            
        # Convert raw observation to numeric dictionary
        numeric_obs = self.wrapper.observation(obs)
        
        # Predict action
        action, _states = self.model.predict(
            numeric_obs, 
            action_masks=numeric_obs['action_mask'],
            deterministic=True
        )
        
        return int(action)
