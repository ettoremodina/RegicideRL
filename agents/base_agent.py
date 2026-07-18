class BaseAgent:
    """
    Abstract base class for all Regicide agents.
    """
    def __init__(self, name="BaseAgent", **kwargs):
        self.name = name
        
    def select_action(self, obs, env=None):
        """
        Takes an observation dictionary from RegicideEnv and returns a chosen action_mask.
        
        Args:
            obs (dict): The observation containing:
                - 'game_state': Raw game state dict
                - 'hand': Current player's hand (list of Card objects)
                - 'current_player': Int
                - 'valid_actions': List of action masks (lists of ints)
                - 'defense_phase': Bool
                - 'required_defense': Int
                
        Returns:
            list: The chosen action_mask.
        """
        raise NotImplementedError("Agents must implement select_action")
