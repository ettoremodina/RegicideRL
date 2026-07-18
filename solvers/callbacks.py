import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in TensorBoard.
    Tracks bosses defeated, invalid actions, and yield rates.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.invalid_actions = 0
        self.yields = 0
        self.total_actions = 0
        self.episodes = 0
        self.wins = 0
        
        # We need to track the last known bosses defeated per environment
        # in case of vectorized environments, but assuming single env for simplicity.
        self.last_bosses_defeated = 0

    def _on_step(self) -> bool:
        # Access info dict from the environment
        infos = self.locals.get("infos", [])
        
        for info in infos:
            self.total_actions += 1
            
            # Record invalid actions (info['success'] == False)
            if not info.get('success', True):
                self.invalid_actions += 1
                
            # Check for yields (you can add a specific flag in env, or infer from message)
            if "Yield" in info.get("message", ""):
                self.yields += 1
                
        # Check if episode terminated to log episodic stats
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                self.episodes += 1
                
                # Fetch env object
                env = self.training_env.envs[i].unwrapped
                
                # Calculate bosses defeated
                # Bosses defeated = 12 - remaining castle deck cards
                if hasattr(env, 'game') and hasattr(env.game, 'castle_deck'):
                    defeated = 12 - len(env.game.castle_deck)
                    if not env.game.victory:
                        defeated = max(0, defeated - 1)
                    self.last_bosses_defeated = defeated
                    
                    if env.game.victory:
                        self.wins += 1
                        
                # Log metrics to TensorBoard
                self.logger.record("custom/bosses_defeated", self.last_bosses_defeated)
                
                win_rate = self.wins / max(1, self.episodes)
                self.logger.record("custom/win_rate", win_rate)
                
                invalid_rate = self.invalid_actions / max(1, self.total_actions)
                self.logger.record("custom/invalid_action_rate", invalid_rate)
                
                yield_rate = self.yields / max(1, self.total_actions)
                self.logger.record("custom/yield_rate", yield_rate)
                
                # Reset counters (except episodes/wins which can be cumulative or episodic)
                self.invalid_actions = 0
                self.yields = 0
                self.total_actions = 0
                
        return True
