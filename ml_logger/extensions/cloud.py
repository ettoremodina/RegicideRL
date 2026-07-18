from typing import Any, Dict
from ..core.plugins import LoggerPlugin

class WandbPlugin(LoggerPlugin):
    """An extension that syncs metrics to Weights & Biases."""
    
    def __init__(self, project_name: str, enabled: bool = True):
        self.project_name = project_name
        self.enabled = enabled
        self._wandb = None
        
        if self.enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self.enabled = False
                print("Warning: 'wandb' package not found. Cloud syncing disabled.")

    def on_startup(self, config: Dict[str, Any]):
        if self.enabled and self._wandb:
            self._wandb.init(project=self.project_name, config=config)
            
    def on_metric_update(self, category: str, name: str, value: Any):
        if self.enabled and self._wandb:
            # We flatten the name, e.g. "Train/Loss"
            metric_key = f"{category}/{name}"
            # Only log numerical values to Wandb
            if isinstance(value, (int, float)):
                self._wandb.log({metric_key: value})

    def on_shutdown(self):
        if self.enabled and self._wandb:
            self._wandb.finish()
