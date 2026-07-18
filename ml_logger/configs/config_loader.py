import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """Loads configuration from a YAML file."""
    if config_path is None:
        # Default to the config file in the same directory
        config_path = Path(__file__).parent / "default_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
