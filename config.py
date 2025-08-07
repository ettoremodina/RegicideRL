"""
Configuration file for Regicide training
Manages file paths and folder organization
"""
import os
from datetime import datetime
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Legacy directories (for compatibility)
LEGACY_MODELS_DIR = PROJECT_ROOT / "regicide_models"
LEGACY_LOGS_DIR = PROJECT_ROOT / "regicide_logs"
LEGACY_TENSORBOARD_DIR = PROJECT_ROOT / "regicide_tensorboard"

class PathManager:
    """
    Manages file paths and ensures directories exist
    """
    
    def __init__(self, experiment_name: str = None):
        """
        Initialize with optional experiment name for organized runs
        """
        if experiment_name is None:
            # Generate default experiment name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"regicide_training_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = OUTPUTS_DIR / experiment_name
        
        # Create all necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            OUTPUTS_DIR,
            MODELS_DIR,
            PLOTS_DIR,
            CHECKPOINTS_DIR,
            LOGS_DIR,
            self.experiment_dir,
            self.experiment_dir / "models",
            self.experiment_dir / "plots",
            self.experiment_dir / "checkpoints",
            self.experiment_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, filename: str, is_checkpoint: bool = False) -> str:
        """
        Get path for model files
        
        Args:
            filename: Name of the model file
            is_checkpoint: If True, save to checkpoints directory
        
        Returns:
            Full path as string
        """
        if is_checkpoint:
            return str(self.experiment_dir / "checkpoints" / filename)
        else:
            return str(self.experiment_dir / "models" / filename)
    
    def get_plot_path(self, filename: str) -> str:
        """
        Get path for plot files
        
        Args:
            filename: Name of the plot file
        
        Returns:
            Full path as string
        """
        return str(self.experiment_dir / "plots" / filename)
    
    def get_log_path(self, filename: str) -> str:
        """
        Get path for log files
        
        Args:
            filename: Name of the log file
        
        Returns:
            Full path as string
        """
        return str(self.experiment_dir / "logs" / filename)
    
    def get_experiment_summary_path(self) -> str:
        """Get path for experiment summary file"""
        return str(self.experiment_dir / "experiment_summary.json")
    
    def get_global_model_path(self, filename: str) -> str:
        """
        Get path for global model storage (not experiment-specific)
        
        Args:
            filename: Name of the model file
        
        Returns:
            Full path as string
        """
        return str(MODELS_DIR / filename)
    
    def get_global_plot_path(self, filename: str) -> str:
        """
        Get path for global plot storage (not experiment-specific)
        
        Args:
            filename: Name of the plot file
        
        Returns:
            Full path as string
        """
        return str(PLOTS_DIR / filename)
    
    def list_experiments(self) -> list:
        """List all available experiments"""
        if not OUTPUTS_DIR.exists():
            return []
        
        experiments = []
        for item in OUTPUTS_DIR.iterdir():
            if item.is_dir() and item.name not in ['models', 'plots', 'checkpoints', 'logs']:
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Clean up old checkpoint files, keeping only the last N
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoint_dir = self.experiment_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return
        
        # Get all checkpoint files sorted by modification time
        checkpoint_files = []
        for file in checkpoint_dir.glob("*.pth"):
            if "episode_" in file.name:
                checkpoint_files.append(file)
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        for file in checkpoint_files[keep_last_n:]:
            try:
                file.unlink()
                print(f"🗑️  Removed old checkpoint: {file.name}")
            except Exception as e:
                print(f"⚠️  Could not remove {file.name}: {e}")
    
    def create_experiment_summary(self, stats: dict, config: dict):
        """
        Create a summary file for the experiment
        
        Args:
            stats: Training statistics
            config: Training configuration
        """
        import json
        
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "final_stats": {
                "total_episodes": stats.get('total_episodes', 0),
                "final_win_rate": stats.get('final_win_rate', 0.0),
                "avg_bosses_killed": stats.get('avg_bosses_killed', 0.0),
                "max_bosses_killed": stats.get('max_bosses_killed', 0)
            },
            "files": {
                "final_model": self.get_model_path("final_model.pth"),
                "training_plot": self.get_plot_path("training_progress.png"),
                "logs": self.get_log_path("training.log")
            }
        }
        
        summary_path = self.get_experiment_summary_path()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Experiment summary saved to: {summary_path}")

# Default path manager instance
default_path_manager = PathManager()

# Convenience functions for backward compatibility
def get_model_path(filename: str) -> str:
    """Get model path using default path manager"""
    return default_path_manager.get_model_path(filename)

def get_plot_path(filename: str) -> str:
    """Get plot path using default path manager"""
    return default_path_manager.get_plot_path(filename)

def get_checkpoint_path(filename: str) -> str:
    """Get checkpoint path using default path manager"""
    return default_path_manager.get_model_path(filename, is_checkpoint=True)

def get_log_path(filename: str) -> str:
    """Get log path using default path manager"""
    return default_path_manager.get_log_path(filename)
