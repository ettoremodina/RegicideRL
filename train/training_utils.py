"""
Training utilities for Regicide - Plotting, evaluation, and helper functions
Extracted from custom_training.py to keep the main training script focused
"""

import torch
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from config import PathManager


class ModelManager:
    """
    Handles model saving, loading, versioning, and automatic resume functionality
    """
    
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        
    def save_model_with_versioning(self, model_state: Dict, filename: str, 
                                 is_checkpoint: bool = False, keep_versions: int = 3) -> str:
        """
        Save model with automatic versioning and cleanup of old versions
        
        Args:
            model_state: Dictionary containing model state and metadata
            filename: Base filename for the model
            is_checkpoint: Whether this is a checkpoint save
            keep_versions: Number of versions to keep (older ones deleted)
            
        Returns:
            Path to saved model
        """
        # Get the appropriate directory
        if is_checkpoint:
            base_dir = self.path_manager.checkpoint_dir
        else:
            base_dir = self.path_manager.model_dir
            
        # Ensure directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Create filename with timestamp for versioning
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        
        # Find existing versions
        pattern = os.path.join(base_dir, f"{base_name}_v*.pth")
        existing_files = glob.glob(pattern)
        
        # Determine next version number
        version_numbers = []
        for file_path in existing_files:
            try:
                # Extract version number from filename like "model_v001.pth"
                basename = os.path.basename(file_path)
                version_str = basename.split('_v')[1].split('.')[0]
                version_numbers.append(int(version_str))
            except (IndexError, ValueError):
                continue
                
        next_version = max(version_numbers, default=0) + 1
        
        # Create new filename with version
        versioned_filename = f"{base_name}_v{next_version:03d}{extension}"
        filepath = os.path.join(base_dir, versioned_filename)
        
        # Save the model
        torch.save(model_state, filepath)
        
        # Clean up old versions
        self._cleanup_old_versions(base_dir, base_name, extension, keep_versions)
        
        return filepath
    
    def _cleanup_old_versions(self, directory: str, base_name: str, extension: str, keep_versions: int):
        """Clean up old model versions, keeping only the most recent ones"""
        pattern = os.path.join(directory, f"{base_name}_v*{extension}")
        existing_files = glob.glob(pattern)
        
        if len(existing_files) <= keep_versions:
            return
            
        # Sort by modification time (newest first)
        existing_files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete older files
        files_to_delete = existing_files[keep_versions:]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"🗑️  Deleted old model version: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"⚠️  Could not delete {file_path}: {e}")
    
    def find_latest_compatible_model(self, model_config: Dict, search_checkpoints: bool = True) -> Optional[str]:
        """
        Find the latest model file that's compatible with the current architecture
        
        Args:
            model_config: Current model configuration to match against
            search_checkpoints: Whether to search checkpoint directory as well
            
        Returns:
            Path to compatible model file or None if not found
        """
        candidate_files = []
        
        # Search model directory
        model_pattern = os.path.join(self.path_manager.model_dir, "*.pth")
        candidate_files.extend(glob.glob(model_pattern))
        
        # Optionally search checkpoint directory
        if search_checkpoints:
            checkpoint_pattern = os.path.join(self.path_manager.checkpoint_dir, "*.pth")
            candidate_files.extend(glob.glob(checkpoint_pattern))
        
        if not candidate_files:
            return None
        
        # Sort by modification time (newest first)
        candidate_files.sort(key=os.path.getmtime, reverse=True)
        
        # Find the most recent compatible model
        for file_path in candidate_files:
            try:
                # Load model metadata without loading the full model
                checkpoint = torch.load(file_path, map_location='cpu')
                
                if 'model_config' not in checkpoint:
                    continue
                    
                saved_config = checkpoint['model_config']
                
                # Check compatibility
                if self._is_config_compatible(model_config, saved_config):
                    return file_path
                    
            except Exception as e:
                print(f"⚠️  Could not load {file_path}: {e}")
                continue
        
        return None
    
    def _is_config_compatible(self, current_config: Dict, saved_config: Dict) -> bool:
        """
        Check if two model configurations are compatible
        
        Args:
            current_config: Current model configuration
            saved_config: Saved model configuration
            
        Returns:
            True if compatible, False otherwise
        """
        # Key parameters that must match for compatibility
        critical_params = [
            'policy_type',
            'max_hand_size', 
            'max_actions',
            'card_embed_dim',
            'hidden_dim'
        ]
        
        for param in critical_params:
            if param in current_config and param in saved_config:
                if current_config[param] != saved_config[param]:
                    return False
            elif param in current_config or param in saved_config:
                # One has the parameter, the other doesn't
                return False
        
        return True
    
    def auto_resume_training(self, model_config: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Automatically find and load the latest compatible model for resuming training
        
        Args:
            model_config: Current model configuration
            
        Returns:
            Tuple of (model_path, loaded_state) or (None, None) if no compatible model found
        """
        latest_model_path = self.find_latest_compatible_model(model_config)
        
        if latest_model_path is None:
            return None, None
        
        try:
            # Load the model state
            checkpoint = torch.load(latest_model_path, map_location='cpu')
            print(f"🔄 Found compatible model for resume: {os.path.basename(latest_model_path)}")
            print(f"   Model config: {checkpoint.get('model_config', {})}")
            
            return latest_model_path, checkpoint
            
        except Exception as e:
            print(f"⚠️  Error loading model {latest_model_path}: {e}")
            return None, None


class TrainingResumer:
    """
    Handles automatic training resume with compatibility checking
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def attempt_resume(self, policy, optimizer, model_config: Dict, 
                      force_resume: bool = False) -> Tuple[int, float, bool]:
        """
        Attempt to resume training from a compatible checkpoint
        
        Args:
            policy: The policy model to load state into
            optimizer: The optimizer to load state into  
            model_config: Current model configuration
            force_resume: If True, skip user confirmation
            
        Returns:
            Tuple of (start_episode, best_score, resumed_successfully)
        """
        model_path, checkpoint = self.model_manager.auto_resume_training(model_config)
        
        if model_path is None:
            print("🆕 No compatible model found - starting fresh training")
            return 0, float('-inf'), False
        
        # Ask user for confirmation unless forced
        if not force_resume:
            response = input(f"\n📄 Found compatible model: {os.path.basename(model_path)}\n"
                           f"   Episodes trained: {checkpoint.get('episode', 'Unknown')}\n"
                           f"   Best score: {checkpoint.get('best_score', 'Unknown')}\n"
                           f"   Resume training? (y/n): ").lower().strip()
            
            if response != 'y' and response != 'yes':
                print("🆕 Starting fresh training")
                return 0, float('-inf'), False
        
        try:
            # Load model state
            policy.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Extract training progress
            start_episode = checkpoint.get('episode', 0)
            best_score = checkpoint.get('best_score', float('-inf'))
            
            print(f"✅ Successfully resumed from episode {start_episode}")
            print(f"   Best score so far: {best_score:.2f}")
            
            return start_episode, best_score, True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("🆕 Starting fresh training instead")
            return 0, float('-inf'), False
    
    def create_resume_config(self, policy, optimizer, episode: int, 
                           best_score: float, model_config: Dict) -> Dict:
        """
        Create a complete checkpoint state for resuming
        
        Args:
            policy: Policy model
            optimizer: Optimizer
            episode: Current episode number
            best_score: Best score achieved
            model_config: Model configuration
            
        Returns:
            Complete checkpoint dictionary
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'model_config': model_config,
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__
        }
        
        return checkpoint


def create_model_config(policy_type: str = "rule_based", **kwargs) -> Dict:
    """
    Create a standardized model configuration dictionary for compatibility checking
    
    Args:
        policy_type: Type of policy ("rule_based", "neural", etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        Model configuration dictionary
    """
    config = {
        'policy_type': policy_type,
        'max_hand_size': kwargs.get('max_hand_size', 30),
        'max_actions': kwargs.get('max_actions', 100),
        'card_embed_dim': kwargs.get('card_embed_dim', 128),
        'hidden_dim': kwargs.get('hidden_dim', 256),
        'created_at': datetime.now().isoformat()
    }
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if key not in config:
            config[key] = value
    
    return config


class TrainingVisualizer:
    """
    Handles all visualization and plotting for training progress
    """
    
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
    
    def plot_training_progress(self, 
                             episode_rewards: List[float],
                             episode_lengths: List[float], 
                             bosses_killed_history: List[int],
                             win_rate_history: List[float] = None,
                             filename: str = "training_progress.png") -> Optional[str]:
        """
        Plot comprehensive training progress
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            bosses_killed_history: List of bosses killed per episode
            filename: Name of the plot file (without path)
        
        Returns:
            Path to saved plot or None if failed
        """
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle('Regicide Training Progress', fontsize=16)
            
            # Episode rewards
            if episode_rewards:
                axes[0, 0].plot(episode_rewards, alpha=0.7)
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add moving average
                if len(episode_rewards) > 10:
                    window = min(50, len(episode_rewards) // 10)
                    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                    axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                                  color='red', linewidth=2, label=f'{window}-episode average')
                    axes[0, 0].legend()
            
            # Episode lengths
            if episode_lengths:
                axes[0, 1].plot(episode_lengths, alpha=0.7, color='orange')
                axes[0, 1].set_title('Episode Lengths')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Steps')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add moving average
                if len(episode_lengths) > 10:
                    window = min(50, len(episode_lengths) // 10)
                    moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg, 
                                  color='red', linewidth=2, label=f'{window}-episode average')
                    axes[0, 1].legend()
            
            # Bosses killed per episode
            if bosses_killed_history:
                axes[0, 2].plot(bosses_killed_history, alpha=0.7, color='green')
                axes[0, 2].set_title('Bosses Killed per Episode')
                axes[0, 2].set_xlabel('Episode')
                axes[0, 2].set_ylabel('Bosses Killed')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Add moving average
                if len(bosses_killed_history) > 10:
                    window = min(50, len(bosses_killed_history) // 10)
                    moving_avg = np.convolve(bosses_killed_history, np.ones(window)/window, mode='valid')
                    axes[0, 2].plot(range(window-1, len(bosses_killed_history)), moving_avg, 
                                  color='red', linewidth=2, label=f'{window}-episode average')
                    axes[0, 2].legend()

            
            # Reward distribution
            if episode_rewards:
                axes[1, 1].hist(episode_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].set_title('Reward Distribution')
                axes[1, 1].set_xlabel('Reward')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Boss kills distribution
            if bosses_killed_history:
                max_bosses = max(bosses_killed_history)
                bins = range(0, max_bosses + 2) if max_bosses > 0 else [0, 1]
                axes[1, 2].hist(bosses_killed_history, bins=bins, align='left', 
                              alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1, 2].set_title('Boss Kills Distribution')
                axes[1, 2].set_xlabel('Bosses Killed')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to organized location
            filepath = self.path_manager.get_plot_path(filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Training progress plot saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"⚠️  Error generating plot: {e}")
            return None
    
    def plot_comparison_chart(self, 
                            experiments_data: Dict[str, Dict],
                            filename: str = "experiments_comparison.png") -> Optional[str]:
        """
        Create a comparison chart between different experiments
        
        Args:
            experiments_data: Dict with experiment names as keys and their stats as values
            filename: Name of the comparison plot file
        
        Returns:
            Path to saved plot or None if failed
        """
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Experiment Comparison', fontsize=16)
            
            experiment_names = list(experiments_data.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_names)))
            
            
            # Average bosses killed
            avg_bosses = [data.get('avg_bosses_killed', 0) for data in experiments_data.values()]
            axes[0, 1].bar(experiment_names, avg_bosses, color=colors)
            axes[0, 1].set_title('Average Bosses Killed')
            axes[0, 1].set_ylabel('Avg Bosses')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Max bosses killed
            max_bosses = [data.get('max_bosses_killed', 0) for data in experiments_data.values()]
            axes[1, 0].bar(experiment_names, max_bosses, color=colors)
            axes[1, 0].set_title('Max Bosses Killed')
            axes[1, 0].set_ylabel('Max Bosses')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Total episodes
            total_episodes = [data.get('total_episodes', 0) for data in experiments_data.values()]
            axes[1, 1].bar(experiment_names, total_episodes, color=colors)
            axes[1, 1].set_title('Training Episodes')
            axes[1, 1].set_ylabel('Episodes')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            filepath = self.path_manager.get_global_plot_path(filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Comparison chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"⚠️  Error generating comparison chart: {e}")
            return None


class TrainingEvaluator:
    """
    Handles evaluation of trained models
    """
    
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
    
    def evaluate_policy(self, num_episodes: int = 100, render: bool = False) -> Dict:
        """
        Evaluate the trained policy
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render some episodes
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"🧪 Evaluating policy for {num_episodes} episodes...")
        
        total_rewards = []
        total_lengths = []
        total_boss_kills = []
        wins = 0
        
        for episode in range(num_episodes):
            episode_reward, episode_length, bosses_killed = self._run_evaluation_episode(
                render and episode < 3
            )
            
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            total_boss_kills.append(bosses_killed)
            
            if total_boss_kills == 12:
                wins += 1
            
            if (episode + 1) % 20 == 0:
                print(f"Evaluation progress: {episode + 1}/{num_episodes}")
        
        results = {
            'avg_reward': np.mean(total_rewards),
            'avg_length': np.mean(total_lengths),
            'avg_bosses_killed': np.mean(total_boss_kills),
            'max_bosses_killed': max(total_boss_kills) if total_boss_kills else 0,
            'total_episodes': num_episodes
        }
        
        self._print_evaluation_results(results)
        return results
    
    def _run_evaluation_episode(self, render: bool = False) -> Tuple[float, int, int]:
        """
        Run a single evaluation episode
        
        Returns:
            Tuple of (episode_reward, episode_length, bosses_killed)
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        if render:
            print(f"\n=== EVALUATION EPISODE ===")
            self.env.render()
        
        while True:
            # Get action from policy (no training)
            with torch.no_grad():
                action, _ = self.policy.get_action(obs, info['action_mask'])
            
            if render:
                meanings = self.env.get_action_meanings()
                action_meaning = meanings[action] if action < len(meanings) else "Invalid"
                print(f"\nStep {episode_length + 1}:")
                print(f"Player {info['current_player'] + 1}, Phase: {info['phase']}")
                print(f"Action {action}: {action_meaning}")
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render and reward != 0:
                print(f"Reward: {reward}")
            
            if render:
                self.env.render()
            
            if terminated or truncated:
                bosses_killed = next_info.get('bosses_killed', 0)
                if render:
                    victory = next_info.get('victory', False)
                    if victory:
                        print(f"\n🎉 VICTORY! Episode length: {episode_length}, Bosses killed: {bosses_killed}")
                    else:
                        print(f"\n💀 DEFEAT! Episode length: {episode_length}, Bosses killed: {bosses_killed}")
                break
            
            obs = next_obs
            info = next_info
        
        return episode_reward, episode_length, next_info.get('bosses_killed', 0)
    
    def _print_evaluation_results(self, results: Dict):
        """Print formatted evaluation results"""
        print(f"\n📊 Evaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.3f}")
        print(f"Average Length: {results['avg_length']:.1f}")
        print(f"Average Bosses Killed: {results['avg_bosses_killed']:.2f}")
        print(f"Max Bosses Killed: {results['max_bosses_killed']}")


class TrainingLogger:
    """
    Handles logging during training
    """
    
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.log_file = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to file"""
        try:
            log_path = self.path_manager.get_log_path("training.log")
            self.log_file = open(log_path, 'w')
            self.log(f"Training log started")
            print(f"📝 Training log: {log_path}")
        except Exception as e:
            print(f"⚠️  Could not setup logging: {e}")
            self.log_file = None
    
    def log(self, message: str):
        """Log a message to file and console"""
        if self.log_file:
            try:
                self.log_file.write(f"{message}\n")
                self.log_file.flush()
            except:
                pass
    
    def log_progress(self, episode: int, results: Dict):
        """Log training progress"""
        message = (f"Episode {episode}: "
                  f"Reward={results['episode_reward']:.2f}, "
                  f"Length={results['episode_length']}, "
                  f"Bosses={results['bosses_killed']}, "
                  )
        self.log(message)
    
    def log_experiment_config(self, config: Dict):
        """Log experiment configuration"""
        self.log("=" * 50)
        self.log("EXPERIMENT CONFIGURATION")
        self.log("=" * 50)
        for key, value in config.items():
            self.log(f"{key}: {value}")
        self.log("=" * 50)
    
    def close(self):
        """Close log file"""
        if self.log_file:
            self.log("Training log ended")
            self.log_file.close()
            self.log_file = None


class TrainingStatistics:
    """
    Tracks and manages training statistics
    """
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.bosses_killed_history = []
        self.max_bosses_killed = 0
        
        # For tracking recent performance
        self.recent_episodes = deque(maxlen=100)
        self.recent_boss_kills = deque(maxlen=100)
    
    def update(self, episode_reward: float, episode_length: int, bosses_killed: int):
        """Update statistics with new episode data"""
        # Record basic stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.bosses_killed_history.append(bosses_killed)
        
        # Update max bosses killed
        if bosses_killed > self.max_bosses_killed:
            self.max_bosses_killed = bosses_killed
        
        # Track recent performance
        won = episode_reward > 0
        self.recent_episodes.append(won)
        self.recent_boss_kills.append(bosses_killed)
    
    
    def get_summary(self) -> Dict:
        """Get summary of all training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'bosses_killed_history': self.bosses_killed_history,
            'max_bosses_killed': self.max_bosses_killed,
            'avg_bosses_killed': np.mean(self.bosses_killed_history) if self.bosses_killed_history else 0.0,
            'total_episodes': len(self.episode_rewards)
        }
    
    def get_recent_averages(self, window: int = 10) -> Dict:
        """Get recent performance averages"""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        
        if window == 0:
            return {
                'avg_reward': 0.0,
                'avg_length': 0.0,
                'avg_bosses': 0.0,
            }
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-window:]),
            'avg_length': np.mean(self.episode_lengths[-window:]),
            'avg_bosses': np.mean(self.bosses_killed_history[-window:]),
        }


