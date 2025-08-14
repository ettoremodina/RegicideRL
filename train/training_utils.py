"""
Training utilities for Regicide - Plotting, evaluation, and helper functions
Extracted from custom_training.py to keep the main training script focused
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt



from config import PathManager


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
                             win_rate_history: List[float],
                             filename: str = "training_progress.png") -> Optional[str]:
        """
        Plot comprehensive training progress
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            bosses_killed_history: List of bosses killed per episode
            win_rate_history: List of rolling win rates
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
            
            # Win rate
            if win_rate_history:
                axes[1, 0].plot(win_rate_history, color='purple', linewidth=2)
                axes[1, 0].set_title('Win Rate (Rolling Average)')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Win Rate')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].grid(True, alpha=0.3)
            
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
            
            # Win rates comparison
            win_rates = [data.get('final_win_rate', 0) for data in experiments_data.values()]
            axes[0, 0].bar(experiment_names, win_rates, color=colors)
            axes[0, 0].set_title('Final Win Rates')
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
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
            
            if episode_reward > 0:
                wins += 1
            
            if (episode + 1) % 20 == 0:
                print(f"Evaluation progress: {episode + 1}/{num_episodes}")
        
        results = {
            'avg_reward': np.mean(total_rewards),
            'avg_length': np.mean(total_lengths),
            'avg_bosses_killed': np.mean(total_boss_kills),
            'max_bosses_killed': max(total_boss_kills) if total_boss_kills else 0,
            'win_rate': wins / num_episodes,
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
        print(f"Win Rate: {results['win_rate']:.2%}")


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
                  f"WinRate={results['win_rate']:.2%}")
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
        self.win_rate_history = []
        
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
        
        # Calculate win rate
        if len(self.recent_episodes) >= 10:
            recent_win_rate = sum(self.recent_episodes) / len(self.recent_episodes)
            self.win_rate_history.append(recent_win_rate)
        else:
            self.win_rate_history.append(0.0)
    
    def get_summary(self) -> Dict:
        """Get summary of all training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'bosses_killed_history': self.bosses_killed_history,
            'max_bosses_killed': self.max_bosses_killed,
            'win_rate_history': self.win_rate_history,
            'final_win_rate': self.win_rate_history[-1] if self.win_rate_history else 0.0,
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
                'win_rate': 0.0
            }
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-window:]),
            'avg_length': np.mean(self.episode_lengths[-window:]),
            'avg_bosses': np.mean(self.bosses_killed_history[-window:]),
            'win_rate': self.win_rate_history[-1] if self.win_rate_history else 0.0
        }


# Import torch only if needed for evaluation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
