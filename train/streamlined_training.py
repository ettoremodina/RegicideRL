"""
Streamlined Training Script for Regicide Environment using PyTorch
Focused only on training logic - utilities moved to training_utils.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Optional

# from card_aware_env import CardAwareRegicideEnv
from regicide_gym_env import RegicideGymEnv
from card_aware_policy import CardAwarePolicy
from config import PathManager
from training_utils import TrainingVisualizer, TrainingEvaluator, TrainingLogger, TrainingStatistics


class CardAwareRegicideTrainer:
    """
    Enhanced trainer for Regicide environment with card-aware architecture
    """
    
    def __init__(self, 
                 env: RegicideGymEnv,
                 learning_rate: float = 0.001,
                 card_embed_dim: int = 16,
                 hidden_dim: int = 64,
                 gamma: float = 0.99,
                 experiment_name: str = None):
        
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Initialize components
        self.path_manager = PathManager(experiment_name)
        self.stats = TrainingStatistics()
        self.logger = TrainingLogger(self.path_manager)
        self.visualizer = TrainingVisualizer(self.path_manager)
    
        # Use new card-aware policy
        self.policy = CardAwarePolicy(
            max_hand_size=env.max_hand_size,
            max_actions=env.max_actions,
            card_embed_dim=card_embed_dim,
            hidden_dim=hidden_dim
        )
    
        # Optimizer - AdamW with weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.policy.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # We want to maximize rewards
            factor=0.8,  # Reduce LR by 20% when plateau
            patience=1000,  # Wait 1000 episodes before reducing
            min_lr=1e-5
        )
        
        # Create evaluator (will need to be updated for card-aware policy)
        # self.evaluator = TrainingEvaluator(env, self.policy)
    
    def collect_episode(self, render: bool = False) -> Tuple[List, List, List, List, float, int, int]:
        """Collect one episode of experience with card-aware observations"""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        if render:
            print(f"\n=== EPISODE START ===")
            self.env.render()
        
        while True:
            action, log_prob = self.policy.get_action(obs)
            
            if render:
                self._render_step(action, episode_length, info, obs)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)  # Store step reward
            
            episode_reward += reward
            episode_length += 1
            
            if render and reward != 0:
                print(f"Reward: {reward:.2f}")
                self.env.render()
            
            if terminated or truncated:
                if render:
                    self._render_episode_end(episode_length, next_info)
                break
            
            obs = next_obs
            info = next_info
        
        return observations, actions, log_probs, rewards, episode_reward, episode_length, next_info.get('bosses_killed', 0)
    
    def _render_step(self, action: int, episode_length: int, info: dict, obs: Dict):
        """Render a single step during episode with enhanced information"""
        meanings = self.env.get_action_meanings()
        action_meaning = meanings[action] if action < len(meanings) else "Invalid"
        
        print(f"\nStep {episode_length + 1}:")
        print(f"Player {info['current_player'] + 1}, Phase: {info['phase']}")
        print(f"Action {action}: {action_meaning}")
        print(f"Bosses killed so far: {info.get('bosses_killed', 0)}")
        
        # Show card analysis if using card-aware policy

        try:
            analysis = self.policy.analyze_decision(obs)
            print(f"Hand size: {analysis['hand_size']}, Valid actions: {analysis['num_valid_actions']}")
            
            # Show top 3 action probabilities
            probs = analysis['action_probabilities']
            if len(probs) > 1:
                sorted_actions = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                print("Top action probabilities:")
                for i, (action_idx, prob) in enumerate(sorted_actions[:3]):
                    if action_idx < len(meanings):
                        print(f"  {action_idx}: {meanings[action_idx]} ({prob:.3f})")
        except Exception as e:
            # Don't crash on analysis errors
            pass
    
    def _render_episode_end(self, episode_length: int, info: dict):
        """Render episode end information"""
        bosses_killed = info.get('bosses_killed', 0)
        victory = info.get('victory', False)
        if victory:
            print(f"\n🎉 VICTORY! Episode length: {episode_length}, Bosses killed: {bosses_killed}")
        else:
            print(f"\n💀 DEFEAT! Episode length: {episode_length}, Bosses killed: {bosses_killed}")
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns properly"""
        returns = []
        G = 0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def train_episode(self, render: bool = False) -> Dict:
        """Train on one episode using REINFORCE with proper returns"""
        # Collect episode - now also collect step rewards
        observations, actions, log_probs, rewards, episode_reward, episode_length, bosses_killed = self.collect_episode(render)
        
        # Compute proper discounted returns from step rewards
        returns = self.compute_returns(rewards)
        
        # Convert to tensors
        log_probs = torch.stack(log_probs)
        returns = torch.FloatTensor(returns)
        
        # Use baseline for variance reduction
        if len(returns) > 1:
            baseline = returns.mean()
            advantages = returns - baseline
        else:
            advantages = returns
        
        # Compute policy loss with entropy regularization
        policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy bonus for exploration (applies to both streamlined and enhanced)
        entropy_bonus = 0.0
        entropy_coefficient = 0.01  # Small entropy bonus
        
        for i, obs in enumerate(observations):
            # Get action probabilities at time of action
            with torch.no_grad():
                action_probs = self.policy.get_action_probabilities(obs)
                if action_probs.numel() > 1:
                    # Compute entropy: -sum(p * log(p))
                    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                    entropy_bonus += entropy
        
        if len(observations) > 0:
            entropy_bonus = entropy_bonus / len(observations)
            policy_loss = policy_loss - entropy_coefficient * entropy_bonus
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Update statistics
        self.stats.update(episode_reward, episode_length, bosses_killed)
        
        # Step learning rate scheduler based on recent performance
        if hasattr(self, 'scheduler') and len(self.stats.episode_rewards) % 100 == 0:
            recent_reward = np.mean(self.stats.episode_rewards[-100:])
            self.scheduler.step(recent_reward)
        
        # Return episode results
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'bosses_killed': bosses_killed,
            'win_rate': self.stats.win_rate_history[-1] if self.stats.win_rate_history else 0.0,
            'policy_loss': policy_loss.item()
        }
    
    def train(self, num_episodes: int, render_every: int = 0, save_every: int = 100, log_every: int = 10):
        """Train the agent for multiple episodes"""
        print(f"🚀 Starting training for {num_episodes} episodes")
        print(f"Experiment: {self.path_manager.experiment_name}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Discount factor: {self.gamma}")
        print(f"Policy type: Card-Aware Attention")
        print("=" * 60)
        
        # Log experiment configuration
        config = {
            'num_episodes': num_episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'policy_type': 'card_aware',
            'num_players': self.env.num_players,
            'max_hand_size': self.env.max_hand_size
        }
        

        config.update({
                'card_embed_dim': self.policy.card_embed_dim,
                'hidden_dim': self.policy.hidden_dim
            })  

            
        self.logger.log_experiment_config(config)
        
        for episode in range(num_episodes):
            # Train one episode
            render = render_every > 0 and (episode + 1) % render_every == 0
            results = self.train_episode(render)
            
            # Log progress
            if (episode + 1) % log_every == 0:
                self.logger.log_progress(episode + 1, results)
            
            # Print progress
            if (episode + 1) % log_every == 0:
                self._print_progress(episode + 1, results)
            
            # Enhanced logging on render episodes
            if render:
                self._print_render_stats(episode + 1, results)
            
            # Save model periodically
            if save_every > 0 and (episode + 1) % save_every == 0:
                self._save_checkpoint(episode + 1)
        
        print("\n🏁 Training completed!")
        self.logger.close()
        return self.stats.get_summary()
    
    def _print_progress(self, episode: int, results: Dict):
        """Print training progress"""
        recent_stats = self.stats.get_recent_averages()
        print(f"Episode {episode:4d}: "
              f"Reward: {recent_stats['avg_reward']:6.2f}, "
              f"Length: {recent_stats['avg_length']:5.1f}, "
              f"Bosses: {recent_stats['avg_bosses']:4.1f}, "
              f"Win Rate: {recent_stats['win_rate']:.2%}, "
              f"Loss: {results['policy_loss']:.4f}")
    
    def _print_render_stats(self, episode: int, results: Dict):
        """Print detailed stats for render episodes"""
        recent_max_bosses = max(self.stats.recent_boss_kills) if self.stats.recent_boss_kills else 0
        print(f"\n📊 RENDER EPISODE {episode} STATS:")
        print(f"  This episode bosses killed: {results['bosses_killed']}")
        print(f"  Max bosses killed ever: {self.stats.max_bosses_killed}")
        print(f"  Max bosses in recent {len(self.stats.recent_boss_kills)} episodes: {recent_max_bosses}")
        print(f"  Recent avg bosses/episode: {np.mean(self.stats.recent_boss_kills) if self.stats.recent_boss_kills else 0:.2f}")
        print("=" * 60)
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        filename = f"regicide_policy_episode_{episode}.pth"
        save_path = self.save_model(filename, is_checkpoint=True)
        print(f"💾 Checkpoint saved: {save_path}")
        print(f"   Current max bosses killed: {self.stats.max_bosses_killed}")
        
        # Clean up old checkpoints
        self.path_manager.cleanup_old_checkpoints(keep_last_n=3)
    
    def save_model(self, filename: str, is_checkpoint: bool = False) -> str:
        """Save model with training statistics"""
        filepath = self.path_manager.get_model_path(filename, is_checkpoint)
        
        # Get current stats
        stats_dict = self.stats.get_summary()
        
        model_config = {
            'policy_type': 'card_aware',
            'max_hand_size': self.policy.max_hand_size,
            'max_actions': self.policy.max_actions,
            'card_embed_dim': self.policy.card_embed_dim,
            'hidden_dim': self.policy.hidden_dim
        }
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'training_stats': stats_dict,
            'model_config': model_config
        }, filepath)
        
        return filepath
    
    def load_model(self, filepath: str):
        """Load model and training statistics"""
        checkpoint = torch.load(filepath)
        
        # Load model
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            if hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training statistics if available
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            self.stats.episode_rewards = stats.get('episode_rewards', [])
            self.stats.episode_lengths = stats.get('episode_lengths', [])
            self.stats.bosses_killed_history = stats.get('bosses_killed_history', [])
            self.stats.max_bosses_killed = stats.get('max_bosses_killed', 0)
            self.stats.win_rate_history = stats.get('win_rate_history', [])
    
    def plot_training_progress(self, filename: str = "training_progress.png") -> Optional[str]:
        """Generate training progress plots"""
        stats_dict = self.stats.get_summary()
        return self.visualizer.plot_training_progress(
            episode_rewards=stats_dict['episode_rewards'],
            episode_lengths=stats_dict['episode_lengths'],
            bosses_killed_history=stats_dict['bosses_killed_history'],
            win_rate_history=stats_dict['win_rate_history'],
            filename=filename
        )
    
    def create_experiment_summary(self):
        """Create experiment summary with all results"""
        stats = self.stats.get_summary()
        config = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'policy_type': 'card_aware',
            'num_players': self.env.num_players,
            'max_hand_size': self.env.max_hand_size
        }
        
        config.update({
            'card_embed_dim': self.policy.card_embed_dim,
            'hidden_dim': self.policy.hidden_dim
        })

            
        self.path_manager.create_experiment_summary(stats, config)

def main():
    """Main training script with card-aware architecture"""
    print("🎮 REGICIDE TRAINING - CARD-AWARE VERSION")
    print("=" * 50)
    
    # Training configuration
    CONFIG = {
        'num_players': 4,
        'max_hand_size': 5,
        'learning_rate': 0.0015,  # Slightly higher for AdamW
        'card_embed_dim': 12,
        'hidden_dim': 32,
        'gamma': 0.95,  # Slightly lower discount for faster learning
        'num_episodes': 50000,
        'render_every': 5000,
        'log_every': 200,
        'save_every': 10000,
        'eval_episodes': 30,
    }
    
    # Create card-aware environment
    env = RegicideGymEnv(
        num_players=CONFIG['num_players'], 
        max_hand_size=CONFIG['max_hand_size']
    )
    
    print(f"Environment: {CONFIG['num_players']} players, {CONFIG['max_hand_size']} max hand size")
    print(f"Observation space: Card-aware structured observations")
    print(f"Action space: {env.action_space.n}")
    print(f"Policy type: Card-Aware Attention")
    
    # Create trainer
    trainer = CardAwareRegicideTrainer(
        env=env,
        learning_rate=CONFIG['learning_rate'],
        card_embed_dim=CONFIG['card_embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        gamma=CONFIG['gamma'],
    )
    
    print(f"Model parameters: {sum(p.numel() for p in trainer.policy.parameters())}")
    
    # Training
    print(f"\n🎯 Training for {CONFIG['num_episodes']} episodes...")
    stats = trainer.train(
        num_episodes=CONFIG['num_episodes'],
        render_every=CONFIG['render_every'],
        save_every=CONFIG['save_every'],
        log_every=CONFIG['log_every']
    )
    
    # Save final model and create visualizations
    final_model_path = trainer.save_model("regicide_policy_final.pth")
    plot_path = trainer.plot_training_progress()
    trainer.create_experiment_summary()
    
    # Print final summary
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"Experiment: {trainer.path_manager.experiment_name}")
    print(f"Episodes: {stats['total_episodes']}")
    print(f"Final win rate: {stats['final_win_rate']:.2%}")
    print(f"Max bosses killed: {stats['max_bosses_killed']}")
    print(f"Policy type:  Card-Aware")
    print(f"Files saved to: {trainer.path_manager.experiment_dir}")
    
    # Quick test to see the model in action
    print(f"\n🎬 Running test episodes...")
    test_episodes = 3
    total_reward = 0
    total_bosses = 0
    
    for i in range(test_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_bosses = 0
        step_count = 0
        
        print(f"\n--- Test Episode {i+1} ---")
        
        while not info.get('game_over', False) and step_count < 200:
            action, _ = trainer.policy.get_action(obs)
            
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_bosses = info.get('bosses_killed', 0)
            step_count += 1
            
            if done:
                break
        
        victory = info.get('victory', False)
        print(f"Result: {'🎉 WIN' if victory else '💀 LOSS'} | "
              f"Reward: {episode_reward:.1f} | "
              f"Bosses: {episode_bosses} | "
              f"Steps: {step_count}")
        
        total_reward += episode_reward
        total_bosses += episode_bosses
    
    avg_reward = total_reward / test_episodes
    avg_bosses = total_bosses / test_episodes
    
    print(f"\n📊 Test Summary:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average bosses killed: {avg_bosses:.2f}")
    print(f"Max bosses in training: {stats['max_bosses_killed']}")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
