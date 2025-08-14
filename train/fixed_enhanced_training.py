"""
Fixed Enhanced Training: Addressing entropy and critic loss issues
Critical fixes:
1. Proper entropy calculation at action time
2. Corrected critic loss computation with bootstrapping
3. Improved advantage calculation
4. Better exploration handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List, Optional

from policy.card_aware_policy import CardAwarePolicy
from streamlined_training import CardAwareRegicideTrainer
from training_utils import TrainingVisualizer, TrainingLogger, TrainingStatistics
from config import PathManager


class FixedActorCriticPolicy(CardAwarePolicy):
    """Fixed actor-critic policy with proper value function integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add critic network for value estimation
        context_dim = self.hidden_dim  # From context encoder output
        self.critic = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize critic weights
        self.critic.apply(self._init_weights)
    
    def get_action_and_value(self, observation: Dict[str, torch.Tensor]) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, value, and entropy in one forward pass"""
        # Ensure batch dimension
        if observation['hand_cards'].dim() == 1:
            for key, value in observation.items():
                if isinstance(value, torch.Tensor):
                    observation[key] = value.unsqueeze(0)
        
        # Forward pass through policy
        logits = self.forward(observation)  # [1, max_actions]
        
        # Get valid actions
        num_valid_actions = observation['num_valid_actions'].item()
        if num_valid_actions == 0:
            # Handle edge case - return dummy values
            return 0, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        # Mask invalid actions and compute probabilities
        valid_logits = logits[:, :num_valid_actions]
        probs = F.softmax(valid_logits, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Compute entropy directly from the distribution used for sampling
        entropy = action_dist.entropy()
        
        # Ensure tensors have consistent shapes (convert to scalars)
        if log_prob.dim() > 0:
            log_prob = log_prob.squeeze()
        if entropy.dim() > 0:
            entropy = entropy.squeeze()
        
        # Get value estimate using same context features
        # Reuse context computation from forward pass to ensure consistency
        batch_size = observation['hand_cards'].size(0)
        device = observation['hand_cards'].device
        
        # Get context features (duplicated from forward pass for value estimation)
        hand_cards = observation['hand_cards']
        if hand_cards.dim() == 1:
            hand_cards = hand_cards.unsqueeze(0)
        
        game_state = observation['game_state']
        if game_state.dim() == 1:
            game_state = game_state.unsqueeze(0)

        discard_pile = observation['discard_pile_cards']
        if discard_pile.dim() == 1:
            discard_pile = discard_pile.unsqueeze(0)
        
        enemy_card = observation['enemy_card']
        if enemy_card.dim() == 1:
            enemy_card = enemy_card.unsqueeze(0)
        
        # Encode context
        hand_embeddings = self.card_embedding(hand_cards)
        hand_lengths = observation['hand_size'].float().unsqueeze(-1)
        
        if (hand_lengths == 0).any():
            hand_context = torch.zeros(batch_size, self.card_embed_dim, device=device)
        else:
            hand_lengths = torch.clamp(hand_lengths, min=1.0)
            hand_context = hand_embeddings.sum(dim=1) / hand_lengths
        
        enemy_embedding = self.enemy_embedding(enemy_card.squeeze(-1))
        game_context = self.game_state_encoder(game_state)
        discard_context = self.discard_pile_encoder(discard_pile)
        
        combined_context = torch.cat([hand_context, enemy_embedding, game_context, discard_context], dim=-1)
        context_features = self.context_encoder(combined_context)
        
        # Get value estimate
        value = self.critic(context_features)
        if value.dim() > 0:
            value = value.squeeze()
        
        return action.item(), log_prob, value, entropy
    
    def get_value(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimate only"""
        _, _, value, _ = self.get_action_and_value(observation)
        return value


class FixedEnhancedCardAwareTrainer(CardAwareRegicideTrainer):
    """Fixed enhanced trainer addressing entropy and critic loss issues"""
    
    def __init__(self, env, entropy_coefficient: float = 0.02, 
                 value_loss_coefficient: float = 0.5, experiment_name: str = None, **kwargs):
        # Initialize base trainer but replace policy with fixed actor-critic
        super().__init__(env, **kwargs)
        
        # Initialize path manager and utilities for plotting and logging
        self.path_manager = PathManager(experiment_name or "fixed_enhanced_training")
        self.visualizer = TrainingVisualizer(self.path_manager)
        self.logger = TrainingLogger(self.path_manager)
        
        # Replace policy with fixed actor-critic version
        self.policy = FixedActorCriticPolicy(
            max_hand_size=env.max_hand_size,
            max_actions=env.max_actions,
            card_embed_dim=kwargs.get('card_embed_dim', 12),
            hidden_dim=kwargs.get('hidden_dim', 32)
        )
        
        # Reinitialize optimizer for new policy
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Loss coefficients
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        
        # Exploration schedule
        self.initial_entropy_coeff = entropy_coefficient
        self.final_entropy_coeff = entropy_coefficient * 0.1
        self.entropy_decay_episodes = 30000
    
    def get_entropy_coefficient(self, episode: int) -> float:
        """Get current entropy coefficient with decay"""
        progress = min(episode / self.entropy_decay_episodes, 1.0)
        return self.initial_entropy_coeff * (self.final_entropy_coeff / self.initial_entropy_coeff) ** progress
    
    def collect_episode_with_values_fixed(self, render: bool = False):
        """Collect episode with proper value estimates and entropy tracking"""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        entropies = []
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Get action, log prob, value, and entropy in one forward pass
            action, log_prob, value, entropy = self.policy.get_action_and_value(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            entropies.append(entropy)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                # Get final state value for bootstrapping if not terminal
                if not terminated and not next_info.get('game_over', False):
                    final_value = self.policy.get_value(next_obs)
                else:
                    final_value = torch.tensor(0.0)
                break
            
            obs = next_obs
            info = next_info
        
        return (observations, actions, log_probs, rewards, values, entropies, 
                final_value, episode_reward, episode_length, next_info.get('bosses_killed', 0))
    
    def compute_gae_returns(self, rewards: List[float], values: List[torch.Tensor], 
                           final_value: torch.Tensor, gamma: float = 0.99, 
                           lambda_gae: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE"""
        values_list = [v.item() for v in values] + [final_value.item()]
        returns = []
        advantages = []
        
        # Compute GAE advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_list[t + 1] - values_list[t]
            gae = delta + gamma * lambda_gae * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_list[t])
        
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return returns_tensor, advantages_tensor
    
    def train_episode_fixed(self, episode_num: int, render: bool = False) -> Dict:
        """Fixed training with proper entropy and critic loss computation"""
        
        # Collect episode with fixed value and entropy collection
        (observations, actions, log_probs, rewards, values, entropies, final_value,
         episode_reward, episode_length, bosses_killed) = self.collect_episode_with_values_fixed(render)
        
        if len(observations) == 0:
            return {'episode_reward': 0, 'episode_length': 0, 'bosses_killed': 0}
        
        # Compute returns and advantages using GAE
        returns, advantages = self.compute_gae_returns(rewards, values, final_value)
        
        # Convert to tensors with shape checking
        if len(log_probs) == 0:
            return {'episode_reward': 0, 'episode_length': 0, 'bosses_killed': 0}
        
        # Ensure all tensors are scalars for consistent stacking
        log_probs_tensor = torch.stack([lp.squeeze() if lp.dim() > 0 else lp for lp in log_probs])
        values_tensor = torch.stack([v.squeeze() if v.dim() > 0 else v for v in values])
        entropies_tensor = torch.stack([e.squeeze() if e.dim() > 0 else e for e in entropies])
        
        # Policy loss with advantages
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        
        # Value loss with proper targets
        value_loss = F.mse_loss(values_tensor, returns.detach())
        
        # Entropy loss for exploration
        entropy_loss = entropies_tensor.mean()
        current_entropy_coeff = self.get_entropy_coefficient(episode_num)
        
        # Combined loss
        total_loss = (policy_loss + 
                     self.value_loss_coefficient * value_loss - 
                     current_entropy_coeff * entropy_loss)
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Update statistics
        self.stats.update(episode_reward, episode_length, bosses_killed)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'bosses_killed': bosses_killed,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy_coefficient': current_entropy_coeff,
            'avg_entropy': entropy_loss.item(),
            'avg_value': values_tensor.mean().item(),
            'win_rate': self.stats.win_rate_history[-1] if self.stats.win_rate_history else 0.0
        }
    
    def train_fixed(self, num_episodes: int, render_every: int = 0, 
                   save_every: int = 100, log_every: int = 10):
        """Fixed training loop with plotting and checkpoint management"""
        print(f"🚀 Starting FIXED enhanced training for {num_episodes} episodes")
        print(f"Experiment: {self.path_manager.experiment_name}")
        print(f"Entropy coefficient: {self.entropy_coefficient} → {self.final_entropy_coeff}")
        print(f"Value loss coefficient: {self.value_loss_coefficient}")
        print("=" * 60)
        
        # Log experiment configuration
        config = {
            'num_episodes': num_episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'policy_type': 'fixed_actor_critic',
            'num_players': self.env.num_players,
            'max_hand_size': self.env.max_hand_size,
            'card_embed_dim': self.policy.card_embed_dim,
            'hidden_dim': self.policy.hidden_dim,
            'entropy_coefficient': self.entropy_coefficient,
            'value_loss_coefficient': self.value_loss_coefficient
        }
        self.logger.log_experiment_config(config)
        
        for episode in range(num_episodes):
            # Train one episode with fixes
            render = render_every > 0 and (episode + 1) % render_every == 0
            results = self.train_episode_fixed(episode, render)
            
            # Log progress
            if (episode + 1) % log_every == 0:
                self.logger.log_progress(episode + 1, results)
                self._print_fixed_progress(episode + 1, results)
            
            # Save model periodically with cleanup
            if save_every > 0 and (episode + 1) % save_every == 0:
                self._save_checkpoint_with_cleanup(episode + 1)
        
        # Generate final plots and save final model
        self._create_final_results(num_episodes)
        
        print("\n🏁 Fixed training completed!")
        self.logger.close()
        return self.stats.get_summary()
    
    def _print_fixed_progress(self, episode: int, results: Dict):
        """Print progress for fixed training"""
        recent_stats = self.stats.get_recent_averages()
        print(f"Episode {episode:4d}: "
              f"R: {recent_stats['avg_reward']:6.2f}, "
              f"L: {recent_stats['avg_length']:5.1f}, "
              f"B: {recent_stats['avg_bosses']:4.1f}, "
              f"Win: {recent_stats['win_rate']:.2%}, "
              f"PL: {results.get('policy_loss', 0):.3f}, "
              f"VL: {results.get('value_loss', 0):.3f}, "
              f"E: {results.get('avg_entropy', 0):.3f}, "
              f"V: {results.get('avg_value', 0):.2f}")
    
    def _save_checkpoint_with_cleanup(self, episode: int):
        """Save model checkpoint and clean up old ones"""
        filename = f"regicide_fixed_enhanced_episode_{episode}.pth"
        save_path = self.save_model(filename, is_checkpoint=True)
        print(f"💾 Checkpoint saved: {save_path}")
        print(f"   Current max bosses killed: {self.stats.max_bosses_killed}")
        
        # Clean up old checkpoints (keep only last 3)
        self.path_manager.cleanup_old_checkpoints(keep_last_n=3)
    
    def save_model(self, filename: str, is_checkpoint: bool = False) -> str:
        """Save model with training statistics"""
        filepath = self.path_manager.get_model_path(filename, is_checkpoint)
        
        # Get current stats
        stats_dict = self.stats.get_summary()
        
        model_config = {
            'policy_type': 'fixed_actor_critic',
            'max_hand_size': self.policy.max_hand_size,
            'max_actions': self.policy.max_actions,
            'card_embed_dim': self.policy.card_embed_dim,
            'hidden_dim': self.policy.hidden_dim,
            'entropy_coefficient': self.entropy_coefficient,
            'value_loss_coefficient': self.value_loss_coefficient
        }
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': stats_dict,
            'model_config': model_config
        }, filepath)
        
        return filepath
    
    def _create_final_results(self, num_episodes: int):
        """Create final plots and experiment summary"""
        # Generate training progress plots
        stats_dict = self.stats.get_summary()
        plot_path = self.visualizer.plot_training_progress(
            episode_rewards=stats_dict['episode_rewards'],
            episode_lengths=stats_dict['episode_lengths'],
            bosses_killed_history=stats_dict['bosses_killed_history'],
            win_rate_history=stats_dict['win_rate_history'],
            filename="fixed_enhanced_training_progress.png"
        )
        
        # Save final model
        final_model_path = self.save_model("regicide_fixed_enhanced_final.pth")
        
        # Create experiment summary
        config = {
            'num_episodes': num_episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'policy_type': 'fixed_actor_critic',
            'entropy_coefficient': self.entropy_coefficient,
            'value_loss_coefficient': self.value_loss_coefficient
        }
        self.path_manager.create_experiment_summary(stats_dict, config)
        
        print(f"\n📈 Training plots saved: {plot_path}")
        print(f"💾 Final model saved: {final_model_path}")
        print(f"📂 All files saved to: {self.path_manager.experiment_dir}")


def test_fixed_training():
    """Test function for fixed enhanced training with plotting"""
    print("🧪 Testing Fixed Enhanced Training with Plotting")
    print("=" * 50)
    
    # Import environment
    from regicide_gym_env import RegicideGymEnv
    
    # Small test configuration
    CONFIG = {
        'num_players': 2,
        'max_hand_size': 7,
        'learning_rate': 0.005,
        'card_embed_dim': 16,
        'hidden_dim': 32,
        'gamma': 0.99,
        'num_episodes': 10,  # Very short test
        'render_every': 0,
        'log_every': 2,
        'save_every': 5,
        'entropy_coefficient': 0.02,
        'value_loss_coefficient': 0.5
    }
    
    # Create environment
    env = RegicideGymEnv(
        num_players=CONFIG['num_players'], 
        max_hand_size=CONFIG['max_hand_size']
    )
    
    # Create trainer
    trainer = FixedEnhancedCardAwareTrainer(
        env=env,
        learning_rate=CONFIG['learning_rate'],
        card_embed_dim=CONFIG['card_embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        gamma=CONFIG['gamma'],
        entropy_coefficient=CONFIG['entropy_coefficient'],
        value_loss_coefficient=CONFIG['value_loss_coefficient'],
        experiment_name="test_fixed_enhanced"
    )
    
    print(f"✅ Trainer created successfully")
    print(f"✅ Experiment directory: {trainer.path_manager.experiment_dir}")
    
    # Run short training
    print(f"\n🎯 Running test training for {CONFIG['num_episodes']} episodes...")
    stats = trainer.train_fixed(
        num_episodes=CONFIG['num_episodes'],
        render_every=CONFIG['render_every'],
        save_every=CONFIG['save_every'],
        log_every=CONFIG['log_every']
    )
    
    print(f"\n✅ Test completed successfully!")
    print(f"Episodes: {stats['total_episodes']}")
    print(f"Max bosses killed: {stats['max_bosses_killed']}")
    print(f"Files saved to: {trainer.path_manager.experiment_dir}")
    
    return True


def main_fixed():
    """Main function to test fixed enhanced training"""
    print("🎮 REGICIDE TRAINING - FIXED ENHANCED VERSION")
    print("=" * 50)
    
    # Import environment
    from regicide_gym_env import RegicideGymEnv
    
    # Training configuration
    CONFIG = {
        'num_players': 2,  # Start with 2 players for faster learning
        'max_hand_size': 7,
        'learning_rate': 0.001,
        'card_embed_dim': 16,
        'hidden_dim': 64,
        'gamma': 0.99,
        'num_episodes': 50000,
        'render_every': 5000,
        'log_every': 100,
        'save_every': 5000,
        'entropy_coefficient': 0.02,
        'value_loss_coefficient': 0.5
    }
    
    # Create environment
    env = RegicideGymEnv(
        num_players=CONFIG['num_players'], 
        max_hand_size=CONFIG['max_hand_size']
    )
    
    print(f"Environment: {CONFIG['num_players']} players, {CONFIG['max_hand_size']} max hand size")
    print(f"Action space: {env.action_space.n}")
    print(f"Policy type: Fixed Actor-Critic with proper entropy and value function")
    
    # Create fixed trainer with experiment name
    trainer = FixedEnhancedCardAwareTrainer(
        env=env,
        learning_rate=CONFIG['learning_rate'],
        card_embed_dim=CONFIG['card_embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        gamma=CONFIG['gamma'],
        entropy_coefficient=CONFIG['entropy_coefficient'],
        value_loss_coefficient=CONFIG['value_loss_coefficient'],
        experiment_name="fixed_enhanced_regicide"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in trainer.policy.parameters())}")
    
    # Training
    print(f"\n🎯 Training for {CONFIG['num_episodes']} episodes...")
    stats = trainer.train_fixed(
        num_episodes=CONFIG['num_episodes'],
        render_every=CONFIG['render_every'],
        save_every=CONFIG['save_every'],
        log_every=CONFIG['log_every']
    )
    
    print(f"\n🏆 FIXED TRAINING COMPLETE!")
    print(f"Final win rate: {stats['final_win_rate']:.2%}")
    print(f"Max bosses killed: {stats['max_bosses_killed']}")
    print(f"Experiment directory: {trainer.path_manager.experiment_dir}")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Allow choosing between test and full training
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_fixed_training()
    else:
        main_fixed()
