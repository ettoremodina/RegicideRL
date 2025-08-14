"""
Proximal Policy Optimization (PPO) Training for Regicide
A more advanced and stable RL algorithm that should perform better than REINFORCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime

from .regicide_gym_env import RegicideGymEnv
from policy.card_aware_policy import CardAwarePolicy
from config import PathManager


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic network that extends CardAwarePolicy with a value function
    """
    
    def __init__(self, max_hand_size: int = 8, max_actions: int = 20, 
                 card_embed_dim: int = 12, hidden_dim: int = 128):
        super(ActorCriticPolicy, self).__init__()
        
        # Actor network (same as CardAwarePolicy)
        self.actor = CardAwarePolicy(max_hand_size, max_actions, card_embed_dim, hidden_dim)
        
        # Critic network (value function)
        # Uses same feature extraction as actor - updated for new context size
        # New context size: card_embed_dim * 2 + game_state_dim (which is now 12 from game_state_encoder + discard_pile_encoder outputs)
        game_state_dim = 12  # This should match the game_state_dim in CardAwarePolicy
        context_input_dim = card_embed_dim * 2 + game_state_dim  # hand + enemy + (game_state + discard_pile)
        
        self.critic = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),  # Updated input size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Value output
        )
        
        # Initialize critic weights
        self.critic.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def get_features(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract shared features for both actor and critic"""
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 1 else 1
        device = observation['hand_cards'].device
        
        # Ensure batch dimension for all tensors
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
        if enemy_card.dim() == 0:
            enemy_card = enemy_card.unsqueeze(0)
        if enemy_card.dim() == 1:
            enemy_card = enemy_card.unsqueeze(0)
        
        # Use actor's feature extraction (up to combined_context)
        hand_embeddings = self.actor.card_embedding(hand_cards)
        # hand_mask = hand_cards != 0
        
        # if hand_mask.any():
        #     attended_hand, _ = self.actor.hand_attention(
        #         hand_embeddings, hand_embeddings, hand_embeddings,
        #         key_padding_mask=~hand_mask
        #     )
        # else:
        attended_hand = hand_embeddings
        
        hand_lengths = observation['hand_size'].float().unsqueeze(0) if observation['hand_size'].dim() == 0 else observation['hand_size'].float()
        hand_lengths = hand_lengths.unsqueeze(-1)
        
        if (hand_lengths == 0).any():
            hand_context = torch.zeros(batch_size, self.actor.card_embed_dim, device=device)
        else:
            hand_lengths = torch.clamp(hand_lengths, min=1.0)
            hand_context = attended_hand.sum(dim=1) / hand_lengths
        
        enemy_embedding = self.actor.enemy_embedding(enemy_card.squeeze(-1))
        game_context = self.actor.game_state_encoder(game_state)
        discard_context = self.actor.discard_pile_encoder(discard_pile)
        
        # Combined context (same as actor) - now includes discard pile
        combined_context = torch.cat([hand_context, enemy_embedding, game_context, discard_context], dim=-1)
        return combined_context
    
    def get_action_and_value(self, observation: Dict[str, torch.Tensor]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value estimate"""
        # Get action from actor
        action, log_prob = self.actor.get_action(observation)
        
        # Get value from critic
        features = self.get_features(observation)
        value = self.critic(features).squeeze()
        
        return action, log_prob, value
    
    def get_value(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimate"""
        features = self.get_features(observation)
        return self.critic(features).squeeze()
    
    def evaluate_actions(self, observations: List[Dict[str, torch.Tensor]], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update"""
        batch_size = len(observations)
        log_probs = []
        values = []
        entropies = []
        
        for i, obs in enumerate(observations):
            # Get action probabilities
            action_probs = self.actor.get_action_probabilities(obs)
            
            # Handle different tensor shapes
            if action_probs.dim() == 0:  # 0-d tensor
                # Single action case
                action_idx = actions[i].item()
                if action_idx == 0:
                    log_prob = torch.log(action_probs + 1e-8)
                else:
                    log_prob = torch.tensor(-1e8)  # Invalid action
            else:
                # Multi-dimensional tensor
                action_probs_flat = action_probs.squeeze()
                action_idx = actions[i].item()
                
                if action_probs_flat.dim() == 0:  # Still 0-d after squeeze
                    if action_idx == 0:
                        log_prob = torch.log(action_probs_flat + 1e-8)
                    else:
                        log_prob = torch.tensor(-1e8)
                else:
                    # Normal case with multiple actions
                    if action_idx < len(action_probs_flat):
                        log_prob = torch.log(action_probs_flat[action_idx] + 1e-8)
                    else:
                        log_prob = torch.tensor(-1e8)  # Invalid action
            
            log_probs.append(log_prob)
            
            # Get entropy - handle 0-d case
            if action_probs.dim() == 0:
                entropy = torch.tensor(0.0)  # No entropy for single action
            else:
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
            entropies.append(entropy)
            
            # Get value
            value = self.get_value(obs)
            values.append(value)
        
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


class PPOBuffer:
    """
    Buffer for storing PPO training data
    """
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (self.observations, 
                torch.tensor(self.actions, dtype=torch.long),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.stack(self.values),
                torch.stack(self.log_probs),
                torch.tensor(self.dones, dtype=torch.bool))
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class PPOTrainer:
    """
    PPO Training Class
    """
    
    def __init__(self, env: RegicideGymEnv, policy: ActorCriticPolicy, 
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
                 ppo_epochs: int = 4, mini_batch_size: int = 64):
        
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=50, factor=0.8
        )
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.bosses_killed_history = deque(maxlen=100)
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        # Convert dones to float for arithmetic operations
        dones_float = dones.float()
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones_float[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones_float[step + 1]
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Detach from computational graph to avoid gradient warnings
            advantages.insert(0, gae.detach())
            returns.insert(0, (gae + values[step]).detach())
        
        return torch.stack(advantages), torch.stack(returns)
    
    def ppo_update(self, buffer: PPOBuffer):
        """Perform PPO update"""
        observations, actions, rewards, values, old_log_probs, dones = buffer.get()
        
        # Compute next value (0 if episode ended)
        if dones[-1]:
            next_value = 0.0
        else:
            next_value = values[-1].item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages and detach from graph
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()  # Ensure no gradients
        returns = returns.detach()  # Ensure no gradients
        old_log_probs = old_log_probs.detach()  # Ensure no gradients
        
        # PPO updates
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate current policy
                log_probs, values_pred, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Store losses for logging
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropy_losses.append(entropy_loss.item())
    
    def collect_trajectory(self, max_steps: int = 200) -> PPOBuffer:
        """Collect a trajectory using current policy"""
        buffer = PPOBuffer()
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Get action and value
            action, log_prob, value = self.policy.get_action_and_value(obs)
            
            # Take step
            next_obs, reward, done, _, next_info = self.env.step(action)
            
            # Store in buffer
            buffer.add(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
            info = next_info
            
            if done:
                break
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.bosses_killed_history.append(info.get('bosses_killed', 0))
        self.episode_count += 1
        
        return buffer
    
    def train(self, total_episodes: int = 1000, update_frequency: int = 10, 
              save_frequency: int = 100, experiment_name: str = "ppo_regicide"):
        """Main training loop"""
        
        # Setup experiment directory
        path_manager = PathManager(experiment_name)
        experiment_dir = str(path_manager.experiment_dir)
        
        print(f"🚀 Starting PPO Training for Regicide")
        print(f"📁 Experiment directory: {experiment_dir}")
        print(f"🎯 Target episodes: {total_episodes}")
        print("=" * 60)
        
        best_performance = 0
        
        for episode in range(total_episodes):
            # Collect trajectory
            buffer = self.collect_trajectory()
            
            # Update policy every update_frequency episodes
            if (episode + 1) % update_frequency == 0:
                self.ppo_update(buffer)
            
            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:])
                avg_bosses = np.mean(list(self.bosses_killed_history)[-10:])
                avg_length = np.mean(list(self.episode_lengths)[-10:])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Bosses: {avg_bosses:4.2f} | "
                      f"Avg Length: {avg_length:5.1f}")
                
                # Update learning rate
                self.scheduler.step(avg_bosses)
            
            # Save model
            if (episode + 1) % save_frequency == 0:
                current_performance = np.mean(list(self.bosses_killed_history)[-50:])
                
                self.save_checkpoint(experiment_dir, episode + 1, current_performance > best_performance)
                
                if current_performance > best_performance:
                    best_performance = current_performance
                
                # Plot progress
                self.plot_training_progress(experiment_dir)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best average bosses killed: {best_performance:.2f}")
        
        # Final save
        self.save_checkpoint(experiment_dir, total_episodes, is_best=True)
        self.plot_training_progress(experiment_dir)
    
    def save_checkpoint(self, experiment_dir: str, episode: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode': episode,
            'total_steps': self.total_steps,
            'training_stats': {
                'episode_rewards': list(self.episode_rewards),
                'episode_lengths': list(self.episode_lengths),
                'bosses_killed_history': list(self.bosses_killed_history),
                'actor_losses': self.actor_losses[-100:],  # Last 100
                'critic_losses': self.critic_losses[-100:],
                'entropy_losses': self.entropy_losses[-100:],
                'total_episodes': episode,
                'max_bosses_killed': max(self.bosses_killed_history) if self.bosses_killed_history else 0
            },
            'model_config': {
                'max_hand_size': self.policy.actor.max_hand_size,
                'max_actions': self.policy.actor.max_actions,
                'card_embed_dim': self.policy.actor.card_embed_dim,
                'hidden_dim': self.policy.actor.hidden_dim
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"ppo_policy_episode_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(experiment_dir, "models", "ppo_policy_best.pth")
            torch.save(checkpoint, best_path)
    
    def plot_training_progress(self, experiment_dir: str):
        """Plot and save training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Bosses killed
        axes[0, 1].plot(self.bosses_killed_history)
        axes[0, 1].set_title('Bosses Killed per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Bosses Killed')
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Losses
        if self.actor_losses:
            axes[1, 1].plot(self.actor_losses, label='Actor Loss', alpha=0.7)
            axes[1, 1].plot(self.critic_losses, label='Critic Loss', alpha=0.7)
            axes[1, 1].plot(self.entropy_losses, label='Entropy Loss', alpha=0.7)
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "plots", "ppo_training_progress.png"), dpi=300)
        plt.close()


def main():
    """Main training function"""
    # Create environment
    env = RegicideGymEnv(
        num_players=4,
        max_hand_size=5,
        observation_mode="card_aware"
    )
    
    # Create actor-critic policy
    policy = ActorCriticPolicy(
        max_hand_size=5,
        max_actions=30,
        card_embed_dim=12,  # Updated to match CardAwarePolicy
        hidden_dim=128
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        mini_batch_size=32
    )
    
    # Train
    trainer.train(
        total_episodes=20000,
        update_frequency=200,
        save_frequency=5000,
        experiment_name="ppo_regicide_enhanced"
    )


if __name__ == "__main__":
    main()
