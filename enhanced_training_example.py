"""
Enhanced Training Example: Adding Replay Buffer and Exploration Control
This demonstrates how to extend your current REINFORCE implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List, Optional

from card_aware_policy import CardAwarePolicy
from streamlined_training import CardAwareRegicideTrainer


class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: Dict, action: int, reward: float, 
             next_state: Dict, done: bool, log_prob: float):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class ActorCriticPolicy(CardAwarePolicy):
    """Extended policy with value function for actor-critic learning"""
    
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
    
    def get_value(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Estimate state value using critic network"""
        # Reuse context encoding from actor
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 1 else 1
        device = observation['hand_cards'].device
        
        # Get context features (same as in forward pass)
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
        
        # Encode context
        hand_embeddings = self.card_embedding(hand_cards)
        hand_lengths = observation['hand_size'].float().unsqueeze(0) if observation['hand_size'].dim() == 0 else observation['hand_size'].float()
        hand_lengths = hand_lengths.unsqueeze(-1)
        
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
        return value.squeeze(-1)


class ExplorationScheduler:
    """Manages exploration parameters during training"""
    
    def __init__(self, method: str = 'entropy', initial_value: float = 1.0, 
                 final_value: float = 0.1, decay_episodes: int = 30000):
        self.method = method
        self.initial = initial_value
        self.final = final_value
        self.decay_episodes = decay_episodes
    
    def get_exploration_param(self, episode: int) -> float:
        """Get current exploration parameter value"""
        progress = min(episode / self.decay_episodes, 1.0)
        
        if self.method == 'linear':
            return self.initial + progress * (self.final - self.initial)
        elif self.method == 'exponential':
            return self.initial * (self.final / self.initial) ** progress
        elif self.method == 'cosine':
            return self.final + 0.5 * (self.initial - self.final) * \
                   (1 + np.cos(np.pi * progress))
        else:  # default to exponential
            return self.initial * (self.final / self.initial) ** progress


class EnhancedCardAwareTrainer(CardAwareRegicideTrainer):
    """Enhanced trainer with replay buffer and exploration control"""
    
    def __init__(self, env, use_replay_buffer: bool = True, 
                 exploration_method: str = 'entropy', **kwargs):
        # Initialize base trainer but replace policy with actor-critic
        super().__init__(env, **kwargs)
        
        # Replace policy with actor-critic version
        self.policy = ActorCriticPolicy(
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
        
        # Replay buffer (optional)
        self.use_replay_buffer = use_replay_buffer
        if use_replay_buffer:
            self.replay_buffer = ReplayBuffer(capacity=50000)
            self.batch_size = 32
            self.update_frequency = 4  # Update every N episodes
        
        # Exploration control
        self.exploration_scheduler = ExplorationScheduler(
            method=exploration_method,
            initial_value=1.0 if exploration_method == 'entropy' else 1.0,
            final_value=0.1 if exploration_method == 'entropy' else 0.1,
            decay_episodes=30000
        )
        self.exploration_method = exploration_method
    
    def collect_episode_with_values(self, render: bool = False):
        """Collect episode with value estimates for actor-critic"""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Get action and value estimate
            action, log_prob = self.policy.get_action(obs)
            value = self.policy.get_value(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            
            # Add to replay buffer if using
            if self.use_replay_buffer:
                self.replay_buffer.push(obs, action, reward, next_obs, 
                                      terminated or truncated, log_prob)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
            
            obs = next_obs
            info = next_info
        
        return observations, actions, log_probs, rewards, values, episode_reward, episode_length, next_info.get('bosses_killed', 0)
    
    def train_episode_enhanced(self, episode_num: int, render: bool = False) -> Dict:
        """Enhanced training with exploration control and optional replay buffer"""
        
        # Get current exploration parameter
        exploration_param = self.exploration_scheduler.get_exploration_param(episode_num)
        
        if self.use_replay_buffer and len(self.replay_buffer) > self.batch_size:
            # Use replay buffer for training
            return self.train_with_replay_buffer(exploration_param)
        else:
            # Standard on-policy training with exploration control
            return self.train_on_policy_with_exploration(exploration_param, render)
    
    def train_on_policy_with_exploration(self, exploration_param: float, render: bool = False) -> Dict:
        """On-policy training with exploration control"""
        
        # Collect episode with values
        observations, actions, log_probs, rewards, values, episode_reward, episode_length, bosses_killed = \
            self.collect_episode_with_values(render)
        
        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        returns_tensor = torch.FloatTensor(returns)
        values_tensor = torch.stack(values)
        
        # Fix tensor shapes - ensure both have same dimensions
        if values_tensor.dim() > 1:
            values_tensor = values_tensor.squeeze(-1)  # Remove extra dimension if present
        
        # Compute advantages using value function
        advantages = returns_tensor - values_tensor.detach()
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with exploration bonus
        log_probs_tensor = torch.stack(log_probs)
        policy_loss = -(log_probs_tensor * advantages).mean()
        
        # Add entropy regularization for exploration
        if self.exploration_method == 'entropy':
            # Get action probabilities for entropy calculation
            entropy_loss = 0
            for obs in observations:
                probs = self.policy.get_action_probabilities(obs)
                if probs.numel() > 1:
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    entropy_loss += entropy
            
            entropy_loss = entropy_loss / len(observations)
            policy_loss = policy_loss - exploration_param * entropy_loss
        
        # Value function loss - now both tensors have shape [episode_length]
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
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
            'exploration_param': exploration_param,
            'win_rate': self.stats.win_rate_history[-1] if self.stats.win_rate_history else 0.0
        }
    
    def train_with_replay_buffer(self, exploration_param: float) -> Dict:
        """Training using experience replay"""
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)
        
        # Convert to tensors (this would need proper batching implementation)
        # This is a simplified version - real implementation needs careful batching
        
        policy_losses = []
        value_losses = []
        
        for state, action, reward, next_state, done, old_log_prob in batch:
            # Get current policy probability and value
            current_action, current_log_prob = self.policy.get_action(state)
            current_value = self.policy.get_value(state)
            
            # Compute target value
            if done:
                target_value = reward
            else:
                next_value = self.policy.get_value(next_state)
                target_value = reward + self.gamma * next_value
            
            # Compute losses
            advantage = target_value - current_value.detach()
            
            # Importance sampling ratio for off-policy correction
            importance_ratio = torch.exp(current_log_prob - old_log_prob.detach())
            
            policy_loss = -importance_ratio * current_log_prob * advantage
            value_loss = F.mse_loss(current_value, target_value.detach())
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        
        # Average losses
        avg_policy_loss = torch.stack(policy_losses).mean()
        avg_value_loss = torch.stack(value_losses).mean()
        total_loss = avg_policy_loss + 0.5 * avg_value_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': avg_policy_loss.item(),
            'value_loss': avg_value_loss.item(),
            'exploration_param': exploration_param,
            'buffer_size': len(self.replay_buffer)
        }
    
    def train_enhanced(self, num_episodes: int, **kwargs):
        """Enhanced training loop with exploration control"""
        
        print(f"🚀 Enhanced Training: {num_episodes} episodes")
        print(f"Replay Buffer: {'Enabled' if self.use_replay_buffer else 'Disabled'}")
        print(f"Exploration Method: {self.exploration_method}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            results = self.train_episode_enhanced(episode, 
                                                render=kwargs.get('render_every', 0) > 0 and 
                                                       (episode + 1) % kwargs.get('render_every', 0) == 0)
            
            # Logging
            if (episode + 1) % kwargs.get('log_every', 200) == 0:
                print(f"Episode {episode + 1:4d}: "
                      f"Reward: {results.get('episode_reward', 0):6.2f}, "
                      f"Exploration: {results.get('exploration_param', 0):.3f}, "
                      f"Policy Loss: {results.get('policy_loss', 0):.4f}")
                
                if 'value_loss' in results:
                    print(f"                Value Loss: {results['value_loss']:.4f}")
                
                if 'buffer_size' in results:
                    print(f"                Buffer Size: {results['buffer_size']}")
        
        return self.stats.get_summary()


# Example usage
if __name__ == "__main__":
    from regicide_gym_env import RegicideGymEnv
    
    # Create environment
    env = RegicideGymEnv(num_players=4, max_hand_size=5)
    
    # Enhanced trainer with exploration control
    trainer = EnhancedCardAwareTrainer(
        env=env,
        learning_rate=0.0015,
        card_embed_dim=12,
        hidden_dim=32,
        gamma=0.95,
        use_replay_buffer=False,  # Start with on-policy
        exploration_method='entropy'
    )
    
    # Train with enhanced features
    stats = trainer.train_enhanced(
        num_episodes=10000,
        render_every=2000,
        log_every=100
    )
    
    print(f"\n🏆 Enhanced Training Complete!")
    print(f"Final win rate: {stats['final_win_rate']:.2%}")
    print(f"Max bosses killed: {stats['max_bosses_killed']}")
