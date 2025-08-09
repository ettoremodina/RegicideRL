"""
Integration Guide: Upgrading to Enhanced Policy
Progressive improvements to achieve even better performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

from enhanced_policy import EnhancedCardAwarePolicy, AdaptiveLearningRate, CurriculumLearning
from fixed_enhanced_training import FixedEnhancedCardAwareTrainer


class SuperiorActorCriticPolicy(EnhancedCardAwarePolicy):
    """Enhanced Actor-Critic with the improved card-aware features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced critic network
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Auxiliary prediction heads for better representation learning
        self.enemy_health_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.bosses_remaining_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 13)  # 0-12 bosses
        )
        
        self.apply(self._init_weights)
    
    def get_action_value_and_predictions(self, observation: Dict[str, torch.Tensor]) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Enhanced action selection with auxiliary predictions"""
        # Get basic forward pass
        logits = self.forward(observation)
        
        # Get context features for auxiliary tasks
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 1 else 1
        
        # ... (reuse context computation from forward pass)
        # This is simplified - in practice, you'd want to refactor to avoid recomputation
        
        # For now, let's use a simplified approach
        num_valid_actions = observation['num_valid_actions'].item()
        if num_valid_actions == 0:
            return 0, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), {}
        
        # Sample action
        valid_logits = logits[:, :num_valid_actions]
        probs = torch.softmax(valid_logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        # Get value (simplified - would need proper context features)
        # For integration, we'll use a dummy value
        value = torch.tensor(0.0)
        
        # Auxiliary predictions (simplified)
        aux_predictions = {
            'enemy_health': torch.tensor(0.0),
            'bosses_remaining': torch.tensor(0.0)
        }
        
        return action.item(), log_prob, value, entropy, aux_predictions


class UltimateRegicideTrainer(FixedEnhancedCardAwareTrainer):
    """Ultimate trainer with all advanced features"""
    
    def __init__(self, env, use_curriculum=True, use_adaptive_lr=True, **kwargs):
        # Initialize with enhanced policy
        super().__init__(env, **kwargs)
        
        # Replace with superior policy
        self.policy = SuperiorActorCriticPolicy(
            max_hand_size=env.max_hand_size,
            max_actions=env.max_actions,
            card_embed_dim=kwargs.get('card_embed_dim', 32),  # Larger embeddings
            hidden_dim=kwargs.get('hidden_dim', 128)  # Larger network
        )
        
        # Reinitialize optimizer with new policy
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Advanced training features
        self.use_curriculum = use_curriculum
        self.use_adaptive_lr = use_adaptive_lr
        
        if use_adaptive_lr:
            self.adaptive_lr = AdaptiveLearningRate(self.optimizer)
        
        if use_curriculum:
            self.curriculum = CurriculumLearning()
            
        # Enhanced loss coefficients
        self.aux_loss_coefficient = 0.1
        
        # Experience quality tracking
        self.episode_quality_buffer = []
    
    def train_episode_ultimate(self, episode_num: int, render: bool = False) -> Dict:
        """Ultimate training with all enhancements"""
        # Check curriculum progression
        if self.use_curriculum and episode_num % 1000 == 0:
            recent_win_rate = np.mean([self.stats.win_rate_history[-100:]]) if len(self.stats.win_rate_history) >= 100 else 0
            if self.curriculum.should_increase_difficulty(recent_win_rate):
                if self.curriculum.increase_difficulty():
                    # Would need to recreate environment with more players
                    print(f"🎓 Curriculum advancement triggered - implement environment recreation")
        
        # Standard episode collection with enhancements
        results = self.train_episode_fixed(episode_num, render)
        
        # Adaptive learning rate
        if self.use_adaptive_lr and episode_num % 100 == 0:
            recent_performance = np.mean(self.stats.bosses_killed_history[-100:]) if len(self.stats.bosses_killed_history) >= 100 else 0
            self.adaptive_lr.step(recent_performance)
        
        # Add auxiliary loss tracking
        results['aux_loss'] = 0.0  # Would be computed from auxiliary predictions
        
        return results


def create_ultimate_trainer(num_players=2, max_hand_size=7):
    """Factory function to create the ultimate trainer"""
    from regicide_gym_env import RegicideGymEnv
    
    # Advanced configuration
    CONFIG = {
        'num_players': num_players,
        'max_hand_size': max_hand_size,
        'learning_rate': 0.0005,  # Lower initial rate for stability
        'card_embed_dim': 32,     # Larger embeddings
        'hidden_dim': 128,        # Larger network
        'gamma': 0.99,
        'entropy_coefficient': 0.03,  # Slightly higher exploration
        'value_loss_coefficient': 0.5,
        'use_curriculum': True,
        'use_adaptive_lr': True
    }
    
    env = RegicideGymEnv(
        num_players=CONFIG['num_players'], 
        max_hand_size=CONFIG['max_hand_size']
    )
    
    trainer = UltimateRegicideTrainer(
        env=env,
        learning_rate=CONFIG['learning_rate'],
        card_embed_dim=CONFIG['card_embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        gamma=CONFIG['gamma'],
        entropy_coefficient=CONFIG['entropy_coefficient'],
        value_loss_coefficient=CONFIG['value_loss_coefficient'],
        use_curriculum=CONFIG['use_curriculum'],
        use_adaptive_lr=CONFIG['use_adaptive_lr'],
        experiment_name="ultimate_regicide"
    )
    
    return trainer, CONFIG


# Performance analysis and suggestions
PERFORMANCE_IMPROVEMENTS = {
    "Current Results Analysis": {
        "Strengths": [
            "100% win rate shows excellent learning",
            "Average 1.88 bosses killed is solid",
            "Max 7 bosses indicates capability for full game completion",
            "Stable learning curve in plots"
        ],
        "Areas for Enhancement": [
            "Could push average bosses killed higher (aim for 2.5+)",
            "Reduce episode length variance",
            "More consistent high-performance episodes",
            "Better handling of complex multi-player scenarios"
        ]
    },
    
    "Recommended Improvements": {
        "1. Enhanced Card Representation": [
            "Separate value/suit embeddings for better understanding",
            "Card type awareness (number/face cards)",
            "Relationship modeling between cards"
        ],
        
        "2. Strategic Reasoning": [
            "Rule-aware combo evaluation",
            "Long-term planning with value estimation",
            "Opponent modeling for multi-player games"
        ],
        
        "3. Advanced Attention": [
            "Multi-head attention for different card aspects",
            "Cross-attention between hand and enemy",
            "Temporal attention for action sequences"
        ],
        
        "4. Curriculum Learning": [
            "Progressive difficulty (2→3→4 players)",
            "Different enemy configurations",
            "Varying game lengths"
        ],
        
        "5. Auxiliary Tasks": [
            "Enemy health prediction",
            "Bosses remaining estimation",
            "Optimal card order prediction"
        ]
    },
    
    "Implementation Priority": [
        "High: Enhanced card embeddings (immediate improvement)",
        "High: Rule-aware combo features (strategic understanding)",
        "Medium: Multi-head attention (representation quality)",
        "Medium: Curriculum learning (generalization)",
        "Low: Auxiliary tasks (representation learning)"
    ]
}


if __name__ == "__main__":
    print("🚀 Ultimate Regicide Policy Improvements")
    print("=" * 60)
    
    # Print current analysis
    for category, items in PERFORMANCE_IMPROVEMENTS.items():
        print(f"\n📊 {category}:")
        if isinstance(items, dict):
            for subcategory, details in items.items():
                print(f"  {subcategory}:")
                for detail in details:
                    print(f"    • {detail}")
        else:
            for item in items:
                print(f"  • {item}")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Integrate enhanced card embeddings")
    print(f"2. Add rule-aware features")
    print(f"3. Test with curriculum learning")
    print(f"4. Monitor for >2.5 average bosses killed")
    print(f"5. Evaluate on 4-player games")
