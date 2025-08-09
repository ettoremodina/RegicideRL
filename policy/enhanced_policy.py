"""
Enhanced Card-Aware Policy with Advanced Features
Improvements for better Regicide performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class EnhancedCardAwarePolicy(nn.Module):
    """
    Enhanced card-aware policy with advanced features for better Regicide performance
    """
    
    def __init__(self, max_hand_size: int = 8, max_actions: int = 30, 
                 card_embed_dim: int = 32, hidden_dim: int = 128):
        super(EnhancedCardAwarePolicy, self).__init__()
        
        self.max_hand_size = max_hand_size
        self.max_actions = max_actions
        self.card_embed_dim = card_embed_dim
        self.hidden_dim = hidden_dim
        
        # 1. IMPROVED CARD EMBEDDINGS with suit/value decomposition
        self.value_embedding = nn.Embedding(14, card_embed_dim // 2, padding_idx=0)  # 0-13 (0=pad, 1-13=values)
        self.suit_embedding = nn.Embedding(5, card_embed_dim // 2, padding_idx=0)   # 0-4 (0=pad, 1-4=suits)
        self.card_type_embedding = nn.Embedding(4, 8)  # Number, Jack, Queen, King
        
        # 2. RULE-AWARE FEATURES
        self.combo_encoder = nn.Sequential(
            nn.Linear(6, 32),  # [combo_size, same_value, total_attack, suits_present, ace_combo, valid_combo]
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 3. ENHANCED ATTENTION with multiple heads for different aspects
        self.hand_attention = nn.MultiheadAttention(
            embed_dim=card_embed_dim, 
            num_heads=8,  # More heads for richer representations
            batch_first=True,
            dropout=0.1
        )
        
        # Cross-attention between hand and enemy
        self.hand_enemy_attention = nn.MultiheadAttention(
            embed_dim=card_embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 4. STRATEGIC CONTEXT ENCODER
        self.strategic_encoder = nn.Sequential(
            nn.Linear(20, 64),  # Enhanced game state: [basic_state + strategic_features]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 5. HIERARCHICAL ACTION EVALUATION
        # First decide action type (attack/yield/specific_strategy)
        self.action_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [aggressive_attack, conservative_attack, strategic_setup, yield]
        )
        
        # Then score specific actions within chosen type
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + card_embed_dim + 16, 64),  # +16 for combo features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 6. CONTEXT INTEGRATION
        context_dim = card_embed_dim * 2 + 32 + 54  # hand + enemy + strategic + discard
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 7. VALUE ESTIMATION COMPONENTS (for the critic)
        self.value_features = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Better weight initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
            if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                torch.nn.init.constant_(m.weight[m.padding_idx], 0)
    
    def _encode_cards_enhanced(self, cards: torch.Tensor) -> torch.Tensor:
        """Enhanced card encoding with value/suit decomposition"""
        # Extract values and suits from card indices
        # Card index encoding: suit * 13 + value (1-indexed)
        values = ((cards - 1) % 13) + 1  # 1-13
        suits = ((cards - 1) // 13) + 1  # 1-4
        
        # Handle padding (0) and jester (53)
        values = torch.where(cards == 0, 0, values)  # Padding
        values = torch.where(cards == 53, 0, values)  # Jester (special case)
        suits = torch.where(cards == 0, 0, suits)
        suits = torch.where(cards == 53, 0, suits)
        
        # Get embeddings
        value_emb = self.value_embedding(values)
        suit_emb = self.suit_embedding(suits)
        
        # Combine value and suit embeddings
        card_emb = torch.cat([value_emb, suit_emb], dim=-1)
        
        # Add card type information
        card_types = torch.zeros_like(values)
        card_types = torch.where((values >= 2) & (values <= 10), 0, card_types)  # Number
        card_types = torch.where(values == 11, 1, card_types)  # Jack
        card_types = torch.where(values == 12, 2, card_types)  # Queen  
        card_types = torch.where(values == 13, 3, card_types)  # King
        
        type_emb = self.card_type_embedding(card_types)  # [batch, seq, 8]
        
        # Pad type embedding to match card embedding dimension
        type_emb_padded = torch.zeros_like(card_emb)
        type_emb_padded[..., :type_emb.size(-1)] = type_emb
        
        return card_emb + type_emb_padded
    
    def _compute_combo_features(self, action_cards: torch.Tensor) -> torch.Tensor:
        """Compute rule-aware combo features for an action"""
        device = action_cards.device if action_cards.numel() > 0 else torch.device('cpu')
        
        if action_cards.numel() == 0:
            return torch.zeros(6, device=device)
        
        # Basic combo statistics
        combo_size = (action_cards != 0).sum().float()
        
        # Extract values for analysis
        values = ((action_cards - 1) % 13) + 1
        values = torch.where(action_cards == 0, 0, values)
        valid_values = values[values != 0]
        
        if valid_values.numel() == 0:
            return torch.zeros(6, device=device)
        
        # Same value check
        same_value = (valid_values == valid_values[0]).all().float()
        
        # Total attack calculation (simplified)
        attack_values = torch.where(valid_values == 1, 1,  # Ace
                        torch.where(valid_values == 11, 10,  # Jack
                        torch.where(valid_values == 12, 15,  # Queen
                        torch.where(valid_values == 13, 20,  # King
                        valid_values.float()))))  # Numbers
        total_attack = attack_values.sum()
        
        # Suits present
        suits = ((action_cards - 1) // 13) + 1
        suits = torch.where(action_cards == 0, 0, suits)
        unique_suits = len(torch.unique(suits[suits != 0]))
        
        # Ace combo detection
        ace_combo = (valid_values == 1).any().float()
        
        # Valid combo (simplified rule check)
        valid_combo = 1.0 if combo_size <= 4 and (same_value or ace_combo) else 0.0
        
        return torch.tensor([combo_size, same_value, total_attack, unique_suits, ace_combo, valid_combo], device=device)
    
    def _compute_strategic_features(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute strategic game state features"""
        game_state = observation['game_state']
        device = game_state.device
        
        # Extract key strategic information
        hand_size = observation['hand_size'].float()
        
        # Compute hand composition features
        hand_cards = observation['hand_cards']
        values = ((hand_cards - 1) % 13) + 1
        values = torch.where(hand_cards == 0, 0, values)
        
        # Count important cards
        aces_count = (values == 1).sum().float()
        faces_count = ((values >= 11) & (values <= 13)).sum().float()
        low_cards = ((values >= 2) & (values <= 6)).sum().float()
        
        # Suit distribution
        suits = ((hand_cards - 1) // 13) + 1
        suits = torch.where(hand_cards == 0, 0, suits)
        hearts = (suits == 1).sum().float()
        diamonds = (suits == 2).sum().float()
        clubs = (suits == 3).sum().float()
        spades = (suits == 4).sum().float()
        
        # Strategic ratios
        high_value_ratio = faces_count / (hand_size + 1e-8)
        suit_diversity = len(torch.unique(suits[suits != 0])) / 4.0
        
        # Combine basic game state with strategic features
        strategic_features = torch.tensor([
            hand_size, aces_count, faces_count, low_cards,
            hearts, diamonds, clubs, spades,
            high_value_ratio, suit_diversity
        ], device=device)
        
        return torch.cat([game_state, strategic_features.unsqueeze(0)])
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced forward pass with strategic reasoning"""
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 1 else 1
        device = observation['hand_cards'].device
        
        # Ensure batch dimensions
        hand_cards = observation['hand_cards']
        if hand_cards.dim() == 1:
            hand_cards = hand_cards.unsqueeze(0)
        
        enemy_card = observation['enemy_card']
        if enemy_card.dim() == 0:
            enemy_card = enemy_card.unsqueeze(0).unsqueeze(0)  # [1, 1]
        elif enemy_card.dim() == 1:
            enemy_card = enemy_card.unsqueeze(1)  # [batch, 1]
        
        # Enhanced card encoding
        hand_embeddings = self._encode_cards_enhanced(hand_cards)
        enemy_embedding = self._encode_cards_enhanced(enemy_card)
        
        # Enhanced attention
        hand_mask = hand_cards != 0
        if hand_mask.any():
            # Self-attention within hand
            attended_hand, _ = self.hand_attention(
                hand_embeddings, hand_embeddings, hand_embeddings,
                key_padding_mask=~hand_mask
            )
            
            # Cross-attention with enemy
            hand_enemy_context, _ = self.hand_enemy_attention(
                attended_hand, enemy_embedding, enemy_embedding
            )
        else:
            hand_enemy_context = hand_embeddings
        
        # Aggregate hand representation
        hand_lengths = observation['hand_size'].float().unsqueeze(-1)
        if hand_lengths.dim() == 1:
            hand_lengths = hand_lengths.unsqueeze(0)
        
        hand_lengths = torch.clamp(hand_lengths, min=1.0)
        hand_context = hand_enemy_context.sum(dim=1) / hand_lengths
        
        # Strategic context
        strategic_state = self._compute_strategic_features(observation)
        strategic_context = self.strategic_encoder(strategic_state.unsqueeze(0))
        
        # Discard pile context (simplified)
        discard_context = observation['discard_pile_cards'].float()
        if discard_context.dim() == 1:
            discard_context = discard_context.unsqueeze(0)
        
        # Combine all contexts
        combined_context = torch.cat([
            hand_context, 
            enemy_embedding.squeeze(1), 
            strategic_context, 
            discard_context
        ], dim=-1)
        
        context_features = self.context_encoder(combined_context)
        
        # Action type classification (strategic prior)
        action_type_logits = self.action_type_classifier(context_features)
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        
        # Score individual actions
        num_valid_actions = observation['num_valid_actions'].item()
        action_scores = []
        
        for action_idx in range(num_valid_actions):
            card_indices = observation['action_card_indices'][action_idx] if action_idx < len(observation['action_card_indices']) else []
            
            if not card_indices:  # Yield action
                action_card_repr = torch.zeros(batch_size, self.card_embed_dim, device=device)
                combo_features = torch.zeros(batch_size, 6, device=device)
            else:
                # Enhanced action representation
                card_tensors = torch.tensor(card_indices, dtype=torch.long, device=device)
                action_card_embeddings = self._encode_cards_enhanced(card_tensors.unsqueeze(0))
                action_card_repr = action_card_embeddings.mean(dim=1)
                action_card_repr = action_card_repr.expand(batch_size, -1)
                
                # Compute combo features
                combo_features = self._compute_combo_features(card_tensors)
                combo_features = combo_features.unsqueeze(0).expand(batch_size, -1)
            
            # Score action with enhanced features
            action_features = torch.cat([context_features, action_card_repr, combo_features], dim=-1)
            score = self.action_scorer(action_features)
            
            # Modulate score based on action type preferences
            # This encourages strategic coherence
            type_bonus = torch.zeros_like(score)
            if not card_indices:  # Yield
                type_bonus += action_type_probs[:, 3:4] * 2.0  # Yield preference
            else:
                # Aggressive vs conservative based on combo strength
                combo_strength = combo_features[:, 2:3] / 20.0  # Normalized attack
                type_bonus += action_type_probs[:, 0:1] * combo_strength  # Aggressive
                type_bonus += action_type_probs[:, 1:2] * (1 - combo_strength)  # Conservative
            
            final_score = score + type_bonus
            action_scores.append(final_score)
        
        # Pad remaining actions
        while len(action_scores) < self.max_actions:
            invalid_score = torch.full((batch_size, 1), -1e8, device=device)
            action_scores.append(invalid_score)
        
        action_logits = torch.cat(action_scores[:self.max_actions], dim=1)
        return action_logits


# Additional utility classes for advanced features
class AdaptiveLearningRate:
    """Adaptive learning rate based on performance"""
    
    def __init__(self, optimizer, patience=1000, factor=0.8, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_performance = float('-inf')
        self.wait = 0
    
    def step(self, performance_metric):
        if performance_metric > self.best_performance:
            self.best_performance = performance_metric
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                if new_lr != old_lr:
                    print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")
            self.wait = 0


class CurriculumLearning:
    """Curriculum learning for progressive difficulty"""
    
    def __init__(self, initial_players=2, max_players=4, threshold_episodes=5000):
        self.current_players = initial_players
        self.max_players = max_players
        self.threshold_episodes = threshold_episodes
        self.episode_count = 0
        self.performance_buffer = []
    
    def should_increase_difficulty(self, win_rate):
        """Check if we should increase difficulty"""
        self.performance_buffer.append(win_rate)
        if len(self.performance_buffer) > 100:
            self.performance_buffer.pop(0)
        
        if (len(self.performance_buffer) >= 100 and 
            np.mean(self.performance_buffer) > 0.8 and 
            self.episode_count > self.threshold_episodes and
            self.current_players < self.max_players):
            return True
        return False
    
    def increase_difficulty(self):
        """Increase number of players"""
        if self.current_players < self.max_players:
            self.current_players += 1
            self.performance_buffer = []  # Reset buffer
            print(f"🎯 Curriculum: Increased difficulty to {self.current_players} players")
            return True
        return False
