"""
Card-Aware Policy Network for Regicide
Uses attention mechanisms and card embeddings to understand card relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class CardAwarePolicy(nn.Module):
    """
    Card-aware policy network that understands individual cards and their relationships
    """
    
    def __init__(self, max_hand_size: int = 8, max_actions: int = 20, 
                 card_embed_dim: int = 64, hidden_dim: int = 128):
        super(CardAwarePolicy, self).__init__()
        
        self.max_hand_size = max_hand_size
        self.max_actions = max_actions
        self.card_embed_dim = card_embed_dim
        self.hidden_dim = hidden_dim
        
        # Card embeddings (53 cards: 52 regular + 1 jester + 1 padding)
        self.card_embedding = nn.Embedding(54, card_embed_dim, padding_idx=0)
        
        # Hand encoder with self-attention
        self.hand_attention = nn.MultiheadAttention(
            embed_dim=card_embed_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
        
        # Game state encoder
        self.game_state_encoder = nn.Sequential(
            nn.Linear(12, 64),  # Game state has 12 features (updated to match environment)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined context encoder
        # Input: card_embed_dim (64) + 32 (game state) = 96
        self.context_encoder = nn.Sequential(
            nn.Linear(card_embed_dim + 32, hidden_dim),  # 96 -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 128 -> 128
            nn.ReLU()
        )

        
        # Final action scorer
        # Input: hidden_dim (128) + card_embed_dim (64) = 192
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + card_embed_dim, 64),  # 192 -> 64
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            # Use a smaller initialization to prevent exploding gradients
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
            if m.padding_idx is not None:
                torch.nn.init.constant_(m.weight[m.padding_idx], 0)
        elif isinstance(m, nn.MultiheadAttention):
            # Initialize attention weights more conservatively
            for param in m.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_normal_(param, gain=0.1)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the card-aware network"""
        batch_size = observation['hand_cards'].size(0) if observation['hand_cards'].dim() > 0 else 1
        device = observation['hand_cards'].device
        
        # Ensure batch dimension for all tensors
        hand_cards = observation['hand_cards']
        if hand_cards.dim() == 1:
            hand_cards = hand_cards.unsqueeze(0)  # [1, hand_size]
        
        game_state = observation['game_state']
        if game_state.dim() == 1:
            game_state = game_state.unsqueeze(0)  # [1, state_dim]
        
        enemy_card = observation['enemy_card']
        if enemy_card.dim() == 0:
            enemy_card = enemy_card.unsqueeze(0)  # [1]
        if enemy_card.dim() == 1:
            enemy_card = enemy_card.unsqueeze(0)  # [1, 1]
        
        # Encode hand cards with attention
        hand_embeddings = self.card_embedding(hand_cards)  # [batch, max_hand, embed_dim]
        
        # Create attention mask for padding
        hand_mask = hand_cards != 0  # [batch, max_hand]
        
        # Check if hand is completely empty (all zeros)
        has_cards = hand_mask.any(dim=1)  # [batch] - True if any cards in hand
        
        # Self-attention over hand cards (only if we have cards)
        if has_cards.any():
            attended_hand, attention_weights = self.hand_attention(
                hand_embeddings, hand_embeddings, hand_embeddings,
                key_padding_mask=~hand_mask
            )
        else:
            attended_hand = torch.zeros_like(hand_embeddings)
        
        # Aggregate hand representation (mean of non-padding cards)
        hand_lengths = observation['hand_size'].float()
        if hand_lengths.dim() == 0:
            hand_lengths = hand_lengths.unsqueeze(0)  # [1]
        hand_lengths = hand_lengths.unsqueeze(-1)  # [batch, 1]
        
        # Handle empty hands explicitly
        if (hand_lengths == 0).any():
            # For empty hands, use zero context
            hand_context = torch.zeros(batch_size, self.card_embed_dim, device=device)
        else:
            # Ensure hand_lengths is at least 1 to avoid division by zero
            hand_lengths = torch.clamp(hand_lengths, min=1.0)
            
            # Safe division for hand context
            hand_context = attended_hand.sum(dim=1) / hand_lengths  # [batch, embed_dim]
   
        # Encode enemy card
        ### NOTE: enemy embedding not used
        # enemy_embedding = self.card_embedding(enemy_card.squeeze(-1))  # [batch, embed_dim]

        
        # Encode game state
        game_context = self.game_state_encoder(game_state)  # [batch, 32]    
        # Combined context
        combined_context = torch.cat([hand_context, game_context], dim=-1)  # [batch, embed_dim + 32]
        context_features = self.context_encoder(combined_context)  # [batch, hidden_dim]
                
        # Score each valid action
        num_valid_actions = observation['num_valid_actions'].item()
        action_scores = []
        
        for action_idx in range(num_valid_actions):
            # Get card indices for this action
            if action_idx < len(observation['action_card_indices']):
                card_indices = observation['action_card_indices'][action_idx]
            else:
                card_indices = []
            
            if not card_indices:  # Yield action
                # Score yield action based on context only
                yield_features = torch.cat([context_features, torch.zeros_like(hand_context)], dim=-1)
                score = self.action_scorer(yield_features)  # [batch, 1]
                action_scores.append(score)
            else:
                # Score card combination
                # Get embeddings for cards in this action
                card_tensors = torch.tensor(card_indices, dtype=torch.long, device=device)
                action_card_embeddings = self.card_embedding(card_tensors)  # [num_cards, embed_dim]

                # Aggregate card embeddings for this action - ensure correct batch dimension
                if action_card_embeddings.dim() == 2:
                    # Multiple cards: [num_cards, embed_dim] -> [1, embed_dim]
                    action_card_repr = action_card_embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
                else:
                    # Single card: [embed_dim] -> [1, embed_dim]
                    action_card_repr = action_card_embeddings.unsqueeze(0)  # [1, embed_dim]
                
                # Expand to match batch size
                action_card_repr = action_card_repr.expand(batch_size, -1)  # [batch, embed_dim]

                # Combine with context and score
                action_features = torch.cat([context_features, action_card_repr], dim=-1)
                score = self.action_scorer(action_features)  # [batch, 1]
                action_scores.append(score)

        # Pad to max_actions if necessary
        while len(action_scores) < self.max_actions:
            # Add very negative scores for invalid actions
            invalid_score = torch.full((batch_size, 1), -1e8, device=device)
            action_scores.append(invalid_score)
        
        # Stack and return logits - ensure all tensors have same shape
        action_logits = torch.cat(action_scores[:self.max_actions], dim=1)  # [batch, max_actions]        
        return action_logits
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   action_mask: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """Get action and log probability from the policy"""
        # Ensure batch dimension
        if observation['hand_cards'].dim() == 1:
            for key, value in observation.items():
                if isinstance(value, torch.Tensor):
                    observation[key] = value.unsqueeze(0)
        
        # Forward pass
        logits = self.forward(observation)  # [1, max_actions]
        
        # Apply action mask (only consider valid actions)
        num_valid_actions = observation['num_valid_actions'].item()
        if num_valid_actions > 0:
            # Mask invalid actions
            masked_logits = logits.clone()
            masked_logits[:, num_valid_actions:] = -1e8
            
            # Get probabilities
            probs = F.softmax(masked_logits, dim=-1)
            
            # Sample from valid actions only
            valid_probs = probs[:, :num_valid_actions]
            action_dist = torch.distributions.Categorical(valid_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.squeeze()
        else:
            # No valid actions (shouldn't happen)
            return 0, torch.tensor(0.0)
    
    def get_action_probabilities(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get action probabilities for analysis"""
        logits = self.forward(observation)
        num_valid_actions = observation['num_valid_actions'].item()
        
        if num_valid_actions > 0:
            # Only return probabilities for valid actions
            valid_logits = logits[:, :num_valid_actions]
            return F.softmax(valid_logits, dim=-1)
        else:
            return torch.zeros(1, 1)
    
    def analyze_decision(self, observation: Dict[str, torch.Tensor], 
                        card_names: Optional[List[str]] = None) -> Dict:
        """Analyze the decision-making process for debugging"""
        with torch.no_grad():
            # Ensure batch dimension
            if observation['hand_cards'].dim() == 1:
                for key, value in observation.items():
                    if isinstance(value, torch.Tensor):
                        observation[key] = value.unsqueeze(0)
            
            try:
                logits = self.forward(observation)
                probs = self.get_action_probabilities(observation)
                
                num_valid_actions = observation['num_valid_actions'].item()
                
                analysis = {
                    'action_probabilities': probs.squeeze().tolist() if probs.numel() > 0 else [],
                    'action_logits': logits[:, :num_valid_actions].squeeze().tolist() if num_valid_actions > 0 else [],
                    'hand_size': observation['hand_size'].item(),
                    'num_valid_actions': num_valid_actions,
                    'game_state': observation['game_state'].squeeze().tolist()
                }
                
                # Add card information if provided
                if card_names:
                    analysis['hand_cards'] = card_names
                
                return analysis
                
            except Exception as e:
                # Return basic info if analysis fails
                return {
                    'action_probabilities': [],
                    'action_logits': [],
                    'hand_size': observation['hand_size'].item(),
                    'num_valid_actions': observation['num_valid_actions'].item(),
                    'game_state': observation['game_state'].squeeze().tolist(),
                    'error': str(e)
                }
