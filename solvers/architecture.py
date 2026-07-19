"""Neural feature extraction used by the MaskablePPO policy."""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RegicideFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch Feature Extractor for RegicideRL.
    Uses Embedding layers for card values and suits instead of simple flattening.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We don't know the exact observation space without passing it, but it should be a Dict space.
        super().__init__(observation_space, features_dim)
        
        # Hand embeddings: 8 cards, values 0-13 (14 classes)
        self.value_embedding = nn.Embedding(14, 16)
        # Hand embeddings: 8 cards, suits 0-4 (5 classes)
        self.suit_embedding = nn.Embedding(5, 8)
        
        # Enemy suit is 0-4 (5 classes)
        self.enemy_suit_embedding = nn.Embedding(5, 8)
        
        # We will process each card: (value_emb + suit_emb) -> size 24
        # 8 cards -> 8 * 24 = 192
        self.hand_fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Enemy: health (1), attack (1), suit (8) = 10
        # Flags: defense_phase (1), required_defense (1) = 2
        # Total non-hand: 12
        self.state_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Final combined: 128 (hand) + 64 (state) = 192
        self.fusion = nn.Sequential(
            nn.Linear(192, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations) -> torch.Tensor:
        """Encode structured card and enemy observations into dense features.

        Args:
            observations: Batched tensor dictionary produced by
                ``NumericObsWrapper``.

        Returns:
            Tensor with shape ``(batch_size, features_dim)``.
        """
        hand_values = observations['hand_values'].long() # (B, 8)
        hand_suits = observations['hand_suits'].long() # (B, 8)
        
        val_emb = self.value_embedding(hand_values) # (B, 8, 16)
        suit_emb = self.suit_embedding(hand_suits) # (B, 8, 8)
        
        # Concat along last dim
        card_features = torch.cat([val_emb, suit_emb], dim=-1) # (B, 8, 24)
        # Flatten for MLP
        card_features = card_features.view(card_features.shape[0], -1) # (B, 192)
        
        hand_out = self.hand_fc(card_features) # (B, 128)
        
        enemy_stats = observations['enemy_stats'] # (B, 3) -> [health, attack, suit]
        # Make sure these are floats since they are continuous/scaled
        enemy_hp = enemy_stats[:, 0:1].float() / 50.0 # Normalize roughly
        enemy_atk = enemy_stats[:, 1:2].float() / 20.0 # Normalize roughly
        enemy_suit = enemy_stats[:, 2].long() # (B,)
        
        enemy_suit_emb = self.enemy_suit_embedding(enemy_suit) # (B, 8)
        
        flags = observations['flags'].float() # (B, 2)
        # We must clone flags before in-place modifications to avoid PyTorch errors during backprop
        flags = flags.clone()
        flags[:, 1] = flags[:, 1] / 20.0 # Normalize required defense
        
        state_features = torch.cat([enemy_hp, enemy_atk, enemy_suit_emb, flags], dim=-1) # (B, 12)
        
        state_out = self.state_fc(state_features) # (B, 64)
        
        combined = torch.cat([hand_out, state_out], dim=-1) # (B, 192)
        
        return self.fusion(combined) # (B, features_dim)
