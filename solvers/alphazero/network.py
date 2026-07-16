"""
Dual-headed neural network for AlphaZero.

Architecture
------------
    Input (state_dim floats)
      │
      ├─ Shared Trunk: [Linear → ReLU] × num_hidden_layers
      │
      ├─ Policy Head: Linear → ReLU → Linear(action_space_size) → masked softmax
      │
      └─ Value Head:  Linear → ReLU → Linear(1) → tanh

The policy head outputs a probability distribution over all 542 actions.
The value head outputs a scalar in [-1, +1] estimating the expected game
outcome (progress-based: ``enemies_defeated / 12 * 2 - 1``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegicideNet(nn.Module):
    """Dual-headed MLP for Regicide AlphaZero."""

    def __init__(self, state_dim: int = 56, action_dim: int = 543,
                 hidden_dim: int = 256, num_hidden_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # --- Shared trunk ---
        layers = []
        in_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # --- Policy head ---
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor):
        """Raw forward pass returning logits and value.

        Args:
            state: ``(B, state_dim)`` float tensor.

        Returns:
            policy_logits: ``(B, action_dim)`` — raw, un-masked logits.
            value: ``(B, 1)`` — predicted value in [-1, 1].
        """
        trunk_out = self.trunk(state)
        policy_logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action_mask: torch.Tensor):
        """Convenience method for MCTS inference.

        Applies the action mask, softmaxes to get prior probabilities,
        and squeezes the value.

        Args:
            state: ``(state_dim,)`` or ``(1, state_dim)`` float tensor.
            action_mask: ``(action_dim,)`` or ``(1, action_dim)`` binary tensor.

        Returns:
            priors: ``(action_dim,)`` numpy array — masked probability dist.
            value: float scalar.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        policy_logits, value = self.forward(state)

        # Mask illegal actions by setting logits to -inf
        masked_logits = policy_logits.clone()
        masked_logits[action_mask == 0] = float('-inf')

        priors = F.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()
        value_scalar = value.squeeze().item()

        return priors, value_scalar
