"""
Fixed-size replay buffer for AlphaZero training.

Stores ``(state, policy_target, value_target)`` tuples generated during
self-play.  Uses a circular buffer with random mini-batch sampling.
"""

import random
import numpy as np


class ReplayBuffer:
    """Circular replay buffer with uniform random sampling.

    Args:
        max_size: Maximum number of samples to retain.  Once full, the
            oldest samples are overwritten.
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self.buffer = []
        self._pos = 0  # Write cursor for circular overwrite

    def __len__(self):
        return len(self.buffer)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training sample.

        Args:
            state: Float32 array of shape ``(state_dim,)``.
            policy: Float32 array of shape ``(action_dim,)`` — MCTS
                visit-count distribution.
            value: Float scalar — game outcome.
        """
        sample = (state, policy, value)
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            self.buffer[self._pos] = sample
        self._pos = (self._pos + 1) % self.max_size

    def add_game(self, game_data):
        """Add all samples from a single self-play game.

        Args:
            game_data: List of ``(state, policy, value)`` tuples.
        """
        for state, policy, value in game_data:
            self.add(state, policy, value)

    def sample_batch(self, batch_size: int):
        """Sample a random mini-batch.

        Returns:
            states: np.ndarray of shape ``(B, state_dim)``.
            policies: np.ndarray of shape ``(B, action_dim)``.
            values: np.ndarray of shape ``(B, 1)``.
        """
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        policies = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)
        values = np.array(
            [[self.buffer[i][2]] for i in indices], dtype=np.float32
        )
        return states, policies, values

    def clear(self):
        """Remove all samples."""
        self.buffer.clear()
        self._pos = 0
