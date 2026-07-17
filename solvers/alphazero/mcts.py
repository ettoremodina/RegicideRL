"""
ISMCTS with PUCT selection and neural network leaf evaluation.

This module upgrades the vanilla ISMCTS (UCB1 + heuristic rollouts) to use:
  1. PUCT formula with network-provided prior probabilities P(s, a).
  2. Network value-head evaluation at leaf nodes (replacing rollouts).
  3. Dirichlet noise at the root for exploration diversity.

The core ISMCTS mechanisms are preserved:
  - Determinization of hidden information (tavern / castle deck).
  - Single tree shared across determinizations.
  - Availability counts for the subset-armed-bandit problem.
"""

import math
import random
import logging

import numpy as np
import torch

from solvers.agents.determinize import determinize_env
from solvers.alphazero.featurizer import (
    encode_state,
    action_mask_to_global_index,
)

logger = logging.getLogger("alphazero.mcts")


class PUCTNode:
    """A node in the ISMCTS tree using PUCT selection.

    Attributes:
        action_tuple: The hand-relative action mask (as tuple) that led here.
            None for the root.
        global_action_id: The corresponding 542-dim global action index.
            -1 for the root.
        parent: Parent PUCTNode (None for root).
        children: Dict mapping action_tuple → child PUCTNode.
        visit_count (N): Number of backpropagation passes through this node.
        total_value (W): Cumulative value from backpropagation.
        prior (P): Network prior probability for this action.
        availability_count: ISMCTS availability counter (how many times
            this action was *legal* during selection at its parent).
    """

    __slots__ = [
        "action_tuple", "global_action_id", "parent", "children",
        "visit_count", "total_value", "prior", "availability_count",
        "noise_added"
    ]

    def __init__(self, action_tuple=None, global_action_id=-1,
                 parent=None, prior=0.0):
        self.action_tuple = action_tuple
        self.global_action_id = global_action_id
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.availability_count = 0
        self.noise_added = False

    @property
    def Q(self):
        """Mean action value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, c_puct):
        """PUCT score adapted for ISMCTS (uses availability_count).

        Score = Q(s,a) + c_puct * P(s,a) * sqrt(availability_count) / (1 + N(s,a))

        Uses ``availability_count`` instead of parent ``visit_count``
        to handle the subset-armed-bandit nature of ISMCTS.
        """
        exploration = c_puct * self.prior * (
            math.sqrt(self.availability_count) / (1 + self.visit_count)
        )
        return self.Q + exploration


def run_mcts(env, network, config, device):
    """Run ISMCTS with PUCT and return a policy vector.

    Args:
        env: A *live* RegicideEnv (will be cloned for each simulation).
        network: The RegicideNet model (in eval mode).
        config: AlphaZeroConfig with MCTS hyperparameters.
        device: torch device for network inference.

    Returns:
        policy: np.ndarray of shape ``(542,)`` — normalized visit counts.
        root_value: float — the mean value at the root after all sims.
    """
    root = PUCTNode()
    root.noise_added = False
    handler = env.handler
    is_defense = env.required_defense > 0

    for _ in range(config.n_simulations):
        # 1. Clone + determinize
        sim_env = env.clone()
        determinize_env(sim_env)

        # 2-4. Select → Expand → Backpropagate
        _run_simulation(root, sim_env, network, config, device, handler)

    # Build policy from visit counts
    policy = np.zeros(config.action_space_size, dtype=np.float32)
    for action_tuple, child in root.children.items():
        policy[child.global_action_id] = child.visit_count

    visit_sum = policy.sum()
    if visit_sum > 0:
        policy /= visit_sum

    root_value = root.Q
    return policy, root_value


def _run_simulation(root, sim_env, network, config, device, handler):
    """One ISMCTS simulation: select → expand/evaluate → backpropagate.

    Descends the tree following PUCT until reaching an unexpanded node
    or a terminal state.  At the leaf, the network's value head provides
    the bootstrap value (no rollout).
    """
    node = root
    path = [node]
    sim_obs = sim_env._get_obs()
    is_defense = sim_env.required_defense > 0

    # --- SELECTION ---
    while not sim_env.game.game_over:
        action_mask_obs = sim_obs["action_mask"]
        legal_actions = np.nonzero(action_mask_obs)[0].tolist()
        if not legal_actions:
            break

        # Update availability counts for existing children
        for at in legal_actions:
            if at in node.children:
                node.children[at].availability_count += 1

        # Check for untried (unexpanded) actions
        untried = [a for a in legal_actions if a not in node.children]

        if untried:
            # --- EXPANSION ---
            # Get network priors for all legal actions at this state
            priors = _get_priors(
                sim_env, network, device, handler, is_defense
            )

            # Expand *all* untried children with their priors
            for at in untried:
                gid = at
                child = PUCTNode(
                    action_tuple=at,
                    global_action_id=gid,
                    parent=node,
                    prior=priors[gid],
                )
                child.availability_count = 1
                node.children[at] = child

            # Pick one untried action (highest prior for efficiency)
            action_tuple = max(
                untried, key=lambda a: node.children[a].prior
            )
            child = node.children[action_tuple]

            # Apply action
            sim_obs, _, terminated, truncated, _ = sim_env.step(action_tuple)
            is_defense = sim_env.required_defense > 0
            path.append(child)
            node = child

            # After expansion, evaluate the leaf and stop descending
            break
        else:
            # All children expanded — select via PUCT
            # Add Dirichlet noise at root for exploration exactly once
            if node is root and config.dirichlet_epsilon > 0 and not root.noise_added:
                _add_dirichlet_noise(node, legal_actions, config)
                root.noise_added = True

            action_tuple = max(
                legal_actions,
                key=lambda a: node.children[a].puct_score(config.c_puct),
            )
            child = node.children[action_tuple]

            sim_obs, _, terminated, truncated, _ = sim_env.step(action_tuple)
            is_defense = sim_env.required_defense > 0
            path.append(child)
            node = child

            if terminated or truncated:
                break

    # --- LEAF EVALUATION ---
    if sim_env.game.game_over:
        value = _evaluate_terminal(sim_env)
    else:
        # Network value-head evaluation (no rollout)
        value = _evaluate_network(sim_env, network, device)

    # --- BACKPROPAGATION ---
    for n in path:
        n.visit_count += 1
        n.total_value += value


def _get_priors(sim_env, network, device, handler, is_defense):
    """Get the network's prior distribution for a state.

    Returns a full 542-dim numpy array of probabilities.
    """
    state = encode_state(sim_env)
    state_t = torch.tensor(state, dtype=torch.float32, device=device)

    # Build the 543-dim action mask
    hand = sim_env.game.get_player_hand(sim_env.game.current_player)
    phase = "defense" if is_defense else "attack"
    raw_state = sim_env.game.get_raw_state()
    game_state = raw_state if phase == "attack" else {"enemy_attack": sim_env.required_defense}
    mask_543 = handler.get_global_action_mask(hand, phase, game_state)
    mask_t = torch.tensor(mask_543, dtype=torch.float32, device=device)

    priors, _ = network.predict(state_t, mask_t)
    return priors


def _evaluate_network(sim_env, network, device):
    """Evaluate a non-terminal leaf node using the network's value head."""
    state = encode_state(sim_env)
    state_t = torch.tensor(state, dtype=torch.float32, device=device)

    # We only need the value; create a dummy mask
    mask_t = torch.ones(543, dtype=torch.float32, device=device)

    _, value = network.predict(state_t, mask_t)
    return value


def _evaluate_terminal(env):
    """Progress-based terminal evaluation in [-1, +1].

    Maps ``enemies_defeated / 12`` into [-1, +1] with a bonus for victory.
    """
    enemies_left = len(env.game.castle_deck) + (
        1 if env.game.current_enemy and not env.game.victory else 0
    )
    enemies_defeated = 12 - enemies_left
    progress = enemies_defeated / 12.0  # [0, 1]

    if env.game.victory:
        return 1.0
    # Map [0, 1) progress to [-1, +1): e.g. 0→-1, 0.5→0, ~1→~1
    return progress * 2.0 - 1.0


def _add_dirichlet_noise(root, legal_actions, config):
    """Add Dirichlet noise to the root priors for exploration."""
    legal_children = [root.children[a] for a in legal_actions]
    noise = np.random.dirichlet(
        [config.dirichlet_alpha] * len(legal_children)
    )
    eps = config.dirichlet_epsilon
    for child, n in zip(legal_children, noise):
        child.prior = (1 - eps) * child.prior + eps * n
