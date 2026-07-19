"""Neural-guided information-set Monte Carlo tree search.

This is the AlphaZero search expert for solo Regicide. It retains the ISMCTS
mechanisms needed for hidden cards and uses network priors with PUCT selection.
During warm-up, rule-based action scores are mixed into the network priors;
leaf values always use the inexpensive network evaluation.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from agents.determinize import determinize_env
from agents.heuristic_agent import HeuristicAgent
from game.action_space import GLOBAL_ACTION_SPACE_SIZE
from solvers.alphazero.featurizer import encode_state, information_state_key
from solvers.alphazero.outcomes import terminal_value


class PUCTEdge:
    """Statistics for one action available from an information-set node."""

    __slots__ = (
        "action_id",
        "visit_count",
        "total_value",
        "prior",
        "availability_count",
        "outcomes",
    )

    def __init__(self, action_id: int, prior: float):
        self.action_id = action_id
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.availability_count = 0
        self.outcomes = {}

    @property
    def mean_value(self) -> float:
        """Return the backed-up mean value for this action."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, exploration_constant: float) -> float:
        """Return PUCT adapted to ISMCTS action-availability counts."""
        exploration = exploration_constant * self.prior * (
            math.sqrt(self.availability_count) / (1 + self.visit_count)
        )
        return self.mean_value + exploration


class PUCTNode:
    """One observable information state in the ISMCTS search tree."""

    __slots__ = (
        "children",
        "visit_count",
        "total_value",
        "noise_added",
    )

    def __init__(self):
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.noise_added = False

    @property
    def mean_value(self) -> float:
        """Return the backed-up mean value for this information state."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass(frozen=True)
class _SearchContext:
    """Dependencies and options shared by all simulations in one search."""

    network: object
    config: object
    device: object
    add_exploration_noise: bool
    heuristic_agent: HeuristicAgent | None


def run_mcts(
    env,
    network,
    config,
    device,
    add_exploration_noise: bool = False,
    use_heuristic_guidance: bool = False,
):
    """Run neural-guided ISMCTS and return root visits and mean value."""
    root = PUCTNode()
    context = _SearchContext(
        network=network,
        config=config,
        device=device,
        add_exploration_noise=add_exploration_noise,
        heuristic_agent=(
            HeuristicAgent(name="AlphaZero warm-up")
            if use_heuristic_guidance
            else None
        ),
    )
    for _ in range(config.n_simulations):
        simulation_env = env.clone()
        determinize_env(simulation_env)
        _run_simulation(root, simulation_env, context)

    policy = np.zeros(config.action_space_size, dtype=np.float32)
    for action_id, edge in root.children.items():
        policy[action_id] = edge.visit_count
    visit_sum = policy.sum()
    if visit_sum <= 0:
        raise RuntimeError("ISMCTS completed without visiting a legal action")
    policy /= visit_sum
    return policy, root.mean_value


def _run_simulation(root, simulation_env, context):
    """Traverse one determinization and back up its leaf evaluation."""
    node = root
    visited_nodes = [root]
    visited_edges = []
    value = None

    while not simulation_env.game.game_over:
        observation = simulation_env._get_obs()
        legal_actions = np.flatnonzero(observation["action_mask"]).tolist()
        if not legal_actions:
            break

        _ensure_action_edges(
            node,
            legal_actions,
            simulation_env,
            context,
        )
        for action_id in legal_actions:
            node.children[action_id].availability_count += 1

        if (
            node is root
            and context.add_exploration_noise
            and not root.noise_added
        ):
            _add_dirichlet_noise(root, legal_actions, context.config)

        edge = max(
            (node.children[action_id] for action_id in legal_actions),
            key=lambda candidate: candidate.puct_score(context.config.c_puct),
        )
        _, _, terminated, truncated, _ = simulation_env.step(edge.action_id)
        visited_edges.append(edge)

        if terminated or truncated or simulation_env.game.game_over:
            value = terminal_value(simulation_env.game)
            break

        outcome_key = information_state_key(simulation_env)
        child = edge.outcomes.get(outcome_key)
        if child is None:
            child = PUCTNode()
            edge.outcomes[outcome_key] = child
            visited_nodes.append(child)
            value = _evaluate_leaf(simulation_env, context)
            break

        node = child
        visited_nodes.append(node)
    else:
        value = terminal_value(simulation_env.game)

    if value is None:
        value = (
            terminal_value(simulation_env.game)
            if simulation_env.game.game_over
            else _evaluate_leaf(simulation_env, context)
        )
    _backpropagate(visited_nodes, visited_edges, value)


def _ensure_action_edges(node, legal_actions, env, context):
    """Create missing legal action edges with network-provided priors."""
    missing_actions = [
        action_id for action_id in legal_actions if action_id not in node.children
    ]
    if not missing_actions:
        return
    priors = _get_priors(env, context)
    for action_id in missing_actions:
        node.children[action_id] = PUCTEdge(
            action_id=action_id,
            prior=float(priors[action_id]),
        )


def _backpropagate(nodes, edges, value):
    """Accumulate one simulation value over visited nodes and actions."""
    for node in nodes:
        node.visit_count += 1
        node.total_value += value
    for edge in edges:
        edge.visit_count += 1
        edge.total_value += value


def _get_priors(env, context):
    """Return network priors, optionally mixed with heuristic guidance."""
    observation = env._get_obs()
    state = torch.tensor(
        encode_state(env),
        dtype=torch.float32,
        device=context.device,
    )
    legal_mask = torch.tensor(
        observation["action_mask"],
        dtype=torch.float32,
        device=context.device,
    )
    network_priors, _ = context.network.predict(state, legal_mask)
    if context.heuristic_agent is None:
        return network_priors
    heuristic_priors = _get_heuristic_priors(
        env,
        context.heuristic_agent,
        observation,
    )
    weight = context.config.heuristic_prior_weight
    return (1 - weight) * network_priors + weight * heuristic_priors


def _get_heuristic_priors(env, agent, observation):
    """Convert rule scores into a normalized legal-action distribution."""
    action_scores = agent.score_actions(observation, env=env)
    priors = np.zeros_like(observation["action_mask"], dtype=np.float32)
    if not action_scores:
        return priors
    minimum_score = min(action_scores.values())
    for action_id, score in action_scores.items():
        priors[action_id] = score - minimum_score + 1.0
    priors /= priors.sum()
    return priors


def _evaluate_network(env, network, device):
    state = torch.tensor(encode_state(env), dtype=torch.float32, device=device)
    dummy_mask = torch.ones(
        GLOBAL_ACTION_SPACE_SIZE,
        dtype=torch.float32,
        device=device,
    )
    _, value = network.predict(state, dummy_mask)
    return value


def _evaluate_leaf(env, context):
    """Evaluate a search leaf using the policy-value network."""
    return _evaluate_network(env, context.network, context.device)


def _add_dirichlet_noise(root, legal_actions, config):
    legal_edges = [root.children[action_id] for action_id in legal_actions]
    noise = np.random.dirichlet(
        [config.dirichlet_alpha] * len(legal_edges)
    )
    epsilon = config.dirichlet_epsilon
    for edge, noise_value in zip(legal_edges, noise):
        edge.prior = (1 - epsilon) * edge.prior + epsilon * noise_value
    root.noise_added = True
