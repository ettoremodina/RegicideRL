"""Recording layer that turns a single ISMCTS decision into a replayable trace.

The tracer is a passive observer: :class:`~agents.ismcts_agent.ISMCTSAgent`
calls a handful of guarded hooks and the tracer stores what happened, without
influencing the search. The resulting dictionary is designed to be replayed
step by step by a viewer, so it records the three properties that distinguish
ISMCTS from plain UCT:

  1. the *determinization* sampled at the start of every iteration;
  2. the *subset of legal actions* at each visited node, which is what makes
     the tree a subset-armed bandit;
  3. the *availability counts*, incremented for every legal child whether or
     not it was selected, which is the denominator used by the UCB formula.

Counters are not stored per iteration. The trace records the increments
(``available`` node ids, the expanded node, the back-propagated path), so a
viewer reproduces ``visit_count``, ``availability_count`` and the mean reward
by replaying the same arithmetic the agent used.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from game.action_space import SOLO_JESTER_ACTION_ID
from ml_logger import get_logger

logger = get_logger(__name__)


def describe_action(handler, action_id: int, hand: List[Any], defense_phase: bool) -> str:
    """Return a short human-readable label for a global action id.

    Args:
        handler: The :class:`~game.action_handler.ActionHandler` used to decode
            the global action space.
        action_id: Global action id to describe.
        hand: Hand the action refers to, needed because the global id maps to
            concrete cards rather than to hand positions.
        defense_phase: Whether the action is a defense discard.

    Returns:
        A label such as ``"5♠+A♥"``, ``"def 7♦"``, ``"yield"`` or ``"jester"``.
    """
    if action_id == SOLO_JESTER_ACTION_ID:
        return "jester"
    if action_id == 0 and not defense_phase:
        return "yield"
    try:
        indices = handler.global_action_to_hand_indices(int(action_id), hand)
    except (ValueError, IndexError, TypeError):
        return f"#{action_id}"
    cards = [str(hand[index]) for index in indices if 0 <= index < len(hand)]
    if not cards:
        return "yield"
    return ("def " if defense_phase else "") + "+".join(cards)


class ISMCTSTracer:
    """Collect one decision worth of ISMCTS iterations as replayable data.

    A tracer instance covers a single decision. The agent registers it before
    the search, calls the hooks during the search and the caller reads
    :meth:`to_dict` afterwards.

    Args:
        max_iterations: Number of iterations to record. Later iterations still
            run normally but are not stored, which keeps traces small when the
            agent uses a realistic budget.
        tavern_preview: Number of upcoming tavern cards stored per
            determinization, in draw order.
        max_depth: Maximum tree depth to record. Deeper descent steps still
            happen, they are simply not written to the trace.
    """

    def __init__(self, max_iterations: int = 200, tavern_preview: int = 8,
                 max_depth: int = 6):
        self.max_iterations = max_iterations
        self.tavern_preview = tavern_preview
        self.max_depth = max_depth
        self.meta: Dict[str, Any] = {}
        self.decision: Dict[str, Any] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.iterations: List[Dict[str, Any]] = []
        self._node_ids: Dict[int, int] = {}
        self._exploration_constant = 0.0
        self._handler = None
        self._current: Optional[Dict[str, Any]] = None
        self._iteration_index = -1

    # ------------------------------------------------------------------
    # Hooks called by ISMCTSAgent
    # ------------------------------------------------------------------
    def begin_decision(self, agent, env, obs, valid_actions: List[int], root) -> None:
        """Capture the real (non-determinized) state the decision starts from."""
        game = env.game
        hand = obs['hand']
        defense_phase = bool(obs['defense_phase'])
        self._handler = env.handler
        self._exploration_constant = agent.exploration_constant
        self._node_ids.clear()
        self.nodes = []
        self.iterations = []
        self._register(root, parent_id=None, action=None, label="root", depth=0)

        enemy = game.current_enemy
        self.meta = {
            "agent": agent.name,
            "n_iterations": agent.n_iterations,
            "recorded_iterations": min(agent.n_iterations, self.max_iterations),
            "exploration_constant": agent.exploration_constant,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.decision = {
            "hand": [str(card) for card in hand],
            "defense_phase": defense_phase,
            "required_defense": obs['required_defense'],
            "tavern_cards": len(game.tavern_deck),
            "discard_cards": len(game.discard_pile),
            "enemies_remaining": len(game.castle_deck),
            "enemy": {
                "card": str(enemy.card),
                "suit": enemy.card.suit.value,
                "health": enemy.health,
                "damage_taken": enemy.damage_taken,
                "attack": enemy.get_effective_attack(),
            } if enemy else None,
            "legal_actions": [
                {
                    "id": action,
                    "label": describe_action(env.handler, action, hand, defense_phase),
                }
                for action in valid_actions
            ],
            "best_action": None,
        }

    def begin_iteration(self, index: int, sim_env) -> None:
        """Open a new iteration record holding the sampled hidden state."""
        self._iteration_index = index
        if index >= self.max_iterations:
            self._current = None
            return
        game = sim_env.game
        # The tavern deck is drawn from its tail, the castle deck from its head.
        upcoming = list(reversed(game.tavern_deck[-self.tavern_preview:]))
        self._current = {
            "i": index,
            "determinization": {
                "tavern_top": [str(card) for card in upcoming],
                "castle": [str(card) for card in game.castle_deck],
            },
            "steps": [],
            "rollout": None,
            "reward": 0.0,
            "path": [],
        }

    def record_step(self, node, child, action: int, legal_actions: List[int],
                    hand: List[Any], defense_phase: bool, expanded: bool,
                    step_reward: float) -> None:
        """Record one descent step, including the untaken but legal siblings.

        Args:
            node: Node the descent is standing on before the step.
            child: Node the step moves to.
            action: Global action id applied to the determinized state.
            legal_actions: Every action legal in this determinization.
            hand: Hand before the action, used to label the actions.
            defense_phase: Whether the node is a forced-defense decision.
            expanded: Whether ``child`` was created by this step.
            step_reward: Environment reward returned by the step.
        """
        if self._current is None:
            return
        parent_id = self._node_ids.get(id(node))
        if parent_id is None or self.nodes[parent_id]["depth"] >= self.max_depth:
            return

        depth = self.nodes[parent_id]["depth"] + 1
        # Availability rises for every legal action that already had a node,
        # which excludes the child created by this very step.
        available = [
            self._node_ids[id(node.children[a])]
            for a in legal_actions
            if a in node.children and not (expanded and a == action)
            and id(node.children[a]) in self._node_ids
        ]
        scores = {}
        if not expanded:
            for a in legal_actions:
                sibling = node.children.get(a)
                sibling_id = self._node_ids.get(id(sibling)) if sibling else None
                if sibling_id is not None:
                    scores[str(sibling_id)] = round(
                        sibling.ucb_score(self._exploration_constant), 4
                    )

        child_id = self._register(
            child,
            parent_id=parent_id,
            action=action,
            label=describe_action(self._handler, action, hand, defense_phase),
            depth=depth,
        )
        self._current["steps"].append({
            "node": parent_id,
            "child": child_id,
            "expanded": expanded,
            "available": available,
            "untried": len([a for a in legal_actions if a not in node.children]),
            "legal": len(legal_actions),
            "scores": scores,
            "reward": round(float(step_reward), 4),
            "defense": bool(defense_phase),
        })

    def record_rollout(self, depth: int, reward: float) -> None:
        """Store the length and the return of the heuristic playout."""
        if self._current is None:
            return
        self._current["rollout"] = {
            "depth": int(depth),
            "reward": round(float(reward), 4),
        }

    def record_backprop(self, path: List[Any], reward: float) -> None:
        """Store the updated path and close the current iteration record."""
        if self._current is None:
            return
        self._current["path"] = [
            self._node_ids[id(node)] for node in path if id(node) in self._node_ids
        ]
        self._current["reward"] = round(float(reward), 4)
        self.iterations.append(self._current)
        self._current = None

    def end_decision(self, root, best_action: Optional[int]) -> None:
        """Record the action the search finally committed to."""
        self.decision["best_action"] = best_action
        self.decision["root_children"] = [
            {
                "id": self._node_ids[id(child)],
                "action": action,
                "visits": child.visit_count,
                "availability": child.availability_count,
                "mean_reward": round(child.mean_reward, 4),
            }
            for action, child in root.children.items()
            if id(child) in self._node_ids
        ]
        logger.info(
            "ISMCTS trace complete: %d iterations recorded, %d nodes, best action %s",
            len(self.iterations), len(self.nodes), best_action,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return the trace as a JSON-serializable dictionary."""
        return {
            "meta": self.meta,
            "decision": self.decision,
            "nodes": self.nodes,
            "iterations": self.iterations,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _register(self, node, parent_id: Optional[int], action: Optional[int],
                  label: str, depth: int) -> int:
        """Assign a stable trace id to a tree node, creating it if needed."""
        existing = self._node_ids.get(id(node))
        if existing is not None:
            return existing
        node_id = len(self.nodes)
        self._node_ids[id(node)] = node_id
        self.nodes.append({
            "id": node_id,
            "parent": parent_id,
            "action": action,
            "label": label,
            "depth": depth,
        })
        return node_id
