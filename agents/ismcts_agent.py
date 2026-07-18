"""
Information Set Monte Carlo Tree Search (ISMCTS) agent for Regicide.

Phase 2: builds a *single* search tree over information sets (shared across
determinizations), using UCB1 selection adapted for "subset-armed bandits"
where available actions vary per determinization. This fixes the "strategy
fusion" flaw of naive PIMC.

Implements the Single-Observer variant (SO-ISMCTS) since solo Regicide is a
1-player game against an environment (the enemy deck).

Reference: Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree
Search", IEEE Transactions on Computational Intelligence and AI in Games, 2012.

Key differences from standard UCT:
  - Nodes represent information sets, not exact game states.
  - Each node tracks an `availability_count` — how many times the action
    leading to it was *available* (legal) during tree descent, regardless
    of whether it was selected. UCB1 uses this instead of parent visits.
  - At each tree node, only the subset of children whose actions are legal
    in the current determinization are considered for selection.
"""

import math
import logging
from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from agents.determinize import determinize_env

logger = logging.getLogger("ismcts_agent")


class ISMCTSNode:
    """A node in the ISMCTS tree representing an information set.

    Attributes:
        action: The action (as tuple) that led to this node from its parent.
                None for the root node.
        parent: Parent ISMCTSNode (None for root).
        children: Dict mapping action_tuple → child ISMCTSNode.
        visit_count: Number of times this node was visited during backprop.
        total_reward: Cumulative reward backpropagated through this node.
        availability_count: Number of times this node's action was *available*
            (legal in the determinized state) during selection at its parent,
            whether or not it was actually chosen. Used in the ISMCTS UCB formula.
    """

    __slots__ = ['action', 'parent', 'children', 'visit_count',
                 'total_reward', 'availability_count']

    def __init__(self, action=None, parent=None):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0.0
        self.availability_count = 0

    @property
    def mean_reward(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb_score(self, exploration_constant):
        """ISMCTS-adapted UCB1 score.

        Uses availability_count instead of parent's visit_count:
            UCB1 = reward_i / visit_i  +  C * sqrt(ln(availability_i) / visit_i)

        This is the key innovation from the Cowling et al. paper that handles
        the subset-armed bandit problem.
        """
        if self.visit_count == 0:
            return float('inf')  # Always try unvisited nodes first
        exploitation = self.total_reward / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.availability_count) / self.visit_count
        )
        return exploitation + exploration


class ISMCTSAgent(BaseAgent):
    """Information Set Monte Carlo Tree Search agent.

    For every decision, runs N ISMCTS iterations building a single shared
    tree, then picks the root child with the highest visit count.

    Each iteration:
      1. Sample one determinization (shuffle hidden decks).
      2. Descend the tree using ISMCTS-UCB1 (subset-armed bandit selection).
      3. Expand one new child node.
      4. Rollout with HeuristicAgent.
      5. Backpropagate reward and update availability counts.

    Args:
        n_iterations: Total ISMCTS iterations per decision (default 1000).
        exploration_constant: UCB exploration parameter C (default √2 ≈ 1.414).
        name: Agent name for logging.
    """

    def __init__(self, n_iterations=200, exploration_constant=10*1.414, name="ISMCTSAgent", status_dict=None, worker_id=None):
        super().__init__(name)
        self.n_iterations = n_iterations
        self.exploration_constant = exploration_constant
        self._rollout_agent = HeuristicAgent(name="ISMCTS_Rollout")
        self.root = None
        self.status_dict = status_dict
        self.worker_id = worker_id
        self.ctx_game = 1
        self.ctx_total_games = 1
        self.ctx_turn = 1
        
    def set_context(self, game, total_games, turn):
        self.ctx_game = game
        self.ctx_total_games = total_games
        self.ctx_turn = turn

    def reset(self):
        """Reset the search tree for a new game."""
        self.root = None

    def select_action(self, obs, env=None):
        """Select the best action via ISMCTS.

        Args:
            obs: Observation dict from RegicideEnv.
            env: The live RegicideEnv instance (required for cloning).

        Returns:
            The action mask with the highest visit count in the ISMCTS tree.
        """
        if env is None:
            raise ValueError("ISMCTSAgent requires the env object for cloning.")

        import numpy as np
        action_mask = obs['action_mask']
        valid_actions = np.nonzero(action_mask)[0].tolist()
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Use existing tree or build a new one
        if self.root is None:
            self.root = ISMCTSNode()
            
        root = self.root

        import time
        start_time = time.time()
        last_update_time = start_time

        for i in range(self.n_iterations):
            current_time = time.time()
            if self.status_dict is not None and current_time - last_update_time > 0.5:
                pct = int((i / self.n_iterations) * 100)
                bar = "#" * (pct // 10) + "-" * (10 - (pct // 10))
                elapsed = current_time - start_time
                its = i / elapsed if elapsed > 0 else 0
                self.status_dict[self.worker_id] = (
                    f"Game {self.ctx_game}/{self.ctx_total_games} | "
                    f"Turn {self.ctx_turn:2d} | "
                    f"ISMCTS [{bar}] {i:4d}/{self.n_iterations} | {its:6.1f} it/s"
                )
                last_update_time = current_time
            # 1. Clone and determinize
            sim_env = env.clone()
            determinize_env(sim_env)

            # 2-5. Select → Expand → Rollout → Backpropagate
            self._run_iteration(root, sim_env)

        # Pick action with highest visit count (most robust choice)
        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visit_count
        )
        
        # Advance the root for the next turn
        self.root = root.children.get(best_action, None)

        return best_action

    def _run_iteration(self, root, sim_env):
        """Run one full ISMCTS iteration: select → expand → rollout → backprop.

        Args:
            root: The root ISMCTSNode.
            sim_env: A cloned + determinized RegicideEnv.
        """
        node = root
        path = [node]
        sim_obs = sim_env._get_obs()

        # --- SELECTION: descend the tree while all children are expanded ---
        import numpy as np
        cumulative_tree_reward = 0.0
        terminated = truncated = False
        
        while not (sim_env.game.game_over):
            action_mask = sim_obs['action_mask']
            legal_actions = np.nonzero(action_mask)[0].tolist()

            if not legal_actions:
                break

            # Update availability counts for all legal actions that have nodes
            for action in legal_actions:
                if action in node.children:
                    node.children[action].availability_count += 1

            # Find untried actions (legal actions with no child node)
            untried = [a for a in legal_actions if a not in node.children]

            if untried:
                # --- EXPANSION: create a new child for an untried action ---
                import random
                action = random.choice(untried)
                child = ISMCTSNode(action=action, parent=node)
                child.availability_count = 1  # First time available
                node.children[action] = child

                # Apply the action
                sim_obs, step_reward, terminated, truncated, info = sim_env.step(action)
                cumulative_tree_reward += step_reward

                path.append(child)
                node = child
                break  # Expand one node, then rollout
            else:
                # All legal actions have been tried — select best via UCB
                action = max(
                    legal_actions,
                    key=lambda a: node.children[a].ucb_score(self.exploration_constant)
                )
                child = node.children[action]

                # Apply the action
                sim_obs, step_reward, terminated, truncated, info = sim_env.step(action)
                cumulative_tree_reward += step_reward

                path.append(child)
                node = child

                if terminated or truncated:
                    break

        # --- ROLLOUT: play to completion with heuristic ---
        if not (sim_env.game.game_over or terminated or truncated):
            reward = cumulative_tree_reward + self._rollout(sim_env, sim_obs)
        else:
            # Game already ended during tree traversal (or invalid action terminated it)
            reward = cumulative_tree_reward + self._evaluate_terminal(sim_env)

        # --- BACKPROPAGATION: update all nodes on the path ---
        for n in path:
            n.visit_count += 1
            n.total_reward += reward

    def _rollout(self, env, obs):
        """Play the game to completion using the heuristic agent.

        Combines cumulative env rewards with a progress-based terminal
        evaluation for better signal in the search tree.

        Args:
            env: A cloned RegicideEnv in mid-game state.
            obs: The current observation.

        Returns:
            Combined reward: cumulative intermediate rewards + progress score.
        """
        done = False
        cumulative_reward = 0.0

        while not done:
            action = self._rollout_agent.select_action(obs, env=env)
            if action is None:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            done = terminated or truncated

        return cumulative_reward + self._evaluate_terminal(env)

    @staticmethod
    def _evaluate_terminal(env):
        """Score a terminal game state by progress.

        Returns:
            Float in [0.0, 2.0]: enemies_defeated/12 + 1.0 if victory.
        """
        enemies_left = len(env.game.castle_deck) + (
            1 if env.game.current_enemy and not env.game.victory else 0
        )
        enemies_defeated = 12 - enemies_left
        progress = enemies_defeated / 12.0

        if env.game.victory:
            return progress + 1.0
        return progress
