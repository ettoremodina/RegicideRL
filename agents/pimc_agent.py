"""
Perfect Information Monte Carlo (PIMC) agent for Regicide.

Phase 1 baseline: for each decision point, sample N determinizations of the
hidden state, evaluate each candidate action by rolling out the game using
the heuristic agent, and pick the action with the highest average outcome.

This is the simplest search-based approach and serves as a control group
to measure how much ISMCTS (Phase 2) improves over naive determinization.

Reference: Discussed in Cowling et al. "Information Set Monte Carlo Tree Search"
and Bjarnason et al. "Lower Bounding Klondike Solitaire with Monte-Carlo Planning".

Known weakness: "Strategy fusion" — the agent evaluates each determinization
independently and implicitly assumes it can make different future choices for
different determinizations. ISMCTS fixes this.
"""

import logging
from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from agents.determinize import determinize_env

logger = logging.getLogger("pimc_agent")


class PIMCAgent(BaseAgent):
    """Perfect Information Monte Carlo search agent.

    For every decision, evaluates each legal action by:
      1. Cloning the environment N times (one per determinization).
      2. Shuffling the hidden decks in each clone.
      3. Applying the candidate action.
      4. Rolling out to game end with the HeuristicAgent.
      5. Averaging the terminal reward across all determinizations.

    Returns the action with the highest average reward.

    Args:
        n_determinizations: Number of random deck shuffles per action (default 50).
        name: Agent name for logging.
    """

    def __init__(self, n_determinizations=50, name="PIMCAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_determinizations = n_determinizations
        self._rollout_agent = HeuristicAgent(name="PIMC_Rollout")

    def select_action(self, obs, env=None):
        """Select the best action via PIMC search.

        Args:
            obs: Observation dict from RegicideEnv.
            env: The live RegicideEnv instance (required for cloning).

        Returns:
            The action mask with the highest average determinized reward.
        """
        if env is None:
            raise ValueError("PIMCAgent requires the env object for cloning.")

        import numpy as np
        action_mask = obs['action_mask']
        valid_actions = np.nonzero(action_mask)[0].tolist()
        
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_action = None
        best_avg_reward = float('-inf')

        for action in valid_actions:
            total_reward = 0.0

            for _ in range(self.n_determinizations):
                # Clone → determinize → apply action → rollout
                sim_env = env.clone()
                determinize_env(sim_env)

                # Apply the candidate action
                sim_obs, step_reward, terminated, truncated, info = sim_env.step(action)

                # If the game didn't end, roll out with heuristic
                if not (terminated or truncated or sim_env.game.game_over):
                    reward = step_reward + self._rollout(sim_env, sim_obs)
                else:
                    reward = step_reward + self._evaluate_terminal(sim_env)

                total_reward += reward

            avg_reward = total_reward / self.n_determinizations
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_action = action

        return best_action

    def _rollout(self, env, obs):
        """Play the game to completion using the heuristic agent.

        Combines cumulative env rewards (intermediate shaping signals like
        +0.1 per enemy defeated) with a progress-based terminal evaluation
        that replaces the sparse +1/-1 win/loss signal.

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

        # Replace the sparse terminal component (+1/-1) with a progress-based
        # score, but keep intermediate shaping rewards accumulated along the way.
        # Strip the terminal reward (last step's reward already in cumulative)
        # and add our own progress evaluation instead.
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

