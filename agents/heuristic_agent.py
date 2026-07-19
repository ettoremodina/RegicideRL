"""Rule-based baseline that scores legal Regicide actions by hand strategy."""

import random
import numpy as np

from game.action_space import SOLO_JESTER_ACTION_ID
from agents.base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    """Score legal actions using defense, damage, suit, and conservation rules."""
    
    def select_action(self, obs, env=None):
        """Return the highest-scoring legal action with randomized tie-breaking.

        Args:
            obs: Observation returned by ``RegicideEnv``.
            env: Live environment used to inspect cards and enemy state.

        Returns:
            Selected global action identifier, or ``None`` if the mask is empty.

        Raises:
            ValueError: If ``env`` is omitted.
        """
        action_scores = self.score_actions(obs, env)
        if not action_scores:
            return None
        return max(
            action_scores,
            key=lambda action_id: (
                action_scores[action_id] + random.uniform(0, 0.1)
            ),
        )

    def score_actions(self, obs, env=None):
        """Return the rule-based score of every legal action."""
        if env is None:
            raise ValueError(
                "HeuristicAgent requires the env object to access game state"
            )
        valid_actions = np.flatnonzero(obs["action_mask"])
        hand = obs["hand"]
        game = env.game
        return {
            int(action_id): self._evaluate_action(
                int(action_id),
                hand,
                game,
                env,
            )
            for action_id in valid_actions
        }
        
    def _evaluate_action(self, action_id, hand, game, env):
        """Assign a strategic score to one legal attack or defense action.

        Exact defense and exact kills receive the strongest bonuses. The policy
        penalizes excess defense, premature yielding, and avoidable use of face
        cards while preferring useful suit powers against threatening enemies.

        Args:
            action_id: Global action-space identifier.
            hand: Current player's sorted cards.
            game: Live ``Game`` state.
            env: Environment whose handler decodes the action.

        Returns:
            Heuristic utility; larger values are preferred.
        """
        # Base score
        score = 0.0
        
        if action_id == SOLO_JESTER_ACTION_ID:
            # Solo jester has no specific card from hand
            action_cards = []
        elif action_id <= 285:
            # Global attack actions are statically defined, avoid decoding
            if action_id == 0:
                action_cards = []
            else:
                action_cards = env.handler._global_attack_actions[action_id]["cards"]
        else:
            hand_indices = env.handler.global_action_to_hand_indices(action_id, hand)
            action_cards = [hand[i] for i in hand_indices]
        
        is_yield = (action_id == 0 and env.required_defense == 0)
        is_defense = env.required_defense > 0
        enemy = game.current_enemy
        
        # --- DEFENSE PHASE ---
        if is_defense:
            if is_yield:
                return -1000.0 # Yielding during defense means death, unless we literally have no choice.
                
            defense_val = sum(c.get_discard_value() for c in action_cards)
            
            # Rule: Perfect Defense (+500)
            if defense_val == env.required_defense:
                score += 500.0
            
            # Rule: Don't waste excess defense. Penalize over-defending.
            over_defense = defense_val - env.required_defense
            if over_defense > 0:
                score -= over_defense * 10
                
            # Rule: Preserve face cards if possible
            for c in action_cards:
                if c.value >= 11:
                    score -= 50.0
                    
            return score
            
        # --- ATTACK PHASE ---
        if is_yield:
            # Yielding when we could attack is usually bad unless our hand is amazing and we want to draw.
            # For a simple baseline, yielding is discouraged unless we have no good cards.
            return -100.0
            
        if enemy is None:
            return 0.0
            
        attack_val = sum(c.get_attack_value() for c in action_cards)
        suits_played = [c.suit.value for c in action_cards]
        
        enemy_health = enemy.health - enemy.damage_taken
        enemy_attack = enemy.attack
        enemy_suit = enemy.card.suit.value
        
        # Check if Jester is played (value 0)
        has_jester = any(c.value == 0 for c in action_cards) or (
            action_id == SOLO_JESTER_ACTION_ID
        )
        if has_jester:
            # Rule: Play Jester against Kings/Queens to cancel immunities (+300)
            if enemy.card.value >= 12:
                score += 300.0
            else:
                score -= 100.0 # Don't waste Jester on weak enemies
            return score # Jester is played alone, so return early
            
        # Check for Clubs doubling
        immunity_cancelled = game.jester_immunity_cancelled
        immune_to_clubs = (enemy_suit == "♣") and not immunity_cancelled
        if "♣" in suits_played and not immune_to_clubs:
            attack_val *= 2
            
        # Rule: Perfect Kill (+500)
        if attack_val == enemy_health:
            score += 500.0
            
        # Rule: Prioritize Clubs if enemy attack is high (+100)
        if enemy_attack >= 15 and "♣" in suits_played and not immune_to_clubs:
            score += 100.0
            
        # Rule: Prioritize Spades if enemy attack is high to shield (+100)
        immune_to_spades = (enemy_suit == "♠") and not immunity_cancelled
        if enemy_attack >= 10 and "♠" in suits_played and not immune_to_spades:
            score += 100.0
            
        # Rule: Prioritize Diamonds if hand is low (+100)
        immune_to_diamonds = (enemy_suit == "♦") and not immunity_cancelled
        if len(hand) < 4 and "♦" in suits_played and not immune_to_diamonds:
            score += 100.0
            
        # Rule: Don't waste face cards unless it's a perfect kill or strong hit
        for c in action_cards:
            if c.value >= 11 and attack_val != enemy_health:
                score -= 50.0
                
        return score
