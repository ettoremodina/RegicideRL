import random
from solvers.agents.base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    A purely rule-based heuristic agent.
    Evaluates every valid action based on a set of hardcoded strategic rules and returns the one with the highest score.
    """
    
    def select_action(self, obs, env=None):
        if env is None:
            raise ValueError("HeuristicAgent requires the env object to be passed in to access game state directly.")
            
        valid_actions = obs['valid_actions']
        if not valid_actions:
            return None
            
        hand = obs['hand']
        game = env.game
        
        # If there's only one action, just take it (often the case when yielding or forced to play one thing)
        if len(valid_actions) == 1:
            return valid_actions[0]
            
        best_action = None
        best_score = float('-inf')
        
        for action_mask in valid_actions:
            score = self._evaluate_action(action_mask, hand, game, env)
            
            # Add a tiny bit of random noise to tie-break equivalent actions
            score += random.uniform(0, 0.1)
            
            if score > best_score:
                best_score = score
                best_action = action_mask
                
        return best_action
        
    def _evaluate_action(self, action_mask, hand, game, env):
        # Base score
        score = 0.0
        
        indices = env.handler.mask_to_card_indices(action_mask, len(hand))
        action_cards = [hand[i] for i in indices]
        
        is_yield = env.handler.is_yield_action(action_mask)
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
        has_jester = any(c.value == 0 for c in action_cards)
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
