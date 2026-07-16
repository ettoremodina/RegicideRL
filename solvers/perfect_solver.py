import time
from typing import Optional, List, Set, Tuple
from solvers.env import RegicideEnv

class GameStateHasher:
    @staticmethod
    def hash_env(env: RegicideEnv) -> tuple:
        """
        Creates a deterministic, immutable hash of the current game state 
        for memoization (transposition tables).
        """
        game = env.game
        
        # Hash players' hands
        # Using string representation of cards for simplicity and immutability
        player_hands = tuple(
            tuple(str(c) for c in sorted(hand, key=lambda card: (card.value, card.suit.value)))
            for hand in game.players
        )
        
        tavern = tuple(str(c) for c in game.tavern_deck)
        discard = tuple(str(c) for c in game.discard_pile)
        castle = tuple(str(c) for c in game.castle_deck)
        
        if game.current_enemy:
            enemy = (
                game.current_enemy.card.value,
                game.current_enemy.card.suit.value,
                game.current_enemy.health,
                game.current_enemy.damage_taken,
                game.current_enemy.spade_protection
            )
        else:
            enemy = None
            
        return (
            env.required_defense,
            game.current_player,
            enemy,
            player_hands,
            tavern,
            discard,
            castle,
            game.jester_immunity_cancelled,
            game.solo_jesters_remaining,
            game.solo_jesters_used,
            tuple(game.players_yielded_this_round),
            game.last_active_player,
        )

class PerfectSolver:
    def __init__(self, verbose: bool = False, callback=None, callback_freq: int = 1000):
        self.visited: Set[tuple] = set()
        self.nodes_evaluated = 0
        self.verbose = verbose
        self.callback = callback
        self.callback_freq = callback_freq
        
    def solve(self, initial_env: RegicideEnv) -> Optional[List[int]]:
        """
        Runs Depth-First Search with Memoization to find a winning sequence of actions.
        Returns the list of action indices if a win is possible, otherwise None.
        """
        self.visited.clear()
        self.nodes_evaluated = 0
        
        start_time = time.time()
        
        # Ensure the environment is running in deterministic mode
        if not initial_env.game.deterministic_hearts:
            if self.verbose:
                print("WARNING: deterministic_hearts is False. The solver assumes a deterministic game.")
                
        result = self._dfs(initial_env)
        
        # Make sure to call callback one last time if it's not a multiple
        if self.callback and self.nodes_evaluated % self.callback_freq != 0:
            self.callback(self.nodes_evaluated % self.callback_freq)
            
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"PerfectSolver finished in {elapsed:.2f}s. Evaluated {self.nodes_evaluated} nodes.")
            print(f"Unique states visited (Transposition Table size): {len(self.visited)}")
            
        return result
        
    def _dfs(self, env: RegicideEnv) -> Optional[List[int]]:
        self.nodes_evaluated += 1
        
        if self.callback and self.nodes_evaluated % self.callback_freq == 0:
            self.callback(self.callback_freq)
        
        # Check termination
        if env.game.victory:
            return []
        if env.game.game_over:
            return None
            
        # Hash state and check memoization
        state_hash = GameStateHasher.hash_env(env)
        if state_hash in self.visited:
            return None
            
        self.visited.add(state_hash)
        
        # Get valid actions
        obs = env._get_obs()
        valid_actions = obs['valid_actions']
        
        # We pair each mask with its integer representation
        action_tuples = []
        for mask in valid_actions:
            idx = sum(val * (1 << j) for j, val in enumerate(mask))
            action_tuples.append((idx, mask))
            
        # Sort actions heuristically to prune the tree faster.
        # This only affects the ORDER we explore, not the final result.
        def action_heuristic(item):
            idx, mask = item
            cards_played = sum(mask)
            is_yield = (cards_played == 0)
            
            if obs['defense_phase']:
                # For defense, try playing fewer cards first
                return (is_yield, cards_played)
            else:
                # For attacks, try playing more cards (stronger combos) first
                return (is_yield, -cards_played)
                
        action_tuples.sort(key=action_heuristic)
        
        for action_idx, mask in action_tuples:
            # Clone env to branch
            next_env = env.clone()
            next_env.step(action_idx)
            
            # Recursive DFS
            path = self._dfs(next_env)
            if path is not None:
                return [action_idx] + path
                
        # All actions led to loss (or cyclic state)
        return None
