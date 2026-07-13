import multiprocessing as mp
import time
from .env import RegicideEnv

def _worker_simulate(agent_cls, agent_kwargs, num_games):
    """
    Worker function to simulate games.
    We instantiate the agent and environment inside the worker to avoid pickling issues.
    """
    # Instantiate agent inside worker
    agent = agent_cls(**agent_kwargs)
    env = RegicideEnv(num_players=1)
    
    results = {
        'victories': 0,
        'enemies_defeated': [],
        'total_turns': [],
        'games_played': num_games
    }
    
    for _ in range(num_games):
        obs, info = env.reset()
        done = False
        turns = 0
        
        while not done:
            action = agent.select_action(obs, env=env)
            if action is None:
                # Agent surrendered / no valid actions
                break
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turns += 1
            
        if env.game.victory:
            results['victories'] += 1
            
        enemies_left = len(env.game.castle_deck) + (1 if env.game.current_enemy and not env.game.victory else 0)
        results['enemies_defeated'].append(12 - enemies_left)
        results['total_turns'].append(turns)
        
    return results

class ParallelSimulator:
    """
    Simulates thousands of games across multiple CPU cores.
    """
    def __init__(self, n_jobs=None):
        if n_jobs is None:
            self.n_jobs = max(1, mp.cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
            
    def run_eval(self, agent_cls, agent_kwargs, total_games):
        """
        Runs `total_games` games in parallel and returns aggregated metrics.
        """
        start_time = time.time()
        
        # Divide games among workers
        games_per_worker = total_games // self.n_jobs
        remainder = total_games % self.n_jobs
        
        worker_args = []
        for i in range(self.n_jobs):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games > 0:
                worker_args.append((agent_cls, agent_kwargs, n_games))
                
        if not worker_args:
            return None
            
        with mp.Pool(self.n_jobs) as pool:
            # Map returns a list of result dicts
            results_list = pool.starmap(_worker_simulate, worker_args)
            
        elapsed = time.time() - start_time
        
        # Aggregate results
        total_victories = sum(r['victories'] for r in results_list)
        all_enemies = []
        all_turns = []
        for r in results_list:
            all_enemies.extend(r['enemies_defeated'])
            all_turns.extend(r['total_turns'])
            
        metrics = {
            'win_rate': total_victories / total_games,
            'avg_enemies_defeated': sum(all_enemies) / total_games,
            'avg_turns': sum(all_turns) / total_games,
            'games_per_second': total_games / elapsed,
            'total_time': elapsed,
            'enemies_distribution': all_enemies,
            'turns_distribution': all_turns
        }
        
        return metrics
