import multiprocessing as mp
import time
from .env import RegicideEnv

def _worker_simulate(args):
    """
    Worker function to simulate games.
    We instantiate the agent and environment inside the worker to avoid pickling issues.
    """
    agent_cls, agent_kwargs, num_games, worker_id = args
    
    status_dict = agent_kwargs.pop('status_dict', None)
    
    agent_kwargs['status_dict'] = status_dict
    agent_kwargs['worker_id'] = worker_id
    
    # Instantiate agent inside worker
    agent = agent_cls(**agent_kwargs)
    env = RegicideEnv(num_players=1)
    
    results = {
        'victories': 0,
        'enemies_defeated': [],
        'total_turns': [],
        'games_played': num_games
    }
    
    for game_idx in range(num_games):
        obs, info = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()
            
        done = False
        turns = 0
        
        while not done:
            if hasattr(agent, 'set_context'):
                agent.set_context(game_idx + 1, num_games, turns + 1)
            elif status_dict is not None:
                status_dict[worker_id] = f"Game {game_idx+1}/{num_games} | Turn {turns+1:2d} | Thinking..."
                
            action = agent.select_action(obs, env=env)
                
            if action is None:
                break
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turns += 1
            
        if env.game.victory:
            results['victories'] += 1
            
        enemies_left = len(env.game.castle_deck) + (1 if env.game.current_enemy and not env.game.victory else 0)
        results['enemies_defeated'].append(12 - enemies_left)
        results['total_turns'].append(turns)
        
    if status_dict is not None:
        status_dict[worker_id] = "Finished"
        
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
        
        manager = mp.Manager()
        status_dict = manager.dict()
        
        for i in range(1, self.n_jobs + 1):
            status_dict[i] = "Starting..."
        
        # Divide games among workers
        games_per_worker = total_games // self.n_jobs
        remainder = total_games % self.n_jobs
        
        worker_args = []
        for i in range(self.n_jobs):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games > 0:
                w_kwargs = dict(agent_kwargs)
                w_kwargs['status_dict'] = status_dict
                worker_args.append((agent_cls, w_kwargs, n_games, i + 1))
                
        if not worker_args:
            return None
            
        import sys
        print("\n" * self.n_jobs)  # Pre-allocate lines for the dashboard
        
        with mp.Pool(self.n_jobs) as pool:
            result = pool.starmap_async(_worker_simulate, [(args,) for args in worker_args])
            
            while not result.ready():
                # Move cursor up and print worker states
                sys.stdout.write(f"\033[{self.n_jobs}F")
                for i in range(1, self.n_jobs + 1):
                    # Clear line and print
                    sys.stdout.write("\033[K")
                    status = status_dict.get(i, "Idle")
                    print(f"Worker {i}: {status}")
                sys.stdout.flush()
                time.sleep(0.5)
                
            results_list = result.get()
            
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
