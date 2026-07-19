"""Parallel solo-game evaluation with optional persistent game recording."""

import multiprocessing as mp
import time
from integrations.regicide_logging import GameRecorder
from ml_logger import RunContext, get_logger
from .env import RegicideEnv

logger = get_logger(__name__)


def _worker_simulate(args):
    """
    Worker function to simulate games.
    We instantiate the agent and environment inside the worker to avoid pickling issues.
    """
    agent_cls, agent_kwargs, num_games, worker_id, recording = args
    
    status_dict = agent_kwargs.pop('status_dict', None)
    
    agent_kwargs['status_dict'] = status_dict
    agent_kwargs['worker_id'] = worker_id
    
    # Instantiate agent inside worker
    agent = agent_cls(**agent_kwargs)
    recorder = None
    if recording:
        recorder = GameRecorder.from_descriptor(
            recording["context"],
            recording["level"],
        )
    env = RegicideEnv(num_players=1, recorder=recorder)
    
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
            elif status_dict is not None and turns == 0 and game_idx % max(1, num_games // 20) == 0:
                status_dict[worker_id] = f"Game {game_idx+1}/{num_games} | Running..."
                
            action = agent.select_action(obs, env=env)
                
            if action is None:
                if recorder and recorder.active:
                    recorder.finish(env.game, reason="agent_returned_no_action")
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
    Uses a persistent pool to avoid massive spawn overhead on Windows.
    """
    def __init__(
        self,
        n_jobs=None,
        run_context: RunContext | None = None,
        recording_level=None,
    ):
        if n_jobs is None:
            self.n_jobs = max(1, mp.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)
        self.recording = None
        if run_context:
            recorder = GameRecorder(run_context, recording_level)
            if recorder.enabled:
                self.recording = {
                    "context": run_context.descriptor(),
                    "level": recorder.recording_level,
                }

        self.manager = None
        self.status_dict = None
        self.pool = None
        if self.n_jobs > 1:
            self.manager = mp.Manager()
            self.status_dict = self.manager.dict()
            self.pool = mp.Pool(self.n_jobs)
        
    def close(self):
        """Join worker processes and stop the multiprocessing manager."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
            
    def run_eval(self, agent_cls, agent_kwargs, total_games):
        """
        Runs `total_games` games in parallel and returns aggregated metrics.
        """
        start_time = time.time()

        if self.n_jobs == 1:
            worker_args = (
                agent_cls,
                dict(agent_kwargs),
                total_games,
                1,
                self.recording,
            )
            results_list = [_worker_simulate(worker_args)]
            return self._aggregate_results(results_list, total_games, start_time)
        
        for i in range(1, self.n_jobs + 1):
            self.status_dict[i] = "Starting..."
        
        # Divide games among workers
        games_per_worker = total_games // self.n_jobs
        remainder = total_games % self.n_jobs
        
        worker_args = []
        for i in range(self.n_jobs):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games > 0:
                w_kwargs = dict(agent_kwargs)
                w_kwargs['status_dict'] = self.status_dict
                worker_args.append(
                    (agent_cls, w_kwargs, n_games, i + 1, self.recording)
                )
                
        if not worker_args:
            return None
            
        result = self.pool.starmap_async(_worker_simulate, [(args,) for args in worker_args])
        
        while not result.ready():
            statuses = [
                f"{worker_id}:{self.status_dict.get(worker_id, 'Idle')}"
                for worker_id in range(1, self.n_jobs + 1)
            ]
            logger.debug("Worker status: %s", " | ".join(statuses))
            time.sleep(0.5)
            
        results_list = result.get()
            
        return self._aggregate_results(results_list, total_games, start_time)

    @staticmethod
    def _aggregate_results(results_list, total_games, start_time):
        """Merge worker summaries into the evaluation metric contract."""
        elapsed = time.time() - start_time
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
