import time
import cProfile
import pstats
from solvers.env import RegicideEnv
from agents.random_agent import RandomAgent

def run_benchmark():
    env = RegicideEnv(num_players=1)
    agent = RandomAgent()
    
    start = time.time()
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, env=env)
            if action is None:
                break
            obs, r, t, trunc, _ = env.step(action)
            done = t or trunc
    print("Elapsed:", time.time() - start)

if __name__ == "__main__":
    cProfile.run("run_benchmark()", "worker_profile.prof")
    p = pstats.Stats("worker_profile.prof")
    p.strip_dirs().sort_stats("tottime").print_stats(30)
