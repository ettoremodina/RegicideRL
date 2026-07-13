import time
from sb3_contrib.ppo_mask import MaskablePPO
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
import torch

def run_benchmark(device):
    print(f"\n--- Benchmarking on {device.upper()} ---")
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=0,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
    )
    
    start_time = time.time()
    model.learn(total_timesteps=10000)
    end_time = time.time()
    
    elapsed = end_time - start_time
    fps = 10000 / elapsed
    print(f"Device: {device.upper()}")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print(f"FPS: {fps:.2f} steps/second")
    
    return fps

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. Cannot benchmark GPU.")
    else:
        fps_cpu = run_benchmark("cpu")
        fps_gpu = run_benchmark("cuda")
        
        print("\n--- Summary ---")
        if fps_gpu > fps_cpu:
            speedup = fps_gpu / fps_cpu
            print(f"GPU is {speedup:.2f}x FASTER than CPU.")
        else:
            speedup = fps_cpu / fps_gpu
            print(f"CPU is {speedup:.2f}x FASTER than GPU (typical for small environments!).")
