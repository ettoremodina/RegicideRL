import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_latest_run(logdir_base="runs/rl_logs"):
    """Finds the most recently modified PPO run directory."""
    runs = glob.glob(os.path.join(logdir_base, "MaskablePPO_*"))
    if not runs:
        return None
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def extract_tb_logs(logdir):
    """
    Extracts key scalar metrics from a TensorBoard event file.
    Returns a dictionary of Pandas DataFrames.
    """
    if not logdir or not os.path.exists(logdir):
        print(f"Log directory not found: {logdir}")
        return {}
        
    print(f"Extracting logs from: {logdir}")
    # Load events
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Check available tags
    tags = event_acc.Tags()['scalars']
    
    metrics = {
        'reward': 'rollout/ep_rew_mean',
        'length': 'rollout/ep_len_mean',
        'loss': 'train/loss',
        'value_loss': 'train/value_loss'
    }
    
    dfs = {}
    for name, tag in metrics.items():
        if tag in tags:
            events = event_acc.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
            dfs[name] = df
        else:
            print(f"Warning: Tag {tag} not found in logs.")
            
    return dfs
