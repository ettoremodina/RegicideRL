import os
import json
import time
from datetime import datetime
import logging

class RunLogger:
    """
    Manages saving logs, metrics, and models for a specific training run.
    """
    def __init__(self, run_name=None, base_dir="runs"):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.models_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.run_dir, "metrics.json")
        self.metrics = []
        
        # Setup file logger
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.run_dir, "logs.txt"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)
        
        self.logger.info(f"Initialized run: {run_name}")
        self.logger.info(f"Run directory: {self.run_dir}")
        
    def log(self, message):
        self.logger.info(message)
        
    def log_metrics(self, step, metrics_dict):
        metrics_dict['step'] = step
        metrics_dict['timestamp'] = time.time()
        
        # Extract distributions to prevent metrics.json from bloating over many episodes
        enemies_dist = metrics_dict.pop('enemies_distribution', [])
        turns_dist = metrics_dict.pop('turns_distribution', [])
        
        self.metrics.append(metrics_dict)
        
        # Save main metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save only the latest distribution
        dist_file = os.path.join(self.run_dir, "latest_distributions.json")
        with open(dist_file, 'w') as f:
            json.dump({
                'step': step,
                'enemies_distribution': enemies_dist,
                'turns_distribution': turns_dist
            }, f, indent=2)
            
    def get_run_dir(self):
        return self.run_dir
