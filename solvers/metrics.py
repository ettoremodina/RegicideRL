import os
import json
import matplotlib.pyplot as plt

def plot_metrics(run_dir):
    """
    Reads metrics.json in run_dir and generates a summary plot.
    """
    metrics_file = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        print(f"No metrics found at {metrics_file}")
        return
        
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        return
        
    # Read distributions if available
    dist_file = os.path.join(run_dir, "latest_distributions.json")
    enemies_dist = []
    turns_dist = []
    if os.path.exists(dist_file):
        with open(dist_file, 'r') as f:
            dist_data = json.load(f)
            enemies_dist = dist_data.get('enemies_distribution', [])
            turns_dist = dist_data.get('turns_distribution', [])
            
    steps = [d.get('step', i) for i, d in enumerate(data)]
    
    win_rates = [d.get('win_rate', 0) * 100 for d in data]
    avg_enemies = [d.get('avg_enemies_defeated', 0) for d in data]
    avg_turns = [d.get('avg_turns', 0) for d in data]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Regicide Training Progress', fontsize=16)
    
    # [0, 0] Win Rate over time
    axes[0, 0].plot(steps, win_rates, 'g-', marker='o')
    axes[0, 0].set_title('Win Rate over Training')
    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # [0, 1] Average Game Length over time
    axes[0, 1].plot(steps, avg_turns, 'orange', marker='o')
    axes[0, 1].set_title('Average Episode Length')
    axes[0, 1].set_ylabel('Steps / Turns')
    axes[0, 1].grid(True, alpha=0.3)
    
    # [0, 2] Average Bosses (Enemies) Killed
    axes[0, 2].plot(steps, avg_enemies, 'b-', marker='o')
    axes[0, 2].set_title('Average Bosses Killed')
    axes[0, 2].set_ylabel('Bosses Killed (Max 12)')
    axes[0, 2].set_ylim(0, 12.5)
    axes[0, 2].grid(True, alpha=0.3)
    
    # [1, 0] Could be reward distribution in RL, but for now we'll plot Speed
    speeds = [d.get('games_per_second', 0) for d in data]
    axes[1, 0].plot(steps, speeds, 'purple', marker='o')
    axes[1, 0].set_title('Simulation Speed')
    axes[1, 0].set_xlabel('Training Step / Episode')
    axes[1, 0].set_ylabel('Games / Second')
    axes[1, 0].grid(True, alpha=0.3)
    
    # [1, 1] Episode Length Distribution (from latest step)
    if turns_dist:
        axes[1, 1].hist(turns_dist, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Latest Game Length Distribution')
        axes[1, 1].set_xlabel('Turns')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    # [1, 2] Boss Kills Distribution (from latest step)
    if enemies_dist:
        max_bosses = max(enemies_dist) if enemies_dist else 0
        bins = range(0, max(max_bosses + 2, 2))
        axes[1, 2].hist(enemies_dist, bins=bins, align='left', alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 2].set_title('Latest Boss Kills Distribution')
        axes[1, 2].set_xlabel('Bosses Killed')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(run_dir, "plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to {plot_file}")
