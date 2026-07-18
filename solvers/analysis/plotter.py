import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from ml_logger import get_logger

logger = get_logger(__name__)

def plot_dashboard(probe_results, tb_data, output_dir):
    """
    Generates a unified dashboard with training curves and evaluation distributions.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Regicide RL Policy Analysis Dashboard", fontsize=18)
    
    # 1. Training Reward Curve
    ax1 = plt.subplot(2, 2, 1)
    if 'reward' in tb_data and not tb_data['reward'].empty:
        sns.lineplot(data=tb_data['reward'], x='step', y='value', ax=ax1, color='blue')
        ax1.set_title("Training Reward Curve (TensorBoard)")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Mean Reward")
    else:
        ax1.text(0.5, 0.5, "No TensorBoard reward data found.", ha='center', va='center')
        ax1.set_title("Training Reward Curve")
        
    # 2. Evaluation Bosses Defeated
    ax2 = plt.subplot(2, 2, 2)
    bosses = probe_results.get('bosses_killed', [])
    if bosses:
        sns.histplot(bosses, bins=range(0, 14), discrete=True, ax=ax2, color='green')
        ax2.set_title(f"Bosses Defeated Distribution (N={len(bosses)} games)")
        ax2.set_xlabel("Bosses Defeated")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, 12)
        
    # 3. Action Distribution
    ax3 = plt.subplot(2, 2, 3)
    actions = probe_results.get('actions_taken', [])
    if actions:
        action_counts = Counter(actions)
        # Sort by frequency
        top_actions = action_counts.most_common(15)
        indices = [str(x[0]) for x in top_actions]
        counts = [x[1] for x in top_actions]
        sns.barplot(x=indices, y=counts, ax=ax3, palette="viridis")
        ax3.set_title("Top 15 Most Used Actions (Action Mask Indices)")
        ax3.set_xlabel("Action ID")
        ax3.set_ylabel("Frequency")
        ax3.tick_params(axis='x', rotation=45)
        
    # 4. Policy Entropy
    ax4 = plt.subplot(2, 2, 4)
    entropies = probe_results.get('entropies', [])
    if entropies:
        sns.histplot(entropies, bins=50, ax=ax4, color='purple', kde=True)
        ax4.set_title("Policy Entropy Distribution (Confidence)")
        ax4.set_xlabel("Entropy (Lower = More Confident)")
        ax4.set_ylabel("Density")
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = f"{output_dir}/comprehensive_policy_analysis.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Dashboard saved to %s", save_path)
    return save_path
