"""Generate legacy text and JSON summaries from PPO probe results."""

import json
import numpy as np
from collections import Counter
from datetime import datetime
from ml_logger import get_logger

logger = get_logger(__name__)

def generate_reports(probe_results, output_dir):
    """
    Takes the raw lists from the prober and generates statistical summaries
    saving them as a JSON and a TXT file in the output directory.
    """
    total_decisions = probe_results['total_decisions']
    if total_decisions == 0:
        return
        
    actions = probe_results['actions_taken']
    entropies = probe_results['entropies']
    bosses = probe_results['bosses_killed']
    rewards = probe_results['rewards']
    
    action_counts = Counter(actions)
    most_common_action = action_counts.most_common(1)[0]
    
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_decisions": total_decisions,
        "total_games": probe_results['total_games'],
        "unique_actions_used": len(action_counts),
        "most_common_action": most_common_action,
        "action_distribution": dict(action_counts),
        "mean_entropy": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "mean_bosses_killed": float(np.mean(bosses)),
        "max_bosses_killed": int(np.max(bosses)),
        "win_rate": float(np.mean(np.array(bosses) == 12)),
        "mean_episode_reward": float(np.mean(rewards)),
        "yield_rate": probe_results['yields'] / total_decisions,
        "scenarios": probe_results['scenarios']
    }
    
    # Save JSON
    with open(f"{output_dir}/analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Save TXT
    report = f"""Regicide RL Policy Analysis Report
Generated: {summary['analysis_timestamp']}

--- High-Level Performance ---
Games Played: {summary['total_games']}
Win Rate: {summary['win_rate'] * 100:.2f}%
Average Bosses Defeated: {summary['mean_bosses_killed']:.2f} / 12
Max Bosses Defeated: {summary['max_bosses_killed']}
Average Episode Reward: {summary['mean_episode_reward']:.2f}

--- Decision Making (Probing) ---
Total Decisions Made: {summary['total_decisions']}
Yield Rate: {summary['yield_rate'] * 100:.2f}% (Action 0)
Unique Actions Used: {summary['unique_actions_used']} / 256
Most Common Action ID: {summary['most_common_action'][0]} (Used {summary['most_common_action'][1]} times)

--- Model Confidence ---
Mean Entropy: {summary['mean_entropy']:.4f} 
(Lower = more decisive, Higher = random/confused)

--- Behavioral Scenarios ---
Jesters Played: {summary['scenarios']['played_jester']}
Yielded During Defense: {summary['scenarios']['yield_on_defense']} (Often fatal)
Wasted Face Cards (No attack/defense required): {summary['scenarios']['wasted_face_card']}
"""

    with open(f"{output_dir}/analysis_report.txt", 'w') as f:
        f.write(report)
        
    logger.info("Reports saved to %s", output_dir)
    return summary
