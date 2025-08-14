"""
Policy Analysis Script for Regicide
Analyzes trained model behavior and decision patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from train.regicide_gym_env import RegicideGymEnv
from policy.card_aware_policy import CardAwarePolicy
from config import PathManager


class PolicyAnalyzer:
    """
    Comprehensive analysis of policy decision-making patterns
    """
    
    def __init__(self, model_path: str, num_players: int = 4, max_hand_size: int = 5):
        self.num_players = num_players
        self.max_hand_size = max_hand_size
        
        # Create environment
        self.env = RegicideGymEnv(
            num_players=num_players,
            max_hand_size=max_hand_size,
            observation_mode="card_aware"
        )
        
        # Load the trained model
        self.policy = self._load_model(model_path)
        self.policy.eval()  # Set to evaluation mode
        
        # Create output directory for analysis
        self.output_dir = f"policy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data collection
        self.reset_statistics()
    
    def _load_model(self, model_path: str) -> CardAwarePolicy:
        """Load the trained model"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            policy = CardAwarePolicy(
                max_hand_size=config['max_hand_size'],
                max_actions=config['max_actions'],
                card_embed_dim=config['card_embed_dim'],
                hidden_dim=config['hidden_dim']
            )
        else:
            # Default configuration if not saved
            policy = CardAwarePolicy(
                max_hand_size=self.max_hand_size,
                max_actions=30,  # Default max actions
                card_embed_dim=64,
                hidden_dim=128
            )
        
        # Load the state dict
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Model loaded successfully!")
        
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            print(f"Model was trained for {stats.get('total_episodes', 'unknown')} episodes")
            print(f"Max bosses killed during training: {stats.get('max_bosses_killed', 'unknown')}")
        
        return policy
    
    def reset_statistics(self):
        """Reset all collected statistics"""
        # Action distribution analysis
        self.action_choices = []
        self.action_probabilities = []
        self.action_entropies = []
        
        # Game state analysis
        self.hand_sizes = []
        self.game_phases = []
        self.bosses_killed_per_game = []
        self.episode_lengths = []
        self.episode_rewards = []
        
        # Decision context analysis
        self.decisions_by_hand_size = defaultdict(list)
        self.decisions_by_phase = defaultdict(list)
        self.decisions_by_enemy = defaultdict(list)
        
        # Yield vs play analysis
        self.yield_decisions = []
        self.play_decisions = []
        
        # Convergence analysis
        self.action_consistency = []
        self.probability_variance = []
        
        print("Statistics reset successfully!")
    
    def analyze_games(self, num_games: int = 100, verbose: bool = True):
        """Run multiple games and collect comprehensive statistics"""
        print(f"\n🔍 Analyzing {num_games} games...")
        print("=" * 50)
        
        for game_idx in range(num_games):
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Analyzing game {game_idx + 1}/{num_games}...")
            
            self._analyze_single_game(game_idx)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
    
    def _analyze_single_game(self, game_idx: int):
        """Analyze a single game in detail"""
        obs, info = self.env.reset()
        episode_reward = 0
        step_count = 0
        game_actions = []
        game_probs = []
        game_entropies = []
        
        while not info.get('game_over', False) and step_count < 200:
            # Get action probabilities and decision analysis
            with torch.no_grad():
                action_probs = self.policy.get_action_probabilities(obs)
                action, log_prob = self.policy.get_action(obs)
                
                # Calculate entropy (measure of decision uncertainty)
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
                
                # Store action decision data
                game_actions.append(action)
                game_probs.append(action_probs.cpu().numpy())
                game_entropies.append(entropy.item())
                
                # Store context-specific decisions
                hand_size = obs['hand_size'].item()
                phase = info.get('phase', 'unknown')
                enemy_card = obs['enemy_card'].item()
                
                self.decisions_by_hand_size[hand_size].append(action)
                self.decisions_by_phase[phase].append(action)
                self.decisions_by_enemy[enemy_card].append(action)
                
                # Analyze yield vs play decisions
                action_meanings = self.env.get_action_meanings()
                if action < len(action_meanings):
                    action_meaning = action_meanings[action].lower()
                    # Get the probability for this action (need to check if it's within valid range)
                    if action < len(action_probs):
                        action_prob_tensor = action_probs[action]
                        # Handle both scalar and tensor cases
                        if action_prob_tensor.numel() == 1:
                            action_prob = action_prob_tensor.item()
                        else:
                            # If it's a multi-element tensor, take the first element or mean
                            action_prob = action_prob_tensor.flatten()[0].item()
                    else:
                        action_prob = 0.0
                    
                    if 'yield' in action_meaning:
                        self.yield_decisions.append({
                            'hand_size': hand_size,
                            'phase': phase,
                            'probability': action_prob,
                            'entropy': entropy.item()
                        })
                    else:
                        self.play_decisions.append({
                            'hand_size': hand_size,
                            'phase': phase,
                            'action_type': action_meaning,
                            'probability': action_prob,
                            'entropy': entropy.item()
                        })
            
            # Take step
            obs, reward, done, _, info = self.env.step(action)
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Store episode-level statistics
        self.action_choices.extend(game_actions)
        self.action_probabilities.extend(game_probs)
        self.action_entropies.extend(game_entropies)
        
        self.episode_lengths.append(step_count)
        self.episode_rewards.append(episode_reward)
        self.bosses_killed_per_game.append(info.get('bosses_killed', 0))
        
        # Analyze action consistency within this game
        if len(game_actions) > 1:
            action_counter = Counter(game_actions)
            most_common_action_count = action_counter.most_common(1)[0][1]
            consistency = most_common_action_count / len(game_actions)
            self.action_consistency.append(consistency)
            
            # Probability variance across decisions
            if len(game_probs) > 1:
                try:
                    # Handle arrays of potentially different lengths
                    max_len = max(len(prob) for prob in game_probs)
                    # Pad shorter arrays with zeros
                    padded_probs = []
                    for prob in game_probs:
                        if len(prob) < max_len:
                            padded = np.zeros(max_len)
                            padded[:len(prob)] = prob
                            padded_probs.append(padded)
                        else:
                            padded_probs.append(prob)
                    
                    prob_arrays = np.array(padded_probs)
                    prob_variance = np.var(prob_arrays, axis=0).mean()
                    self.probability_variance.append(prob_variance)
                except Exception as e:
                    # If there's still an issue, just calculate variance of max probabilities
                    max_probs = [np.max(prob) for prob in game_probs]
                    prob_variance = np.var(max_probs)
                    self.probability_variance.append(prob_variance)
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis plots and report"""
        print("\n📊 Generating analysis report...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Action Distribution Analysis
        self._plot_action_distribution(fig, 1)
        
        # 2. Action Probability Analysis
        self._plot_probability_analysis(fig, 2)
        
        # 3. Decision Entropy Analysis
        self._plot_entropy_analysis(fig, 3)
        
        # 4. Context-based Decision Analysis
        self._plot_context_analysis(fig, 4)
        
        # 5. Yield vs Play Analysis
        self._plot_yield_analysis(fig, 5)
        
        # 6. Game Performance Analysis
        self._plot_performance_analysis(fig, 6)
        
        # 7. Policy Consistency Analysis
        self._plot_consistency_analysis(fig, 7)
        
        # 8. Decision Quality Heatmap
        self._plot_decision_heatmap(fig, 8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comprehensive_policy_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary statistics
        self._generate_summary_stats()
        
        print(f"✅ Analysis complete! All files saved to: {self.output_dir}")
    
    def _plot_action_distribution(self, fig, plot_num):
        """Plot action choice distribution"""
        ax = plt.subplot(4, 2, plot_num)
        
        action_counts = Counter(self.action_choices)
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        
        plt.bar(actions, counts, alpha=0.7, color='skyblue')
        plt.title('Action Choice Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Action Index')
        plt.ylabel('Frequency')
        
        # Add percentage labels
        total_actions = sum(counts)
        for i, (action, count) in enumerate(zip(actions, counts)):
            percentage = (count / total_actions) * 100
            plt.text(action, count + max(counts) * 0.01, f'{percentage:.1f}%', 
                    ha='center', fontsize=8)
        
        plt.grid(axis='y', alpha=0.3)
        
        # Check for action collapse
        if len(actions) > 0:
            max_action_pct = max(counts) / total_actions
            if max_action_pct > 0.8:
                plt.text(0.5, 0.95, f'⚠️ Potential Collapse: {max_action_pct:.1%} on Action {actions[counts.index(max(counts))]}', 
                        transform=ax.transAxes, ha='center', va='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    def _plot_probability_analysis(self, fig, plot_num):
        """Plot action probability distributions"""
        ax = plt.subplot(4, 2, plot_num)
        
        if self.action_probabilities:
            try:
                # Handle arrays of potentially different lengths
                max_len = max(len(prob) for prob in self.action_probabilities)
                min_len = min(len(prob) for prob in self.action_probabilities)
                
                # If all arrays have the same length, create matrix normally
                if max_len == min_len:
                    prob_matrix = np.array(self.action_probabilities)
                else:
                    # Pad shorter arrays with zeros
                    padded_probs = []
                    for prob in self.action_probabilities:
                        if len(prob) < max_len:
                            padded = np.zeros(max_len)
                            padded[:len(prob)] = prob
                            padded_probs.append(padded)
                        else:
                            padded_probs.append(prob)
                    prob_matrix = np.array(padded_probs)
                
                # Plot heatmap of probabilities over time (limit to first 100 decisions for readability)
                plot_data = prob_matrix[:100] if len(prob_matrix) > 100 else prob_matrix
                im = plt.imshow(plot_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
                plt.colorbar(im, label='Probability')
                plt.title('Action Probabilities Over Time (First 100 Decisions)', fontsize=14, fontweight='bold')
                plt.xlabel('Decision Number')
                plt.ylabel('Action Index')
                
            except Exception as e:
                # Fallback: plot entropy over time instead
                plt.plot(self.action_entropies[:100])
                plt.title('Decision Entropy Over Time (First 100 Decisions)', fontsize=14, fontweight='bold')
                plt.xlabel('Decision Number')
                plt.ylabel('Entropy')
                plt.text(0.02, 0.98, f'Note: Probability matrix visualization failed\nShowing entropy instead', 
                        transform=ax.transAxes, va='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def _plot_entropy_analysis(self, fig, plot_num):
        """Plot decision entropy analysis"""
        plt.subplot(4, 2, plot_num)
        
        if self.action_entropies:
            plt.hist(self.action_entropies, bins=30, alpha=0.7, color='lightcoral')
            plt.axvline(np.mean(self.action_entropies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.action_entropies):.3f}')
            plt.title('Decision Entropy Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Entropy (bits)')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Add interpretation
            mean_entropy = np.mean(self.action_entropies)
            if mean_entropy < 0.5:
                interpretation = "Low entropy: Decisive/Deterministic"
            elif mean_entropy < 1.5:
                interpretation = "Medium entropy: Balanced decisions"
            else:
                interpretation = "High entropy: Uncertain/Exploratory"
            
            plt.text(0.02, 0.95, interpretation, transform=plt.gca().transAxes, 
                    va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _plot_context_analysis(self, fig, plot_num):
        """Plot context-based decision analysis"""
        plt.subplot(4, 2, plot_num)
        
        # Analyze decisions by hand size
        hand_sizes = sorted(self.decisions_by_hand_size.keys())
        action_diversity = []
        
        for hand_size in hand_sizes:
            actions = self.decisions_by_hand_size[hand_size]
            unique_actions = len(set(actions))
            action_diversity.append(unique_actions)
        
        plt.bar(hand_sizes, action_diversity, alpha=0.7, color='lightgreen')
        plt.title('Action Diversity by Hand Size', fontsize=14, fontweight='bold')
        plt.xlabel('Hand Size')
        plt.ylabel('Number of Unique Actions')
        plt.grid(axis='y', alpha=0.3)
        
        # Add trend line
        if len(hand_sizes) > 1:
            z = np.polyfit(hand_sizes, action_diversity, 1)
            p = np.poly1d(z)
            plt.plot(hand_sizes, p(hand_sizes), "r--", alpha=0.8, label=f'Trend: slope={z[0]:.2f}')
            plt.legend()
    
    def _plot_yield_analysis(self, fig, plot_num):
        """Plot yield vs play decision analysis"""
        plt.subplot(4, 2, plot_num)
        
        yield_count = len(self.yield_decisions)
        play_count = len(self.play_decisions)
        total_decisions = yield_count + play_count
        
        if total_decisions > 0:
            labels = ['Yield', 'Play Cards']
            sizes = [yield_count, play_count]
            colors = ['orange', 'lightblue']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Yield vs Play Decisions\n(Total: {total_decisions})', fontsize=14, fontweight='bold')
            
            # Add yield rate by hand size analysis
            if self.yield_decisions:
                yield_by_hand = defaultdict(int)
                total_by_hand = defaultdict(int)
                
                for decision in self.yield_decisions:
                    yield_by_hand[decision['hand_size']] += 1
                
                for decision in self.play_decisions:
                    total_by_hand[decision['hand_size']] += 1
                
                for hand_size in yield_by_hand:
                    total_by_hand[hand_size] += yield_by_hand[hand_size]
    
    def _plot_performance_analysis(self, fig, plot_num):
        """Plot game performance analysis"""
        plt.subplot(4, 2, plot_num)
        
        if self.bosses_killed_per_game:
            boss_counts = Counter(self.bosses_killed_per_game)
            bosses = sorted(boss_counts.keys())
            frequencies = [boss_counts[b] for b in bosses]
            
            plt.bar(bosses, frequencies, alpha=0.7, color='gold')
            plt.title('Bosses Killed Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Bosses Killed per Game')
            plt.ylabel('Number of Games')
            
            # Add statistics
            mean_bosses = np.mean(self.bosses_killed_per_game)
            max_bosses = max(self.bosses_killed_per_game)
            win_rate = sum(1 for b in self.bosses_killed_per_game if b >= 12) / len(self.bosses_killed_per_game)
            
            stats_text = f'Mean: {mean_bosses:.2f}\nMax: {max_bosses}\nWin Rate: {win_rate:.1%}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            
            plt.grid(axis='y', alpha=0.3)
    
    def _plot_consistency_analysis(self, fig, plot_num):
        """Plot policy consistency analysis"""
        plt.subplot(4, 2, plot_num)
        
        if self.action_consistency:
            plt.hist(self.action_consistency, bins=20, alpha=0.7, color='mediumpurple')
            plt.axvline(np.mean(self.action_consistency), color='purple', linestyle='--', 
                       label=f'Mean: {np.mean(self.action_consistency):.3f}')
            plt.title('Action Consistency Within Games', fontsize=14, fontweight='bold')
            plt.xlabel('Consistency Score (0=diverse, 1=always same action)')
            plt.ylabel('Number of Games')
            plt.legend()
            
            # Interpretation
            mean_consistency = np.mean(self.action_consistency)
            if mean_consistency > 0.8:
                interpretation = "High consistency: May be stuck in patterns"
            elif mean_consistency > 0.5:
                interpretation = "Moderate consistency: Reasonable variety"
            else:
                interpretation = "Low consistency: Good action diversity"
            
            plt.text(0.02, 0.95, interpretation, transform=plt.gca().transAxes, 
                    va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))
    
    def _plot_decision_heatmap(self, fig, plot_num):
        """Plot decision quality heatmap"""
        plt.subplot(4, 2, plot_num)
        
        # Create a heatmap of action choices by game state
        if self.decisions_by_hand_size and self.decisions_by_phase:
            # Prepare data for heatmap
            hand_sizes = sorted(self.decisions_by_hand_size.keys())
            phases = sorted(self.decisions_by_phase.keys())
            
            # Create matrix of action diversity
            matrix = np.zeros((len(phases), len(hand_sizes)))
            
            for i, phase in enumerate(phases):
                for j, hand_size in enumerate(hand_sizes):
                    # Get actions for this combination
                    phase_actions = set(self.decisions_by_phase[phase])
                    hand_actions = set(self.decisions_by_hand_size[hand_size])
                    # Use intersection as a measure of context-specific diversity
                    matrix[i, j] = len(phase_actions.intersection(hand_actions))
            
            sns.heatmap(matrix, xticklabels=hand_sizes, yticklabels=phases, 
                       annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title('Action Diversity by Phase and Hand Size', fontsize=14, fontweight='bold')
            plt.xlabel('Hand Size')
            plt.ylabel('Game Phase')
    
    def _generate_summary_stats(self):
        """Generate and save summary statistics"""
        stats = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_decisions': len(self.action_choices),
            'total_games': len(self.episode_lengths),
            
            # Action distribution
            'unique_actions_used': len(set(self.action_choices)),
            'most_common_action': Counter(self.action_choices).most_common(1)[0] if self.action_choices else None,
            'action_distribution': dict(Counter(self.action_choices)),
            
            # Decision quality
            'mean_entropy': float(np.mean(self.action_entropies)) if self.action_entropies else 0,
            'entropy_std': float(np.std(self.action_entropies)) if self.action_entropies else 0,
            
            # Performance
            'mean_bosses_killed': float(np.mean(self.bosses_killed_per_game)) if self.bosses_killed_per_game else 0,
            'max_bosses_killed': int(max(self.bosses_killed_per_game)) if self.bosses_killed_per_game else 0,
            'win_rate': float(sum(1 for b in self.bosses_killed_per_game if b >= 12) / len(self.bosses_killed_per_game)) if self.bosses_killed_per_game else 0,
            'mean_episode_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            'mean_episode_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            
            # Consistency
            'mean_action_consistency': float(np.mean(self.action_consistency)) if self.action_consistency else 0,
            'mean_probability_variance': float(np.mean(self.probability_variance)) if self.probability_variance else 0,
            
            # Context analysis
            'yield_rate': len(self.yield_decisions) / (len(self.yield_decisions) + len(self.play_decisions)) if (self.yield_decisions or self.play_decisions) else 0,
            
            # Potential issues
            'potential_action_collapse': self._check_action_collapse(),
            'decision_quality_assessment': self._assess_decision_quality()
        }
        
        # Save to JSON
        with open(os.path.join(self.output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create human-readable report
        self._create_text_report(stats)
        
        return stats
    
    def _check_action_collapse(self) -> Dict:
        """Check if the policy has collapsed to always choosing the same actions"""
        if not self.action_choices:
            return {"collapsed": False, "reason": "No actions recorded"}
        
        action_counts = Counter(self.action_choices)
        total_actions = len(self.action_choices)
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        
        collapse_threshold = 0.8  # If 80% of actions are the same
        is_collapsed = (most_common_count / total_actions) > collapse_threshold
        
        return {
            "collapsed": is_collapsed,
            "dominant_action": int(most_common_action),
            "dominant_action_percentage": float(most_common_count / total_actions),
            "threshold_used": collapse_threshold,
            "unique_actions_used": len(action_counts),
            "action_distribution": {int(k): v for k, v in action_counts.most_common()}
        }
    
    def _assess_decision_quality(self) -> Dict:
        """Assess overall decision quality"""
        assessment = {
            "entropy_based": "unknown",
            "consistency_based": "unknown", 
            "performance_based": "unknown",
            "overall": "unknown"
        }
        
        if self.action_entropies:
            mean_entropy = np.mean(self.action_entropies)
            if mean_entropy < 0.5:
                assessment["entropy_based"] = "very_decisive"
            elif mean_entropy < 1.0:
                assessment["entropy_based"] = "decisive" 
            elif mean_entropy < 2.0:
                assessment["entropy_based"] = "balanced"
            else:
                assessment["entropy_based"] = "highly_uncertain"
        
        if self.action_consistency:
            mean_consistency = np.mean(self.action_consistency)
            if mean_consistency > 0.8:
                assessment["consistency_based"] = "overly_consistent"
            elif mean_consistency > 0.5:
                assessment["consistency_based"] = "reasonably_varied"
            else:
                assessment["consistency_based"] = "highly_varied"
        
        if self.bosses_killed_per_game:
            mean_bosses = np.mean(self.bosses_killed_per_game)
            if mean_bosses > 8:
                assessment["performance_based"] = "excellent"
            elif mean_bosses > 5:
                assessment["performance_based"] = "good"
            elif mean_bosses > 2:
                assessment["performance_based"] = "fair"
            else:
                assessment["performance_based"] = "poor"
        
        return assessment
    
    def _create_text_report(self, stats: Dict):
        """Create a human-readable text report"""
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("REGICIDE POLICY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {stats['analysis_timestamp']}\n")
            f.write(f"Total Games Analyzed: {stats['total_games']}\n")
            f.write(f"Total Decisions Recorded: {stats['total_decisions']}\n\n")
            
            f.write("ACTION DISTRIBUTION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Unique Actions Used: {stats['unique_actions_used']}\n")
            if stats['most_common_action']:
                action_idx, count = stats['most_common_action']
                percentage = (count / stats['total_decisions']) * 100
                f.write(f"Most Common Action: {action_idx} ({percentage:.1f}% of decisions)\n")
            
            collapse_info = stats['potential_action_collapse']
            if collapse_info['collapsed']:
                f.write(f"ACTION COLLAPSE DETECTED!\n")
                f.write(f"   Action {collapse_info['dominant_action']} used {collapse_info['dominant_action_percentage']:.1%} of the time\n")
            else:
                f.write(" No action collapse detected - good action diversity\n")
            
            f.write(f"\nDECISION QUALITY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Decision Entropy: {stats['mean_entropy']:.3f} ± {stats['entropy_std']:.3f}\n")
            f.write(f"Action Consistency: {stats['mean_action_consistency']:.3f}\n")
            
            quality = stats['decision_quality_assessment']
            f.write(f"Entropy Assessment: {quality['entropy_based']}\n")
            f.write(f"Consistency Assessment: {quality['consistency_based']}\n")
            
            f.write(f"\nPERFORMANCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Bosses Killed: {stats['mean_bosses_killed']:.2f}\n")
            f.write(f"Maximum Bosses Killed: {stats['max_bosses_killed']}\n")
            f.write(f"Win Rate: {stats['win_rate']:.1%}\n")
            f.write(f"Average Episode Length: {stats['mean_episode_length']:.1f} steps\n")
            f.write(f"Average Episode Reward: {stats['mean_episode_reward']:.2f}\n")
            f.write(f"Performance Assessment: {quality['performance_based']}\n")
            
            f.write(f"\nSTRATEGIC ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Yield Rate: {stats['yield_rate']:.1%}\n")
            
            f.write(f"\n RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            if collapse_info['collapsed']:
                f.write("• CRITICAL: Address action collapse - policy is too deterministic\n")
                f.write("• Consider reducing learning rate or adding exploration\n")
            
            if stats['mean_entropy'] < 0.5:
                f.write("• Policy is very decisive - might be overfitting\n")
            elif stats['mean_entropy'] > 2.0:
                f.write("• Policy is highly uncertain - might need more training\n")
            
            if stats['win_rate'] < 0.1:
                f.write("• Low win rate - policy needs significant improvement\n")
            elif stats['win_rate'] > 0.5:
                f.write("• High win rate - policy is performing well!\n")
        
        print(f" Detailed report saved to: {report_path}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Regicide policy behavior")
    parser.add_argument("--model_path", required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to analyze")
    parser.add_argument("--num_players", type=int, default=4, help="Number of players")
    parser.add_argument("--max_hand_size", type=int, default=5, help="Maximum hand size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🎮 REGICIDE POLICY ANALYZER")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PolicyAnalyzer(
        model_path=args.model_path,
        num_players=args.num_players,
        max_hand_size=args.max_hand_size
    )
    
    # Run analysis
    analyzer.analyze_games(num_games=args.num_games, verbose=args.verbose)
    
    # Generate report
    stats = analyzer.generate_analysis_report()
    
    print("\n🎉 Analysis Complete!")
    print(f"📁 Results saved to: {analyzer.output_dir}")
    
    # Quick summary
    collapse_info = stats['potential_action_collapse']
    if collapse_info['collapsed']:
        print(f"⚠️  WARNING: Action collapse detected! Action {collapse_info['dominant_action']} used {collapse_info['dominant_action_percentage']:.1%} of the time")
    else:
        print("✅ No action collapse - policy shows good diversity")
    
    print(f"🏆 Win rate: {stats['win_rate']:.1%}")
    print(f"👑 Average bosses killed: {stats['mean_bosses_killed']:.2f}")


if __name__ == "__main__":
    main()
