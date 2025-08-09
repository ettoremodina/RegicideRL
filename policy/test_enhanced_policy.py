"""
Test Enhanced Policy Performance
Compare current policy vs enhanced version
"""

import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

from regicide_gym_env import RegicideGymEnv
from fixed_enhanced_training import FixedEnhancedCardAwareTrainer
from enhanced_policy import EnhancedCardAwarePolicy


class PolicyComparison:
    """Compare different policy architectures"""
    
    def __init__(self):
        self.results = {}
        
    def run_policy_test(self, policy_class, policy_name: str, episodes: int = 100) -> Dict:
        """Test a policy for specified episodes"""
        print(f"🧪 Testing {policy_name} for {episodes} episodes...")
        
        env = RegicideGymEnv(num_players=2, max_hand_size=7)
        
        if policy_class == EnhancedCardAwarePolicy:
            policy = EnhancedCardAwarePolicy(
                max_hand_size=7,
                max_actions=30,
                card_embed_dim=32,
                hidden_dim=128
            )
        else:
            # Would use current CardAwarePolicy
            from card_aware_policy import CardAwarePolicy
            policy = CardAwarePolicy(max_hand_size=7, max_actions=30)
        
        policy.eval()
        
        results = {
            'wins': 0,
            'bosses_killed': [],
            'episode_lengths': [],
            'rewards': []
        }
        
        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    if hasattr(policy, 'get_action_and_value'):
                        # Enhanced policy
                        action, _, _, _, _ = policy.get_action_and_value(obs)
                    else:
                        # Basic policy
                        action_probs = policy(obs)
                        num_valid = obs['num_valid_actions'].item()
                        if num_valid > 0:
                            valid_probs = action_probs[:num_valid]
                            action = torch.multinomial(torch.softmax(valid_probs, dim=0), 1).item()
                        else:
                            action = 0
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                episode_length += 1
            
            # Record results
            if info.get('won', False):
                results['wins'] += 1
            results['bosses_killed'].append(info.get('bosses_killed', 0))
            results['episode_lengths'].append(episode_length)
            results['rewards'].append(total_reward)
            
            if (episode + 1) % 20 == 0:
                current_win_rate = results['wins'] / (episode + 1)
                current_avg_bosses = np.mean(results['bosses_killed'])
                print(f"  Episode {episode + 1}: Win Rate: {current_win_rate:.1%}, Avg Bosses: {current_avg_bosses:.2f}")
        
        # Calculate final metrics
        final_results = {
            'policy_name': policy_name,
            'episodes': episodes,
            'win_rate': results['wins'] / episodes,
            'avg_bosses_killed': np.mean(results['bosses_killed']),
            'max_bosses_killed': max(results['bosses_killed']),
            'avg_episode_length': np.mean(results['episode_lengths']),
            'avg_reward': np.mean(results['rewards']),
            'bosses_std': np.std(results['bosses_killed']),
            'raw_results': results
        }
        
        self.results[policy_name] = final_results
        return final_results
    
    def compare_policies(self, episodes: int = 100):
        """Compare current vs enhanced policies"""
        print("🔍 Starting Policy Comparison")
        print("=" * 50)
        
        # Test policies (if they exist)
        try:
            from card_aware_policy import CardAwarePolicy
            current_results = self.run_policy_test(CardAwarePolicy, "Current Policy", episodes)
        except ImportError:
            print("⚠️  Current policy not found, using random baseline")
            current_results = self._random_baseline(episodes)
        
        # Test enhanced policy
        enhanced_results = self.run_policy_test(EnhancedCardAwarePolicy, "Enhanced Policy", episodes)
        
        # Print comparison
        self._print_comparison(current_results, enhanced_results)
        
        # Plot comparison
        self._plot_comparison()
        
        return current_results, enhanced_results
    
    def _random_baseline(self, episodes: int) -> Dict:
        """Generate random baseline for comparison"""
        print(f"🎲 Running random baseline for {episodes} episodes...")
        
        env = RegicideGymEnv(num_players=2, max_hand_size=7)
        
        results = {
            'wins': 0,
            'bosses_killed': [],
            'episode_lengths': [],
            'rewards': []
        }
        
        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                num_valid = obs['num_valid_actions'].item()
                action = np.random.randint(0, max(1, num_valid))
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                episode_length += 1
            
            if info.get('won', False):
                results['wins'] += 1
            results['bosses_killed'].append(info.get('bosses_killed', 0))
            results['episode_lengths'].append(episode_length)
            results['rewards'].append(total_reward)
        
        return {
            'policy_name': "Random Baseline",
            'episodes': episodes,
            'win_rate': results['wins'] / episodes,
            'avg_bosses_killed': np.mean(results['bosses_killed']),
            'max_bosses_killed': max(results['bosses_killed']),
            'avg_episode_length': np.mean(results['episode_lengths']),
            'avg_reward': np.mean(results['rewards']),
            'bosses_std': np.std(results['bosses_killed']),
            'raw_results': results
        }
    
    def _print_comparison(self, current: Dict, enhanced: Dict):
        """Print detailed comparison"""
        print("\n📊 Policy Comparison Results")
        print("=" * 60)
        
        metrics = [
            ('Win Rate', 'win_rate', ':.1%'),
            ('Avg Bosses Killed', 'avg_bosses_killed', ':.2f'),
            ('Max Bosses Killed', 'max_bosses_killed', ':d'),
            ('Avg Episode Length', 'avg_episode_length', ':.1f'),
            ('Avg Reward', 'avg_reward', ':.2f'),
            ('Bosses Std Dev', 'bosses_std', ':.2f')
        ]
        
        for metric_name, key, fmt in metrics:
            current_val = current[key]
            enhanced_val = enhanced[key]
            improvement = enhanced_val - current_val
            improvement_pct = (improvement / max(current_val, 0.001)) * 100
            
            print(f"{metric_name:18} | {current['policy_name']:15} | {enhanced['policy_name']:15} | Improvement")
            # print(f"{'':<18} | {current_val{fmt}:>15} | {enhanced_val{fmt}:>15} | {improvement:+.3f} ({improvement_pct:+.1f}%)")
            print("-" * 70)
        
        # Highlight key improvements
        print(f"\n🎯 Key Improvements:")
        if enhanced['win_rate'] > current['win_rate']:
            print(f"  ✅ Win rate improved by {(enhanced['win_rate'] - current['win_rate']) * 100:.1f} percentage points")
        if enhanced['avg_bosses_killed'] > current['avg_bosses_killed']:
            print(f"  ✅ Average bosses killed improved by {enhanced['avg_bosses_killed'] - current['avg_bosses_killed']:.2f}")
        if enhanced['avg_episode_length'] < current['avg_episode_length']:
            print(f"  ✅ Faster games by {current['avg_episode_length'] - enhanced['avg_episode_length']:.1f} steps average")
        if enhanced['bosses_std'] < current['bosses_std']:
            print(f"  ✅ More consistent performance (lower std dev)")
    
    def _plot_comparison(self):
        """Create comparison plots"""
        if len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Policy Architecture Comparison', fontsize=16, fontweight='bold')
        
        policies = list(self.results.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(policies)]
        
        # Win Rate Comparison
        win_rates = [self.results[p]['win_rate'] for p in policies]
        axes[0, 0].bar(policies, win_rates, color=colors)
        axes[0, 0].set_title('Win Rate Comparison')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, rate in enumerate(win_rates):
            axes[0, 0].text(i, rate + 0.01, f'{rate:.1%}', ha='center', fontweight='bold')
        
        # Average Bosses Killed
        avg_bosses = [self.results[p]['avg_bosses_killed'] for p in policies]
        axes[0, 1].bar(policies, avg_bosses, color=colors)
        axes[0, 1].set_title('Average Bosses Killed')
        axes[0, 1].set_ylabel('Bosses Killed')
        for i, bosses in enumerate(avg_bosses):
            axes[0, 1].text(i, bosses + 0.05, f'{bosses:.2f}', ha='center', fontweight='bold')
        
        # Episode Length Distribution
        for i, policy in enumerate(policies):
            lengths = self.results[policy]['raw_results']['episode_lengths']
            axes[1, 0].hist(lengths, alpha=0.7, label=policy, bins=20, color=colors[i])
        axes[1, 0].set_title('Episode Length Distribution')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Bosses Killed Distribution
        for i, policy in enumerate(policies):
            bosses = self.results[policy]['raw_results']['bosses_killed']
            axes[1, 1].hist(bosses, alpha=0.7, label=policy, bins=range(0, 14), color=colors[i])
        axes[1, 1].set_title('Bosses Killed Distribution')
        axes[1, 1].set_xlabel('Bosses Killed')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"policy_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📈 Comparison plot saved as: {filename}")
        
        plt.show()


def quick_enhancement_test():
    """Quick test of enhancement features"""
    print("🚀 Quick Enhancement Test")
    print("=" * 40)
    
    # Test enhanced card embeddings
    enhanced_policy = EnhancedCardAwarePolicy(
        max_hand_size=7,
        max_actions=30,
        card_embed_dim=32,
        hidden_dim=128
    )
    
    # Create dummy observation
    dummy_obs = {
        'hand_cards': torch.randint(0, 52, (7,)),
        'enemy_health': torch.tensor([20.0]),
        'enemy_attack': torch.tensor([15.0]),
        'players_health': torch.tensor([15.0, 12.0]),
        'deck_size': torch.tensor([30]),
        'suit_powers': torch.tensor([2, 1, 0, 3]),
        'num_valid_actions': torch.tensor([8]),
        'bosses_killed': torch.tensor([2]),
        'jacks_played': torch.tensor([1]),
        'queens_played': torch.tensor([0]),
        'kings_played': torch.tensor([1]),
        'aces_played': torch.tensor([0])
    }
    
    print("Testing enhanced policy forward pass...")
    try:
        with torch.no_grad():
            logits = enhanced_policy(dummy_obs)
        print(f"✅ Enhanced policy works! Output shape: {logits.shape}")
        
        print("\nTesting enhanced action selection...")
        action, log_prob, value, entropy, aux_pred = enhanced_policy.get_action_and_value(dummy_obs)
        print(f"✅ Action selection works!")
        print(f"   Action: {action}")
        print(f"   Log prob: {log_prob.item():.4f}")
        print(f"   Value: {value.item():.4f}")
        print(f"   Entropy: {entropy.item():.4f}")
        
    except Exception as e:
        print(f"❌ Error in enhanced policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🧪 Enhanced Policy Testing Suite")
    print("=" * 50)
    
    # Quick test first
    if not quick_enhancement_test():
        print("❌ Enhanced policy has issues, stopping tests")
        exit(1)
    
    print("\n" + "=" * 50)
    print("🏁 Starting full policy comparison...")
    
    # Full comparison
    comparison = PolicyComparison()
    
    # Run comparison with fewer episodes for speed
    current_results, enhanced_results = comparison.compare_policies(episodes=50)
    
    print(f"\n🎯 Recommendation:")
    if enhanced_results['avg_bosses_killed'] > current_results['avg_bosses_killed']:
        improvement = enhanced_results['avg_bosses_killed'] - current_results['avg_bosses_killed']
        print(f"✅ Enhanced policy shows {improvement:.2f} improvement in average bosses killed")
        print(f"   Consider integrating enhanced features into main training!")
    else:
        print(f"⚠️  Enhanced policy needs more tuning")
        print(f"   Current architecture might already be well-optimized")
    
    print(f"\n🚀 To achieve >2.5 average bosses killed:")
    print(f"   • Current best: {max(current_results['avg_bosses_killed'], enhanced_results['avg_bosses_killed']):.2f}")
    print(f"   • Target: 2.5")
    print(f"   • Gap: {max(0, 2.5 - max(current_results['avg_bosses_killed'], enhanced_results['avg_bosses_killed'])):.2f}")
    print(f"   • Recommended: Use curriculum learning + longer training")
