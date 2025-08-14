#!/usr/bin/env python3
"""
Random Policy Evaluation for Regicide
Runs N games with a uniform random policy and generates histograms of performance metrics
"""

import numpy as np
import torch
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from collections import defaultdict
from math import ceil
try:
    # For PyTorch-heavy environments, avoid thread oversubscription across processes
    torch.set_num_threads(1)
except Exception:
    pass

# Import our modules
from train.regicide_gym_env import make_regicide_env
from train.training_utils import TrainingVisualizer
from config import PathManager
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - plots will not be generated")


class RandomPolicy:
    """Simple random policy that selects uniformly from valid actions"""
    
    def __init__(self):
        self.name = "Uniform Random Policy"
    
    def get_action(self, observation: Dict[str, torch.Tensor]) -> Tuple[int, torch.Tensor]:
        """Select random action from valid actions"""
        num_valid_actions = observation['num_valid_actions'].item()
        
        if num_valid_actions > 0:
            # Uniform random selection from valid actions
            action = np.random.randint(0, num_valid_actions)
            # Return dummy log_prob for compatibility
            log_prob = torch.tensor(-np.log(num_valid_actions))
            return action, log_prob
        else:
            return 0, torch.tensor(0.0)


def _run_single_game_worker(args: Tuple[int, int, int, int]) -> Dict:
    """Top-level worker to run a single game. Safe for Windows process spawning.

    Args tuple: (num_players, max_hand_size, max_steps, seed)
    Returns a dict with the same keys used by the evaluator aggregation.
    """
    num_players, max_hand_size, max_steps, seed = args

    # Local imports already resolved at module import; ensure independent RNG
    rng = np.random.default_rng(seed if seed is not None else None)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    env = make_regicide_env(
        num_players=num_players,
        max_hand_size=max_hand_size,
        observation_mode="card_aware",
        render_mode=None,
    )

    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    next_info = {}

    # Fast random policy inline to avoid extra pickling
    while episode_length < max_steps:
        num_valid_actions = int(obs['num_valid_actions'].item())
        action = int(rng.integers(0, num_valid_actions)) if num_valid_actions > 0 else 0
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        # Ensure reward is not None
        if reward is not None:
            episode_reward += reward
        episode_length += 1
        if terminated or truncated:
            break
        obs = next_obs
        info = next_info

    game = env.game
    bosses_killed = next_info.get('bosses_killed', 0) if next_info else 0
    victory = next_info.get('victory', False) if next_info else False

    # Ensure bosses_killed is not None
    if bosses_killed is None:
        bosses_killed = 0

    # Determine defeat reason
    defeat_reason = "max_steps"
    if terminated:
        if victory:
            defeat_reason = "victory"
        elif getattr(game, 'game_over', False):
            if hasattr(game, 'current_enemy') and getattr(game, 'current_enemy'):
                try:
                    defeat_reason = f"defeated_by_{game.current_enemy.card.value}"
                except Exception:
                    defeat_reason = "game_over"
            else:
                defeat_reason = "game_over"

    # Final enemy type
    final_enemy_type = "none"
    if hasattr(game, 'current_enemy') and getattr(game, 'current_enemy'):
        try:
            final_enemy_type = f"{game.current_enemy.card.value}_{game.current_enemy.card.suit.name}"
        except Exception:
            final_enemy_type = "unknown"

    return {
        'reward': float(episode_reward),
        'length': int(episode_length),
        'bosses_killed': int(bosses_killed),
        'victory': bool(victory),
        'defeat_reason': defeat_reason,
        'final_enemy_type': final_enemy_type,
    }


class RandomPolicyEvaluator:
    """Evaluates random policy performance across multiple games"""
    
    def __init__(self, num_players: int = 2, max_hand_size: int = 7, max_steps: int = 200, num_workers: int | None = None):
        self.num_players = num_players
        self.max_hand_size = max_hand_size
        self.max_steps = max_steps
        # Default to physical cores if unspecified
        self.num_workers = max(1, int(num_workers)) if num_workers else max(1, os.cpu_count() or 1)
        
        # Create environment
        self.env = make_regicide_env(
            num_players=num_players,
            max_hand_size=max_hand_size,
            observation_mode="card_aware",
            render_mode=None
        )
        
        # Create random policy
        self.policy = RandomPolicy()
        
        # Setup paths for saving results
        self.path_manager = PathManager("random_policy_eval")
        
        # Results storage
        self.results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'bosses_killed': [],
            'victories': [],
            'defeat_reasons': [],
            'final_enemy_types': []
        }
    
    def run_evaluation(self, num_games: int) -> Dict:
        """Run N games with random policy (parallelized across processes when possible)."""
        print(f"🎮 Running {num_games} games with random policy...")
        print(f"Configuration: {self.num_players} players, max {self.max_hand_size} cards, max {self.max_steps} steps")
        print(f"Workers: {self.num_workers}")
        print("=" * 60)

        if self.num_workers <= 1:
            # Sequential fallback (original behavior)
            for game_idx in tqdm(range(num_games)):
                game_result = self._run_single_game(game_idx)
                self._accumulate_result(game_result)
        else:
            # Parallel execution
            # Derive deterministic-ish seeds per game to avoid RNG collisions
            base_seed = np.random.SeedSequence().entropy
            seeds = [int((base_seed + i) % (2**32 - 1)) for i in range(num_games)]

            args_iter = (
                (self.num_players, self.max_hand_size, self.max_steps, seeds[i])
                for i in range(num_games)
            )

            # Choose a reasonable chunksize to reduce IPC overhead
            chunksize = max(1, num_games // (self.num_workers * 8))

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for game_result in tqdm(executor.map(_run_single_game_worker, args_iter, chunksize=chunksize), total=num_games):
                    if isinstance(game_result, dict) and '__error__' in game_result:
                        raise RuntimeError(f"Worker error: {game_result['__error__']}")
                    self._accumulate_result(game_result)

        # Generate summary statistics
        summary = self._generate_summary()
        self._print_summary(summary)
        
        # Generate plots
        if MATPLOTLIB_AVAILABLE:
            self._generate_plots()
        
        return summary

    def _accumulate_result(self, game_result: Dict):
        """Append a single game's results into the aggregation lists."""
        self.results['episode_rewards'].append(game_result['reward'])
        self.results['episode_lengths'].append(game_result['length'])
        self.results['bosses_killed'].append(game_result['bosses_killed'])
        self.results['victories'].append(game_result['victory'])
        self.results['defeat_reasons'].append(game_result['defeat_reason'])
        self.results['final_enemy_types'].append(game_result['final_enemy_type'])
    
    def _run_single_game(self, game_idx: int) -> Dict:
        """Run a single game and return results"""
        obs, info = self.env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        
        while episode_length < self.max_steps:
            # Get random action
            action, _ = self.policy.get_action(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Ensure reward is not None
            if reward is not None:
                episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
            
            obs = next_obs
            info = next_info
        
        # Extract final game state
        game = self.env.game
        bosses_killed = next_info.get('bosses_killed', 0) if next_info else 0
        victory = next_info.get('victory', False) if next_info else False
        
        # Ensure bosses_killed is not None
        if bosses_killed is None:
            bosses_killed = 0
        
        # Determine defeat reason
        defeat_reason = "max_steps"
        if terminated:
            if victory:
                defeat_reason = "victory"
            elif game.game_over:
                if hasattr(game, 'current_enemy') and game.current_enemy:
                    defeat_reason = f"defeated_by_{game.current_enemy.card.value}"
                else:
                    defeat_reason = "game_over"
        
        # Get final enemy type
        final_enemy_type = "none"
        if hasattr(game, 'current_enemy') and game.current_enemy:
            final_enemy_type = f"{game.current_enemy.card.value}_{game.current_enemy.card.suit.name}"
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'bosses_killed': bosses_killed,
            'victory': victory,
            'defeat_reason': defeat_reason,
            'final_enemy_type': final_enemy_type
        }
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        results = self.results
        
        summary = {
            'total_games': len(results['episode_rewards']),
            'victories': sum(results['victories']),
            'victory_rate': sum(results['victories']) / len(results['victories']) if results['victories'] else 0,
            'avg_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'avg_length': np.mean(results['episode_lengths']),
            'std_length': np.std(results['episode_lengths']),
            'avg_bosses_killed': np.mean(results['bosses_killed']),
            'std_bosses_killed': np.std(results['bosses_killed']),
            'max_bosses_killed': max(results['bosses_killed']) if results['bosses_killed'] else 0,
            'min_bosses_killed': min(results['bosses_killed']) if results['bosses_killed'] else 0,
            'boss_kill_distribution': self._get_distribution(results['bosses_killed']),
            'length_distribution': self._get_distribution(results['episode_lengths']),
            'defeat_reasons': self._get_defeat_reason_stats(results['defeat_reasons'])
        }
        
        return summary
    
    def _get_distribution(self, values: List) -> Dict:
        """Get distribution statistics for a list of values"""
        if not values:
            return {}
        
        unique, counts = np.unique(values, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        return {
            'distribution': distribution,
            'percentiles': {
                '25th': np.percentile(values, 25),
                '50th': np.percentile(values, 50),
                '75th': np.percentile(values, 75),
                '90th': np.percentile(values, 90)
            }
        }
    
    def _get_defeat_reason_stats(self, defeat_reasons: List[str]) -> Dict:
        """Get statistics on defeat reasons"""
        reason_counts = defaultdict(int)
        for reason in defeat_reasons:
            reason_counts[reason] += 1
        
        total = len(defeat_reasons)
        reason_percentages = {reason: (count / total) * 100 
                            for reason, count in reason_counts.items()}
        
        return {
            'counts': dict(reason_counts),
            'percentages': reason_percentages
        }
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary statistics"""
        print("\n" + "=" * 60)
        print("🏆 RANDOM POLICY EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"Total Games: {summary['total_games']}")
        print(f"Victories: {summary['victories']} ({summary['victory_rate']:.2%})")
        print()
        
        print("📊 REWARD STATISTICS:")
        print(f"  Average Reward: {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
        print()
        
        print("⏱️  EPISODE LENGTH STATISTICS:")
        print(f"  Average Length: {summary['avg_length']:.1f} ± {summary['std_length']:.1f} steps")
        length_dist = summary['length_distribution']
        if 'percentiles' in length_dist:
            p = length_dist['percentiles']
            print(f"  Percentiles: 25th={p['25th']:.0f}, 50th={p['50th']:.0f}, 75th={p['75th']:.0f}, 90th={p['90th']:.0f}")
        print()
        
        print("🐉 BOSSES KILLED STATISTICS:")
        print(f"  Average Bosses: {summary['avg_bosses_killed']:.2f} ± {summary['std_bosses_killed']:.2f}")
        print(f"  Range: {summary['min_bosses_killed']} - {summary['max_bosses_killed']}")
        
        boss_dist = summary['boss_kill_distribution']
        if 'percentiles' in boss_dist:
            p = boss_dist['percentiles']
            print(f"  Percentiles: 25th={p['25th']:.0f}, 50th={p['50th']:.0f}, 75th={p['75th']:.0f}, 90th={p['90th']:.0f}")
        
        if 'distribution' in boss_dist:
            print("  Distribution:")
            for bosses, count in sorted(boss_dist['distribution'].items()):
                percentage = (count / summary['total_games']) * 100
                print(f"    {bosses} bosses: {count} games ({percentage:.1f}%)")
        print()
        
        print("💀 DEFEAT REASONS:")
        defeat_stats = summary['defeat_reasons']
        for reason, percentage in sorted(defeat_stats['percentages'].items(), key=lambda x: x[1], reverse=True):
            count = defeat_stats['counts'][reason]
            print(f"  {reason}: {count} games ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def _generate_plots(self):
        """Generate histogram plots"""
        print("\n📈 Generating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Random Policy Performance Analysis', fontsize=16)
        
        # Bosses killed histogram
        axes[0, 0].hist(self.results['bosses_killed'], bins=range(max(self.results['bosses_killed']) + 2), 
                       alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Bosses Killed')
        axes[0, 0].set_xlabel('Number of Bosses Killed')
        axes[0, 0].set_ylabel('Number of Games')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        avg_bosses = np.mean(self.results['bosses_killed'])
        max_bosses = max(self.results['bosses_killed'])
        axes[0, 0].axvline(avg_bosses, color='red', linestyle='--', 
                          label=f'Mean: {avg_bosses:.2f}')
        axes[0, 0].legend()
        
        # Episode length histogram
        axes[0, 1].hist(self.results['episode_lengths'], bins=20, 
                       alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Episode Lengths')
        axes[0, 1].set_xlabel('Episode Length (steps)')
        axes[0, 1].set_ylabel('Number of Games')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        avg_length = np.mean(self.results['episode_lengths'])
        axes[0, 1].axvline(avg_length, color='red', linestyle='--', 
                          label=f'Mean: {avg_length:.1f}')
        axes[0, 1].legend()
        
        # Reward histogram
        axes[1, 0].hist(self.results['episode_rewards'], bins=20, 
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Distribution of Episode Rewards')
        axes[1, 0].set_xlabel('Episode Reward')
        axes[1, 0].set_ylabel('Number of Games')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        avg_reward = np.mean(self.results['episode_rewards'])
        axes[1, 0].axvline(avg_reward, color='red', linestyle='--', 
                          label=f'Mean: {avg_reward:.2f}')
        axes[1, 0].legend()
        
        # Victory pie chart
        victories = sum(self.results['victories'])
        defeats = len(self.results['victories']) - victories
        
        if victories > 0:
            labels = ['Victories', 'Defeats']
            sizes = [victories, defeats]
            colors = ['lightcoral', 'lightskyblue']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Victory Rate')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Victories\nAchieved', 
                           ha='center', va='center', fontsize=14, 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Victory Rate: 0%')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.path_manager.get_plot_path("random_policy_histograms.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Histograms saved to: {plot_path}")
        
        # Create detailed bosses killed plot
        self._create_detailed_boss_plot()
        
        plt.show()
    
    def _create_detailed_boss_plot(self):
        """Create a more detailed analysis of boss kills"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Detailed histogram with better binning
        max_bosses = max(self.results['bosses_killed']) if self.results['bosses_killed'] else 0
        bins = np.arange(-0.5, max_bosses + 1.5, 1)
        
        counts, bin_edges, patches = axes[0].hist(self.results['bosses_killed'], bins=bins, 
                                                 alpha=0.7, color='skyblue', edgecolor='black')
        
        # Color bars differently for better visualization
        for i, patch in enumerate(patches):
            if i == 0:  # 0 bosses
                patch.set_facecolor('lightcoral')
            elif i <= 3:  # 1-3 bosses
                patch.set_facecolor('orange')
            elif i <= 6:  # 4-6 bosses
                patch.set_facecolor('yellow')
            else:  # 7+ bosses
                patch.set_facecolor('lightgreen')
        
        axes[0].set_title('Detailed Bosses Killed Distribution')
        axes[0].set_xlabel('Number of Bosses Killed')
        axes[0].set_ylabel('Number of Games')
        axes[0].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        total_games = len(self.results['bosses_killed'])
        for i, count in enumerate(counts):
            if count > 0:
                percentage = (count / total_games) * 100
                axes[0].text(i, count + 0.5, f'{percentage:.1f}%', 
                           ha='center', va='bottom', fontsize=9)
        
        # Cumulative distribution
        sorted_bosses = sorted(self.results['bosses_killed'])
        cumulative_probs = np.arange(1, len(sorted_bosses) + 1) / len(sorted_bosses)
        
        axes[1].plot(sorted_bosses, cumulative_probs, marker='o', markersize=3)
        axes[1].set_title('Cumulative Distribution of Bosses Killed')
        axes[1].set_xlabel('Number of Bosses Killed')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Add percentile lines
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(self.results['bosses_killed'], p)
            axes[1].axvline(value, color='red', linestyle='--', alpha=0.7)
            axes[1].text(value, 0.1 + (p/100 * 0.8), f'{p}th', 
                        rotation=90, va='bottom', ha='right')
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_plot_path = self.path_manager.get_plot_path("random_policy_boss_analysis.png")
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed boss analysis saved to: {detailed_plot_path}")


def main():
    """Main evaluation function"""
    print("🎯 Random Policy Evaluation for Regicide")
    print("=" * 50)
    
    # Configuration
    NUM_GAMES = 1000000  # Number of games to run
    NUM_PLAYERS = 3
    MAX_HAND_SIZE = 6
    MAX_STEPS = 500  # Maximum steps per game
    
    print(f"Configuration:")
    print(f"  Games to run: {NUM_GAMES}")
    print(f"  Players: {NUM_PLAYERS}")
    print(f"  Max hand size: {MAX_HAND_SIZE}")
    print(f"  Max steps per game: {MAX_STEPS}")
    print()
    
    # Create evaluator
    evaluator = RandomPolicyEvaluator(
        num_players=NUM_PLAYERS,
        max_hand_size=MAX_HAND_SIZE,
        max_steps=MAX_STEPS
    )
    
    try:
        # Run evaluation
        summary = evaluator.run_evaluation(NUM_GAMES)
        
        print(f"\n✅ Evaluation complete!")
        print(f"Results saved to: {evaluator.path_manager.experiment_dir}")
        
        # Save summary to file
        summary_path = evaluator.path_manager.get_model_path("evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Random Policy Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Summary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation stopped by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
