"""
Quick Policy Analysis Script
Easy-to-use script for analyzing trained Regicide models
"""

import os
import glob
from tests.policy_analysis import PolicyAnalyzer

def find_latest_model(base_dir="outputs"):
    """Find the most recent trained model"""
    model_patterns = [
        "outputs/**/models/regicide_policy_final.pth",
        "outputs/**/models/*.pth", 
        "outputs/**/checkpoints/*.pth"
    ]
    
    all_models = []
    for pattern in model_patterns:
        all_models.extend(glob.glob(pattern, recursive=True))
    
    if not all_models:
        print("❌ No trained models found!")
        print("   Looking in: outputs/**/models/*.pth and outputs/**/checkpoints/*.pth")
        return None
    
    # Sort by modification time (most recent first)
    all_models.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📁 Found {len(all_models)} model(s)")
    for i, model in enumerate(all_models[:5]):  # Show top 5
        mtime = os.path.getmtime(model)
        print(f"   {i+1}. {model} (modified: {os.path.getctime(model)})")
    
    return all_models[0]

def quick_analysis(model_path=None, num_games=50):
    """Run a quick analysis of a trained model"""
    
    if model_path is None:
        print("🔍 Searching for trained models...")
        model_path = find_latest_model()
        if model_path is None:
            return
        print(f"✅ Using latest model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n🎮 Starting Policy Analysis")
    print(f"Model: {model_path}")
    print(f"Games to analyze: {num_games}")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PolicyAnalyzer(
        model_path=model_path,
        num_players=4,
        max_hand_size=5
    )
    
    # Run analysis
    analyzer.analyze_games(num_games=num_games, verbose=True)
    
    # Generate report
    stats = analyzer.generate_analysis_report()
    
    print("\n" + "="*50)
    print("📊 QUICK SUMMARY")
    print("="*50)
    
    # Action diversity check
    collapse_info = stats['potential_action_collapse']
    if collapse_info['collapsed']:
        print(f"⚠️  ACTION COLLAPSE DETECTED!")
        print(f"   Action {collapse_info['dominant_action']} used {collapse_info['dominant_action_percentage']:.1%} of decisions")
        print(f"   Only {collapse_info['unique_actions_used']} unique actions used")
    else:
        print(f"✅ Good action diversity")
        print(f"   {collapse_info['unique_actions_used']} unique actions used")
        print(f"   Most common action: {collapse_info['dominant_action']} ({collapse_info['dominant_action_percentage']:.1%})")
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE:")
    print(f"   Win rate: {stats['win_rate']:.1%}")
    print(f"   Average bosses killed: {stats['mean_bosses_killed']:.2f}")
    print(f"   Max bosses killed: {stats['max_bosses_killed']}")
    
    # Decision quality
    print(f"\n🎯 DECISION QUALITY:")
    print(f"   Average entropy: {stats['mean_entropy']:.3f}")
    print(f"   Action consistency: {stats['mean_action_consistency']:.3f}")
    print(f"   Yield rate: {stats['yield_rate']:.1%}")
    
    # Assessment
    quality = stats['decision_quality_assessment']
    print(f"\n💡 ASSESSMENT:")
    print(f"   Entropy: {quality['entropy_based']}")
    print(f"   Consistency: {quality['consistency_based']}")
    print(f"   Performance: {quality['performance_based']}")
    
    print(f"\n📁 Detailed results saved to: {analyzer.output_dir}")
    
    return analyzer.output_dir

if __name__ == "__main__":
    import sys
    
    model_path = None
    num_games = 50
    
    # Simple command line argument parsing
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_games = int(sys.argv[2])
    
    print("🎮 REGICIDE POLICY QUICK ANALYSIS")
    print("=" * 40)
    
    quick_analysis(model_path, num_games)
