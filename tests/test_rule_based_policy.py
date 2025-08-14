"""
Test script for the Rule-Based Policy
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.rule_based_policy import RuleBasedPolicy
from train.regicide_gym_env import make_regicide_env


def test_rule_based_policy():
    """Test the rule-based policy with the Regicide environment"""
    print("🧪 Testing Rule-Based Policy")
    print("=" * 50)
    
    # Create environment and policy
    env = make_regicide_env(num_players=2, observation_mode='card_aware')
    policy = RuleBasedPolicy(max_hand_size=8, max_actions=30, game_state_dim=19)
    
    print(f"Policy created with {policy.num_rules} rules:")
    for i, rule_func in enumerate(policy.rules):
        print(f"  {i+1:2d}. {rule_func.__name__}")
    
    # Test reset and first observation
    obs, info = env.reset()
    print(f"\nInitial observation:")
    print(f"  Hand size: {obs['hand_size'].item()}")
    print(f"  Valid actions: {obs['num_valid_actions'].item()}")
    print(f"  Game state shape: {obs['game_state'].shape}")
    
    # Test policy forward pass
    print(f"\nTesting policy forward pass...")
    try:
        logits = policy.forward(obs)
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Test action selection
        action, log_prob = policy.get_action(obs)
        print(f"  Selected action: {action}")
        print(f"  Log probability: {log_prob.item():.3f}")
        
        # Test action probabilities
        probs = policy.get_action_probabilities(obs)
        print(f"  Action probabilities shape: {probs.shape}")
        print(f"  Probabilities sum: {probs.sum().item():.3f}")
        
        # Test decision analysis
        analysis = policy.analyze_decision(obs)
        print(f"\nDecision analysis:")
        print(f"  Rule weights: {[f'{w:.3f}' for w in analysis['rule_weights'][:5]]}... (first 5)")
        print(f"  Rule temperature: {analysis['rule_temperature']:.3f}")
        
        print(f"\n✅ Policy forward pass successful!")
        
    except Exception as e:
        print(f"❌ Policy forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a few game steps
    print(f"\nTesting game steps...")
    try:
        for step in range(3):
            if info['valid_actions'] > 0:
                action, log_prob = policy.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                print(f"  Step {step+1}: action={action}, reward={reward:.3f}, done={done}")
                
                if done:
                    print("  Game ended!")
                    break
            else:
                print("  No valid actions available!")
                break
        
        print(f"✅ Game steps successful!")
        
    except Exception as e:
        print(f"❌ Game steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_rule_based_policy()
