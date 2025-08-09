"""Test the enhanced policy with tensor fixes"""

import torch
import numpy as np
from enhanced_policy import EnhancedCardAwarePolicy
from regicide_gym_env import RegicideGymEnv

def test_enhanced_policy():
    """Test enhanced policy with environment"""
    print("Testing Enhanced Policy with Tensor Fixes...")
    
    # Create environment and policy
    env = RegicideGymEnv()
    policy = EnhancedCardAwarePolicy(
        input_dim=34,  # Based on observation space
        action_dim=env.action_space.n,
        max_actions=env.action_space.n
    )
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Hand cards shape: {obs['hand_cards'].shape}")
    print(f"Enemy card shape: {obs['enemy_card'].shape}")
    print(f"Game state shape: {obs['game_state'].shape}")
    
    # Test policy forward pass
    try:
        with torch.no_grad():
            action_logits = policy(obs)
            print(f"Action logits shape: {action_logits.shape}")
            print(f"Action logits range: [{action_logits.min():.3f}, {action_logits.max():.3f}]")
            
            # Test action selection
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            print(f"Selected action: {action}")
            
            # Test value estimation if available
            if hasattr(policy, 'value_head'):
                # Create a simple value head test
                enhanced_features = policy._compute_strategic_features(obs)
                print(f"Enhanced features shape: {enhanced_features.shape}")
            
        print("✓ Enhanced policy works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced policy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\nTesting Batch Processing...")
    
    env = RegicideGymEnv()
    policy = EnhancedCardAwarePolicy(
        input_dim=34,
        action_dim=env.action_space.n,
        max_actions=env.action_space.n
    )
    
    # Create batch observations
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    
    # Stack observations for batch processing
    batch_obs = {
        'hand_cards': torch.stack([obs1['hand_cards'], obs2['hand_cards']]),
        'enemy_card': torch.stack([obs1['enemy_card'], obs2['enemy_card']]),
        'game_state': torch.stack([obs1['game_state'], obs2['game_state']]),
        'hand_size': torch.stack([obs1['hand_size'], obs2['hand_size']])
    }
    
    try:
        with torch.no_grad():
            batch_logits = policy(batch_obs)
            print(f"Batch logits shape: {batch_logits.shape}")
            print(f"Expected shape: [2, {env.action_space.n}]")
            
            assert batch_logits.shape[0] == 2, f"Expected batch size 2, got {batch_logits.shape[0]}"
            
        print("✓ Batch processing works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    single_test = test_enhanced_policy()
    batch_test = test_batch_processing()
    
    if single_test and batch_test:
        print("\n🎉 All tests passed! Enhanced policy is ready for training.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
