"""
Test script to verify that the updated ActorCriticPolicy works with the new discard pile information
"""

import torch
import numpy as np
from regicide_gym_env import make_regicide_env
from ppo_training import ActorCriticPolicy

def test_actor_critic_with_discard_pile():
    """Test that the updated ActorCriticPolicy works with discard pile information"""
    print("🧪 Testing Updated ActorCriticPolicy with Discard Pile")
    print("="*60)
    
    # Create environment
    env = make_regicide_env(
        num_players=2,
        max_hand_size=5,
        observation_mode="card_aware"
    )
    
    # Create actor-critic policy
    policy = ActorCriticPolicy(
        max_hand_size=5,
        max_actions=30,
        card_embed_dim=12,
        hidden_dim=128
    )
    
    print(f"✓ Environment and policy created successfully")
    
    # Reset environment and get initial observation
    obs, info = env.reset()
    print(f"✓ Environment reset. Observation keys: {list(obs.keys())}")
    
    # Test that observation includes discard pile
    if 'discard_pile_cards' in obs:
        print(f"✓ Discard pile cards present: shape {obs['discard_pile_cards'].shape}")
    else:
        print("✗ Discard pile cards missing from observation!")
        return False
    
    # Test actor (get action)
    try:
        action, log_prob = policy.actor.get_action(obs)
        print(f"✓ Actor forward pass successful. Action: {action}, Log prob: {log_prob}")
    except Exception as e:
        print(f"✗ Actor forward pass failed: {str(e)}")
        return False
    
    # Test critic (get value)
    try:
        value = policy.get_value(obs)
        print(f"✓ Critic forward pass successful. Value: {value}")
    except Exception as e:
        print(f"✗ Critic forward pass failed: {str(e)}")
        return False
    
    # Test combined action and value
    try:
        action, log_prob, value = policy.get_action_and_value(obs)
        print(f"✓ Combined action and value successful. Action: {action}, Value: {value}")
    except Exception as e:
        print(f"✗ Combined action and value failed: {str(e)}")
        return False
    
    # Test with a few environment steps
    print(f"\n🔄 Testing with environment steps...")
    for step in range(3):
        if info['valid_actions'] > 0:
            try:
                action, log_prob, value = policy.get_action_and_value(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                discard_count = obs['discard_pile_cards'].sum().item()
                print(f"  Step {step+1}: Action {action}, Value {value:.3f}, Discard cards: {discard_count}")
                
                if terminated:
                    print(f"  Episode terminated at step {step+1}")
                    break
            except Exception as e:
                print(f"✗ Step {step+1} failed: {str(e)}")
                return False
        else:
            print(f"  No valid actions at step {step+1}")
            break
    
    env.close()
    print(f"\n✅ All tests passed! ActorCriticPolicy successfully updated for discard pile.")
    return True

def test_feature_dimensions():
    """Test that the feature dimensions are correct"""
    print(f"\n🔍 Testing Feature Dimensions")
    print("="*40)
    
    # Create policy
    policy = ActorCriticPolicy(
        max_hand_size=5,
        max_actions=30,
        card_embed_dim=12,
        hidden_dim=128
    )
    
    # Create a mock observation
    mock_obs = {
        'hand_cards': torch.zeros(5, dtype=torch.long),
        'hand_size': torch.tensor(3, dtype=torch.long),
        'enemy_card': torch.tensor(10, dtype=torch.long),
        'game_state': torch.zeros(12, dtype=torch.float32),
        'discard_pile_cards': torch.zeros(54, dtype=torch.bool),
        'action_mask': torch.zeros(30, dtype=torch.bool),
        'num_valid_actions': torch.tensor(5, dtype=torch.long),
        'action_card_indices': [[], [1], [2], [1, 2], []]
    }
    
    # Test feature extraction
    try:
        features = policy.get_features(mock_obs)
        print(f"✓ Feature extraction successful. Shape: {features.shape}")
        
        # Expected dimensions:
        # hand_context: 12 (card_embed_dim)
        # enemy_embedding: 12 (card_embed_dim) 
        # game_context: 6 (game_state_dim//2)
        # discard_context: 6 (game_state_dim//2)
        # Total: 12 + 12 + 6 + 6 = 36
        expected_dim = 12 + 12 + 6 + 6  # card_embed_dim * 2 + game_state_dim
        
        if features.shape[-1] == expected_dim:
            print(f"✓ Feature dimensions correct: {features.shape[-1]} (expected {expected_dim})")
        else:
            print(f"✗ Feature dimensions incorrect: {features.shape[-1]} (expected {expected_dim})")
            return False
            
    except Exception as e:
        print(f"✗ Feature extraction failed: {str(e)}")
        return False
    
    print(f"✅ Feature dimension test passed!")
    return True

if __name__ == "__main__":
    success1 = test_actor_critic_with_discard_pile()
    success2 = test_feature_dimensions()
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! ActorCriticPolicy is ready for training with discard pile information.")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
