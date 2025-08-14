"""
Example of how to use the new ModelManager and TrainingResumer utilities
"""

import torch
import torch.optim as optim
from config import PathManager
from training_utils import ModelManager, TrainingResumer, create_model_config

def example_usage():
    """Demonstrate the model management functionality"""
    
    # Initialize path manager (assuming this exists in your config)
    path_manager = PathManager()
    
    # Create model manager and training resumer
    model_manager = ModelManager(path_manager)
    training_resumer = TrainingResumer(model_manager)
    
    # Example 1: Create model configuration
    model_config = create_model_config(
        policy_type="rule_based",
        max_hand_size=30,
        max_actions=100,
        card_embed_dim=128,
        hidden_dim=256
    )
    print("📋 Model configuration created:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")
    
    # Example 2: Save model with versioning
    # (Assuming you have a policy model)
    # policy = YourPolicyModel()
    # optimizer = optim.Adam(policy.parameters())
    
    # Create example state to save
    model_state = {
        'model_state_dict': {},  # policy.state_dict() in real usage
        'optimizer_state_dict': {},  # optimizer.state_dict() in real usage
        'episode': 1000,
        'best_score': 85.5,
        'model_config': model_config,
        'training_metadata': {
            'total_episodes': 1000,
            'learning_rate': 0.001,
            'batch_size': 64
        }
    }
    
    # Save with versioning (keeps 3 most recent versions)
    saved_path = model_manager.save_model_with_versioning(
        model_state=model_state,
        filename="regicide_policy.pth",
        is_checkpoint=True,
        keep_versions=3
    )
    print(f"\n💾 Model saved to: {saved_path}")
    
    # Example 3: Find compatible models
    compatible_model = model_manager.find_latest_compatible_model(
        model_config=model_config,
        search_checkpoints=True
    )
    
    if compatible_model:
        print(f"\n🔍 Found compatible model: {compatible_model}")
    else:
        print("\n🔍 No compatible models found")
    
    # Example 4: Auto-resume functionality
    # (This would be used in your training script)
    # start_episode, best_score, resumed = training_resumer.attempt_resume(
    #     policy=policy,
    #     optimizer=optimizer,
    #     model_config=model_config,
    #     force_resume=False  # Will ask user for confirmation
    # )
    
    print("\n✅ Model management example completed!")

if __name__ == "__main__":
    example_usage()
