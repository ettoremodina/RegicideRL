"""
Integration example: How to modify your training script to use ModelManager and TrainingResumer

This shows the key changes needed in streamlined_training.py
"""

# Key imports to add:
from config import PathManager
from training_utils import ModelManager, TrainingResumer, create_model_config

def enhanced_training_with_model_management():
    """
    Example of integrating model management into your training loop
    """
    
    # 1. Initialize model management
    path_manager = PathManager()
    model_manager = ModelManager(path_manager)
    training_resumer = TrainingResumer(model_manager)
    
    # 2. Create model configuration for compatibility tracking
    model_config = create_model_config(
        policy_type="rule_based",
        max_hand_size=30,
        max_actions=100,
        card_embed_dim=128,
        hidden_dim=256,
        temperature_decay=config.TEMPERATURE_DECAY,
        learning_rate=config.LEARNING_RATE
    )
    
    # 3. Initialize your policy and optimizer
    # policy = RuleBasedPolicy(...)
    # optimizer = optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
    
    # 4. Attempt to resume from compatible checkpoint
    start_episode, best_score, resumed = training_resumer.attempt_resume(
        policy=policy,
        optimizer=optimizer,
        model_config=model_config,
        force_resume=False  # Will ask user for confirmation
    )
    
    if resumed:
        print(f"🔄 Resumed training from episode {start_episode}")
    else:
        print("🆕 Starting fresh training")
        start_episode = 0
        best_score = float('-inf')
    
    # 5. Training loop (your existing logic)
    for episode in range(start_episode, config.NUM_EPISODES):
        
        # ... your training logic here ...
        
        # 6. Enhanced checkpoint saving with versioning
        if episode % config.CHECKPOINT_INTERVAL == 0:
            # Create comprehensive checkpoint
            checkpoint_state = training_resumer.create_resume_config(
                policy=policy,
                optimizer=optimizer,
                episode=episode,
                best_score=best_score,
                model_config=model_config
            )
            
            # Save with automatic versioning and cleanup
            checkpoint_path = model_manager.save_model_with_versioning(
                model_state=checkpoint_state,
                filename=f"checkpoint_episode_{episode}.pth",
                is_checkpoint=True,
                keep_versions=5  # Keep 5 most recent checkpoints
            )
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # 7. Save best model with versioning
        if current_score > best_score:
            best_score = current_score
            
            best_model_state = training_resumer.create_resume_config(
                policy=policy,
                optimizer=optimizer,
                episode=episode,
                best_score=best_score,
                model_config=model_config
            )
            
            best_model_path = model_manager.save_model_with_versioning(
                model_state=best_model_state,
                filename="best_regicide_policy.pth",
                is_checkpoint=False,  # Save to models directory
                keep_versions=3  # Keep 3 best versions
            )
            
            print(f"🏆 New best model saved: {best_model_path}")


# Integration points for your streamlined_training.py:

"""
STEP 1: Add imports at the top
from training_utils import ModelManager, TrainingResumer, create_model_config

STEP 2: Replace your policy/optimizer initialization section with:
    # Model management setup
    model_manager = ModelManager(path_manager)
    training_resumer = TrainingResumer(model_manager)
    
    # Create model config
    model_config = create_model_config(
        policy_type="rule_based",
        max_hand_size=30,
        max_actions=100,
        card_embed_dim=128,
        hidden_dim=256,
        temperature_decay=config.TEMPERATURE_DECAY,
        learning_rate=config.LEARNING_RATE
    )
    
    # Initialize policy and optimizer
    policy = RuleBasedPolicy(...)
    optimizer = optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
    
    # Auto-resume
    start_episode, best_score, resumed = training_resumer.attempt_resume(
        policy, optimizer, model_config, force_resume=False
    )

STEP 3: Replace your checkpoint saving with:
    if episode % CHECKPOINT_INTERVAL == 0:
        checkpoint_state = training_resumer.create_resume_config(
            policy, optimizer, episode, best_score, model_config
        )
        model_manager.save_model_with_versioning(
            checkpoint_state, f"checkpoint_episode_{episode}.pth", 
            is_checkpoint=True, keep_versions=5
        )

STEP 4: Replace your best model saving with:
    if current_score > best_score:
        best_score = current_score
        best_state = training_resumer.create_resume_config(
            policy, optimizer, episode, best_score, model_config
        )
        model_manager.save_model_with_versioning(
            best_state, "best_regicide_policy.pth", 
            is_checkpoint=False, keep_versions=3
        )
"""
