import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sb3_contrib.ppo_mask import MaskablePPO

from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
from solvers.architecture import RegicideFeatureExtractor
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_bc(data_path="bc_data.npz", config_path="config.yaml", epochs=10, batch_size=64):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    # Convert to tensors
    hand_values = torch.tensor(data['hand_values'], dtype=torch.long)
    hand_suits = torch.tensor(data['hand_suits'], dtype=torch.long)
    enemy_stats = torch.tensor(data['enemy_stats'], dtype=torch.float32)
    flags = torch.tensor(data['flags'], dtype=torch.float32)
    action_masks = torch.tensor(data['action_masks'], dtype=torch.float32) # not strictly needed for CE loss, but good to have
    actions = torch.tensor(data['actions'], dtype=torch.long)
    
    dataset = TensorDataset(hand_values, hand_suits, enemy_stats, flags, action_masks, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Initializing model architecture...")
    config = load_config(config_path)
    ppo_cfg = config["ppo"]
    
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    # Custom architecture kwargs
    policy_kwargs = dict(
        features_extractor_class=RegicideFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=ppo_cfg.get("net_arch", [256, 256]), vf=ppo_cfg.get("net_arch", [256, 256]))
    )
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=ppo_cfg["device"],
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    policy = model.policy
    policy.train()
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Behavioral Cloning pre-training...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            b_hand_v, b_hand_s, b_enemy, b_flags, b_masks, b_acts = batch
            
            # Send to device
            b_hand_v = b_hand_v.to(model.device)
            b_hand_s = b_hand_s.to(model.device)
            b_enemy = b_enemy.to(model.device)
            b_flags = b_flags.to(model.device)
            b_acts = b_acts.to(model.device).squeeze()
            
            obs_dict = {
                'hand_values': b_hand_v,
                'hand_suits': b_hand_s,
                'enemy_stats': b_enemy,
                'flags': b_flags,
                # action_mask is in obs but we don't need it for extract_features
                'action_mask': b_masks.to(model.device)
            }
            
            # Forward pass
            features = policy.extract_features(obs_dict)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            
            
            loss = criterion(logits, b_acts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_acts).sum().item()
            total += b_acts.size(0)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {correct/total*100:.2f}%")
        
    os.makedirs("models", exist_ok=True)
    out_path = "models/ppo_bc_pretrained"
    model.save(out_path)
    print(f"Pre-trained model saved to {out_path}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="bc_data.npz", help="Path to BC data")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for pre-training")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_bc(data_path=args.data, epochs=args.epochs, batch_size=args.batch)
