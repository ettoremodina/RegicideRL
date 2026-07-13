import numpy as np
import torch
from solvers.env import RegicideEnv
from solvers.wrappers import NumericObsWrapper
from sb3_contrib.ppo_mask import MaskablePPO
from game.action_handler import ActionHandler

def probe_policy(model_path, num_games=50):
    print(f"Probing model: {model_path} for {num_games} games...")
    
    # Initialize environment
    raw_env = RegicideEnv(num_players=1)
    env = NumericObsWrapper(raw_env)
    
    # Load model
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model at {model_path}: {e}")
        return None
        
    handler = ActionHandler()
    
    # Metrics
    total_decisions = 0
    actions_taken = []
    entropies = []
    bosses_killed = []
    rewards = []
    yields = 0
    
    # Scenarios
    scenarios = {
        'played_jester': 0,
        'yield_on_defense': 0,
        'wasted_face_card': 0 # playing J/Q/K unnecessarily
    }
    
    for game_idx in range(num_games):
        obs, info = env.reset()
        done = False
        game_reward = 0
        invalid_count = 0
        
        while not done:
            # We must use torch to evaluate the distribution explicitly to get entropy
            with torch.no_grad():
                # Format obs for PyTorch (add batch dimension)
                obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(model.device) for k, v in obs.items()}
                
                # Get the action distribution from the policy
                distribution = model.policy.get_distribution(obs_tensor)
                
                # MaskablePPO distributions take the action mask into account
                # Calculate entropy
                entropy = distribution.entropy().item()
                entropies.append(entropy)
                
            # Get the actual action
            action_masks = np.expand_dims(obs['action_mask'], axis=0)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action = int(action)
            actions_taken.append(action)
            
            # Record scenarios
            action_mask_list = [(action >> i) & 1 for i in range(handler.max_hand_size)]
            raw_obs = raw_env._get_obs()
            current_hand = raw_obs['hand']
            cards_played = [c for i, c in enumerate(current_hand) if i < len(current_hand) and action_mask_list[i]]
            
            if not cards_played:
                yields += 1
                if raw_env.required_defense > 0:
                    scenarios['yield_on_defense'] += 1
            else:
                for c in cards_played:
                    if c.value == 0: # Jester
                        scenarios['played_jester'] += 1
                    if c.value in [11, 12, 13]: # Face cards
                        # Simple heuristic: if attack is 0 and defense is 0, playing a face card might be a waste
                        # (Though could be valid to cycle hand)
                        if raw_env.game.current_enemy and raw_env.game.current_enemy.attack == 0 and raw_env.required_defense == 0:
                            scenarios['wasted_face_card'] += 1
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            game_reward += reward
            done = terminated or truncated
            total_decisions += 1
            
            if reward == -1.0: # Invalid action was taken
                invalid_count += 1
                if invalid_count > 10:
                    print(f"Game {game_idx} stuck in infinite loop of invalid actions! Breaking...")
                    break
            else:
                invalid_count = 0
                
        rewards.append(game_reward)
        bosses = raw_env.game.discard_pile
        bosses_killed.append(len([c for c in bosses if c.value in [11, 12, 13]])) 
        # Actually a better way to count bosses defeated is max(0, 12 - len(castle_deck))
        defeated = 12 - len(raw_env.game.castle_deck)
        bosses_killed[-1] = defeated if raw_env.game.victory else max(0, defeated - 1)
        
    results = {
        'total_decisions': total_decisions,
        'total_games': num_games,
        'entropies': entropies,
        'actions_taken': actions_taken,
        'bosses_killed': bosses_killed,
        'rewards': rewards,
        'yields': yields,
        'scenarios': scenarios
    }
    return results
