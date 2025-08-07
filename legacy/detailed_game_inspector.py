#!/usr/bin/env python3
"""
Detailed Game Inspector for Regicide Training
Shows comprehensive game state information to verify correct gameplay mechanics
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from regicide_gym_env import make_regicide_env
from card_aware_policy import CardAwarePolicy


class DetailedGameInspector:
    """Comprehensive game state inspector for training verification"""
    
    def __init__(self, num_players: int = 2, max_hand_size: int = 6):
        self.env = make_regicide_env(
            num_players=num_players,
            max_hand_size=max_hand_size,
            observation_mode="card_aware",
            render_mode=None
        )
        
        # Create a random policy for testing
        self.policy = CardAwarePolicy(
            max_hand_size=max_hand_size,
            max_actions=self.env.max_actions
        )
        
        self.step_count = 0
        self.episode_count = 0
    
    def run_detailed_episode(self, max_steps: int = 100, use_policy: bool = True) -> Dict:
        """Run a single episode with detailed logging"""
        print("\n" + "🎮" + "="*80 + "🎮")
        print(f"🎯 STARTING DETAILED EPISODE {self.episode_count + 1}")
        print("🎮" + "="*80 + "🎮")
        
        obs, info = self.env.reset()
        self.step_count = 0
        episode_reward = 0.0
        
        self._show_initial_game_state()
        
        while self.step_count < max_steps and not info.get('game_over', False):
            print(f"\n🔥 STEP {self.step_count + 1} 🔥")
            print("─" * 60)
            
            # Show comprehensive pre-action state
            self._show_comprehensive_state(obs, info)
            
            # Get action
            if use_policy:
                action, log_prob = self.policy.get_action(obs)
                print(f"🤖 POLICY DECISION: Action {action} (log_prob: {log_prob:.4f})")
            else:
                # Manual action selection for testing
                action = self._get_manual_action(info)
            
            # Show what action means
            self._explain_action(action, info)
            
            # Execute action
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            episode_reward += reward
            
            # Show post-action results
            self._show_action_results(reward, next_obs, next_info)
            
            # Update for next iteration
            obs = next_obs
            info = next_info
            self.step_count += 1
            
            if terminated or truncated:
                print(f"\n🏁 EPISODE ENDED: {'Victory!' if info.get('victory', False) else 'Defeat'}")
                break
            
            # Pause for readability (optional)
            if self.step_count % 10 == 0:
                input("\nPress Enter to continue (or Ctrl+C to stop)...")
        
        self.episode_count += 1
        return {
            'episode_reward': episode_reward,
            'episode_length': self.step_count,
            'victory': info.get('victory', False),
            'bosses_killed': info.get('bosses_killed', 0)
        }
    
    def _show_initial_game_state(self):
        """Show the initial game setup"""
        game = self.env.game
        print(f"\n🏰 INITIAL GAME SETUP 🏰")
        print(f"Players: {game.num_players}")
        print(f"Castle deck (enemies): {len(game.castle_deck)} cards")
        print(f"Tavern deck: {len(game.tavern_deck)} cards")
        print(f"Current enemy: {game.current_enemy}")
        print(f"Enemy health: {game.current_enemy.health}")
        print(f"Enemy attack: {game.current_enemy.attack}")
        
        print(f"\n👥 PLAYER HANDS:")
        for i, hand in enumerate(game.players):
            hand_str = ', '.join(str(card) for card in hand) if hand else "(empty)"
            print(f"  Player {i+1}: {hand_str} ({len(hand)} cards)")
    
    def _show_comprehensive_state(self, obs: Dict, info: Dict):
        """Show detailed current game state"""
        game = self.env.game
        current_player = info['current_player']
        
        print(f"🎯 CURRENT TURN: Player {current_player + 1}")
        print(f"🎲 Phase: {info['phase'].upper()}")
        print(f"📊 Episode length: {info['episode_length']}")
        
        # Enemy status
        print(f"\n🐉 ENEMY STATUS:")
        print(f"  Card: {game.current_enemy}")
        print(f"  Health: {game.current_enemy.health - game.current_enemy.damage_taken}/{game.current_enemy.health}")
        print(f"  Damage taken: {game.current_enemy.damage_taken}")
        print(f"  Attack power: {game.current_enemy.get_effective_attack()}")
        print(f"  Spade protection: {game.current_enemy.spade_protection}")
        print(f"  Is defeated: {game.current_enemy.is_defeated()}")
        
        # Deck status
        print(f"\n📚 DECK STATUS:")
        print(f"  Enemies remaining: {len(game.castle_deck)}")
        print(f"  Tavern deck: {len(game.tavern_deck)} cards")
        print(f"  Discard pile: {len(game.discard_pile)} cards")
        
        # Player status
        print(f"\n👥 ALL PLAYERS:")
        for i, hand in enumerate(game.players):
            is_current = (i == current_player)
            has_yielded = game.players_yielded_this_round[i]
            
            prefix = "🎯 " if is_current else "   "
            suffix = " [YIELDED]" if has_yielded else ""
            suffix += " ⭐ CURRENT" if is_current else ""
            
            hand_str = ', '.join(str(card) for card in hand) if hand else "(empty)"
            print(f"{prefix}Player {i+1}{suffix}: {hand_str} ({len(hand)} cards)")
        
        # Yield status
        print(f"\n⚡ YIELD STATUS:")
        print(f"  Last active player: {game.last_active_player + 1 if game.last_active_player is not None else 'None'}")
        print(f"  Players yielded this round: {[i+1 for i, yielded in enumerate(game.players_yielded_this_round) if yielded]}")
        print(f"  Current player can yield: {'✅ YES' if game.can_yield() else '❌ NO'}")
        
        if not game.can_yield():
            all_others_yielded = all(game.players_yielded_this_round[i] for i in range(game.num_players) if i != current_player)
            print(f"  Reason: {'All other players have yielded' if all_others_yielded else 'Cannot yield in current state'}")
        
        # Game flags
        print(f"\n🏁 GAME FLAGS:")
        print(f"  Jester immunity cancelled: {game.jester_immunity_cancelled}")
        print(f"  Game over: {game.game_over}")
        print(f"  Victory: {getattr(game, 'victory', False)}")
        
        # Action information
        print(f"\n🎮 ACTION INFO:")
        print(f"  Valid actions: {info['valid_actions']}")
        print(f"  Total actions available: {len(self.env.valid_actions)}")
        
        # Show valid actions in detail
        if info['valid_actions'] > 0:
            meanings = self.env.get_action_meanings()
            print(f"  Available actions:")
            for i, meaning in enumerate(meanings):  # Show all actions
                print(f"    {i}: {meaning}")
                
        
        # Observation tensor info
        print(f"\n🧠 OBSERVATION INFO:")
        print(f"  Hand cards tensor: {obs['hand_cards']}")
        print(f"  Hand size: {obs['hand_size'].item()}")
        print(f"  Enemy card index: {obs['enemy_card'].item()}")
        print(f"  Game state tensor: {obs['game_state']}")
        print(f"  Number of valid actions: {obs['num_valid_actions'].item()}")
    
    def _get_manual_action(self, info: Dict) -> int:
        """Get manual action input for testing"""
        valid_actions = info['valid_actions']
        if valid_actions == 0:
            return 0
        
        meanings = self.env.get_action_meanings()
        print(f"\nChoose action (0-{valid_actions-1}):")
        for i in range(min(valid_actions, 10)):
            print(f"  {i}: {meanings[i]}")
        
        while True:
            try:
                action = int(input("Enter action: "))
                if 0 <= action < valid_actions:
                    return action
                else:
                    print(f"Invalid action. Must be 0-{valid_actions-1}")
            except ValueError:
                print("Please enter a number")
    
    def _explain_action(self, action: int, info: Dict):
        """Explain what the chosen action does"""
        if action >= info['valid_actions']:
            print(f"❌ INVALID ACTION: {action} (only {info['valid_actions']} valid actions)")
            return
        
        meanings = self.env.get_action_meanings()
        action_meaning = meanings[action] if action < len(meanings) else "Unknown"
        
        print(f"🎯 ACTION EXPLANATION:")
        print(f"  Action {action}: {action_meaning}")
        
        # Get the actual card indices for this action
        action_mask = self.env.valid_actions[action]
        current_hand = self.env.game.players[self.env.game.current_player]
        card_indices = self.env.action_handler.mask_to_card_indices(action_mask, len(current_hand))
        
        if not card_indices:
            print(f"  Type: YIELD TURN")
            print(f"  Effect: Pass turn to next player")
            if self.env.current_phase == "attack":
                enemy_attack = self.env.game.current_enemy.get_effective_attack()
                if enemy_attack > 0:
                    print(f"  Warning: Enemy will attack for {enemy_attack} damage!")
        else:
            cards_to_play = [current_hand[idx] for idx in card_indices]
            print(f"  Type: PLAY CARDS")
            print(f"  Cards: {', '.join(str(card) for card in cards_to_play)}")
            
            # Calculate potential damage
            total_attack = sum(card.get_attack_value() for card in cards_to_play)
            print(f"  Total attack value: {total_attack}")
            
            # Check for special effects
            suits = [card.suit for card in cards_to_play]
            if any(card.value == 0 for card in cards_to_play):
                print(f"  Special: JESTER - Cancels immunity, choose next player")
            
            for suit in set(suits):
                if suit.value == "♥":
                    print(f"  Special: HEARTS - Heal cards from discard")
                elif suit.value == "♦":
                    print(f"  Special: DIAMONDS - Draw cards for all players")
                elif suit.value == "♣":
                    print(f"  Special: CLUBS - Deal double damage")
                elif suit.value == "♠":
                    print(f"  Special: SPADES - Reduce enemy's future attack")
    
    def _show_action_results(self, reward: float, next_obs: Dict, next_info: Dict):
        """Show the results of the action"""
        print(f"\n📊 ACTION RESULTS:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Game over: {next_info.get('game_over', False)}")
        print(f"  Victory: {next_info.get('victory', False)}")
        print(f"  Phase changed to: {next_info['phase']}")
        print(f"  Current player: {next_info['current_player'] + 1}")
        print(f"  Bosses killed this episode: {next_info.get('bosses_killed', 0)}")
        print(f"  Last damage dealt: {next_info.get('last_damage', 0)}")
        
        # Show enemy status changes
        game = self.env.game
        if game.current_enemy:
            current_health = game.current_enemy.health - game.current_enemy.damage_taken
            print(f"  Enemy health now: {current_health}/{game.current_enemy.health}")
        
        # Show if any players changed yield status
        current_player = next_info['current_player']
        print(f"  Yield status: {[i+1 for i, yielded in enumerate(game.players_yielded_this_round) if yielded]}")
    
    def run_multiple_episodes(self, num_episodes: int = 3, max_steps_per_episode: int = 50):
        """Run multiple episodes for comprehensive testing"""
        print(f"\n🎯 RUNNING {num_episodes} DETAILED EPISODES")
        print("="*80)
        
        results = []
        
        for episode in range(num_episodes):
            try:
                result = self.run_detailed_episode(max_steps=max_steps_per_episode, use_policy=False)
                results.append(result)
                
                print(f"\n📈 EPISODE {episode + 1} SUMMARY:")
                print(f"  Reward: {result['episode_reward']:.2f}")
                print(f"  Length: {result['episode_length']} steps")
                print(f"  Victory: {'✅' if result['victory'] else '❌'}")
                print(f"  Bosses killed: {result['bosses_killed']}")
                
                if episode < num_episodes - 1:
                    input(f"\nPress Enter to start episode {episode + 2}...")
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopping at episode {episode + 1}")
                break
        
        # Final summary
        print(f"\n🏆 FINAL SUMMARY ({len(results)} episodes):")
        if results:
            avg_reward = sum(r['episode_reward'] for r in results) / len(results)
            avg_length = sum(r['episode_length'] for r in results) / len(results)
            victories = sum(1 for r in results if r['victory'])
            total_bosses = sum(r['bosses_killed'] for r in results)
            
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Average length: {avg_length:.1f} steps")
            print(f"  Victory rate: {victories}/{len(results)} ({victories/len(results)*100:.1f}%)")
            print(f"  Total bosses killed: {total_bosses}")
            print(f"  Average bosses per episode: {total_bosses/len(results):.1f}")
        
        return results


def test_specific_scenarios():
    """Test specific game scenarios"""
    print("\n🧪 TESTING SPECIFIC SCENARIOS")
    print("="*50)
    
    inspector = DetailedGameInspector(num_players=2, max_hand_size=6)
    
    print("Testing yielding mechanics...")
    
    # You can add specific test scenarios here
    # For example, force certain game states and verify behavior
    
    return inspector.run_detailed_episode(max_steps=20, use_policy=True)


def main():
    """Main inspection routine"""
    print("🔍 REGICIDE GAME INSPECTOR")
    print("=" * 50)
    print("This tool shows comprehensive game state information")
    print("to verify that training episodes follow correct game logic.")
    print("=" * 50)
    
    # Configuration
    NUM_PLAYERS = 2
    MAX_HAND_SIZE = 7
    NUM_EPISODES = 2
    MAX_STEPS = 30
    
    inspector = DetailedGameInspector(
        num_players=NUM_PLAYERS,
        max_hand_size=MAX_HAND_SIZE
    )
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"  Players: {NUM_PLAYERS}")
    print(f"  Max hand size: {MAX_HAND_SIZE}")
    print(f"  Episodes to run: {NUM_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    
    try:
        # Run detailed episodes
        results = inspector.run_multiple_episodes(
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS
        )
        
        print(f"\n✅ INSPECTION COMPLETE!")
        print("Check the detailed logs above to verify game mechanics.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Inspection stopped by user")
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
