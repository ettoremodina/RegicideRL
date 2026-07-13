#!/usr/bin/env python3
"""
Test script for the updated Regicide game flow with proper yielding mechanics.
This demonstrates the corrected game flow based on the flow diagram.
"""

from .regicide import Game
import random

class NewGameInterface():
    """Brand new game interface built from scratch for the updated flow mechanics"""
    
    def __init__(self):
        self.game = None
    
    def start_game(self, num_players: int):
        """Initialize a new game with the specified number of players"""
        if not (1 <= num_players <= 4):
            raise ValueError("Number of players must be between 1 and 4")
        
        self.game = Game(num_players)
        print(f"\n🎮 New Regicide game started with {num_players} player(s)!")
        self._show_game_rules()
    
    def _show_game_rules(self):
        """Display the key rules and flow information"""
        print("\n📜 KEY GAME FLOW RULES:")
        print("• YIELDING: You can only yield if NOT ALL other players have yielded since the last active turn")
        print("• ACTIVE TURN: Playing cards resets everyone's yield status")
        print("• DEFENSE: When enemy attacks, you must discard cards with total value >= enemy's attack")
        print("• VICTORY: Defeat all Jacks, Queens, and Kings to win!")
        print("• DEFEAT: If you cannot defend against an attack, you lose")
    
    def run_game_loop(self):
        """Main game loop"""
        if not self.game:
            raise ValueError("No game initialized. Call start_game() first.")
        
        while not self.game.game_over:
            self._display_full_game_state()
            self._handle_current_player_turn()
        
        self._display_game_results()
    
    def _display_full_game_state(self):
        """Display comprehensive game state information"""
        state = self.game.get_game_state()
        
        print("\n" + "🏰" + "="*68 + "🏰")
        print(f"🐉 ENEMY: {state['current_enemy']}")
        print(f"🏰 Enemies remaining: {state['enemies_remaining']}")
        print(f"🎴 Tavern: {state['tavern_cards']} cards | 🗂️ Discard: {state['discard_cards']} cards")
        
        # Detailed yield status
        print(f"\n⚡ YIELD STATUS:")
        if state['last_active_player'] is not None:
            print(f"   Last active player: Player {state['last_active_player'] + 1}")
        else:
            print("   No active turns taken yet")
        
        # Show each player's yield status
        yield_info = []
        for i in range(self.game.num_players):
            if state['players_yielded_this_round'][i]:
                status = "YIELDED"
            else:
                status = "ready"
            
            if i == state['current_player']:
                status = f">>> {status.upper()} <<<"
            
            yield_info.append(f"P{i+1}: {status}")
        
        print(f"   {' | '.join(yield_info)}")
        print(f"   Current player can yield: {'✅ YES' if state['can_yield'] else '❌ NO'}")
        
        if not state['can_yield']:
            print(f"   🚫 Reason: All other players have yielded since last active turn")
        
        # Player hands
        print(f"\n👥 PLAYER HANDS:")
        for i, hand in enumerate(state['player_hands']):
            is_current = i == state['current_player']
            has_yielded = state['players_yielded_this_round'][i]
            
            prefix = "🎯 " if is_current else "   "
            suffix = " [YIELDED]" if has_yielded else ""
            
            print(f"{prefix}Player {i+1}{suffix}: {', '.join(hand) if hand else '(no cards)'}")
        
        print("=" * 70)
    
    def _handle_current_player_turn(self):
        """Handle the current player's turn with all options"""
        current_player = self.game.current_player
        hand = self.game.players[current_player]
        
        print(f"\n🎯 PLAYER {current_player + 1}'S TURN")
        print("─" * 40)
        
        # Show player's hand with indices
        print("🎴 Your cards:")
        if not hand:
            print("   (No cards in hand)")
        else:
            for i, card in enumerate(hand):
                print(f"   {i}: {card}")
        
        # Main action loop
        while True:
            # Get fresh state each iteration
            state = self.game.get_game_state()
            
            # Show tactical information
            print(f"\n⚔️ TACTICAL INFO:")
            print(f"   Enemy attack power: {state['enemy_attack_damage']} damage")
            print(f"   You can defend: {'✅' if state['can_defend'] else '❌'}")
            
            if state['can_defend'] and state['enemy_attack_damage'] > 0:
                min_defense = self.game.get_minimum_defense_cards()
                if min_defense:
                    min_total = sum(c.get_discard_value() for c in min_defense)
                    min_cards_str = ', '.join(str(c) for c in min_defense)
                    print(f"   Min defense needed: {min_cards_str} (total: {min_total})")
            
            # Determine available actions based on current state
            actions = ["play"]
            if state['can_yield']:
                actions.append("yield")
            actions.extend(["help", "quit"])
            
            print(f"\n🎮 Available actions: {' | '.join(actions)}")
            action = input(f"Choose action: ").strip().lower()
            
            if action == "quit":
                print("👋 Quitting game...")
                self.game.game_over = True
                return
            
            elif action == "help":
                self._show_help()
                continue
            
            elif action == "yield":
                if not state['can_yield']:
                    print("❌ You cannot yield! All other players have already yielded.")
                    continue
                
                if self._handle_yield_action():
                    return
            
            elif action == "play":
                if not hand:
                    print("❌ You have no cards to play!")
                    continue
                
                if self._handle_play_action(hand):
                    return
            
            else:
                print(f"❌ Unknown action '{action}'. Try: {', '.join(actions)}")
    
    def _handle_yield_action(self):
        """Handle yielding and return True if turn should end"""
        print("\n🔄 Yielding turn...")
        
        # Double-check that yielding is still allowed
        if not self.game.can_yield():
            print("❌ Cannot yield! All other players have already yielded.")
            return False  # Don't end turn, let player try again
        
        result = self.game.yield_turn()
        
        print(f"   {result['message']}")
        
        if not result['success']:
            print(f"   ❌ Yield failed: {result['message']}")
            return False  # Don't end turn, let player try again
        
        if result['defense_required'] > 0:
            print(f"   ⚔️ Enemy attacks for {result['defense_required']} damage!")
            return self._handle_defense_phase(result['defense_required'])
        else:
            print(f"   ✅ Turn passed to Player {self.game.current_player + 1}")
            return True
    
    def _handle_play_action(self, hand):
        """Handle card playing and return True if turn should end"""
        print("\n🎴 Playing cards...")
        
        # Get card selection
        try:
            selection = input("Enter card indices (e.g., '0' or '0,1,2'): ").strip()
            if not selection:
                print("❌ No cards selected!")
                return False
            
            indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            if not all(0 <= i < len(hand) for i in indices):
                print("❌ Invalid card indices!")
                return False
            
            # Show what will be played
            cards_to_play = [hand[i] for i in indices]
            cards_str = ', '.join(str(c) for c in cards_to_play)
            print(f"   Playing: {cards_str}")
            
            # Execute the play
            result = self.game.play_card(indices)
            
            if not result['success']:
                print(f"❌ {result['message']}")
                return False
            
            # Handle the result
            return self._process_play_result(result)
            
        except ValueError:
            print("❌ Invalid input! Use numbers separated by commas.")
            return False
    
    def _process_play_result(self, result):
        """Process the result of playing cards"""
        print(f"\n⚔️ {result['message']}")
        
        if result['enemy_damage'] > 0:
            print(f"   💥 Damage dealt: {result['enemy_damage']}")
        
        if result['cards_played']:
            print(f"   🎴 Cards played: {', '.join(result['cards_played'])}")
        
        # Handle different phases
        if result['phase'] == 'next_player_choice':
            return self._handle_jester_choice()
        
        elif result['phase'] == 'defense_needed':
            print(f"   🛡️ You must defend against {result['defense_required']} damage!")
            return self._handle_defense_phase(result['defense_required'])
        
        elif result['phase'] == 'enemy_defeated':
            print("   🏆 Enemy defeated! You get another turn!")
            return False  # Continue with same player
        
        elif result['phase'] == 'turn_complete':
            print(f"   ✅ Turn complete! Next: Player {self.game.current_player + 1}")
            return True
        
        elif result['phase'] == 'victory':
            print("   � VICTORY! All enemies defeated!")
            return True
        
        return True
    
    def _handle_jester_choice(self):
        """Handle Jester effect - choosing next player"""
        print("\n🃏 JESTER EFFECT!")
        print("   You must choose who goes next.")
        
        while True:
            try:
                choice = int(input(f"Choose next player (0-{self.game.num_players-1}): "))
                if self.game.choose_next_player(choice):
                    print(f"   ✅ Player {choice + 1} will go next!")
                    return True
                else:
                    print("❌ Invalid player number!")
            except ValueError:
                print("❌ Please enter a valid number!")
    
    def _handle_defense_phase(self, damage_required):
        """Handle defense phase and return True if turn should end"""
        print(f"\n🛡️ DEFENSE PHASE")
        print("─" * 25)
        print(f"You must defend against {damage_required} damage!")
        
        if not self.game.can_defend():
            print("💀 GAME OVER - Cannot defend!")
            self.game.game_over = True
            return True
        
        hand = self.game.players[self.game.current_player]
        
        # Show defense options
        print("\n🎴 Your cards (defense values):")
        total_defense = 0
        for i, card in enumerate(hand):
            defense_val = card.get_discard_value()
            total_defense += defense_val
            print(f"   {i}: {card} (defense: {defense_val})")
        
        print(f"\nTotal available defense: {total_defense}")
        
        
        # Defense selection loop
        while True:
            try:
                selection = input(f"\n🛡️ Choose cards to discard (need >= {damage_required}): ").strip()
                
                if not selection:
                    print("❌ You must choose cards to defend!")
                    continue
                
                indices = [int(x.strip()) for x in selection.split(',')]
                
                if not all(0 <= i < len(hand) for i in indices):
                    print("❌ Invalid card indices!")
                    continue
                
                # Preview the defense
                defense_cards = [hand[i] for i in indices]
                defense_value = sum(c.get_discard_value() for c in defense_cards)
                cards_preview = ', '.join(str(c) for c in defense_cards)
                
                print(f"   Discarding: {cards_preview}")
                print(f"   Defense value: {defense_value}")
                
                if defense_value < damage_required:
                    print(f"❌ Not enough! Need {damage_required}, but you only have {defense_value}")
                    continue
                
                # Execute defense
                result = self.game.defend_with_card_indices(indices)
                print(f"\n✅ {result['message']}")
                
                if result['success']:
                    print(f"🔄 Turn passed to Player {self.game.current_player + 1}")
                    return True
                elif result['game_over']:
                    print("💀 GAME OVER!")
                    return True
                
            except ValueError:
                print("❌ Invalid input! Use numbers separated by commas.")
    
    def _show_help(self):
        """Display help information"""
        print("\n📖 HELP")
        print("─" * 20)
        print("🎮 ACTIONS:")
        print("  • play - Play one or more cards")
        print("  • yield - Skip turn (if allowed)")
        print("  • help - Show this help")
        print("  • quit - End game")
        print("\n🎴 CARD COMBOS:")
        print("  • Single card: Any one card")
        print("  • Same value: Multiple cards with same number (total ≤ 10)")
        print("  • Animal Companion: Ace + any other card, or two Aces")
        print("\n⚔️ SUIT POWERS:")
        print("  • Hearts ♥: Heal cards from discard back to tavern")
        print("  • Diamonds ♦: Draw cards for all players")
        print("  • Clubs ♣: Deal double damage")
        print("  • Spades ♠: Reduce enemy's future attack")
        print("\n🃏 SPECIAL:")
        print("  • Jester: Cancels immunity, choose next player")
        print("  • Immunity: Cards matching enemy suit are immune to suit powers")
    
    def _display_game_results(self):
        """Display final game results"""
        print("\n" + "🏰" + "="*68 + "🏰")
        if self.game.victory:
            print("🎉 VICTORY! 🎉")
            print("You have successfully defeated all the corrupted monarchs!")
            print("The kingdom is saved!")
        else:
            print("💀 DEFEAT 💀")
            print("The corruption has overwhelmed you...")
            print("The kingdom falls to darkness.")
        print("🏰" + "="*68 + "🏰")


def run_demo_game():
    """Run a demonstration game with the new interface"""
    interface = NewGameInterface()
    
    while True:
        try:
            num_players = int(input("\nEnter number of players (1-4): "))
            if 1 <= num_players <= 4:
                break
            print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    interface.start_game(num_players)
    print("\nWatch how the yield mechanics work:")
    print("- Try yielding multiple times to see when it's blocked")
    print("- Notice how yield status resets when someone plays cards")
    print("- Observe the detailed defense phase mechanics")
    
    interface.run_game_loop()


if __name__ == "__main__":
    print("\n" + "="*40)
    run_demo_game()
    
    print("\n🎮 Demo complete!")
