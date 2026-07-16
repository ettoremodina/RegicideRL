from enum import Enum
from typing import List, Optional, Dict
import random
import logging

logger = logging.getLogger("regicide")
logger.addHandler(logging.NullHandler())

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Card:
    def __init__(self, value: int, suit: Suit):
        self.value = value
        self.suit = suit
        self._hash = hash((self.value, self.suit))
        self._suit_order = {Suit.HEARTS: 0, Suit.DIAMONDS: 1, Suit.CLUBS: 2, Suit.SPADES: 3}[self.suit]
    
    def __str__(self):
        if self.value == 1:
            return f"A{self.suit.value}"
        elif self.value == 11:
            return f"J{self.suit.value}"
        elif self.value == 12:
            return f"Q{self.suit.value}"
        elif self.value == 13:
            return f"K{self.suit.value}"
        else:
            return f"{self.value}{self.suit.value}"
    
    def __lt__(self, other):
        """Define sorting order for cards: first by value, then by suit"""
        if self.value != other.value:
            return self.value < other.value
        return self._suit_order < other._suit_order
    
    def __eq__(self, other):
        """Define equality for cards"""
        return self.value == other.value and self.suit == other.suit
    
    def __hash__(self):
        """Make cards hashable"""
        return self._hash
    
    def get_attack_value(self):
        if self.value == 1:  # Animal Companion
            return 1
        elif self.value == 11:  # Jack
            return 10
        elif self.value == 12:  # Queen
            return 15
        elif self.value == 13:  # King
            return 20
        else:
            return self.value
    
    def get_discard_value(self):
        if self.value == 1:  # Animal Companion
            return 1
        elif self.value == 11:  # Jack
            return 10
        elif self.value == 12:  # Queen
            return 15
        elif self.value == 13:  # King
            return 20
        else:
            return self.value

class Enemy:
    def __init__(self, card: Card):
        self.card = card
        self.health = self._get_health()
        self.attack = self._get_attack()
        self.damage_taken = 0
        self.spade_protection = 0
    
    def _get_health(self):
        if self.card.value == 11:  # Jack
            return 20
        elif self.card.value == 12:  # Queen
            return 30
        elif self.card.value == 13:  # King
            return 40
    
    def _get_attack(self):
        if self.card.value == 11:  # Jack
            return 10
        elif self.card.value == 12:  # Queen
            return 15
        elif self.card.value == 13:  # King
            return 20
    
    def is_defeated(self):
        return self.damage_taken >= self.health
    
    def get_effective_attack(self):
        return max(0, self.attack - self.spade_protection)
    
    def __str__(self):
        return f"{self.card} (HP: {self.health - self.damage_taken}/{self.health}, ATK: {self.get_effective_attack()})"

class Game:
    def __init__(self, num_players: int, deterministic_hearts: bool = False):
        self.num_players = num_players
        self.deterministic_hearts = deterministic_hearts
        self.players = [[] for _ in range(num_players)]
        self.current_player = 0
        self.tavern_deck = []
        self.discard_pile = []
        self.castle_deck = []
        self.current_enemy: Optional[Enemy] = None
        self.jester_immunity_cancelled = False
        self.game_over = False
        self.victory = False
        # Instrumentation counters for reward shaping / analytics
        self.last_hearts_healed = 0
        self.last_diamonds_drawn = 0
        self.last_spade_protection_added = 0
        # Track yield status for each player - resets when it becomes their turn
        self.players_yielded_this_round = [False] * num_players
        # Track who had the last non-yield turn (for yield eligibility)
        self.last_active_player = None
        # Track spade protection blocked by immunity (for Jester retroactive effect)
        self.blocked_spade_value = 0
        # Solo Jester rules: 2 jester tokens set aside for single-player mode
        self.solo_jesters_remaining = 2 if num_players == 1 else 0
        self.solo_jesters_used = 0
        
        self._setup_game()
    
    def _sort_hand(self, player_index: int):
        """Sort a player's hand in place"""
        self.players[player_index].sort()
    
    def _sort_all_hands(self):
        """Sort all players' hands"""
        for i in range(self.num_players):
            self._sort_hand(i)
    
    def get_player_hand(self, player_index: int) -> List[Card]:
        """Get a player's hand, ensuring it's sorted"""
        if 0 <= player_index < self.num_players:
            self._sort_hand(player_index)
            return self.players[player_index].copy()
        return []
    
    def get_current_player_hand(self) -> List[Card]:
        """Get the current player's hand, ensuring it's sorted"""
        return self.get_player_hand(self.current_player)
    
    def _setup_game(self):
        # Create castle deck (Jacks, Queens, Kings)
        for suit in Suit:
            self.castle_deck.extend([Card(11, suit), Card(12, suit), Card(13, suit)])
        
        # Separate and shuffle each rank
        jacks = [c for c in self.castle_deck if c.value == 11]
        queens = [c for c in self.castle_deck if c.value == 12]
        kings = [c for c in self.castle_deck if c.value == 13]
        
        random.shuffle(jacks)
        random.shuffle(queens)
        random.shuffle(kings)
        
        self.castle_deck = jacks + queens + kings
        logger.info(f"Castle deck prepared with 12 enemies.")
        
        # Create tavern deck
        for suit in Suit:
            self.tavern_deck.append(Card(1, suit))  # Animal Companions
            for value in range(2, 11):
                self.tavern_deck.append(Card(value, suit))
        
        # Add Jesters (value 0)
        jester_counts = {1: 0, 2: 0, 3: 1, 4: 2}
        for _ in range(jester_counts[self.num_players]):
            self.tavern_deck.append(Card(0, Suit.HEARTS))  # Jester
        
        random.shuffle(self.tavern_deck)
        
        # Deal initial hands
        hand_sizes = {1: 8, 2: 7, 3: 6, 4: 5}
        max_hand_size = hand_sizes[self.num_players]
        
        for player in self.players:
            for _ in range(max_hand_size):
                if self.tavern_deck:
                    player.append(self.tavern_deck.pop())
        
        # Sort all hands after dealing
        self._sort_all_hands()
        
        # Reveal first enemy
        if self.castle_deck:
            self.current_enemy = Enemy(self.castle_deck.pop(0))
            logger.info(f"First enemy revealed: {self.current_enemy}")
        # Reset any lingering attack buffer at start
        self.attack_cards_buffer = []
    
    def clone(self) -> 'Game':
        """Create a fast deep copy of the game state for search simulation.
        
        Manually copies all mutable fields instead of using copy.deepcopy,
        which is significantly faster for MCTS/PIMC where thousands of
        clones are created per decision.
        
        Returns:
            A new Game instance with identical but independent state.
        """
        # Bypass __init__ and _setup_game entirely
        new = object.__new__(Game)
        new.num_players = self.num_players
        new.deterministic_hearts = self.deterministic_hearts
        new.current_player = self.current_player
        new.game_over = self.game_over
        new.victory = self.victory
        new.jester_immunity_cancelled = self.jester_immunity_cancelled
        new.last_active_player = self.last_active_player
        new.blocked_spade_value = self.blocked_spade_value
        new.solo_jesters_remaining = self.solo_jesters_remaining
        new.solo_jesters_used = self.solo_jesters_used
        
        # Instrumentation counters
        new.last_hearts_healed = self.last_hearts_healed
        new.last_diamonds_drawn = self.last_diamonds_drawn
        new.last_spade_protection_added = self.last_spade_protection_added
        
        # Deep-copy card lists (Card objects are immutable, so shallow-copy of lists suffices)
        new.players = [hand[:] for hand in self.players]
        new.tavern_deck = self.tavern_deck[:]
        new.discard_pile = self.discard_pile[:]
        new.castle_deck = self.castle_deck[:]
        new.attack_cards_buffer = self.attack_cards_buffer[:]
        new.players_yielded_this_round = self.players_yielded_this_round[:]
        
        # Clone the current enemy (Enemy is mutable — has damage_taken, spade_protection)
        if self.current_enemy is not None:
            new_enemy = object.__new__(Enemy)
            new_enemy.card = self.current_enemy.card  # Card is immutable
            new_enemy.health = self.current_enemy.health
            new_enemy.attack = self.current_enemy.attack
            new_enemy.damage_taken = self.current_enemy.damage_taken
            new_enemy.spade_protection = self.current_enemy.spade_protection
            new.current_enemy = new_enemy
        else:
            new.current_enemy = None
        
        return new

    def get_max_hand_size(self):
        hand_sizes = {1: 8, 2: 7, 3: 6, 4: 5}
        return hand_sizes[self.num_players]
    
    def play_card(self, card_indices: List[int]) -> Dict[str, any]:
        """
        Play cards and return detailed result information about what happened.
        Returns a dictionary with the following structure:
        {
            'success': bool,          # Whether the action was successful
            'phase': str,             # Current phase: 'invalid', 'next_player_choice', 'enemy_defeated', 'defense_needed', 'turn_complete'
            'message': str,           # Description of what happened
            'enemy_damage': int,      # Damage dealt to enemy (if any)
            'cards_played': List[str], # String representations of cards played
            'next_player': int,       # Who should act next (-1 if not applicable)
            'defense_required': int   # Damage that needs to be defended against (0 if none)
        }
        """
        if self.game_over:
            return {
                'success': False,
                'phase': 'invalid',
                'message': 'Game is over',
                'enemy_damage': 0,
                'cards_played': [],
                'next_player': -1,
                'defense_required': 0
            }
        
        # Handle yield action (empty card_indices)
        if not card_indices:
            return self._handle_yield()
        
        player_hand = self.players[self.current_player]
        
        # Validate indices
        if not all(0 <= i < len(player_hand) for i in card_indices):
            return {
                'success': False,
                'phase': 'invalid',
                'message': 'Invalid card indices',
                'enemy_damage': 0,
                'cards_played': [],
                'next_player': -1,
                'defense_required': 0
            }
        
        cards_to_play = [player_hand[i] for i in sorted(card_indices, reverse=True)]
        
        # Validate combo
        if not self._is_valid_combo(cards_to_play):
            return {
                'success': False,
                'phase': 'invalid',
                'message': 'Invalid card combination',
                'enemy_damage': 0,
                'cards_played': [],
                'next_player': -1,
                'defense_required': 0
            }
        
        # Mark this player as having taken an active turn
        self.last_active_player = self.current_player
        self._reset_yield_tracking()
        
        # Remove cards from hand (keep references in cards_to_play)
        for i in sorted(card_indices, reverse=True):
            player_hand.pop(i)
        
        # Sort hand after removing cards
        self._sort_hand(self.current_player)
        
        cards_played_str = [str(card) for card in cards_to_play]
        logger.info(f"Player {self.current_player + 1} plays: {', '.join(cards_played_str)}")

        # Accumulate attack cards; they will be discarded only once the enemy is defeated
        self.attack_cards_buffer.extend(cards_to_play)
        
        # Handle Jester (multiplayer — solo jesters are separate tokens)
        if any(card.value == 0 for card in cards_to_play):
            self.jester_immunity_cancelled = True
            # Retroactive spade protection: per rules, "spades played prior to
            # the Jester will begin reducing the attack value of the enemy"
            if self.blocked_spade_value > 0:
                self.current_enemy.spade_protection += self.blocked_spade_value
                self.blocked_spade_value = 0
            return {
                'success': True,
                'phase': 'next_player_choice',
                'message': 'Jester played! Immunity cancelled. Choose next player.',
                'enemy_damage': 0,
                'cards_played': cards_played_str,
                'next_player': -1,  # Indicates choice needed
                'defense_required': 0
            }
        
        # Reset instrumentation counters for this play
        self.last_hearts_healed = 0
        self.last_diamonds_drawn = 0
        self.last_spade_protection_added = 0
        
        total_attack = sum(card.get_attack_value() for card in cards_to_play)
        
        # Apply suit powers
        total_attack = self._apply_suit_powers(cards_to_play, total_attack)
        
        # Deal damage
        initial_damage = self.current_enemy.damage_taken
        self.current_enemy.damage_taken += total_attack
        actual_damage = self.current_enemy.damage_taken - initial_damage
        logger.info(f"Enemy takes {actual_damage} damage. (Total HP: {self.current_enemy.health - self.current_enemy.damage_taken}/{self.current_enemy.health})")
        
        # Check if enemy defeated
        if self.current_enemy.is_defeated():
            self._reset_yield_tracking()
            self._defeat_enemy()
            if not self.castle_deck:
                self.victory = True
                self.game_over = True
                return {
                    'success': True,
                    'phase': 'victory',
                    'message': 'Victory! All enemies defeated!',
                    'enemy_damage': actual_damage,
                    'cards_played': cards_played_str,
                    'next_player': -1,
                    'defense_required': 0
                }
            return {
                'success': True,
                'phase': 'enemy_defeated',
                'message': 'Enemy defeated! Same player continues.',
                'enemy_damage': actual_damage,
                'cards_played': cards_played_str,
                'next_player': self.current_player,
                'defense_required': 0
            }
        else:
            # Enemy attacks - check if defense is needed
            enemy_attack = self.current_enemy.get_effective_attack()
            logger.info(f"Enemy counter-attacks for {enemy_attack} damage.")
            if enemy_attack == 0:
                self._next_player()
                return {
                    'success': True,
                    'phase': 'turn_complete',
                    'message': 'Enemy attack negated by Spades protection. Turn complete.',
                    'enemy_damage': actual_damage,
                    'cards_played': cards_played_str,
                    'next_player': self.current_player,
                    'defense_required': 0
                }
            else:
                return {
                    'success': True,
                    'phase': 'defense_needed',
                    'message': f'Enemy attacks for {enemy_attack} damage. Defense required.',
                    'enemy_damage': actual_damage,
                    'cards_played': cards_played_str,
                    'next_player': self.current_player,
                    'defense_required': enemy_attack
                }
    
    @staticmethod
    def _is_valid_combo(cards: List[Card]) -> bool:
        """Validate a play.

        Jester behaviour intentionally left unchanged per user request (no extra restrictions).

        Animal Companion (Ace, value 1) rules enforced:
          - Single Ace allowed
          - Exactly two-card combos allowed: (Ace + Ace) or (Ace + one other card)
          - Ace cannot participate in larger combos
        Regular numbered combos (no Aces present):
          - All cards same value
          - Total combined value <= 10
        """
        if not cards:
            return False
        if len(cards) == 1:
            return True

        aces = [c for c in cards if c.value == 1]
        others = [c for c in cards if c.value != 1]

        if aces:
            # Any combo with Ace must be exactly two cards only
            if len(cards) != 2:
                return False
            # Two Aces
            if len(aces) == 2:
                return True
            # One Ace + one other card (rules explicitly forbid Ace + Jester pairing)
            if len(aces) == 1 and len(others) == 1 and others[0].value != 0:
                return True
            return False

        # No Aces: uniform value combo with sum <= 10
        values = {c.value for c in cards}
        if len(values) != 1:
            return False
        return sum(c.value for c in cards) <= 10
    
    def _apply_suit_powers(self, cards: List[Card], total_attack: int):
        """Apply suit powers with corrected ordering and stacking.

        Adjustments:
          - Hearts resolved before Diamonds.
          - Clubs: damage doubled once if any effective club present (not per club).
          - Spades: each effective spade adds protection (stacks).
          - Ace pairing (two-card play containing Ace): suit power for each distinct suit counted once.
          - Immunity: cards matching enemy suit (unless Jester cancelled) skip suit effects.
        """
        if not cards or not self.current_enemy:
            return total_attack

        effective = [c for c in cards if not self._is_immune(c)]
        immune_cards = [c for c in cards if self._is_immune(c)]

        # Track blocked spade protection for potential Jester retroactive effect
        has_immune_spade = any(c.suit == Suit.SPADES for c in immune_cards)
        has_effective_spade = any(c.suit == Suit.SPADES for c in effective)
        if has_immune_spade and not has_effective_spade:
            self.blocked_spade_value += total_attack

        if not effective:
            return total_attack

        suits = {c.suit for c in effective}

        # All suit powers use the base attack value (before clubs doubling)
        power_value = total_attack

        # Hearts first (heal from discard)
        if Suit.HEARTS in suits:
            self._hearts_power(power_value)
        # Diamonds next (draw cards)
        if Suit.DIAMONDS in suits:
            self._diamonds_power(power_value)
        # Spades protection (cumulative, uses base attack value)
        if Suit.SPADES in suits:
            self.current_enemy.spade_protection += power_value
            self.last_spade_protection_added += power_value
        # Clubs: doubles DAMAGE only (does not affect other suit power values)
        if Suit.CLUBS in suits:
            total_attack *= 2

        return total_attack

    def _is_immune(self, card: Card) -> bool:
        if self.jester_immunity_cancelled:
            return False
        return card.suit == self.current_enemy.card.suit
    
    def _hearts_power(self, value: int):
        if not self.discard_pile:
            return
            
        if self.deterministic_hearts:
            # Sort discard pile deterministically (e.g., by value then suit) 
            # to make perfect information solving possible without random shuffles.
            self.discard_pile.sort(key=lambda c: (c.value, c.suit.value))
        else:
            random.shuffle(self.discard_pile)
            
        cards_to_heal = min(value, len(self.discard_pile))
        healed_cards = self.discard_pile[:cards_to_heal]
        self.discard_pile = self.discard_pile[cards_to_heal:]
        # Place healed cards UNDER the tavern deck (bottom). Top is end of list.
        self.tavern_deck = healed_cards + self.tavern_deck
        self.last_hearts_healed += cards_to_heal
    
    def _diamonds_power(self, value: int):
        cards_drawn = 0
        player_idx = self.current_player
        
        while cards_drawn < value and self.tavern_deck:
            if len(self.players[player_idx]) < self.get_max_hand_size():
                self.players[player_idx].append(self.tavern_deck.pop())
                cards_drawn += 1
            
            player_idx = (player_idx + 1) % self.num_players
            if player_idx == self.current_player and cards_drawn < value:
                # Made full circle, check if anyone can still draw
                if all(len(hand) >= self.get_max_hand_size() for hand in self.players):
                    break
        
        # Sort all hands after drawing cards
        self._sort_all_hands()
        self.last_diamonds_drawn += cards_drawn
    
    def _defeat_enemy(self):
        # When an enemy is defeated, move all attack-played cards to the discard pile
        if self.attack_cards_buffer:
            self.discard_pile.extend(self.attack_cards_buffer)
            self.attack_cards_buffer = []

        # Exact kill -> enemy card goes to TOP of tavern deck (end of list)
        if self.current_enemy.damage_taken == self.current_enemy.health:
            self.tavern_deck.append(self.current_enemy.card)
            logger.info(f"Enemy exactly defeated! {self.current_enemy.card} joins the Tavern.")
        else:
            self.discard_pile.append(self.current_enemy.card)
            logger.info(f"Enemy defeated! {self.current_enemy.card} moves to discard.")

        # Reset for next enemy
        if self.castle_deck:
            self.current_enemy = Enemy(self.castle_deck.pop(0))
            logger.info(f"New enemy revealed: {self.current_enemy}")
            self.jester_immunity_cancelled = False
            self.attack_cards_buffer = []
            self.blocked_spade_value = 0
        else:
            self.current_enemy = None
    
    def defend_against_attack(self, cards_to_discard: List[Card]) -> Dict[str, any]:
        """
        Defend against enemy attack using specific cards.
        Returns a dictionary with result information:
        {
            'success': bool,          # Whether defense was successful
            'message': str,           # Description of what happened
            'defense_value': int,     # Total defense value of discarded cards
            'damage_blocked': int,    # Amount of damage blocked
            'cards_discarded': List[str], # String representations of discarded cards
            'next_player': int,       # Who should act next (-1 if game over)
            'game_over': bool         # Whether the game ended due to failed defense
        }
        """
        enemy_damage = self.current_enemy.get_effective_attack()
        
        if enemy_damage == 0:
            self._next_player()
            return {
                'success': True,
                'message': 'No damage to defend against',
                'defense_value': 0,
                'damage_blocked': 0,
                'cards_discarded': [],
                'next_player': self.current_player,
                'game_over': False
            }
        
        if not cards_to_discard:
            # No cards provided for defense - game over
            self.game_over = True
            return {
                'success': False,
                'message': 'No defense provided - game over',
                'defense_value': 0,
                'damage_blocked': 0,
                'cards_discarded': [],
                'next_player': -1,
                'game_over': True
            }
        
        # Calculate total defense value
        defense_value = sum(card.get_discard_value() for card in cards_to_discard)
        cards_discarded_str = [str(card) for card in cards_to_discard]
        logger.info(f"Player {self.current_player + 1} defends with {', '.join(cards_discarded_str)} (value: {defense_value})")
        
        # Check if defense is sufficient
        if defense_value < enemy_damage:
            self.game_over = True
            return {
                'success': False,
                'message': f'Insufficient defense: {defense_value} < {enemy_damage}',
                'defense_value': defense_value,
                'damage_blocked': 0,
                'cards_discarded': cards_discarded_str,
                'next_player': -1,
                'game_over': True
            }
        
        # Remove cards from hand and add to discard pile
        player_hand = self.players[self.current_player]
        for card in cards_to_discard:
            if card in player_hand:
                player_hand.remove(card)
                self.discard_pile.append(card)
        
        # Sort hand after removing cards
        self._sort_hand(self.current_player)
        
        self._next_player()
        return {
            'success': True,
            'message': f'Defense successful: {defense_value} >= {enemy_damage}',
            'defense_value': defense_value,
            'damage_blocked': enemy_damage,
            'cards_discarded': cards_discarded_str,
            'next_player': self.current_player,
            'game_over': False
        }
    
    def defend_with_card_indices(self, card_indices: List[int]) -> Dict[str, any]:
        """
        Defend against enemy attack using card indices from current player's hand.
        This is a convenience method that converts indices to cards and calls defend_against_attack.
        """
        if not card_indices:
            return self.defend_against_attack([])
        
        player_hand = self.players[self.current_player]
        
        # Validate indices
        if not all(0 <= i < len(player_hand) for i in card_indices):
            return {
                'success': False,
                'message': 'Invalid card indices',
                'defense_value': 0,
                'damage_blocked': 0,
                'cards_discarded': [],
                'next_player': -1,
                'game_over': False
            }
        
        cards_to_discard = [player_hand[i] for i in card_indices]
        return self.defend_against_attack(cards_to_discard)
    
    def can_defend(self) -> bool:
        """Check if current player has enough cards to defend against enemy attack"""
        damage = self.current_enemy.get_effective_attack()
        if damage == 0:
            return True
        player_hand = self.players[self.current_player]
        total_defense = sum(card.get_discard_value() for card in player_hand)
        return total_defense >= damage
    
    def _next_player(self):
        """Move to next player and reset their yield status for this round"""
        self.current_player = (self.current_player + 1) % self.num_players
        # Reset yield status for the new current player
        self.players_yielded_this_round[self.current_player] = False
        # Per rules: "players also lose if any player is unable to play a card or yield"
        self._check_player_can_act()
    
    def choose_next_player(self, chosen_player: int) -> bool:
        """Set the next player (used when Jester is played). Returns True if valid."""
        if 0 <= chosen_player < self.num_players:
            self.current_player = chosen_player
            # Reset yield status for the new current player
            self.players_yielded_this_round[self.current_player] = False
            # Per rules: "players also lose if any player is unable to play a card or yield"
            self._check_player_can_act()
            return True
        return False
    
    def yield_turn(self) -> Dict[str, any]:
        """
        Yield the current turn. Returns result information:
        {
            'success': bool,          # Whether yield was successful
            'message': str,           # Description of what happened
            'defense_required': int,  # Damage that needs to be defended against (0 if none)
            'next_player': int,       # Who should act next
            'can_yield': bool         # Whether yielding was allowed
        }
        """
        if self.game_over:
            return {
                'success': False,
                'message': 'Game is over',
                'defense_required': 0,
                'next_player': -1,
                'can_yield': False
            }
        
        if not self.can_yield():
            return {
                'success': False,
                'message': 'Cannot yield: all other players have already yielded since the last active turn',
                'defense_required': 0,
                'next_player': self.current_player,
                'can_yield': False
            }
        
        # Mark current player as yielded
        self.players_yielded_this_round[self.current_player] = True
        logger.info(f"Player {self.current_player + 1} yields their turn.")
        
        # Enemy attacks
        enemy_damage = self.current_enemy.get_effective_attack()
        logger.info(f"Enemy counter-attacks for {enemy_damage} damage.")
        if enemy_damage == 0:
            self._next_player()
            return {
                'success': True,
                'message': 'Yield successful. No enemy damage due to Spades protection.',
                'defense_required': 0,
                'next_player': self.current_player,
                'can_yield': True
            }
        else:
            return {
                'success': True,
                'message': f'Yield successful. Enemy attacks for {enemy_damage} damage.',
                'defense_required': enemy_damage,
                'next_player': self.current_player,
                'can_yield': True
            }
    
    def get_raw_state(self):
        """
        Return the game state without string allocations. 
        Used by the environment and search agents for performance.
        """
        return {
            'current_player': self.current_player,
            'current_enemy': self.current_enemy,
            'player_hands': self.players,
            'tavern_cards': len(self.tavern_deck),
            'discard_cards': len(self.discard_pile),
            'enemies_remaining': len(self.castle_deck),
            'game_over': self.game_over,
            'victory': self.victory,
            'victory_tier': self.get_victory_tier(),
            'players_yielded_this_round': self.players_yielded_this_round.copy(),
            'last_active_player': self.last_active_player,
            'can_yield': self.can_yield(),
            'can_defend': self.can_defend() if self.current_enemy else True,
            'enemy_attack_damage': self.current_enemy.get_effective_attack() if self.current_enemy else 0,
            
            # Keys required for the survivability filter
            'enemy_attack': self.current_enemy.get_effective_attack() if self.current_enemy else 0,
            'enemy_health': self.current_enemy.health if self.current_enemy else 0,
            'enemy_damage_taken': self.current_enemy.damage_taken if self.current_enemy else 0,
            'enemy_suit': self.current_enemy.card.suit.value if self.current_enemy else None,
            'jester_immunity_cancelled': self.jester_immunity_cancelled,
            
            'has_legal_action': self.has_legal_action(),
            'solo_jesters_remaining': self.solo_jesters_remaining,
            'solo_jesters_used': self.solo_jesters_used,
            'can_use_solo_jester': self.can_use_solo_jester(),
        }

    def get_game_state(self):
        # Hands are kept sorted after every mutation (play_card, defend, draw),
        # so no need to re-sort here. This avoids O(n log n) overhead per call
        # during MCTS simulation where this is called thousands of times.
        
        return {
            'current_player': self.current_player,
            'current_enemy': str(self.current_enemy) if self.current_enemy else None,
            'player_hands': [[str(card) for card in hand] for hand in self.players],
            'tavern_cards': len(self.tavern_deck),
            'discard_cards': len(self.discard_pile),
            'enemies_remaining': len(self.castle_deck),
            'game_over': self.game_over,
            'victory': self.victory,
            'victory_tier': self.get_victory_tier(),
            'players_yielded_this_round': self.players_yielded_this_round.copy(),
            'last_active_player': self.last_active_player,
            'can_yield': self.can_yield(),
            'can_defend': self.can_defend() if self.current_enemy else True,
            'enemy_attack_damage': self.current_enemy.get_effective_attack() if self.current_enemy else 0,
            'has_legal_action': self.has_legal_action(),
            'solo_jesters_remaining': self.solo_jesters_remaining,
            'solo_jesters_used': self.solo_jesters_used,
            'can_use_solo_jester': self.can_use_solo_jester(),
        }
    
    def _handle_yield(self) -> Dict[str, any]:
        """Handle yield action and return result information"""
        return self.yield_turn()
    
    def can_yield(self) -> bool:
        """
        Check if the current player can yield.
        A player can yield if not all other players have yielded since the last active turn.
        Special case: At start of game, only allow yielding if not everyone has yielded yet.
        """
        # Get other players (excluding current player)
        other_players = [i for i in range(self.num_players) if i != self.current_player]
        
        # Check if all other players have yielded this round
        all_others_yielded = all(self.players_yielded_this_round[i] for i in other_players)
        
        # If all other players have yielded, current player cannot yield
        return not all_others_yielded
    
    def _reset_yield_tracking(self):
        """Reset yield tracking when a player takes an active turn"""
        self.players_yielded_this_round = [False] * self.num_players

    def _check_player_can_act(self):
        """Check if current player has any legal action. If not, game over.
        
        Per rules: 'The players also lose if any player is unable to play
        a card or yield on their turn.'
        """
        if self.game_over:
            return
        hand = self.players[self.current_player]
        # A player can always play any single card from hand.
        # The only stuck state is: empty hand AND cannot yield.
        if not hand and not self.can_yield():
            self.game_over = True

    def has_legal_action(self) -> bool:
        """Check if the current player has any legal action available."""
        if self.game_over:
            return False
        hand = self.players[self.current_player]
        return bool(hand) or self.can_yield()

    # --- Solo Jester Rules ---

    def can_use_solo_jester(self) -> bool:
        """Check if the solo player can use a jester token.
        
        Solo jesters are separate from the deck jesters used in multiplayer.
        They can be used at the start of Step 1 (before playing) or
        at the start of Step 4 (before defending).
        """
        return self.num_players == 1 and self.solo_jesters_remaining > 0

    def use_solo_jester(self, timing: str) -> Dict[str, any]:
        """Use a solo jester token: discard entire hand and refill to 8 cards.
        
        Per rules: 'Discard your hand and refill to 8 cards - this does not
        count as drawing for the purpose of enemy diamond immunity.'
        Solo jesters do NOT cancel enemy immunity.
        
        Args:
            timing: 'step1' (before playing a card) or 'step4' (before defending)
        
        Returns:
            Dict with result information
        """
        if not self.can_use_solo_jester():
            return {
                'success': False,
                'message': 'No solo jesters remaining' if self.num_players == 1 else 'Solo jesters only available in single-player',
                'jesters_remaining': self.solo_jesters_remaining
            }

        if timing not in ('step1', 'step4'):
            return {
                'success': False,
                'message': f'Invalid timing: {timing}. Must be "step1" or "step4".',
                'jesters_remaining': self.solo_jesters_remaining
            }

        # Discard entire hand to discard pile
        hand = self.players[0]
        self.discard_pile.extend(hand)
        hand.clear()

        # Refill to max hand size (8 for solo) — NOT treated as diamond drawing
        max_hand = self.get_max_hand_size()
        while len(hand) < max_hand and self.tavern_deck:
            hand.append(self.tavern_deck.pop())

        self._sort_hand(0)

        self.solo_jesters_remaining -= 1
        self.solo_jesters_used += 1

        return {
            'success': True,
            'message': f'Solo jester used! Hand refilled to {len(hand)} cards. '
                       f'{self.solo_jesters_remaining} jester(s) remaining.',
            'jesters_remaining': self.solo_jesters_remaining,
            'cards_drawn': len(hand)
        }

    def get_victory_tier(self) -> Optional[str]:
        """Get the victory tier (solo play only).
        
        Returns:
            'gold' (0 jesters used), 'silver' (1 used), 'bronze' (2 used),
            'victory' (multiplayer), or None (game not won).
        """
        if not self.victory:
            return None
        if self.num_players != 1:
            return 'victory'
        if self.solo_jesters_used == 0:
            return 'gold'
        elif self.solo_jesters_used == 1:
            return 'silver'
        else:
            return 'bronze'
