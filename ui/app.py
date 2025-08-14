import tkinter as tk
from tkinter import ttk, messagebox
from typing import List

from game.regicide import Game, Card, Suit
from .theme import DARK_THEME
from .widgets import CardButton, LogBox, EnemyView
from .sound import play_thud, play_draw, play_shimmer, play_clang, play_victory, play_defeat


class RegicideApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Regicide")
        self.geometry("1100x760")
        self.configure(bg=DARK_THEME["bg"]) 
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=DARK_THEME["bg"])
        self.style.configure("Panel.TFrame", background=DARK_THEME["panel"]) 
        self.style.configure("TLabel", background=DARK_THEME["bg"], foreground=DARK_THEME["text"], font=("Merriweather", 11))
        self.style.configure("H1.TLabel", font=("Uncial Antiqua", 20, "bold"))
        self.style.configure("H2.TLabel", font=("Uncial Antiqua", 16, "bold"))
        self.style.configure("Muted.TLabel", foreground=DARK_THEME["muted"]) 

        self.game = None
        self.card_buttons = []
        self.selected_indices = []
        # Track last enemy HP for animations
        self._last_hp = None

        self._build_start_menu()

    # ----- UI Build -----
    def _build_start_menu(self):
        self.menu_frame = ttk.Frame(self)
        self.menu_frame.pack(fill="both", expand=True)
        title = ttk.Label(self.menu_frame, text="Regicide", style="H1.TLabel")
        subtitle = ttk.Label(self.menu_frame, text="A digital tabletop of corrupted royalty.", style="Muted.TLabel")
        title.pack(pady=(80, 6))
        subtitle.pack(pady=(0, 20))

        ttk.Label(self.menu_frame, text="Choose number of players:").pack(pady=8)
        self.players_var = tk.IntVar(value=1)
        controls = ttk.Frame(self.menu_frame)
        controls.pack()
        for n in [1, 2, 3, 4]:
            rb = ttk.Radiobutton(controls, text=str(n), variable=self.players_var, value=n)
            rb.pack(side=tk.LEFT, padx=10)

        ttk.Button(self.menu_frame, text="Start", command=self._start_game).pack(pady=18)
        ttk.Button(self.menu_frame, text="Rules / Help", command=self._show_help).pack()

    def _build_board(self):
        self.menu_frame.destroy()
        self.board = ttk.Frame(self)
        self.board.pack(fill="both", expand=True)

        # Layout: left main area (enemy + hand), right side panel (log)
        body = ttk.Frame(self.board)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill="both", expand=True)
        right = ttk.Frame(body, style="Panel.TFrame")
        right.pack(side=tk.LEFT, fill="y", padx=(10, 0))

        # Enemy + decks on top left
        top_left = ttk.Frame(left)
        top_left.pack(fill="x")

        self.enemy_panel = ttk.Frame(top_left, style="Panel.TFrame")
        self.enemy_panel.pack(side=tk.LEFT, padx=(0, 10), ipadx=8, ipady=8)
        ttk.Label(self.enemy_panel, text="Enemy", style="H2.TLabel").pack(anchor="w")
        self.enemy_view = EnemyView(self.enemy_panel)
        self.enemy_view.pack()

        decks = ttk.Frame(top_left, style="Panel.TFrame")
        decks.pack(side=tk.LEFT, fill="x", expand=True, ipadx=8, ipady=8)
        ttk.Label(decks, text="Decks & Piles", style="H2.TLabel").pack(anchor="w")
        self.castle_label = ttk.Label(decks, text="Castle: 0 enemies remain")
        self.castle_label.pack(anchor="w", pady=(8, 2))
        self.tavern_label = ttk.Label(decks, text="Tavern: 0 cards")
        self.tavern_label.pack(anchor="w")
        self.discard_label = ttk.Label(decks, text="Discard: 0 cards (click to view)")
        self.discard_label.pack(anchor="w", pady=(2, 0))
        self.discard_label.bind("<Button-1>", lambda _e: self._show_discard())

        # Right side: smaller action log
        ttk.Label(right, text="Action Log", style="H2.TLabel").pack(anchor="w", padx=8, pady=6)
        self.log = LogBox(right)
        self.log.pack(fill="y", expand=False, padx=8, pady=(0, 8))

        # Bottom: Player area
        bottom = ttk.Frame(left)
        bottom.pack(fill="x", padx=16, pady=8)
        info = ttk.Frame(bottom)
        info.pack(fill="x")
        self.player_title = ttk.Label(info, text="Player 1  |  Hand: 0/0")
        self.player_title.pack(side=tk.LEFT)

        btns = ttk.Frame(bottom)
        btns.pack(fill="x", pady=8)
        self.play_btn = ttk.Button(btns, text="Play Selected", command=self._on_play)
        self.yield_btn = ttk.Button(btns, text="Yield", command=self._on_yield)
        self.help_btn = ttk.Button(btns, text="Help", command=self._show_help)
        for b in (self.play_btn, self.yield_btn, self.help_btn):
            b.pack(side=tk.LEFT, padx=6)

        self.hand_frame = ttk.Frame(bottom, style="Panel.TFrame")
        self.hand_frame.pack(fill="x", pady=(6, 10))

    # ----- Game wiring -----
    def _start_game(self):
        n = self.players_var.get()
        self.game = Game(n)
        self._build_board()
        self._refresh_all(first=True)
        self.log.log(f"Started game with {n} player(s).")

    def _refresh_all(self, first: bool = False):
        if not self.game:
            return
        state = self.game.get_game_state()

        # Enemy area (large view)
        if self.game.current_enemy:
            enemy = self.game.current_enemy
            hp_left = enemy.health - enemy.damage_taken
            # Animate in the enemy view
            if first or self._last_hp is None:
                self.enemy_view.set_enemy(
                    value_label=self._enemy_value_label(enemy.card.value),
                    suit_char=enemy.card.suit.value,
                    hp_current=hp_left,
                    hp_max=enemy.health,
                    atk=enemy.get_effective_attack(),
                    shield=enemy.spade_protection,
                    immunity=enemy.card.suit.value
                )
            else:
                # Update static fields and animate hp
                self.enemy_view.set_enemy(
                    value_label=self._enemy_value_label(enemy.card.value),
                    suit_char=enemy.card.suit.value,
                    hp_current=hp_left,
                    hp_max=enemy.health,
                    atk=enemy.get_effective_attack(),
                    shield=enemy.spade_protection,
                    immunity=enemy.card.suit.value
                )
                self.enemy_view.animate_health(self._last_hp, hp_left)
            self._last_hp = hp_left
        else:
            # Clear enemy view
            self.enemy_view.set_enemy(value_label="-", suit_char="-", hp_current=0, hp_max=1, atk=0, shield=0, immunity="-")
            self._last_hp = 0

        # Decks/piles
        self.castle_label.config(text=f"Castle: {state['enemies_remaining']} enemies remain")
        self.tavern_label.config(text=f"Tavern: {state['tavern_cards']} cards")
        self.discard_label.config(text=f"Discard: {state['discard_cards']} cards (click to view)")

        # Player info
        current = state["current_player"]
        max_hand = self.game.get_max_hand_size()
        self.player_title.config(text=f"Player {current+1}  |  Hand: {len(self.game.players[current])}/{max_hand}")

        # Yield availability
        self.yield_btn.config(state=(tk.NORMAL if state["can_yield"] else tk.DISABLED))

        # Hand
        for w in self.card_buttons:
            w.destroy()
        self.card_buttons = []
        self.selected_indices = []

        hand = self.game.get_current_player_hand()

        # Simple legal highlight: enable/disable by checking if any card solo is legal, and pairing same values
        counts = {}
        for c in hand:
            counts[c.value] = counts.get(c.value, 0) + 1

        def make_click(idx: int):
            def _cb():
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.append(idx)
                # Visual feedback
                self.card_buttons[idx].btn.config(relief=(tk.SUNKEN if idx in self.selected_indices else tk.RAISED))
                self._update_play_preview()
            return _cb

        for i, c in enumerate(hand):
            tooltip = self._card_tooltip([c])
            btn = CardButton(self.hand_frame, str(c), on_click=make_click(i), tooltip=tooltip)
            btn.pack(side=tk.LEFT, padx=4, pady=6)
            self.card_buttons.append(btn)

        self._update_play_preview()

    # No longer used (handled by EnemyView)
    def _animate_health(self, start: int, end: int, maximum: int):
        pass

    def _card_tooltip(self, cards: List[Card]) -> str:
        # Predict simple outcome - mirrors some rules in Game
        total_attack = sum(c.get_attack_value() for c in cards)
        effects = []
        for c in cards:
            if self.game and self.game.current_enemy and not self.game._is_immune(c):
                if c.suit == Suit.HEARTS:
                    effects.append(f"Heal {total_attack}")
                elif c.suit == Suit.DIAMONDS:
                    effects.append(f"Draw {total_attack}")
                elif c.suit == Suit.CLUBS:
                    effects.append(f"Double +{total_attack}")
                elif c.suit == Suit.SPADES:
                    effects.append(f"Shield +{total_attack}")
        base = f"Deal {total_attack}"
        if effects:
            base += " | " + ", ".join(effects)
        return base

    def _update_play_preview(self):
        # Enable play button only if selection forms a valid combo per Game._is_valid_combo
        if not self.game:
            self.play_btn.config(state=tk.DISABLED)
            return
        hand = self.game.get_current_player_hand()
        cards = [hand[i] for i in sorted(self.selected_indices)]
        ok = False
        try:
            # Use private method cautiously for preview; fallback conservative
            ok = self.game._is_valid_combo(cards) if cards else False
        except Exception:
            ok = False
        self.play_btn.config(state=(tk.NORMAL if ok else tk.DISABLED))

        # Update tooltips of selected as combined
        if cards:
            tip = self._card_tooltip(cards)
            for idx in self.selected_indices:
                self.card_buttons[idx].set_tooltip(tip)

    def _enemy_value_label(self, value: int) -> str:
        if value == 11:
            return "J"
        if value == 12:
            return "Q"
        if value == 13:
            return "K"
        if value == 1:
            return "A"
        return str(value)

    # ----- Actions -----
    def _on_play(self):
        if not self.game:
            return
        indices = sorted(self.selected_indices)
        if not indices:
            return
        res = self.game.play_card(indices)
        self._log_result_on_play(res)
        self._handle_phase_after_play(res)

    def _on_yield(self):
        if not self.game:
            return
        res = self.game.yield_turn()
        self.log.log(res["message"]) 
        if res.get("defense_required", 0) > 0:
            self._prompt_defense(res["defense_required"]) 
        else:
            self._refresh_all()

    def _log_result_on_play(self, res: dict):
        if not res:
            return
        if res.get("cards_played"):
            self.log.log(f"Played: {', '.join(res['cards_played'])}")
        if res.get("enemy_damage", 0) > 0:
            self.log.log(f"Dealt {res['enemy_damage']} damage.")
            self._damage_popup(res['enemy_damage'])
            play_thud()
            # Bonus suit sounds based on played cards
            cps = res.get("cards_played", [])
            if any("♦" in c for c in cps):
                play_draw()
            if any("♥" in c for c in cps):
                play_shimmer()
            if any("♠" in c for c in cps):
                play_clang()
        if msg := res.get("message"):
            self.log.log(msg)

    def _handle_phase_after_play(self, res: dict):
        phase = res.get("phase")
        if phase == "next_player_choice":
            self._prompt_jester()
        elif phase == "defense_needed":
            self._prompt_defense(res.get("defense_required", 0))
        elif phase in ("enemy_defeated", "turn_complete", "victory"):
            self._refresh_all()
            if phase == "victory":
                messagebox.showinfo("Victory", "All enemies defeated!")
                play_victory()
        else:
            self._refresh_all()

    def _prompt_jester(self):
        if not self.game:
            return
        win = tk.Toplevel(self)
        win.title("Jester - choose next player")
        ttk.Label(win, text="Choose the next player:").pack(padx=12, pady=8)
        row = ttk.Frame(win)
        row.pack(padx=12, pady=6)
        for i in range(self.game.num_players):
            def make(i=i):
                return lambda: self._choose_player_and_close(win, i)
            ttk.Button(row, text=f"Player {i+1}", command=make()).pack(side=tk.LEFT, padx=6)

    def _choose_player_and_close(self, dialog, idx: int):
        if self.game and self.game.choose_next_player(idx):
            self.log.log(f"Jester chose Player {idx+1}.")
            dialog.destroy()
            self._refresh_all()

    def _prompt_defense(self, required: int):
        if not self.game:
            return
        win = tk.Toplevel(self)
        win.title(f"Suffer {required} Damage")
        ttk.Label(win, text=f"Suffer {required} damage. Select cards to discard.").pack(padx=10, pady=8)
        selected: List[int] = []

        frame = ttk.Frame(win)
        frame.pack(padx=10, pady=8)
        hand = self.game.get_current_player_hand()
        buttons: List[CardButton] = []

        def update_total():
            cards = [hand[i] for i in selected]
            total = sum(c.get_discard_value() for c in cards)
            total_label.config(text=f"Selected: {total}/{required}")
            confirm.config(state=(tk.NORMAL if total >= required else tk.DISABLED))
            # If no possible defense (sum of all cards < required), auto-defeat
            all_total = sum(c.get_discard_value() for c in hand)
            if all_total < required:
                # immediate defeat as per rules
                win.after(100, lambda: self._resolve_auto_defeat(win))

        for i, c in enumerate(hand):
            def make(i=i):
                return lambda: (selected.remove(i) if i in selected else selected.append(i),
                                btn_set(i))
            btn = CardButton(frame, str(c), on_click=make())
            btn.pack(side=tk.LEFT, padx=3, pady=3)
            buttons.append(btn)

        def btn_set(i: int):
            buttons[i].btn.config(relief=(tk.SUNKEN if i in selected else tk.RAISED))
            update_total()

        total_label = ttk.Label(win, text=f"Selected: 0/{required}")
        total_label.pack()

        actions = ttk.Frame(win)
        actions.pack(pady=6)
        def on_confirm():
            res = self.game.defend_with_card_indices(sorted(selected))
            self.log.log(res.get("message", "Defense resolved."))
            win.destroy()
            self._refresh_all()
            if res.get("game_over"):
                play_defeat()
                messagebox.showerror("Defeat", "The party has fallen...")
        confirm = ttk.Button(actions, text="Discard & Resolve", command=on_confirm, state=tk.DISABLED)
        confirm.pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=4)

        update_total()

    def _resolve_auto_defeat(self, dialog):
        # Set game over if cannot defend
        self.game.game_over = True
        self._refresh_all()
        dialog.destroy()
        play_defeat()
        messagebox.showerror("Defeat", "No possible defense. The party has fallen...")

    def _show_discard(self):
        if not self.game:
            return
        win = tk.Toplevel(self)
        win.title("Discard Pile")
        if not self.game.discard_pile:
            ttk.Label(win, text="Discard is empty.").pack(padx=12, pady=12)
            return
        text = tk.Text(win, height=12, width=40, bg="#111", fg=DARK_THEME["text"]) 
        text.pack(padx=12, pady=12)
        for c in self.game.discard_pile:
            text.insert("end", str(c) + "\n")
        text.config(state="disabled")

    def _show_help(self):
        txt = (
            "Suits: Hearts heal from discard. Diamonds draw.\n"
            "Clubs double damage. Spades reduce enemy attack.\n"
            "Play: select cards then Play Selected. Yield only allowed if not all others yielded.\n"
            "Combos: Same value totaling ≤10, or Ace combos, or single card."
        )
        messagebox.showinfo("Rules / Help", txt)

    def _damage_popup(self, amount: int):
        # Delegate to enemy view for cleaner animation
        try:
            self.enemy_view.show_damage(amount)
        except Exception:
            pass


if __name__ == "__main__":
    app = RegicideApp()
    app.mainloop()
