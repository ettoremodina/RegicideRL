import pygame
from typing import Optional

from game.regicide import Game
from game.action_handler import ActionHandler
from ml_logger import GameRecorder, RunContext, get_logger, start_run
from ml_logger.serialization import serialize_game
from .theme import DARK_THEME
from .ui_elements import Button, HealthBar, draw_card
from .sound import play_thud, play_draw, play_shimmer, play_clang, play_victory, play_defeat

# Font utilities
pygame.font.init()
logger = get_logger(__name__)

def get_font(name: str, size: int, bold=False):
    # Fallbacks if specific fonts aren't available
    try:
        if name == "Uncial Antiqua":
            # Using standard pygame fallback
            f = pygame.font.SysFont("timesnewroman", size, bold=bold)
        elif name == "Merriweather":
            f = pygame.font.SysFont("timesnewroman", size, bold=bold)
        else:
            f = pygame.font.SysFont("arial", size, bold=bold)
        return f
    except:
        return pygame.font.Font(None, size)

class RegicideApp:
    def __init__(self, run_context: RunContext | None = None):
        self.run_context = run_context or start_run("ui-session")
        self.recorder = GameRecorder(self.run_context)
        pygame.init()
        infoObject = pygame.display.Info()
        self.width, self.height = infoObject.current_w, infoObject.current_h
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Regicide")
        self.clock = pygame.time.Clock()

        # Fonts
        self.title_font = get_font("Uncial Antiqua", 64, bold=True)
        self.h1_font = get_font("Uncial Antiqua", 32, bold=True)
        self.h2_font = get_font("Uncial Antiqua", 24, bold=True)
        self.big_font = get_font("Uncial Antiqua", 54, bold=True)
        self.body_font = get_font("Merriweather", 18)
        self.bold_font = get_font("Merriweather", 18, bold=True)
        self.card_font = get_font("Merriweather", 24, bold=True)

        self.state = "MENU"
        self.game: Optional[Game] = None
        self.num_players = 1

        self.selected_indices = []
        
        # Popups
        self.defense_required = 0
        self.jester_active = False
        
        self.action_scroll_y = 0
        self.action_handler = None

        self._build_menu()

    def _build_menu(self):
        self.menu_buttons = []
        # Player count buttons
        btn_w, btn_h = 160, 50
        start_y = self.height // 2
        spacing = 180
        start_x = (self.width - (4 * btn_w + 3 * spacing)) // 2
        if start_x < 0: start_x = 50

        for i in range(1, 5):
            def make_on_click(n=i):
                return lambda: self.set_players_and_start(n)
            
            b = Button(start_x + (i-1)*spacing, start_y, btn_w, btn_h, f"{i} Player{'s' if i>1 else ''}", self.bold_font, on_click=make_on_click())
            self.menu_buttons.append(b)

    def set_players_and_start(self, n: int):
        self.num_players = n
        self.game = Game(n)
        self.state = "GAME"
        self.selected_indices = []
        self.defense_required = 0
        self.jester_active = False
        self.action_scroll_y = 0
        self.action_handler = ActionHandler(max_hand_size=self.game.get_max_hand_size())
        self._build_game_ui()
        self.recorder.begin_game(
            self.game,
            metadata={"source": "pygame", "num_players": n},
        )
        self.action_log = ["Game started!"]
        logger.info("Started %d-player UI game", n)

    def _build_game_ui(self):
        # Action Buttons
        self.play_btn = Button(50, self.height - 100, 180, 50, "Play Selected", self.bold_font, self._on_play)
        self.yield_btn = Button(250, self.height - 100, 120, 50, "Yield", self.bold_font, self._on_yield)
        self.jester_btn = Button(390, self.height - 100, 180, 50, "Solo Jester", self.bold_font, self._on_jester)
        self.defend_btn = Button(self.width//2 - 90, self.height - 100, 180, 50, "Defend", self.bold_font, self._on_defend)
        
        self.enemy_hp_bar = HealthBar(self.width // 2 - 150, 280, 300, 30, self.bold_font)

    def log(self, text: str):
        self.action_log.append(text)
        if len(self.action_log) > 8:
            self.action_log.pop(0)
        logger.info("%s", text)

    # --- Actions ---
    def _on_play(self):
        if not self.selected_indices or self.state != "GAME": return
        indices = sorted(self.selected_indices)
        cards = [str(self.game.get_current_player_hand()[index]) for index in indices]
        res = self._record_action(
            {"kind": "play", "phase": "attack", "card_indices": indices, "cards": cards},
            lambda: self.game.play_card(indices),
        )
        self.selected_indices = []
        self._handle_result(res)

    def _on_yield(self):
        if self.state != "GAME": return
        res = self._record_action(
            {"kind": "yield", "phase": "attack"},
            self.game.yield_turn,
        )
        self._handle_result(res)

    def _on_jester(self):
        if self.state not in ("GAME", "DEFENSE"): return
        timing = "step1" if self.state == "GAME" else "step4"
        res = self._record_action(
            {"kind": "solo_jester", "phase": self.state.lower(), "timing": timing},
            lambda: self.game.use_solo_jester(timing),
        )
        if res.get("success"):
            self.selected_indices = []
            self.log("Used Solo Jester! Refilled hand.")
            if self.state == "DEFENSE" and not self.game.can_defend():
                self.game.game_over = True
                self.state = "GAME_OVER"
                self.game_over_msg = "Defeat... No possible defense."
                play_defeat()
                self._finish_recording("no_possible_defense")

    def _on_defend(self):
        if self.state != "DEFENSE": return
        indices = sorted(self.selected_indices)
        cards = [str(self.game.get_current_player_hand()[index]) for index in indices]
        res = self._record_action(
            {"kind": "defend", "phase": "defense", "card_indices": indices, "cards": cards},
            lambda: self.game.defend_with_card_indices(indices),
        )
        self.selected_indices = []
        self.log(res.get("message", "Defense resolved."))
        
        if res.get("game_over"):
            self.state = "GAME_OVER"
            self.game_over_msg = "Defeat... The party has fallen."
        else:
            self.state = "GAME"
        self._finish_recording_if_over(res)

    def _record_action(self, action, operation):
        if not self.recorder.enabled:
            return operation()
        state_before = serialize_game(self.game)
        result = operation()
        self.recorder.record_event(action, result, self.game, state_before)
        return result

    def _finish_recording_if_over(self, result):
        if self.game.game_over and self.recorder.active:
            self._finish_recording(result.get("message"))

    def _finish_recording(self, reason):
        if self.recorder.active:
            self.recorder.finish(self.game, reason=reason)

    def _handle_result(self, res: dict):
        if not res: return
        
        if res.get("cards_played"):
            self.log(f"Played: {', '.join(res['cards_played'])}")
            cps = res["cards_played"]
            if any("♦" in c for c in cps): play_draw()
            if any("♥" in c for c in cps): play_shimmer()
            if any("♠" in c for c in cps): play_clang()

        if res.get("enemy_damage", 0) > 0:
            self.log(f"Dealt {res['enemy_damage']} damage.")
            play_thud()

        if msg := res.get("message"):
            self.log(msg)

        phase = res.get("phase")
        if phase == "next_player_choice":
            self.jester_active = True
        elif phase == "defense_needed":
            self.state = "DEFENSE"
            self.defense_required = res.get("defense_required", 0)
            
            # Auto-defeat check
            hand = self.game.get_current_player_hand()
            all_total = sum(c.get_discard_value() for c in hand)
            if all_total < self.defense_required:
                if self.game.can_use_solo_jester():
                    self.log("No possible defense, but you can use Solo Jester!")
                else:
                    self.game.game_over = True
                    self.state = "GAME_OVER"
                    self.game_over_msg = "Defeat... No possible defense."
                    play_defeat()
        elif phase in ("victory", "game_over") or self.game.victory or self.game.game_over:
            self.state = "GAME_OVER"
            if phase == "victory" or self.game.victory:
                self.game_over_msg = "Victory! The King is dead!"
                play_victory()
            else:
                self.game_over_msg = "Defeat... The party has fallen."
                play_defeat()
        self._finish_recording_if_over(res)

    # --- Updates ---
    def _update_ui_state(self):
        if self.state != "GAME" and self.state != "DEFENSE": return
        gstate = self.game.get_game_state()
        
        # Enable play button only if valid
        hand = self.game.get_current_player_hand()
        cards = [hand[i] for i in sorted(self.selected_indices)]
        ok = False
        try:
            ok = self.game._is_valid_combo(cards) if cards else False
        except:
            ok = False
        self.play_btn.is_disabled = not ok
        self.yield_btn.is_disabled = not gstate["can_yield"]
        
        # Solo Jester
        show_jester = (self.num_players == 1 and gstate.get('solo_jesters_remaining', 0) > 0)
        self.jester_btn.is_disabled = False if show_jester else True

        # Defend button
        if self.state == "DEFENSE":
            total = sum(c.get_discard_value() for c in cards)
            self.defend_btn.is_disabled = total < self.defense_required

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
                
            if event.type == pygame.MOUSEWHEEL:
                self.action_scroll_y += event.y * 25
                if self.action_scroll_y > 0:
                    self.action_scroll_y = 0
            
            if self.state == "MENU":
                for b in self.menu_buttons:
                    b.handle_event(event)

            elif self.state in ("GAME", "DEFENSE"):
                if self.state == "GAME":
                    if not self.jester_active:
                        self.play_btn.handle_event(event)
                        self.yield_btn.handle_event(event)
                        if not self.jester_btn.is_disabled:
                            self.jester_btn.handle_event(event)
                else:
                    self.defend_btn.handle_event(event)
                    if not self.jester_btn.is_disabled:
                        self.jester_btn.handle_event(event)
                
                # Card clicks
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.jester_active:
                        # Handle jester player selection
                        px, py = event.pos
                        for i in range(self.game.num_players):
                            rect = pygame.Rect(self.width//2 - 100 + i*60, self.height//2, 50, 40)
                            if rect.collidepoint(px, py):
                                result = self._record_action(
                                    {
                                        "kind": "choose_next_player",
                                        "phase": "jester_choice",
                                        "chosen_player": i,
                                    },
                                    lambda: {
                                        "success": self.game.choose_next_player(i),
                                        "message": f"Chose player {i + 1}",
                                    },
                                )
                                if result["success"]:
                                    self.log(f"Jester chose Player {i+1}")
                                    self.jester_active = False
                                    self.selected_indices = []
                    else:
                        # Handle card selection
                        hand = self.game.get_current_player_hand()
                        num_cards = len(hand)
                        card_w, card_h = 90, 130
                        spacing = 10
                        total_w = num_cards * card_w + (num_cards - 1) * spacing
                        start_x = (self.width - total_w) // 2
                        start_y = self.height - card_h - 120

                        px, py = event.pos
                        for i in range(num_cards):
                            rect = pygame.Rect(start_x + i * (card_w + spacing), start_y, card_w, card_h)
                            if rect.collidepoint(px, py):
                                if i in self.selected_indices:
                                    self.selected_indices.remove(i)
                                else:
                                    self.selected_indices.append(i)

            elif self.state == "GAME_OVER":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.state = "MENU"
        
        return True

    def draw(self):
        self.screen.fill(DARK_THEME["bg"])
        
        if self.state == "MENU":
            self.draw_menu()
        elif self.state in ("GAME", "DEFENSE"):
            self.draw_game()
        elif self.state == "GAME_OVER":
            self.draw_game_over()
        
        pygame.display.flip()

    def draw_menu(self):
        title = self.title_font.render("REGICIDE", True, DARK_THEME["danger"])
        title_rect = title.get_rect(center=(self.width//2, 200))
        self.screen.blit(title, title_rect)
        
        sub = self.body_font.render("A digital tabletop of corrupted royalty.", True, DARK_THEME["muted"])
        self.screen.blit(sub, sub.get_rect(center=(self.width//2, 260)))

        for b in self.menu_buttons:
            b.draw(self.screen)

    def draw_game(self):
        gstate = self.game.get_game_state()
        self._update_ui_state()

        # Top Bar (Piles info)
        top_bar_y = 20
        castle = self.body_font.render(f"Castle: {gstate['enemies_remaining']}", True, DARK_THEME["text"])
        tavern = self.body_font.render(f"Tavern: {gstate['tavern_cards']}", True, DARK_THEME["text"])
        discard = self.body_font.render(f"Discard: {gstate['discard_cards']}", True, DARK_THEME["text"])
        self.screen.blit(castle, (20, top_bar_y))
        self.screen.blit(tavern, (20, top_bar_y + 25))
        self.screen.blit(discard, (20, top_bar_y + 50))
        
        # Possible Actions Panel (Left side)
        panel_x = 20
        panel_y = 120
        panel_w = 280
        panel_h = self.height - 240
        pygame.draw.rect(self.screen, DARK_THEME["panel"], (panel_x, panel_y, panel_w, panel_h), border_radius=8)
        actions_title = self.bold_font.render("Possible Actions", True, DARK_THEME["muted"])
        self.screen.blit(actions_title, (panel_x + 10, panel_y + 10))
        
        hand = self.game.get_current_player_hand()
        if self.action_handler and hand:
            phase = "attack" if self.state == "GAME" else "defense"
            try:
                actions = self.action_handler.get_all_possible_actions(hand, phase, gstate)
                
                clip_rect = pygame.Rect(panel_x, panel_y + 40, panel_w, panel_h - 40)
                self.screen.set_clip(clip_rect)
                
                list_y = panel_y + 40 + self.action_scroll_y
                for i, action_mask in enumerate(actions):
                    if self.action_handler.is_yield_action(action_mask):
                        act_str = "Yield"
                    else:
                        indices = self.action_handler.mask_to_card_indices(action_mask, len(hand))
                        cards_played = [hand[idx] for idx in indices]
                        act_str = "Play: " + ", ".join(str(c) for c in cards_played)
                        
                        if phase == "attack" and self.game.current_enemy:
                            total_attack = sum(card.get_attack_value() for card in cards_played)
                            enemy_suit = self.game.current_enemy.card.suit
                            jester_cancelled = self.game.jester_immunity_cancelled
                            
                            has_clubs = False
                            for card in cards_played:
                                is_immune = (not jester_cancelled) and (card.suit == enemy_suit)
                                if not is_immune and card.suit.value == "♣":
                                    has_clubs = True
                                    break
                                    
                            expected_damage = total_attack * 2 if has_clubs else total_attack
                            act_str += f" (Dmg: {expected_damage})"
                    
                    if list_y + 25 > clip_rect.top and list_y < clip_rect.bottom:
                        act_surf = self.body_font.render(act_str, True, DARK_THEME["text"])
                        self.screen.blit(act_surf, (panel_x + 10, list_y))
                    list_y += 25
                
                self.screen.set_clip(None)
                
                max_scroll = min(0, (panel_h - 40) - (len(actions) * 25))
                if self.action_scroll_y < max_scroll:
                    self.action_scroll_y = max_scroll
                if self.action_scroll_y > 0:
                    self.action_scroll_y = 0
            except Exception as e:
                err_surf = self.body_font.render("Error loading actions", True, DARK_THEME["danger"])
                self.screen.blit(err_surf, (panel_x + 10, panel_y + 40))

        # Enemy Area
        if self.game.current_enemy:
            enemy = self.game.current_enemy
            # Draw a big enemy card
            ex, ey = self.width//2 - 100, 50
            draw_card(self.screen, ex, ey, 200, 200, str(enemy.card), self.big_font, selected=False)
            
            # Enemy Stats
            atk = self.body_font.render(f"ATK: {enemy.get_effective_attack()}", True, DARK_THEME["text"])
            shield = self.body_font.render(f"Shield: {enemy.spade_protection}", True, DARK_THEME["text"])
            self.screen.blit(atk, (ex + 220, ey + 30))
            self.screen.blit(shield, (ex + 220, ey + 60))

            # Health Bar
            hp_left = enemy.health - enemy.damage_taken
            self.enemy_hp_bar.set(hp_left, enemy.health)
            self.enemy_hp_bar.draw(self.screen)

        # Action Log (Right side)
        log_x = self.width - 450
        pygame.draw.rect(self.screen, DARK_THEME["panel"], (log_x, 20, 430, 200), border_radius=8)
        log_title = self.bold_font.render("Action Log", True, DARK_THEME["muted"])
        self.screen.blit(log_title, (log_x + 10, 30))
        for i, log_str in enumerate(self.action_log):
            log_surf = self.body_font.render(log_str, True, DARK_THEME["text"])
            self.screen.blit(log_surf, (log_x + 10, 60 + i * 20))

        # Hand / Player Area
        hand = self.game.get_current_player_hand()
        num_cards = len(hand)
        card_w, card_h = 90, 130
        spacing = 10
        total_w = num_cards * card_w + (num_cards - 1) * spacing
        start_x = (self.width - total_w) // 2
        start_y = self.height - card_h - 120

        player_info = self.h1_font.render(f"Player {gstate['current_player']+1} Turn", True, DARK_THEME["accent2"])
        self.screen.blit(player_info, (start_x, start_y - 40))
        
        # Expected Damage
        cards = [hand[i] for i in sorted(self.selected_indices)]
        if self.state == "GAME" and cards and not self.play_btn.is_disabled and self.game.current_enemy:
            total_attack = sum(card.get_attack_value() for card in cards)
            
            enemy_suit = self.game.current_enemy.card.suit
            jester_cancelled = self.game.jester_immunity_cancelled
            
            has_clubs = False
            for card in cards:
                is_immune = (not jester_cancelled) and (card.suit == enemy_suit)
                if not is_immune and card.suit.value == "♣":
                    has_clubs = True
                    break
                    
            expected_damage = total_attack * 2 if has_clubs else total_attack
            dmg_text = self.bold_font.render(f"Expected Damage: {expected_damage}", True, DARK_THEME["danger"])
            # Display it to the left or right of player_info
            self.screen.blit(dmg_text, (start_x + player_info.get_width() + 20, start_y - 35))

        for i, c in enumerate(hand):
            cx = start_x + i * (card_w + spacing)
            draw_card(self.screen, cx, start_y, card_w, card_h, str(c), self.card_font, selected=(i in self.selected_indices))

        # Bottom Buttons
        if self.state == "GAME":
            if not self.jester_active:
                self.play_btn.draw(self.screen)
                self.yield_btn.draw(self.screen)
                if not self.jester_btn.is_disabled:
                    self.jester_btn.draw(self.screen)
            else:
                msg = self.h2_font.render("Jester Active: Choose next player", True, DARK_THEME["accent2"])
                self.screen.blit(msg, (self.width//2 - msg.get_width()//2, self.height//2 - 40))
                for i in range(self.game.num_players):
                    rect = pygame.Rect(self.width//2 - 100 + i*60, self.height//2, 50, 40)
                    pygame.draw.rect(self.screen, DARK_THEME["wood"], rect, border_radius=5)
                    ts = self.body_font.render(f"P{i+1}", True, DARK_THEME["text"])
                    self.screen.blit(ts, ts.get_rect(center=rect.center))

        elif self.state == "DEFENSE":
            # Defense overlay
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg1 = self.h1_font.render("DEFENSE REQUIRED!", True, DARK_THEME["danger"])
            msg2 = self.h2_font.render(f"Take {self.defense_required} damage. Select cards to discard.", True, DARK_THEME["text"])
            self.screen.blit(msg1, (self.width//2 - msg1.get_width()//2, self.height//2 - 100))
            self.screen.blit(msg2, (self.width//2 - msg2.get_width()//2, self.height//2 - 40))

            # Redraw hand on top of overlay
            for i, c in enumerate(hand):
                cx = start_x + i * (card_w + spacing)
                draw_card(self.screen, cx, start_y, card_w, card_h, str(c), self.card_font, selected=(i in self.selected_indices))
            
            cards = [hand[i] for i in sorted(self.selected_indices)]
            total = sum(c.get_discard_value() for c in cards)
            t_color = DARK_THEME["success"] if total >= self.defense_required else DARK_THEME["danger"]
            total_lbl = self.bold_font.render(f"Selected: {total} / {self.defense_required}", True, t_color)
            self.screen.blit(total_lbl, (self.width//2 - total_lbl.get_width()//2, self.height//2 + 20))
            
            self.defend_btn.draw(self.screen)
            if not self.jester_btn.is_disabled:
                self.jester_btn.draw(self.screen)

    def draw_game_over(self):
        title = self.title_font.render(self.game_over_msg, True, DARK_THEME["danger"] if "Defeat" in self.game_over_msg else DARK_THEME["success"])
        self.screen.blit(title, title.get_rect(center=(self.width//2, 200)))
        
        msg = self.body_font.render("Click anywhere to return to Menu", True, DARK_THEME["muted"])
        self.screen.blit(msg, msg.get_rect(center=(self.width//2, 300)))

    def run(self):
        try:
            running = True
            while running:
                running = self.handle_events()
                self.draw()
                self.clock.tick(60)
        except Exception as error:
            self.run_context.fail(error)
            logger.exception("UI session failed")
            raise
        finally:
            if self.recorder.active:
                self.recorder.abort("ui_closed")
            if self.run_context.manifest["status"] == "running":
                self.run_context.complete()
            pygame.quit()

if __name__ == "__main__":
    app = RegicideApp()
    app.run()
