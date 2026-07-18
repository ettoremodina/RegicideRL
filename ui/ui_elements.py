import pygame
from typing import Callable
from .theme import DARK_THEME, SUIT_COLORS

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, font: pygame.font.Font, on_click: Callable = None, bg_color=DARK_THEME["wood"], hover_color=DARK_THEME["wood_hover"], text_color=DARK_THEME["text"]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.on_click = on_click
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.is_disabled = False

    def draw(self, surface: pygame.Surface):
        color = self.bg_color
        if self.is_disabled:
            color = (50, 50, 50)
        elif self.is_hovered:
            color = self.hover_color

        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, DARK_THEME["accent2"], self.rect, width=2, border_radius=8)

        text_surf = self.font.render(self.text, True, self.text_color if not self.is_disabled else (100, 100, 100))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event: pygame.event.Event):
        if self.is_disabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered and self.on_click:
                self.on_click()
                return True
        return False


class Label:
    def __init__(self, x: int, y: int, text: str, font: pygame.font.Font, color=DARK_THEME["text"], anchor="topleft"):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.anchor = anchor
        self._render()

    def set_text(self, text: str, color=None):
        self.text = text
        if color:
            self.color = color
        self._render()

    def _render(self):
        self.surf = self.font.render(self.text, True, self.color)
        self.rect = self.surf.get_rect()
        setattr(self.rect, self.anchor, (self.x, self.y))

    def draw(self, surface: pygame.Surface):
        surface.blit(self.surf, self.rect)


class HealthBar:
    def __init__(self, x: int, y: int, width: int, height: int, font: pygame.font.Font):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.current = 0
        self.maximum = 1

    def set(self, current: int, maximum: int):
        self.current = max(0, current)
        self.maximum = max(1, maximum)

    def draw(self, surface: pygame.Surface):
        # Background
        pygame.draw.rect(surface, (59, 59, 59), self.rect, border_radius=4)
        # Foreground
        frac = min(1.0, self.current / self.maximum)
        if frac > 0:
            fg_rect = pygame.Rect(self.rect.x, self.rect.y, int(self.rect.width * frac), self.rect.height)
            pygame.draw.rect(surface, DARK_THEME["danger"], fg_rect, border_radius=4)
        
        # Text
        text_surf = self.font.render(f"HP {self.current}/{self.maximum}", True, DARK_THEME["text"])
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


def draw_card(surface: pygame.Surface, x: int, y: int, width: int, height: int, label: str, font: pygame.font.Font, selected: bool = False):
    rect = pygame.Rect(x, y, width, height)
    suit = label[-1] if label else "?"
    color = SUIT_COLORS.get(suit, DARK_THEME["text"])

    bg_color = DARK_THEME["card_selected"] if selected else DARK_THEME["wood"]
    
    # Drop shadow
    shadow_rect = rect.copy()
    shadow_rect.x += 4
    shadow_rect.y += 4
    pygame.draw.rect(surface, DARK_THEME["shadow"], shadow_rect, border_radius=10)

    # Card background
    pygame.draw.rect(surface, bg_color, rect, border_radius=10)
    # Card border
    pygame.draw.rect(surface, color if selected else DARK_THEME["accent2"], rect, width=2, border_radius=10)

    # Text
    text_surf = font.render(label, True, color)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)
