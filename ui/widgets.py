import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Optional
from .theme import DARK_THEME, SUIT_COLORS


class HealthBar(ttk.Frame):
    def __init__(self, master, max_value: int = 40, **kwargs):
        super().__init__(master, **kwargs)
        self.max_value = max_value
        self.canvas = tk.Canvas(self, height=18, bg=DARK_THEME["panel"], highlightthickness=0)
        self.canvas.pack(fill="x")
        self.text = self.canvas.create_text(4, 9, anchor="w", fill=DARK_THEME["text"], font=("Merriweather", 10))
        self.rect_bg = self.canvas.create_rectangle(60, 4, 260, 14, fill="#3b3b3b", outline="")
        self.rect_fg = self.canvas.create_rectangle(60, 4, 60, 14, fill=DARK_THEME["danger"], outline="")

    def set(self, current: int, maximum: Optional[int] = None):
        if maximum is not None:
            self.max_value = maximum
        maximum = self.max_value
        current = max(0, min(current, maximum))
        # Draw
        self.canvas.itemconfig(self.text, text=f"Health: {current}/{maximum}")
        width = self.canvas.winfo_width() or 300
        bar_left, bar_right = 60, max(120, width - 10)
        frac = current / maximum if maximum else 0
        x = bar_left + int((bar_right - bar_left) * frac)
        self.canvas.coords(self.rect_bg, bar_left, 4, bar_right, 14)
        self.canvas.coords(self.rect_fg, bar_left, 4, x, 14)


class CardButton(ttk.Frame):
    def __init__(self, master, label: str, on_click: Callable[[None], None], tooltip: str = ""):
        super().__init__(master, padding=2)
        suit = label[-1] if label else "?"
        color = SUIT_COLORS.get(suit, DARK_THEME["text"]) 
        self.btn = tk.Button(
                self,
                text=label,
                fg=color,
                bg=DARK_THEME["wood"],
                activebackground="#4a3b3b",
                relief=tk.RAISED,
                bd=2,
            font=("Merriweather", 14, "bold"),
                command=on_click,
                padx=8, pady=6,
                takefocus=0,
            )
        self.btn.pack(fill="both", expand=True)
        # Make whole frame clickable too
        self.bind("<Button-1>", lambda e: on_click())
        self.btn.bind("<Button-1>", lambda e: on_click())
        self._tooltip = None
        if tooltip:
            self._tooltip = ToolTip(self.btn, tooltip)

    def set_state(self, state: str):
        self.btn.config(state=state)

    def set_tooltip(self, text: str):
        if self._tooltip is None:
            self._tooltip = ToolTip(self.btn, text)
        else:
            self._tooltip.text = text


class LogBox(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, height=9, bg="#111", fg=DARK_THEME["muted"], wrap="word")
        self.text.pack(fill="both", expand=True)
        self.text.config(state="disabled")

    def log(self, line: str):
        self.text.config(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.config(state="disabled")


# Simple tooltip helper
class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 300):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self._after_id = None
        self.delay_ms = delay_ms
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self.hide)
        widget.bind("<Button-1>", self.hide)

    def _schedule(self, _=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self.show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def show(self, _=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") or (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 20
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#222", foreground=DARK_THEME["text"],
                         relief=tk.SOLID, borderwidth=1,
                         font=("Merriweather", 9))
        label.pack(ipadx=6, ipady=4)

    def hide(self, _=None):
        self._cancel()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


def create_tooltip(widget, text: str):
    ToolTip(widget, text)


class EnemyView(ttk.Frame):
    """Large enemy display: big suit/value, health bar, attack/shield/immunity."""
    def __init__(self, master, width=440, height=260):
        super().__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, width=width, height=height, bg=DARK_THEME["panel"], highlightthickness=0)
        self.canvas.pack()
        self._hp = 0
        self._hp_max = 1
        # Pre-draw static card background
        self._card_rect = self.canvas.create_rectangle(16, 16, 220, height-16, fill=DARK_THEME["wood"], outline=DARK_THEME["accent2"], width=3)
        self._value_text = self.canvas.create_text(118, 70, text="?", font=("Uncial Antiqua", 44, "bold"), fill=DARK_THEME["text"])
        self._suit_text = self.canvas.create_text(118, 130, text="?", font=("Uncial Antiqua", 48, "bold"), fill=DARK_THEME["text"]) 
        # Right panel texts
        self._name = self.canvas.create_text(250, 28, text="Enemy", anchor="nw", font=("Uncial Antiqua", 18, "bold"), fill=DARK_THEME["text"])
        self._atk = self.canvas.create_text(250, 70, text="ATK: 0", anchor="nw", font=("Merriweather", 14, "bold"), fill=DARK_THEME["text"]) 
        self._shield = self.canvas.create_text(250, 100, text="Shield: 0", anchor="nw", font=("Merriweather", 12), fill=DARK_THEME["muted"]) 
        self._immune = self.canvas.create_text(250, 130, text="Immunity: -", anchor="nw", font=("Merriweather", 12), fill=DARK_THEME["muted"]) 
        # Health bar
        self._hp_label = self.canvas.create_text(250, 168, text="HP 0/0", anchor="nw", font=("Merriweather", 12), fill=DARK_THEME["text"]) 
        self._hp_bg = self.canvas.create_rectangle(250, 192, width-20, 212, fill="#3b3b3b", outline="")
        self._hp_fg = self.canvas.create_rectangle(250, 192, 250, 212, fill=DARK_THEME["danger"], outline="")

    def set_enemy(self, value_label: str, suit_char: str, hp_current: int, hp_max: int, atk: int, shield: int, immunity: str):
        # Update text colors based on suit
        suit_color = SUIT_COLORS.get(suit_char, DARK_THEME["text"])
        self.canvas.itemconfig(self._value_text, text=value_label)
        self.canvas.itemconfig(self._suit_text, text=suit_char, fill=suit_color)
        self.canvas.itemconfig(self._name, text=f"Enemy: {value_label}{suit_char}")
        self.canvas.itemconfig(self._atk, text=f"ATK: {atk}")
        self.canvas.itemconfig(self._shield, text=f"Shield: {shield}")
        self.canvas.itemconfig(self._immune, text=f"Immunity: {immunity}")
        self._hp = hp_current
        self._hp_max = max(1, hp_max)
        self._redraw_hp()

    def _redraw_hp(self):
        frac = max(0.0, min(1.0, self._hp / self._hp_max))
        x0, y0, x1, y1 = self.canvas.coords(self._hp_bg)
        x = x0 + (x1 - x0) * frac
        self.canvas.coords(self._hp_fg, x0, y0, x, y1)
        self.canvas.itemconfig(self._hp_label, text=f"HP {self._hp}/{self._hp_max}")

    def animate_health(self, start: int, end: int):
        steps = max(1, min(25, abs(start - end)))
        delta = (end - start) / steps if steps else 0
        current = float(start)
        def step(i=0):
            nonlocal current
            if i >= steps:
                self._hp = end
                self._redraw_hp()
                return
            current += delta
            self._hp = int(round(current))
            self._redraw_hp()
            self.after(20, lambda: step(i+1))
        step(0)

    def show_damage(self, amount: int):
        # Float red number near card
        lbl = self.canvas.create_text(200, 30, text=f"-{amount}", fill=DARK_THEME["danger"], font=("Merriweather", 18, "bold"))
        def anim(i=0):
            if i > 20:
                self.canvas.delete(lbl)
                return
            self.canvas.move(lbl, 0, -2)
            self.after(30, lambda: anim(i+1))
        anim(0)
