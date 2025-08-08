# Regicide GUI

A simple Tkinter-based GUI to play the Regicide engine in `regicide.py`.

Features:
- Enemy panel with health and spade shield.
- Castle/Tavern/Discard counts and discard viewer.
- Player hand with selectable cards, legal-combo preview gating Play button.
- Yield button (auto-disabled if illegal), Jester chooser dialog.
- Defense dialog with running total (e.g., "Selected: 9/12").

Run:
```
python -m ui.app
```

Notes:
- This is a base UI; animations/sounds and fancier theming can be added later.
- Fonts: installs are optional; app falls back to defaults.
