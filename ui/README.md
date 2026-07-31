# Pygame desktop client

`ui` is the full-screen desktop client for the shared Regicide rules engine. It
supports one to four local players and records completed sessions through
`ml_logger`.

## Run

From the repository root:

```bash
python -m ui
```

Choose the player count on the opening screen. During a game, select cards in
the current hand and use the phase-appropriate action:

- **Play Selected** attacks with the selected legal card or combination;
- **Yield** passes when the rules allow it;
- **Defend** discards the selected cards against the enemy attack;
- **Solo Jester** uses the solo-mode Jester at a legal timing.

The client displays the current enemy, castle/tavern/discard counts, player
hands, and a bounded action log. Use the mouse wheel where a panel is
scrollable. Press `Esc` to close the application.

The UI calls `game.Game` and `game.ActionHandler` directly; it does not maintain
a separate copy of the rules. System fonts are optional because Pygame's default
font is used as a fallback. The sound hooks are currently no-ops until audio
assets are connected.

See the project [usage guide](../docs/USAGE.md) for installation, logging, and
artifact inspection.
