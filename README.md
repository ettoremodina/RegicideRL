# Regicide

A Python implementation of the cooperative card game **Regicide**. 
This repository contains a full rules-enforcing game engine and a graphical user interface built with PyQt6.

## Features
- **Accurate Rules Engine**: Fully implements the official Regicide mechanics, including Jester rules, suit powers, enemy immunity, combinations, yielding, and defense.
- **Solo & Multiplayer Support**: Supports 1-4 players following the official scaling mechanics (Tavern Deck size, Jester behavior, Hand Limits).
- **Graphical Interface**: A complete desktop GUI to play the game natively.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Regicide.git
cd Regicide

# Install requirements
pip install -r requirements.txt
```

## Running the Game

To launch the graphical game client:
```bash
python -m ui
```

## Running Tests

The rules engine is thoroughly tested to guarantee mechanics accuracy.
```bash
# Run the test suite
python -m pytest tests/ -v
```

## Project Structure
- `game/`: The core mechanics engine (`regicide.py`, `action_handler.py`, `card.py`)
- `ui/`: PyQt6-based graphical interface (`main.py`, `gui.py`, `widgets.py`)
- `tests/`: Comprehensive test suite (`pytest`)
- `rules/`: Text files containing reference rulebooks.
- `archive/`: Legacy AI and reinforcement learning (gymnasium) experiments.
