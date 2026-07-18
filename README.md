# Regicide AI: ISMCTS Solver

A Python implementation of the cooperative card game **Regicide** featuring a fully-functional rules engine, graphical interface, and a highly capable AI agent based on Information Set Monte Carlo Tree Search (ISMCTS).

## Features
- **Accurate Rules Engine**: Fully implements the official Regicide mechanics, including Jester rules, suit powers, enemy immunity, combinations, yielding, and defense.
- **Solo & Multiplayer Support**: Supports 1-4 players following the official scaling mechanics.
- **Graphical Interface**: A complete desktop GUI to play the game natively.
- **ISMCTS AI Solver**: A powerful autonomous agent capable of solving the game in solo mode by planning moves under hidden information (the deck).
- **Advanced Logging & Telemetry**: Integrated `ml_logger` for terminal dashboards, JSONL logging, metrics tracking, and hardware monitoring.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Regicide.git
cd Regicide

# Install requirements
pip install -r requirements.txt
```

## Running the Game (GUI)

To launch the graphical game client:
```bash
python -m ui
```

## Running the AI Agent

To simulate a game played autonomously by the ISMCTS agent:
```bash
python -m scripts.log_game
```
This script will leverage the `ml_logger` dashboard to provide real-time updates on game state, agent actions, and memory consumption.

To run the quick benchmark:
```bash
python benchmark.py
```
The complete suite, including training benchmarks, is opt-in:
```bash
python benchmark.py --mode all
```

## Analyzing Results

Every command creates an isolated run under `artifacts/runs/<date>/<run_id>`.
To analyze recorded games and persist the aggregate result:
```bash
python -m scripts.analyze_runs
```

To inspect the catalog or replay the recorded action sequence:
```bash
python -m scripts.runs list
python -m scripts.runs games <run_id>
python -m scripts.runs replay <game_id>
```

## Logging and artifacts

`ml_logger` is the only application logging and persistence entry point.
Modules acquire a logger with `get_logger(__name__)`; executable commands
create a `RunContext` with `start_run(...)`. Application code must not use
`print()` or write directly to `stdout`.

Before every run, `ml_logger` reads [`logger_config.yaml`](logger_config.yaml).
The main switches are:

- `logging.enabled`, `console`, `file`, and `level`;
- `terminal.colors`, `timezone`, visibility, and traceback rendering;
- `saving.enabled`, `metrics`, `results`, and `telemetry`;
- `games.enabled` and `recording_level`;
- `run_type_overrides` for high-volume commands.

`benchmark`, PPO, AlphaZero, tuning, and BC data generation disable individual
game recording by default. To record benchmark games, change:

```yaml
run_type_overrides:
  benchmark:
    games:
      enabled: true
```

`recording_level` accepts `summary`, `actions`, or `full`. A different
configuration file can be selected without editing the repository by setting
the `ML_LOGGER_CONFIG` environment variable. A minimal manifest and catalog
entry are always retained; `saving.enabled` controls optional logger-managed
artifacts, not explicitly requested outputs such as trained models.

Generated data is organized as follows:

```text
artifacts/
├── catalog.sqlite
├── runs/<date>/<run_id>/
│   ├── manifest.json
│   ├── logs/run.log
│   ├── metrics/
│   ├── games/
│   ├── datasets/
│   ├── models/
│   ├── checkpoints/
│   └── analysis/
├── datasets/
├── promoted_models/
└── legacy/
```

To migrate old `runs`, `logs`, `models`, `outputs`, `experiments`, and
`archive` directories without deleting their contents:

```bash
python -m scripts.migrate_artifacts
```

## Generating Documentation

We use `pdoc` to generate the HTML documentation from docstrings automatically. To build it:
```bash
python -m scripts.generate_docs
```
This will place the documentation in the `docs/` folder.

## Running the AI Training

To train new RL models (e.g. AlphaZero) or test advanced solvers, use the module runner from the root directory:
```bash
python -m solvers.train --help
```

## Project Structure
- `game/`: The core mechanics engine (`regicide.py`, `action_handler.py`).
- `agents/`: Core interfaces and heuristic bots (e.g., Random, ISMCTS, PIMC).
- `solvers/`: Advanced RL training loops, AlphaZero networks, self-play logic, and TensorBoard logging.
- `ui/`: Pygame-based graphical interface.
- `ml_logger/`: Custom Rich-based telemetry, dashboard, and metrics tracker.
- `scripts/`: Diagnostic tools, game runners, and documentation generators (run them with `python -m scripts.<name>`).
- `tests/`: Comprehensive test suite (`pytest`).
- `rules/`: Text files containing reference rulebooks.
- `artifacts/`: Canonical generated runs, games, models, datasets, analyses, and legacy outputs.

## Future Work
- Implementing a multi-agent cooperative version of ISMCTS for 2-4 player scaling.
- Adding a "hints" feature to the UI powered by the ISMCTS tree.
- Further optimizing the tree determinization for speed.
