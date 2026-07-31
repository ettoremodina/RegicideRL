# Regicide

Regicide is a Python implementation of the cooperative card game, built around
an explicit rules engine and a fixed global action space. The repository includes
a Pygame desktop client, a local browser control panel, reproducible experiment
tracking, and several agents for solo-game evaluation.

The release focuses on the completed rules engine and the comparison of Random,
Heuristic, PIMC, and Information Set Monte Carlo Tree Search (ISMCTS). PPO was
investigated but proved expensive to train and weak in the available experiments;
it is not part of the final comparison. AlphaZero and a lightweight,
interpretable agent distilled from ISMCTS decisions remain future work.

## Quick start

Regicide requires Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m ui
```

The desktop client supports one to four players and records games through the
project's local-first logging system. Press `Esc` to leave the full-screen UI.

On Windows, `control_panel.pyw` opens the browser control panel. It can also be
started from a terminal on any supported platform:

```bash
python -m control_panel
```

The panel binds to `127.0.0.1`; it is intended for local use only.

## Documentation

- [Project and experimental report](docs/site/index.html) — navigable overview,
  rules, architecture, agents, results, and future work.
- [Results entry point](docs/RESULTS.md) — provenance, dataset, and report links.
- [Usage guide](docs/USAGE.md) — installation and verified command-line workflows.
- [Repository structure](docs/REPOSITORY_STRUCTURE.md) — where the main components
  and generated artifacts live.
- [API documentation policy](docs/DOCUMENTATION.md) — docstring conventions and
  reference-generation checks.
- [Logging guide](ml_logger/README.md) — runs, metrics, telemetry, and artifacts.
- [Testing guide](TESTING.md) — focused and complete validation commands.

Build the generated API reference with:

```bash
python -m scripts.generate_docs
```

Then open `docs/api/index.html` locally.

## Common workflows

Run a recorded random game:

```bash
python -m scripts.log_game
```

Run the reproducible four-agent comparison:

```bash
python -m scripts.experimental_report.orchestrator \
  --agents random heuristic pimc ismcts \
  --games 100 \
  --base-seed 20260718 \
  --jobs 1
```

Using one worker avoids CPU contention in timing comparisons. Runs can be
resumed from their artifact directory; see the [usage guide](docs/USAGE.md) for
the full command.

Run the test suite:

```bash
python -m pytest
```

## Architecture at a glance

- `game/` owns rules, state transitions, legal-action generation, and global
  action encoding.
- `agents/` contains the common agent interface and the implemented policies.
- `solvers/` contains the Gymnasium adapter and the experimental training and
  evaluation workflows.
- `ui/` provides the Pygame game client; `control_panel/` provides the local web
  launcher and run browser.
- `ml_logger/` and `integrations/` store reproducible run metadata and
  Regicide-specific game histories under `artifacts/`.
- `scripts/` exposes reporting, inspection, migration, and documentation tools.

See [Repository structure](docs/REPOSITORY_STRUCTURE.md) for a more complete map.
