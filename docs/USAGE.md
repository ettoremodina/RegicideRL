# Usage

All commands in this guide are run from the repository root. Regicide supports
Python 3.10 and newer.

## Install

Create and activate a virtual environment using the mechanism for your platform,
then install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For contributors, install the local logger package and test dependency as well:

```bash
python -m pip install -e ".[dev]"
```

The main requirements include Pygame, Gymnasium, Stable-Baselines3, pdoc, and
the scientific Python packages used by the evaluation pipeline. Model training
is not required to run the game or reproduce the final non-neural comparison.

## Play in the desktop client

```bash
python -m ui
```

The Pygame client opens full-screen, supports one to four players, and uses the
same rules engine as the agents. Press `Esc` to exit. Actions are enabled only
when the selected cards form a legal play for the current phase.

## Use the local control panel

On Windows, double-click `control_panel.pyw`, or start the server explicitly:

```bash
python -m control_panel
```

The application opens a local browser interface for approved commands,
configuration editing, process monitoring, artifacts, and reports. The server
is deliberately bound to `127.0.0.1`; do not change it to a public interface.
Closing the browser does not stop active jobs.

## Record and inspect a game

Play one random solo game and save its run data:

```bash
python -m scripts.log_game
```

Inspect the run catalog and its recorded games:

```bash
python -m scripts.runs list
python -m scripts.runs show <run_id>
python -m scripts.runs games <run_id>
python -m scripts.runs replay <game_id>
```

Aggregate completed game records across the catalog, or restrict the analysis
to one run:

```bash
python -m scripts.analyze_runs
python -m scripts.analyze_runs --run-id <run_id>
```

Generated data belongs under `artifacts/`. Each run has an immutable identifier,
a manifest, logs, and only the result folders required by that workload.

## Generate the experimental comparison

The publication comparison uses Random, Heuristic, PIMC, and ISMCTS with the
same sequence of game seeds:

```bash
python -m scripts.experimental_report.orchestrator \
  --agents random heuristic pimc ismcts \
  --games 100 \
  --base-seed 20260718 \
  --jobs 1
```

`--jobs 1` is recommended when comparing execution time because concurrent
search agents compete for CPU resources. For a quick pipeline check, lower
`--games` to `2`; that smoke run is not a publication result.

If a run is interrupted, resume it without repeating completed agent/seed pairs:

```bash
python -m scripts.experimental_report.orchestrator \
  --resume-run artifacts/runs/<date>/<run_id> \
  --jobs 1
```

The effective configuration and raw per-game data are saved with the run. The
pipeline's configuration lives under `experimental_report` in `config.yaml`.

PPO is retained as an experimental implementation, but its training cost and
weak available results exclude it from the final comparison. AlphaZero is
unfinished future work. Neither needs to be trained to close or use this
project.

## Run benchmarks

The default benchmark exercises the rules engine without launching training:

```bash
python benchmark.py --mode normal --games 1000
```

Other non-training modes are `env` and `parallel`. The `cpu`, `gpu`, and `all`
modes include PPO training throughput and are unnecessary for release
validation.

## Build the API reference

```bash
python -m scripts.generate_docs
```

The command rebuilds `docs/api/` from Google-style docstrings with pdoc. It does
not overwrite curated Markdown or the navigable report in `docs/site/`. Open
`docs/api/index.html` in a browser after generation.

## Run tests

Run the entire suite:

```bash
python -m pytest
```

Run the API documentation contract and build smoke test alone:

```bash
python -m pytest tests/test_documentation.py
```

More focused commands and test expectations are listed in
[Testing](../TESTING.md).

## Logging and artifacts

Every executable workflow uses `ml_logger` and writes canonical runs under:

```text
artifacts/
|-- catalog.sqlite
|-- runs/<date>/<run_id>/
|-- datasets/
|-- promoted_models/
`-- legacy/
```

The configuration is read from `logger_config.yaml`. Set the
`ML_LOGGER_CONFIG` environment variable to use another logger configuration
without modifying the repository. See the [logger reference](../ml_logger/README.md)
for event, metrics, telemetry, and retention details.

