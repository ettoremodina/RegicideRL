# Testing

Run tests from the repository root with Python 3.10 or newer. Install the runtime
and contributor dependencies first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Release gate

Run the complete automated suite:

```bash
python -m pytest
```

The suite covers the rules engine, global action encoding, Gymnasium adapter,
agents, experimental-report pipeline, logging, control panel, and documentation
contract. AlphaZero tests protect the retained development code; passing them
does not mean AlphaZero is part of the finished agent comparison.

Build the API reference after the tests:

```bash
python -m scripts.generate_docs
```

Open `docs/api/index.html` and confirm that the package index and representative
pages render correctly.

## Focused suites

Use a focused command while changing one subsystem:

```bash
python -m pytest tests/test_game_rules.py tests/test_solo_rules.py
python -m pytest tests/test_action_handler.py tests/test_env.py
python -m pytest tests/test_experimental_report.py
python -m pytest tests/test_ml_logger.py tests/test_ml_logger_runtime.py
python -m pytest tests/test_control_panel.py
python -m pytest tests/test_documentation.py
```

`tests/test_documentation.py` performs both an AST docstring audit and an
isolated pdoc build for every published package. New public modules, classes,
functions, and methods require docstrings; non-trivial private lifecycle or
algorithm helpers do as well. The detailed policy is in
[`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Manual smoke checks

After the automated gate passes, verify each shipped entry point:

1. Start `python -m ui`, begin a one-player game, select a legal card, and exit
   with `Esc`.
2. Start `python -m control_panel`; confirm it opens on `127.0.0.1`, launch an
   approved short job, and inspect its logs.
3. Run `python -m scripts.log_game`, then use `python -m scripts.runs list` to
   confirm that the run and game were cataloged.
4. Exercise the report pipeline with a non-publication sample:

   ```bash
   python -m scripts.experimental_report.orchestrator \
     --agents random heuristic \
     --games 2 \
     --base-seed 42 \
     --jobs 1
   ```

5. Open `docs/site/index.html` and check navigation, charts, responsive layout,
   and links to the API and repository guides.

Do not treat the two-game report smoke check as an experimental result.
Publication results must use the recorded protocol and a completed run.

## Reproducibility checks

Before publishing results, verify that:

- the effective report configuration is stored with the run;
- every compared agent receives the same ordered seeds;
- timing measurements use `--jobs 1` on one otherwise idle machine;
- raw per-game data and generated analyses remain under the same run directory;
- no PPO or AlphaZero checkpoint is silently substituted into the final
  Random/Heuristic/PIMC/ISMCTS comparison.

PPO remains a costly, weak experiment and AlphaZero remains future work; neither
requires additional training for the release gate.
