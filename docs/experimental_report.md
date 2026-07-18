# Regicide — experimental report protocol

The pipeline in `scripts/experimental_report/` compares the implemented agents
with a reproducible protocol and automatically generates the report materials:
method descriptions, raw data, tables, plots, and statistical analyses.

## Full run

```bash
python -m scripts.experimental_report.orchestrator
```

Configuration lives under `experimental_report` in `config.yaml`. The sample
size, base seed, and agents can be overridden from the command line:

```bash
python -m scripts.experimental_report.orchestrator \
  --agents random heuristic pimc ismcts \
  --games 100 \
  --base-seed 20260718
```

Random, Heuristic, PIMC, and ISMCTS are enabled initially. PPO and AlphaZero
are registered but remain disabled until their checkpoints are available. To
include them, configure the model path and set `enabled: true`.

To compare PIMC and ISMCTS with equal simulation budgets, use the same total
budget per decision:

```yaml
pimc:
  kwargs:
    rollout_budget: 3000
ismcts:
  kwargs:
    n_iterations: 3000
```

`rollout_budget` is distributed uniformly across PIMC's legal actions. Setting
`n_determinizations: 3000` instead would allocate 3,000 rollouts to every action
and favor PIMC with a much higher computational cost.

## Protocol

Every agent plays with the same sequence of seeds. The comparison is therefore
paired: for each seed, every method starts from the same initial card order.
Each game records:

- outcome and number of bosses defeated;
- number of decisions and defense phases;
- cumulative reward;
- yields and invalid actions;
- total time;
- mean and 95th-percentile decision latency;
- completion or decision-limit status.

The pipeline measures methods sequentially on the same machine. For reliable
timing comparisons, close applications with variable workloads and record the
hardware used.

## Statistical analysis

- win rate: Wilson confidence interval;
- continuous metrics: bootstrap interval for the mean;
- pairwise win difference: exact McNemar test;
- difference in bosses defeated: paired Wilcoxon test;
- multiple comparisons: Holm correction;
- effect size: Cohen *dz* and probability of superiority.

P-values are not interpreted alone: the report also presents confidence
intervals, effect size, and computational cost.

## Output

Each execution creates a run under:

```text
artifacts/runs/<date>/experimental-report-<id>/
|-- config.yaml
|-- experimental_report_config.yaml
|-- datasets/
|   `-- games.csv
|-- metrics/
|   `-- metrics.jsonl
`-- analysis/
    |-- experimental_report.md
    |-- summary.csv
    |-- pairwise_tests.csv
    |-- statistics.json
    |-- tables.md
    |-- tables.tex
    |-- comprehensive_dashboard.png
    |-- win_rate.png
    |-- bosses_defeated.png
    |-- execution_time.png
    `-- quality_cost_tradeoff.png
```

The configuration copy preserves the original parameters, while
`experimental_report_config.yaml` stores the effective configuration after any
command-line overrides.

## Regenerating the report

Plots and tables can be regenerated from the raw data without repeating the
games:

```bash
python -m scripts.experimental_report.analysis \
  artifacts/runs/<date>/<run_id>
```

To produce only `datasets/games.csv`:

```bash
python -m scripts.experimental_report.runner --agents random heuristic
```

## Resuming an interrupted run

`datasets/games.csv` is updated atomically after every game. If a run is
interrupted, the pipeline also recovers rows already present in
`metrics/metrics.jsonl` and skips every completed agent/seed pair:

```bash
python -m scripts.experimental_report.orchestrator \
  --resume-run artifacts/runs/<date>/<run_id> \
  --jobs 4
```

To preserve reproducibility, `--resume-run` uses the configuration saved with
the run and cannot be combined with `--agents`, `--games`, or `--base-seed`.
`--jobs` may still be changed on each resume. Games run in separate processes,
while only the main process updates metrics and checkpoints, preventing
concurrent writes.

With multiple workers, per-game timings for slow methods are affected by CPU
contention. Win rates and bosses defeated remain comparable, but a rigorous
execution-time benchmark requires `--jobs 1` for every agent.
