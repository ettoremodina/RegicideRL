# Results

The publication results are presented in the navigable English report:

- [Open the results page](site/results.html)
- [Start from the project introduction](site/index.html)
- [Download the 400-game dataset](site/data/games.csv)
- [Inspect machine-readable provenance](site/data/results-summary.json)

The dataset contains 100 completed solo games for each of Random, Heuristic,
PIMC, and ISMCTS. Its parent run is marked `failed` because PPO was attempted
after those 400 games without a valid checkpoint, so the site labels the
publication dataset as **partial** and excludes PPO and AlphaZero from every
measured chart.

Regenerate the checked-in charts and publication data from the source artifact:

```bash
python -m scripts.build_report_site
```

The generated site remains the canonical narrative report; this short Markdown
page exists as a stable entry point and does not duplicate its tables or prose.
