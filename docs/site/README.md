# Regicide AI static report

Open `index.html` to browse the English project field guide. The site is fully
static and uses only local HTML, CSS, JavaScript, SVG, CSV, and JSON assets.

## Rebuild result assets

From the repository root:

```bash
python -m scripts.build_report_site
```

The builder reads the checked-in 400-game artifact from
`artifacts/runs/2026-07-18/experimental-report-20260718T160605-4dcbb706/`,
regenerates the three accessible SVG charts, copies the exact game-level CSV,
and writes a machine-readable provenance summary.

The dataset is deliberately labelled **partial**. All four measured agents
completed 100 games, but the enclosing run failed afterward when PPO attempted
to load a missing checkpoint. PPO and AlphaZero are not present in the dataset
and must not be added to its charts without new completed game rows.
