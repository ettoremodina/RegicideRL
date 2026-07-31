"""Publish the static Regicide report charts from an existing run artifact.

The chart builder uses only the Python standard library; the project logger
records publication without requiring the game or plotting stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_logger import get_logger


DEFAULT_RUN = Path(
    "artifacts/runs/2026-07-18/"
    "experimental-report-20260718T160605-4dcbb706"
)
SITE_ROOT = Path("docs/site")
COLORS = {
    "Random": "#6B8498",
    "Heuristic": "#D0A84C",
    "PIMC": "#123D3A",
    "ISMCTS": "#B64038",
}
logger = get_logger(__name__)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of string-keyed dictionaries."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def svg_document(title: str, description: str, body: str, height: int) -> str:
    """Wrap chart markup in an accessible, responsive SVG document."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 {height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <style>
    text {{ font-family: "Segoe UI", Arial, sans-serif; fill: #101A24; }}
    .utility {{ font-family: "Cascadia Code", Consolas, monospace; font-size: 13px; fill: #536577; }}
    .label {{ font-size: 16px; font-weight: 650; }}
    .value {{ font-size: 15px; font-weight: 700; }}
    .grid {{ stroke: #D7D8D2; stroke-width: 1; }}
  </style>
  <rect width="920" height="{height}" rx="14" fill="#FBF9F4"/>
  {body}
</svg>
"""


def win_rate_chart(rows: list[dict[str, str]]) -> str:
    """Render win rates with Wilson 95% confidence intervals."""
    ordered = sorted(rows, key=lambda row: float(row["win_rate"]), reverse=True)
    left, right = 185, 850
    parts = []
    for tick in range(0, 61, 10):
        x = left + (right - left) * tick / 60
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="45" x2="{x:.1f}" y2="330"/>')
        parts.append(f'<text class="utility" x="{x:.1f}" y="355" text-anchor="middle">{tick}%</text>')
    for index, row in enumerate(ordered):
        y = 82 + index * 72
        rate = float(row["win_rate"]) * 100
        low = float(row["win_ci_low"]) * 100
        high = float(row["win_ci_high"]) * 100
        label = row["label"]
        x_rate = left + (right - left) * rate / 60
        x_low = left + (right - left) * low / 60
        x_high = left + (right - left) * high / 60
        parts.extend(
            [
                f'<text class="label" x="24" y="{y + 5}">{escape(label)}</text>',
                f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="#101A24" stroke-width="3"/>',
                f'<line x1="{x_low:.1f}" y1="{y - 8}" x2="{x_low:.1f}" y2="{y + 8}" stroke="#101A24" stroke-width="2"/>',
                f'<line x1="{x_high:.1f}" y1="{y - 8}" x2="{x_high:.1f}" y2="{y + 8}" stroke="#101A24" stroke-width="2"/>',
                f'<circle cx="{x_rate:.1f}" cy="{y}" r="9" fill="{COLORS[label]}" stroke="#FBF9F4" stroke-width="3"/>',
                f'<text class="value" x="{x_rate:.1f}" y="{y - 17}" text-anchor="middle">{rate:.0f}%</text>',
            ]
        )
    return svg_document(
        "Solo win rate by agent",
        "ISMCTS won 42 percent, PIMC 24 percent, Heuristic and Random zero percent; horizontal lines show Wilson 95 percent confidence intervals.",
        "\n  ".join(parts),
        385,
    )


def castle_progress_chart(rows: list[dict[str, str]]) -> str:
    """Render mean defeated enemies against the twelve-card castle."""
    ordered = sorted(rows, key=lambda row: float(row["bosses_mean"]), reverse=True)
    cell_width, gap, start_x = 49, 6, 187
    parts = []
    for cell in range(12):
        rank = "J" if cell < 4 else "Q" if cell < 8 else "K"
        x = start_x + cell * (cell_width + gap)
        parts.append(f'<text class="utility" x="{x + cell_width / 2:.1f}" y="42" text-anchor="middle">{rank}</text>')
    for index, row in enumerate(ordered):
        y = 70 + index * 70
        mean = float(row["bosses_mean"])
        low = float(row["bosses_ci_low"])
        high = float(row["bosses_ci_high"])
        label = row["label"]
        parts.append(f'<text class="label" x="24" y="{y + 28}">{escape(label)}</text>')
        for cell in range(12):
            x = start_x + cell * (cell_width + gap)
            fraction = max(0.0, min(1.0, mean - cell))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="38" rx="5" fill="#E7E4DC"/>')
            if fraction:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_width * fraction:.1f}" height="38" rx="5" fill="{COLORS[label]}"/>')
        low_x = start_x + low / 12 * (12 * cell_width + 11 * gap)
        high_x = start_x + high / 12 * (12 * cell_width + 11 * gap)
        parts.extend(
            [
                f'<line x1="{low_x:.1f}" y1="{y + 49}" x2="{high_x:.1f}" y2="{y + 49}" stroke="#101A24" stroke-width="2"/>',
                f'<circle cx="{start_x + mean / 12 * (12 * cell_width + 11 * gap):.1f}" cy="{y + 49}" r="4" fill="#101A24"/>',
                f'<text class="value" x="880" y="{y + 28}" text-anchor="end">{mean:.2f} / 12</text>',
            ]
        )
    parts.append('<text class="utility" x="187" y="374">Mean castle progress · whisker = bootstrap 95% CI</text>')
    return svg_document(
        "Mean castle progress by agent",
        "A twelve-cell Jack Queen King castle ladder. ISMCTS defeated 10.01 enemies on average, PIMC 9.25, Heuristic 4.37, and Random 2.97.",
        "\n  ".join(parts),
        400,
    )


def quality_cost_chart(rows: list[dict[str, str]]) -> str:
    """Render progress against mean decision latency on a log scale."""
    left, right, top, bottom = 100, 850, 45, 340
    min_log, max_log = -2, 4
    parts = []
    for exponent in range(min_log, max_log + 1):
        x = left + (exponent - min_log) / (max_log - min_log) * (right - left)
        label = { -2: "0.01 ms", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1 s", 4: "10 s"}[exponent]
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<text class="utility" x="{x:.1f}" y="370" text-anchor="middle">{label}</text>',
            ]
        )
    for tick in range(0, 13, 2):
        y = bottom - tick / 12 * (bottom - top)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text class="utility" x="82" y="{y + 5:.1f}" text-anchor="end">{tick}</text>',
            ]
        )
    offsets = {"Random": (12, 22), "Heuristic": (12, -12), "PIMC": (-66, -14), "ISMCTS": (12, -14)}
    for row in rows:
        label = row["label"]
        latency = float(row["decision_ms_mean"])
        bosses = float(row["bosses_mean"])
        x = left + (math.log10(latency) - min_log) / (max_log - min_log) * (right - left)
        y = bottom - bosses / 12 * (bottom - top)
        dx, dy = offsets[label]
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{COLORS[label]}" stroke="#FBF9F4" stroke-width="3"/>',
                f'<text class="label" x="{x + dx:.1f}" y="{y + dy:.1f}">{escape(label)}</text>',
            ]
        )
    parts.extend(
        [
            '<text class="utility" x="475" y="400" text-anchor="middle">Mean decision latency · logarithmic scale</text>',
            '<text class="utility" transform="translate(28 195) rotate(-90)" text-anchor="middle">Mean enemies defeated</text>',
        ]
    )
    return svg_document(
        "Castle progress versus decision latency",
        "Search agents reach much deeper into the castle but require several seconds per decision. Timing was collected with five parallel workers and is descriptive only.",
        "\n  ".join(parts),
        425,
    )


def publish(run_dir: Path, site_root: Path) -> None:
    """Generate report assets and copy the 400-game source dataset."""
    summary_path = run_dir / "analysis" / "summary.csv"
    games_path = run_dir / "datasets" / "games.csv"
    manifest_path = run_dir / "manifest.json"
    if not all(path.is_file() for path in (summary_path, games_path, manifest_path)):
        raise FileNotFoundError(f"Incomplete report artifact: {run_dir}")

    rows = read_rows(summary_path)
    games = read_rows(games_path)
    if len(rows) != 4 or len(games) != 400:
        raise ValueError("Publication expects four agents and exactly 400 game rows")
    if {row["label"] for row in rows} != set(COLORS):
        raise ValueError("Publication dataset must contain Random, Heuristic, PIMC, and ISMCTS only")

    charts_dir = site_root / "assets" / "charts"
    data_dir = site_root / "data"
    charts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "win-rate.svg": win_rate_chart(rows),
        "castle-progress.svg": castle_progress_chart(rows),
        "quality-cost.svg": quality_cost_chart(rows),
    }
    for filename, content in charts.items():
        (charts_dir / filename).write_text(content, encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = {
        "publication_status": "partial",
        "run_id": manifest["run_id"],
        "run_status": manifest["status"],
        "run_started_at": manifest["started_at"],
        "git_commit": manifest["git"]["commit"],
        "base_seed": manifest["config"]["protocol"]["base_seed"],
        "games_per_agent": manifest["config"]["protocol"]["games_per_agent"],
        "completed_game_rows": len(games),
        "agents": rows,
        "failure_reason": manifest["result"]["error"],
    }
    (data_dir / "results-summary.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(games_path, data_dir / "games.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the source artifact and output site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--site-root", type=Path, default=SITE_ROOT)
    return parser.parse_args()


def main() -> None:
    """Build the checked-in static report assets."""
    args = parse_args()
    publish(args.run, args.site_root)
    logger.info("Published static report assets to %s", args.site_root)


if __name__ == "__main__":
    main()
