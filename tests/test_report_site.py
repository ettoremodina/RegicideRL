"""Verify reproducible publication of the static experimental-report assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_report_site import publish


def test_publish_builds_accessible_charts_and_provenance(tmp_path: Path) -> None:
    """Build all public data assets from a minimal four-agent run artifact."""
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    datasets_dir = run_dir / "datasets"
    analysis_dir.mkdir(parents=True)
    datasets_dir.mkdir(parents=True)

    summary_rows = []
    for index, label in enumerate(("Random", "Heuristic", "PIMC", "ISMCTS")):
        summary_rows.append(
            {
                "label": label,
                "win_rate": str(index / 10),
                "win_ci_low": str(max(0.0, index / 10 - 0.05)),
                "win_ci_high": str(index / 10 + 0.05),
                "bosses_mean": str(3 + index * 2),
                "bosses_ci_low": str(2.8 + index * 2),
                "bosses_ci_high": str(3.2 + index * 2),
                "decision_ms_mean": str(0.03 * (100**index)),
            }
        )
    _write_csv(analysis_dir / "summary.csv", summary_rows)
    _write_csv(
        datasets_dir / "games.csv",
        [{"agent": f"agent-{index % 4}", "seed": str(index)} for index in range(400)],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "experimental-report-test",
                "status": "failed",
                "started_at": "2026-07-18T16:06:05+00:00",
                "git": {"commit": "abc123"},
                "config": {
                    "protocol": {"base_seed": 42, "games_per_agent": 100}
                },
                "result": {"error": "missing PPO checkpoint"},
            }
        ),
        encoding="utf-8",
    )

    site_root = tmp_path / "site"
    publish(run_dir, site_root)

    for filename in ("win-rate.svg", "castle-progress.svg", "quality-cost.svg"):
        chart = (site_root / "assets" / "charts" / filename).read_text(
            encoding="utf-8"
        )
        assert 'role="img"' in chart
        assert "<title" in chart
        assert "<desc" in chart
    win_rate_chart = (site_root / "assets" / "charts" / "win-rate.svg").read_text(
        encoding="utf-8"
    )
    assert '<text class="value" x="517.5" y="65" text-anchor="middle">30%</text>' in win_rate_chart
    provenance = json.loads(
        (site_root / "data" / "results-summary.json").read_text(encoding="utf-8")
    )
    assert provenance["publication_status"] == "partial"
    assert provenance["completed_game_rows"] == 400
    assert (site_root / "data" / "games.csv").is_file()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows with a stable header for the publication smoke test."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
