from types import SimpleNamespace

from rich.console import Console
from rich.text import Text

from ml_logger.core.layout import render_logs_panel
from solvers.alphazero import self_play


def test_log_stream_renders_the_newest_rows_within_available_height():
    console = Console(
        width=30,
        height=6,
        record=True,
        color_system=None,
    )
    logs = [Text(f"log line {index}") for index in range(1, 11)]

    console.print(render_logs_panel(logs), height=6)
    rendered = console.export_text()

    assert "│ log line 1 " not in rendered
    assert "log line 7" in rendered
    assert "log line 10" in rendered


def test_self_play_reports_cumulative_metrics_before_iteration_end(monkeypatch):
    game_number = 0

    def fake_game(*_args, **_kwargs):
        nonlocal game_number
        game_number += 1
        return [f"sample-{game_number}"], {
            "enemies_defeated": game_number,
            "victory": game_number % 2 == 0,
        }

    monkeypatch.setattr(self_play, "run_self_play_game", fake_game)
    progress_updates = []
    config = SimpleNamespace(games_per_iteration=12)

    _, final_stats = self_play.generate_self_play_data(
        network=None,
        config=config,
        device=None,
        progress_callback=lambda completed, total, stats: progress_updates.append(
            (completed, total, stats)
        ),
    )

    assert [update[0] for update in progress_updates] == [10, 12]
    assert progress_updates[0][2]["total_samples"] == 10
    assert progress_updates[-1][2] == final_stats
