import logging
from datetime import datetime, timezone
from io import StringIO

from rich.console import Console
from rich.text import Text

from ml_logger.configs.config_loader import load_config
from ml_logger.terminal import (
    ConfiguredRichHandler,
    UtcOffsetFormatter,
    parse_utc_offset,
)
from ml_logger.utils.formatting import Highlighter


def test_rich_console_is_compact_colored_and_uses_utc_plus_two():
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=120,
    )
    handler = ConfiguredRichHandler(
        {
            "timezone": "+02:00",
            "show_time": True,
            "show_level": True,
            "show_source": False,
            "omit_repeated_times": False,
            "rich_tracebacks": True,
        },
        [{"pattern": r"[\d.]+", "style": "cyan"}],
        console=console,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = _make_record()

    handler.handle(record)
    rendered = output.getvalue()
    plain_text = Text.from_ansi(rendered).plain

    assert "16:14:09 UTC+2" in plain_text
    assert "INFO" in plain_text
    assert "1000 games at 1641.12 games/s" in plain_text
    assert "noisy.module" not in plain_text
    assert "\x1b[" in rendered


def test_file_formatter_uses_explicit_utc_plus_two_offset():
    formatter = UtcOffsetFormatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s",
        None,
        parse_utc_offset("+02:00"),
    )

    rendered = formatter.format(_make_record())

    assert rendered.startswith("2026-07-18T16:14:09+02:00 INFO noisy.module")


def test_utc_offset_parser_accepts_compact_and_named_values():
    assert parse_utc_offset("+02:00").utcoffset(None).total_seconds() == 7200
    assert parse_utc_offset("UTC+2").utcoffset(None).total_seconds() == 7200


def test_numeric_highlight_does_not_color_letters_inside_words():
    highlighter = Highlighter(load_config()["highlights"])

    highlighted = highlighter.apply(
        "Environment benchmark: 100 games, 1.5e-3 loss, run b4e25362"
    )
    cyan_fragments = [
        highlighted.plain[span.start : span.end]
        for span in highlighted.spans
        if span.style == "cyan"
    ]

    assert cyan_fragments == ["100", "1.5e-3"]


def _make_record():
    record = logging.LogRecord(
        name="noisy.module",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processed %d games at %.2f games/s",
        args=(1000, 1641.12),
        exc_info=None,
    )
    record.created = datetime(
        2026,
        7,
        18,
        14,
        14,
        9,
        tzinfo=timezone.utc,
    ).timestamp()
    return record
