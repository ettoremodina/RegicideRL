"""Central logging configuration backed by a :class:`RunContext`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .configs.config_loader import load_config
from .storage import RunContext
from .terminal import ConfiguredRichHandler, UtcOffsetFormatter, parse_utc_offset

FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
FILE_DATE_FORMAT = None
PLAIN_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
PLAIN_CONSOLE_DATE_FORMAT = "%H:%M:%S %Z"
_MANAGED_HANDLER_ATTRIBUTE = "_ml_logger_managed"


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger routed through ml_logger configuration."""
    return logging.getLogger(name)


def configure_logging(
    context: RunContext | None = None,
    level: int | str | None = None,
    console: bool | None = None,
) -> logging.Logger:
    """Configure ml_logger handlers without disturbing third-party handlers."""
    root_logger = logging.getLogger()
    _remove_managed_handlers(root_logger)
    settings = context.settings if context else load_config()
    logging_settings = settings.get("logging", {})
    if not logging_settings.get("enabled", True):
        logging.disable(logging.CRITICAL)
        return root_logger
    logging.disable(logging.NOTSET)
    effective_level = level or logging_settings.get("level", "INFO")
    root_logger.setLevel(effective_level)
    terminal_settings = settings.get("terminal", {})
    utc_offset = parse_utc_offset(
        terminal_settings.get("timezone", "+02:00")
    )
    console_enabled = (
        logging_settings.get("console", True) if console is None else console
    )
    if console_enabled:
        console_handler = _create_console_handler(
            terminal_settings,
            settings.get("highlights", []),
            utc_offset,
        )
        _mark_managed(console_handler)
        root_logger.addHandler(console_handler)
    if context and logging_settings.get("file", True):
        log_path = context.run_dir / "logs" / "run.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            UtcOffsetFormatter(
                FILE_LOG_FORMAT,
                FILE_DATE_FORMAT,
                utc_offset,
            )
        )
        _mark_managed(file_handler)
        root_logger.addHandler(file_handler)
    return root_logger


def _create_console_handler(terminal_settings, highlights, utc_offset):
    """Create either the configured Rich handler or a plain stream fallback."""
    if terminal_settings.get("colors", True):
        handler = ConfiguredRichHandler(terminal_settings, highlights)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler
    handler = logging.StreamHandler()
    handler.setFormatter(
        UtcOffsetFormatter(
            PLAIN_CONSOLE_FORMAT,
            PLAIN_CONSOLE_DATE_FORMAT,
            utc_offset,
        )
    )
    return handler


def start_run(
    run_type: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    root_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    logger_config_path: str | Path | None = None,
) -> RunContext:
    """Create a run and route application logging into it."""
    settings = load_config(logger_config_path, run_type=run_type)
    effective_root = (
        root_dir
        or os.environ.get("REGICIDE_ARTIFACTS_DIR")
        or settings.get("artifacts", {}).get("root_dir")
    )
    context = RunContext.create(
        run_type=run_type,
        name=name,
        config=config,
        root_dir=effective_root,
        metadata=metadata,
        settings=settings,
    )
    configure_logging(context)
    runtime_logger = get_logger(__name__)
    runtime_logger.info(
        "Run started  %-22s %s",
        run_type,
        context.run_id.rsplit("-", maxsplit=1)[-1],
    )
    runtime_logger.info(
        "Output       log:%s  metrics:%s  results:%s  games:%s",
        _switch(settings["logging"]["file"]),
        _switch(context.saving_enabled("metrics")),
        _switch(context.saving_enabled("results")),
        _switch(context.game_recording_enabled),
    )
    runtime_logger.debug(
        "Run directory: %s | logger config: %s | recording: %s",
        context.run_dir,
        settings["config_path"],
        context.game_recording_level,
    )
    return context


class RunLogger:
    """Small compatibility facade for solver loops using the unified storage."""

    def __init__(
        self,
        context: RunContext | None = None,
        run_type: str = "solver",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.context = context or start_run(
            run_type=run_type,
            name=run_name,
            config=config,
        )
        self.logger = get_logger(run_name or run_type)

    def log(self, message: str, *args: Any) -> None:
        """Log an informational message through the compatibility facade."""
        self.logger.info(message, *args)

    def info(self, message: str, *args: Any) -> None:
        """Log an informational message."""
        self.logger.info(message, *args)

    def log_metrics(self, step: int, metrics_dict: dict[str, Any]) -> None:
        """Persist metrics for one solver step."""
        self.context.log_metrics(step, metrics_dict)

    def get_run_dir(self) -> str:
        """Return the active run directory as a legacy string path."""
        return str(self.context.run_dir)

    @property
    def models_dir(self) -> str:
        """Return the run-local model directory as a legacy string path."""
        return str(self.context.run_dir / "models")


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Close handlers installed by this package without touching third parties."""
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()


def _mark_managed(handler: logging.Handler) -> None:
    setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)


def _switch(enabled: bool) -> str:
    return "on" if enabled else "off"
