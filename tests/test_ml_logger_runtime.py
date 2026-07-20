import json
import logging

import pytest

from ml_logger import EventKind, RunContext, run_scope, start_run


class EventCollector:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_run_scope_finalizes_and_generates_configured_report(tmp_path):
    config_path = _runtime_config(tmp_path)

    with run_scope(
        "scope-test",
        config={"seed": 4},
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    ) as context:
        context.log_params({"learning_rate": 0.01})
        context.log_metrics(1, {"train/loss": 2.0})
        context.log_metrics(2, {"train/loss": 1.0})
        source = tmp_path / "model.bin"
        source.write_bytes(b"model")
        artifact = context.log_artifact(source, kind="model")
        run_dir = context.run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = run_dir / "reports" / "run_report.html"
    assert manifest["status"] == "completed"
    assert manifest["params"]["learning_rate"] == 0.01
    assert artifact.read_bytes() == b"model"
    assert report.exists()
    assert "train/loss" in report.read_text(encoding="utf-8")
    assert context.listener_errors == ()
    assert _managed_handlers() == []


def test_run_scope_marks_uncaught_exception_as_failed(tmp_path):
    config_path = _runtime_config(tmp_path)

    with pytest.raises(RuntimeError, match="training failed"):
        with run_scope(
            "failure-test",
            root_dir=tmp_path / "artifacts",
            logger_config_path=config_path,
        ) as context:
            raise RuntimeError("training failed")

    manifest = json.loads(
        (context.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["result"]["error"] == "training failed"
    assert _managed_handlers() == []


def test_run_scope_finalizes_keyboard_interrupt_with_meaningful_error(tmp_path):
    config_path = _runtime_config(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        with run_scope(
            "interrupted-test",
            root_dir=tmp_path / "artifacts",
            logger_config_path=config_path,
        ) as context:
            raise KeyboardInterrupt

    manifest = json.loads(
        (context.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["result"]["error"] == "KeyboardInterrupt"
    assert _managed_handlers() == []


def test_metric_event_reaches_storage_and_listener_once(tmp_path):
    config_path = _runtime_config(tmp_path)
    context = start_run(
        "event-test",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    )
    collector = EventCollector()
    context.subscribe(collector)

    context.log_metrics(7, {"eval/accuracy": 0.9})
    context.complete()

    listener_metrics = [
        event for event in collector.events if event.kind == EventKind.METRICS
    ]
    stored_metrics = context.catalog.list_events(
        context.run_id,
        EventKind.METRICS,
    )
    assert len(listener_metrics) == 1
    assert len(stored_metrics) == 1
    assert stored_metrics[0].payload == {"eval/accuracy": 0.9}
    assert stored_metrics[0].step == 7


def test_attached_writer_uses_sqlite_and_parent_materializes_jsonl(tmp_path):
    config_path = _runtime_config(tmp_path)
    context = start_run(
        "multiprocess-test",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    )
    descriptor = context.descriptor()
    attached = RunContext.attach(**descriptor)

    attached.log_metrics(3, {"worker/throughput": 12.5})
    context.complete()

    metrics_path = context.run_dir / "metrics" / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "timestamp": records[0]["timestamp"],
            "step": 3,
            "worker/throughput": 12.5,
        }
    ]


def test_metric_storage_filters_are_independent_from_call_site(tmp_path):
    config_path = _runtime_config(
        tmp_path,
        """
metrics:
  include: ["train/*"]
  exclude: ["*/private"]
""",
    )
    context = start_run(
        "filter-test",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    )

    context.log_metrics(
        1,
        {
            "train/loss": 0.5,
            "train/private": 99,
            "eval/accuracy": 0.9,
        },
    )
    context.complete()

    events = context.catalog.list_events(context.run_id, EventKind.METRICS)
    assert events[0].payload == {"train/loss": 0.5}


def _runtime_config(tmp_path, extra=""):
    config_path = tmp_path / "logger.yaml"
    config_path.write_text(
        """
logging:
  console: false
  file: true
dashboard:
  mode: off
telemetry:
  enabled: false
report:
  enabled: true
  visualization: auto
"""
        + extra,
        encoding="utf-8",
    )
    return config_path


def _managed_handlers():
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_ml_logger_managed", False)
    ]
