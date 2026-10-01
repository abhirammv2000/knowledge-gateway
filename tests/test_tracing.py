import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway import tracing
from gateway.tracing import JsonlSpanExporter, get_tracer, setup_tracing, shutdown_tracing


@pytest.fixture(autouse=True)
def tracing_off_afterwards():
    yield
    shutdown_tracing()


def clear_trace_env(monkeypatch):
    for name in ("KG_TRACE_FILE", "KG_TRACE_CONSOLE", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


def test_tracing_stays_off_when_nothing_asks_for_it(monkeypatch):
    clear_trace_env(monkeypatch)
    assert setup_tracing() is None
    with get_tracer().start_as_current_span("anything") as span:  # must not fail
        span.set_attribute("x", 1)


def test_spans_go_to_the_exporter_that_was_passed_in(monkeypatch):
    clear_trace_env(monkeypatch)
    exporter = InMemorySpanExporter()
    setup_tracing(exporter)
    with get_tracer().start_as_current_span("outer"):
        with get_tracer().start_as_current_span("inner"):
            pass
    names = [s.name for s in exporter.get_finished_spans()]
    assert names == ["inner", "outer"]  # a span is exported when it ends, so the inner one comes first


def test_the_trace_file_gets_one_json_line_per_span_with_parent_links(tmp_path, monkeypatch):
    clear_trace_env(monkeypatch)
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("KG_TRACE_FILE", str(path))
    assert setup_tracing() is not None

    with get_tracer().start_as_current_span("outer") as outer:
        outer.set_attribute("chunks", 12)
        with get_tracer().start_as_current_span("inner"):
            pass

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    inner, outer_row = rows
    assert (inner["name"], outer_row["name"]) == ("inner", "outer")
    assert inner["parent_id"] == outer_row["span_id"] and outer_row["parent_id"] is None
    assert inner["trace_id"] == outer_row["trace_id"]
    assert outer_row["attributes"] == {"chunks": 12}
    assert outer_row["duration_ms"] >= 0 and outer_row["status"] == "UNSET"


def test_otlp_endpoint_without_the_exporter_package_says_what_to_install(monkeypatch):
    clear_trace_env(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    def missing(endpoint):
        raise RuntimeError("pip install opentelemetry-exporter-otlp-proto-http")

    monkeypatch.setattr(tracing, "_otlp_exporter", missing)
    with pytest.raises(RuntimeError, match="opentelemetry-exporter-otlp-proto-http"):
        setup_tracing()


def test_jsonl_exporter_appends_across_runs(tmp_path):
    path = tmp_path / "t.jsonl"
    for _ in range(2):
        setup_tracing(JsonlSpanExporter(path))
        with get_tracer().start_as_current_span("s"):
            pass
        shutdown_tracing()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
