"""OpenTelemetry tracing, off unless you turn it on with one of:

    KG_TRACE_FILE=traces.jsonl          one json line per span
    KG_TRACE_CONSOLE=1                  spans to stderr
    OTEL_EXPORTER_OTLP_ENDPOINT=...     any OTLP/HTTP collector (needs opentelemetry-exporter-otlp-proto-http)

The documents can contain personal data, so spans only carry sizes, counts and timings, never the text of
a query or a document. Keep it that way if you add attributes. Console output goes to stderr because stdout
is the MCP protocol. The provider is kept here and not set as the global one, since OpenTelemetry only
allows that once per process and the tests need a fresh exporter each time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

SERVICE_NAME = "knowledge-gateway"

_provider: TracerProvider | None = None


def get_tracer() -> trace.Tracer:
    if _provider is not None:
        return _provider.get_tracer(SERVICE_NAME)
    return trace.get_tracer(SERVICE_NAME)  # the no-op tracer unless someone set a global provider


class JsonlSpanExporter(SpanExporter):
    """Appends each finished span to a file as a json line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with open(self.path, "a", encoding="utf-8") as f:
            for s in spans:
                ctx = s.get_span_context()
                f.write(json.dumps({
                    "name": s.name,
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                    "attributes": dict(s.attributes or {}),
                    "status": s.status.status_code.name,
                }) + "\n")
        return SpanExportResult.SUCCESS


def _otlp_exporter(endpoint: str) -> SpanExporter:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as e:
        raise RuntimeError(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http "
            "is not installed. pip install opentelemetry-exporter-otlp-proto-http"
        ) from e
    return OTLPSpanExporter()  # reads the endpoint from the same environment variable


def setup_tracing(exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Start tracing with the exporter you pass, or whatever the environment asks for. Returns None if tracing stays off."""
    global _provider

    if exporter is None:
        if os.environ.get("KG_TRACE_FILE"):
            exporter = JsonlSpanExporter(os.environ["KG_TRACE_FILE"])
        elif os.environ.get("KG_TRACE_CONSOLE"):
            exporter = ConsoleSpanExporter(out=sys.stderr)
        elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            exporter = _otlp_exporter(os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])

    if exporter is None:
        _provider = None
        return None

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
