"""OpenTelemetry tracing for the gateway.

Off by default. Importing this module and creating spans costs next to nothing
when no exporter is configured, because the spans go to a tracer that does
nothing. Turn it on with one of:

    KG_TRACE_FILE=traces.jsonl          one JSON line per finished span
    KG_TRACE_CONSOLE=1                  spans printed to stderr
    OTEL_EXPORTER_OTLP_ENDPOINT=...     send to any OTLP/HTTP collector (needs
                                        opentelemetry-exporter-otlp-proto-http)

The gateway sits in front of documents that contain personal data, so spans carry
sizes, counts and timings and never the text of a query or a document. If you add
a span attribute, keep that rule.

stdout is off limits here. An MCP server on stdio uses stdout for the protocol
itself, so anything printed there corrupts it. The console option writes to
stderr for that reason.

The provider is kept in this module and is not registered as the global one.
OpenTelemetry only allows setting the global provider once per process, which
would make tests that need a fresh exporter awkward, and only this package
creates spans here anyway.
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
    """Appends each finished span to a file as one JSON line."""

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
    """Start tracing, using `exporter` if given and otherwise whatever the environment asks for.

    Returns the provider, or None when nothing was asked for and tracing stays off.
    """
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
