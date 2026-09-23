"""
observability.py
-----------------
Lightweight OpenTelemetry instrumentation for the RAG pipeline.

Spans are exported two places:
  1. Console (ConsoleSpanExporter) -- shows up in your terminal locally, and
     in Streamlit Cloud's app logs when deployed. This is real OpenTelemetry
     output, not a mock.
  2. An in-memory recorder -- so the Streamlit UI itself can show a short
     trace summary (which model was used, how many retries, latency) without
     needing a real observability backend wired up.

In production you'd swap the console exporter for a real OTLP exporter
pointed at Jaeger / Grafana Cloud / Honeycomb / etc. -- that's a one-line
change (see the commented-out import below), not a rewrite. That's the
whole point of OpenTelemetry: the instrumentation code (the spans you saw
added in rag_engine.py and report_agent.py) never changes, only where the
finished spans get sent.

Usage:
    from observability import tracer, get_recent_spans

    with tracer.start_as_current_span("some.operation") as span:
        span.set_attribute("key", "value")
        ... do the thing ...
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# -- production swap-in: replace ConsoleSpanExporter() below with
#    OTLPSpanExporter(endpoint="https://your-collector:4318/v1/traces")


class _InMemorySpanRecorder(SpanExporter):
    """Keeps the last N finished spans in memory so the Streamlit UI can
    show a trace summary without a real observability backend."""

    def __init__(self, max_spans: int = 30):
        self.spans: list[dict] = []
        self.max_spans = max_spans

    def export(self, spans):
        for span in spans:
            duration_ms = (span.end_time - span.start_time) / 1_000_000 if span.end_time else None
            self.spans.append(
                {
                    "name": span.name,
                    "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
                    "attributes": dict(span.attributes or {}),
                    "status": "ERROR" if span.status.status_code.name == "ERROR" else "OK",
                }
            )
        self.spans = self.spans[-self.max_spans :]
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


_recorder = _InMemorySpanRecorder()
_provider = TracerProvider(resource=Resource.create({"service.name": "meef-rag-pipeline"}))
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
_provider.add_span_processor(SimpleSpanProcessor(_recorder))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("meef.rag_pipeline")


def get_recent_spans() -> list[dict]:
    """Return recent trace spans (most recent last) for display in the UI."""
    return list(_recorder.spans)
