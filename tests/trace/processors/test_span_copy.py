"""Tests for ReadableSpan copy helpers preserving SDK drop counters."""

from __future__ import annotations

from typing import Sequence, cast

from coralogix_opentelemetry.trace.processors.span_copy import (
    copy_with_attributes,
    copy_with_parent,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class _ListExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _ended_span_with_drops() -> ReadableSpan:
    exporter = _ListExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "test"}),
        span_limits=SpanLimits(max_attributes=3, max_events=2, max_links=2),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("limited") as span:
        for key in ("a", "b", "c", "d"):
            span.set_attribute(key, 1)
        span.add_event("e1")
        span.add_event("e2")
        span.add_event("e3")
    provider.force_flush()
    provider.shutdown()  # type: ignore[no-untyped-call]
    assert len(exporter.spans) == 1
    return cast(ReadableSpan, exporter.spans[0])


def test_copy_with_parent_preserves_dropped_counts() -> None:
    original = _ended_span_with_drops()
    assert original.dropped_attributes >= 1
    assert original.dropped_events >= 1

    copied = copy_with_parent(original, original.parent)
    assert copied.dropped_attributes == original.dropped_attributes
    assert copied.dropped_events == original.dropped_events
    assert copied.dropped_links == original.dropped_links


def test_copy_with_attributes_preserves_dropped_counts() -> None:
    original = _ended_span_with_drops()
    attrs = dict(original.attributes or {})
    attrs["cgx.transaction"] = "txn"

    copied = copy_with_attributes(original, attrs)
    assert copied.dropped_attributes >= original.dropped_attributes
    assert copied.dropped_events == original.dropped_events
    assert (copied.attributes or {})["cgx.transaction"] == "txn"
