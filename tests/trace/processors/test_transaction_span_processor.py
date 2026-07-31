"""Tests for exclusive self-time and TransactionSpanProcessor."""

from __future__ import annotations

import threading
import time
from typing import Sequence

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors import (
    METRIC_SELF_TIME,
    SELF_TIME_ATTRIBUTE,
    TransactionSpanProcessor,
)
from coralogix_opentelemetry.trace.processors.self_time import self_time_by_span_id
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind


class ListSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def test_self_time_nested_fixture() -> None:
    spans = [
        ("1", "", "root", 0, 100),
        ("2", "1", "child", 20, 80),
    ]
    assert self_time_by_span_id(spans) == {"1": 40, "2": 60}


def test_processor_tags_and_self_time_without_sampler() -> None:
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, meter_provider=meter_provider, completion_holdback_millis=0
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("GET /orders", kind=SpanKind.SERVER):
        with tracer.start_as_current_span("db"):
            time.sleep(0.01)

    provider.force_flush()
    meter_provider.force_flush()

    by_name = {span.name: span for span in exporter.spans}
    root = by_name["GET /orders"]
    child = by_name["db"]
    assert root.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "GET /orders"
    assert root.attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
    assert SELF_TIME_ATTRIBUTE in root.attributes
    assert child.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "GET /orders"
    assert CoralogixAttributes.TRANSACTION_ROOT not in (child.attributes or {})
    assert SELF_TIME_ATTRIBUTE in child.attributes

    names = []
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.append(metric.name)
    assert METRIC_SELF_TIME in names

    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_force_flush_does_not_finalize_incomplete_traces() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    # max_regular_traces=0 disables harvest sampling so a completed trace is
    # exported immediately (no waiting on the next harvest flush).
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=0
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("parent"):
        with tracer.start_as_current_span("child"):
            pass
        assert exporter.spans == [], "child must stay buffered while parent is live"
        assert processor.force_flush() is True
        assert (
            exporter.spans == []
        ), "ForceFlush must not finalize incomplete local traces"

    assert {span.name for span in exporter.spans} == {"parent", "child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_shutdown_waits_for_in_flight_spans() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=0
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    parent = tracer.start_span("parent")
    with trace.use_span(parent, end_on_exit=False):
        with tracer.start_as_current_span("child"):
            pass
    assert exporter.spans == []

    def end_parent() -> None:
        time.sleep(0.03)
        parent.end()

    thread = threading.Thread(target=end_parent)
    thread.start()
    processor.shutdown()
    thread.join()

    assert {span.name for span in exporter.spans} == {"parent", "child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_export_is_serialized_across_concurrent_callers() -> None:
    """OTel forbids concurrent Export on one SpanExporter."""
    in_export = threading.Event()
    release = threading.Event()
    concurrent = 0
    max_concurrent = 0
    guard = threading.Lock()

    class BlockingExporter(SpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            nonlocal concurrent, max_concurrent
            with guard:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            in_export.set()
            if not release.wait(timeout=2.0):
                with guard:
                    concurrent -= 1
                raise RuntimeError("release not signaled")
            with guard:
                concurrent -= 1
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=0
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    # End the first span on a worker so MainThread can orchestrate while
    # export is blocked (on_end calls export synchronously).
    def first_trace() -> None:
        with tracer.start_as_current_span("a", kind=SpanKind.SERVER):
            pass

    t1 = threading.Thread(target=first_trace)
    t1.start()
    assert in_export.wait(timeout=1.0), "first export should block"

    def second_trace() -> None:
        with tracer.start_as_current_span("b", kind=SpanKind.SERVER):
            pass

    t2 = threading.Thread(target=second_trace)
    t2.start()
    time.sleep(0.05)  # second export should be waiting on _export_lock
    with guard:
        assert concurrent == 1
        assert max_concurrent == 1
    release.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert not t1.is_alive() and not t2.is_alive()
    with guard:
        assert max_concurrent == 1
    processor.shutdown()
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_shutdown_tracks_post_stop_child_of_in_flight_trace() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=0
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    parent = tracer.start_span("parent", kind=SpanKind.SERVER)
    parent_ctx = trace.set_span_in_context(parent)

    def run_shutdown() -> None:
        time.sleep(0.02)
        processor.shutdown()

    thread = threading.Thread(target=run_shutdown)
    thread.start()
    time.sleep(0.05)  # let shutdown set _stopped

    child = tracer.start_span("late-child", context=parent_ctx)
    parent.end()
    time.sleep(0.02)
    assert (
        exporter.spans == []
    ), "parent must not finalize while post-stop child is live"
    child.end()
    thread.join()

    names = {span.name for span in exporter.spans}
    assert names == {"parent", "late-child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_processor_records_metrics_even_when_trace_not_harvested() -> None:
    """Self-time metrics fire for every completed local trace; harvest only gates export."""
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            max_regular_traces=1,
            harvest_period_millis=3_600_000,
            completion_holdback_millis=0,
            meter_provider=meter_provider,
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("fast", kind=SpanKind.SERVER):
        time.sleep(0.005)
    with tracer.start_as_current_span("slow", kind=SpanKind.SERVER):
        time.sleep(0.04)

    # Fast loser is stub-exported when displaced by slow; slow waits for harvest.
    assert len(exporter.spans) == 1
    assert exporter.spans[0].name == "fast"

    meter_provider.force_flush()
    span_names = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != METRIC_SELF_TIME:
                    continue
                for point in metric.data.data_points:
                    span_names.add(dict(point.attributes).get("span.name"))
    assert "fast" in span_names
    assert "slow" in span_names

    provider.force_flush()
    roots = [
        span
        for span in exporter.spans
        if (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT)
    ]
    assert {span.name for span in roots} == {"fast", "slow"}
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_processor_immediate_export_when_max_regular_traces_zero() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            max_regular_traces=0,
            harvest_period_millis=0,
            completion_holdback_millis=0,
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("now", kind=SpanKind.SERVER):
        pass

    assert any(span.name == "now" for span in exporter.spans)
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_processor_trims_to_max_nodes_keeping_root() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, max_nodes=2, max_regular_traces=0, completion_holdback_millis=0
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root", kind=SpanKind.SERVER):
        with tracer.start_as_current_span("short"):
            pass
        with tracer.start_as_current_span("long"):
            time.sleep(0.02)

    names = {span.name for span in exporter.spans}
    assert "root" in names
    assert "long" in names
    assert "short" not in names
    assert len(exporter.spans) == 2
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_shutdown_flushes_pending_harvest_winner() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    processor = TransactionSpanProcessor(
        exporter,
        max_regular_traces=1,
        harvest_period_millis=3_600_000,
        completion_holdback_millis=0,
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root", kind=SpanKind.SERVER):
        pass

    # Harvest winner has not been flushed yet (no periodic harvester, no explicit flush).
    assert exporter.spans == []
    processor.shutdown()
    assert any(span.name == "root" for span in exporter.spans)


def test_processor_server_under_local_parent_starts_new_transaction() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, max_regular_traces=0, completion_holdback_millis=0
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("parent-flow"):
        with tracer.start_as_current_span("GET /checkout", kind=SpanKind.SERVER):
            with tracer.start_as_current_span("child"):
                pass

    provider.force_flush()
    by_name = {span.name: span for span in exporter.spans}
    assert (
        by_name["parent-flow"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "parent-flow"
    )
    assert (
        by_name["GET /checkout"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /checkout"
    )
    assert (
        by_name["GET /checkout"].attributes[CoralogixAttributes.TRANSACTION_ROOT]
        is True
    )
    assert (
        CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER
        not in by_name["GET /checkout"].attributes
    )
    assert (
        by_name["child"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /checkout"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_nested_server_finalizes_while_outer_still_open() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, max_regular_traces=0, completion_holdback_millis=0
        )
    )
    tracer = provider.get_tracer("test")

    outer = tracer.start_span("outer", kind=SpanKind.SERVER)
    with trace.use_span(outer, end_on_exit=False):
        with tracer.start_as_current_span("inner", kind=SpanKind.SERVER):
            with tracer.start_as_current_span("db"):
                time.sleep(0.005)

        names = {span.name for span in exporter.spans}
        assert names == {
            "inner",
            "db",
        }, "nested local transaction must finalize before the outer SERVER ends"

    outer.end()
    provider.force_flush()
    names = {span.name for span in exporter.spans}
    assert "outer" in names
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_completion_holdback_keeps_fire_and_forget_child() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=80
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    parent = tracer.start_span("parent", kind=SpanKind.SERVER)
    parent_ctx = trace.set_span_in_context(parent)
    parent.end()
    assert exporter.spans == [], "must not finalize while holdback is open"

    child = tracer.start_span("late-child", context=parent_ctx)
    child.end()
    # Holdback should have been cancelled by late-child on_start; after child
    # ends the TraceID is idle again — wait past holdback for leftover finalize.
    time.sleep(0.12)
    provider.force_flush()
    assert {span.name for span in exporter.spans} == {"parent", "late-child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_select_slowest_protects_all_transaction_roots() -> None:
    from coralogix_opentelemetry.trace.processors.trace_heap import select_slowest_spans
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import (
        SpanContext,
        SpanKind,
        Status,
        StatusCode,
        TraceFlags,
    )

    def ctx(span_id: int) -> SpanContext:
        return SpanContext(
            trace_id=1,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    def span(
        name: str,
        span_id: int,
        start: int,
        end: int,
        parent: int | None = None,
        root: bool = False,
    ) -> ReadableSpan:
        return ReadableSpan(
            name=name,
            context=ctx(span_id),
            parent=ctx(parent) if parent is not None else None,
            resource=Resource.create({}),
            attributes=(
                {CoralogixAttributes.TRANSACTION_ROOT.value: True} if root else {}
            ),
            events=(),
            links=(),
            kind=SpanKind.INTERNAL,
            status=Status(StatusCode.UNSET),
            start_time=start,
            end_time=end,
        )

    spans = [
        span("root-a", 1, 0, 1, root=True),
        span("root-b", 2, 0, 1, parent=1, root=True),
        span("slow", 3, 0, 100, parent=1),
    ]
    kept = select_slowest_spans(
        spans,
        max_nodes=2,
        root_span_ids=["0000000000000001", "0000000000000002"],
    )
    assert {s.name for s in kept} == {"root-a", "root-b"}


def _readable(
    name: str,
    *,
    trace_id: int = 1,
    span_id: int,
    parent_id: int | None = None,
    root: bool = False,
    start: int = 0,
    end: int = 1,
) -> ReadableSpan:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import (
        SpanContext,
        SpanKind,
        Status,
        StatusCode,
        TraceFlags,
    )

    def ctx(sid: int) -> SpanContext:
        return SpanContext(
            trace_id=trace_id,
            span_id=sid,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    attrs: dict[str, bool | str] = {}
    if root:
        attrs[CoralogixAttributes.TRANSACTION_ROOT.value] = True
        attrs[CoralogixAttributes.TRANSACTION_IDENTIFIER.value] = name
    return ReadableSpan(
        name=name,
        context=ctx(span_id),
        parent=ctx(parent_id) if parent_id is not None else None,
        resource=Resource.create({}),
        attributes=attrs,
        events=(),
        links=(),
        kind=SpanKind.SERVER if root else SpanKind.INTERNAL,
        status=Status(StatusCode.UNSET),
        start_time=start,
        end_time=end,
    )


def test_stale_cancelled_timer_does_not_pop_replacement() -> None:
    """A cancelled holdback timer must not pop a replacement timer for the same TraceID."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=60_000
    )
    trace_id = 0xABC
    outer = _readable("outer", span_id=1, root=True, end=10)

    with processor._lock:
        processor._buffers[trace_id] = [outer]
        processor._schedule_completion_locked(trace_id)
        stale = processor._pending_completions[trace_id]
        processor._cancel_pending_completion_locked(trace_id)
        processor._schedule_completion_locked(trace_id)
        replacement = processor._pending_completions[trace_id]
        assert replacement is not stale

    # Simulate the cancelled timer racing in after a replacement was armed.
    stale.function(*stale.args, **stale.kwargs)

    with processor._lock:
        assert processor._pending_completions.get(trace_id) is replacement
        assert processor._buffers.get(trace_id) == [outer]
    assert exporter.spans == []

    replacement.cancel()
    processor.shutdown()


def test_extract_completed_roots_deepest_first_excludes_extracted() -> None:
    """Nested SERVER root extracts before outer; outer must not re-export nested IDs.

    Buffer order deliberately puts the outer root first so buffer-order
    iteration would let the outer swallow the nested subtree.
    """
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(
        exporter, max_regular_traces=0, completion_holdback_millis=0
    )
    trace_id = 0xDEF
    outer = _readable("outer", span_id=1, root=True, start=0, end=100)
    nested = _readable("inner", span_id=2, parent_id=1, root=True, start=10, end=60)
    child = _readable("db", span_id=3, parent_id=2, start=20, end=50)

    with processor._lock:
        # Outer first — the defect case for buffer-order iteration.
        processor._buffers[trace_id] = [outer, nested, child]
        batches = processor._extract_completed_local_transactions_locked(
            trace_id, flush_leftover=True
        )

    assert len(batches) == 2
    assert {s.name for s in batches[0]} == {"inner", "db"}
    assert {s.name for s in batches[1]} == {"outer"}
    exported_ids = [s.context.span_id for batch in batches for s in batch]
    assert len(exported_ids) == len(set(exported_ids)), "no span exported twice"
    assert processor._buffers.get(trace_id) is None
    processor.shutdown()


def test_nested_and_outer_finalize_separately_when_trace_goes_idle() -> None:
    """When the whole TraceID goes idle with both roots buffered, nested extracts alone."""
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, max_regular_traces=0, completion_holdback_millis=0
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("outer", kind=SpanKind.SERVER):
        with tracer.start_as_current_span("inner", kind=SpanKind.SERVER):
            with tracer.start_as_current_span("db"):
                pass

    provider.force_flush()
    roots = [
        span
        for span in exporter.spans
        if (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT)
    ]
    assert {r.name for r in roots} == {"outer", "inner"}
    assert len(exporter.spans) == 3
    by_name = {span.name: span for span in exporter.spans}
    assert (
        by_name["inner"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "inner"
    )
    assert (
        by_name["db"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "inner"
    )
    assert (
        by_name["outer"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "outer"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]
