"""Tests for exclusive self-duration and TransactionSpanProcessor."""

from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from typing import Sequence

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors import (
    METRIC_SELF_DURATION,
    SELF_DURATION_ATTRIBUTE,
    TransactionSpanProcessor,
    start_new_transaction,
)
from coralogix_opentelemetry.trace.processors.self_duration import (
    self_duration_by_span_id,
)
from coralogix_opentelemetry.trace.processors.transaction_naming import (
    TransactionMembership,
)
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind
from pytest import MonkeyPatch


class ListSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _self_duration_metric_span_names(reader: InMemoryMetricReader) -> set[str]:
    names: set[str] = set()
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return names
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != METRIC_SELF_DURATION:
                    continue
                for point in metric.data.data_points:
                    name = dict(point.attributes).get("span.name")
                    if isinstance(name, str):
                        names.add(name)
    return names


def test_self_duration_nested_fixture() -> None:
    spans = [
        ("1", "", "root", 0, 100),
        ("2", "1", "child", 20, 80),
    ]
    assert self_duration_by_span_id(spans) == {"1": 40, "2": 60}


def test_processor_tags_and_self_duration_without_sampler() -> None:
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
    assert SELF_DURATION_ATTRIBUTE in root.attributes
    assert child.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "GET /orders"
    assert CoralogixAttributes.TRANSACTION_ROOT not in (child.attributes or {})
    assert SELF_DURATION_ATTRIBUTE in child.attributes

    names = []
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.append(metric.name)
    assert METRIC_SELF_DURATION in names

    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_inherits_transaction_from_parent_tracestate_when_parent_has_no_attributes() -> (
    None
):
    from opentelemetry.trace import SpanContext, TraceFlags, set_span_in_context
    from opentelemetry.trace.span import NonRecordingSpan, TraceState

    from coralogix_opentelemetry.trace.common import CoralogixTraceState

    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    parent_sc = SpanContext(
        trace_id=0x1,
        span_id=0x1,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
        trace_state=TraceState(
            [(CoralogixTraceState.TRANSACTION_IDENTIFIER, "from-tracestate")]
        ),
    )
    parent_ctx = set_span_in_context(NonRecordingSpan(parent_sc))
    with tracer.start_as_current_span(
        "internal-child", kind=SpanKind.INTERNAL, context=parent_ctx
    ):
        pass

    provider.force_flush()
    assert len(exporter.spans) == 1
    child = exporter.spans[0]
    assert (
        child.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "from-tracestate"
    )
    assert CoralogixAttributes.TRANSACTION_ROOT not in (child.attributes or {})
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_transaction_name_uses_final_root_name_after_update_name() -> None:
    """Express-style rename: early name must not freeze cgx.transaction."""
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("GET", kind=SpanKind.SERVER)
    with trace.use_span(root, end_on_exit=False):
        root.update_name("GET /myroute")
        with tracer.start_as_current_span("handler"):
            pass
    root.end()
    provider.force_flush()

    by_name = {span.name: span for span in exporter.spans}
    assert (
        by_name["GET /myroute"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /myroute"
    )
    assert (
        by_name["handler"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /myroute"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_start_new_transaction_override_wins_over_span_name() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("raw-name", kind=SpanKind.INTERNAL)
    start_new_transaction(root, "fulfill")
    with trace.use_span(root, end_on_exit=False):
        with tracer.start_as_current_span("child"):
            pass
    root.end()
    provider.force_flush()

    by_name = {span.name: span for span in exporter.spans}
    assert (
        by_name["raw-name"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "fulfill"
    )
    assert (
        by_name["child"].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "fulfill"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_sampler_echo_does_not_block_update_name() -> None:
    """Legacy sampler copies start name into cgx.transaction; update_name must still win."""
    from coralogix_opentelemetry.trace.samplers import CoralogixTransactionSampler

    exporter = ListSpanExporter()
    provider = TracerProvider(sampler=CoralogixTransactionSampler())
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("GET", kind=SpanKind.SERVER)
    root.update_name("GET /myroute")
    root.end()
    provider.force_flush()

    assert len(exporter.spans) == 1
    assert (
        exporter.spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /myroute"
    )
    assert CoralogixAttributes.TRANSACTION_EXPLICIT not in (
        exporter.spans[0].attributes or {}
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_start_new_transaction_equal_name_survives_rename() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("flow", kind=SpanKind.INTERNAL)
    start_new_transaction(root, "flow")
    root.update_name("flow-renamed")
    root.end()
    provider.force_flush()

    assert exporter.spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == (
        "flow"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_nested_server_with_sampler_does_not_inherit_outer_override() -> None:
    """Sampler copies outer txn onto nested SERVER; nested must keep its own name."""
    from coralogix_opentelemetry.trace.samplers import CoralogixTransactionSampler

    exporter = ListSpanExporter()
    provider = TracerProvider(sampler=CoralogixTransactionSampler())
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    outer = tracer.start_span("outer", kind=SpanKind.SERVER)
    with trace.use_span(outer, end_on_exit=False):
        with tracer.start_as_current_span("inner", kind=SpanKind.SERVER):
            pass
        provider.force_flush()
        assert exporter.spans == []

    outer.end()
    provider.force_flush()
    inner_spans = [span for span in exporter.spans if span.name == "inner"]
    assert len(inner_spans) == 1
    assert (
        inner_spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "inner"
    )
    assert inner_spans[0].attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
    outer_spans = [span for span in exporter.spans if span.name == "outer"]
    assert (
        outer_spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "outer"
    )


def test_nested_server_ignores_sampler_name_before_parent_override() -> None:
    from coralogix_opentelemetry.trace.samplers import CoralogixTransactionSampler

    exporter = ListSpanExporter()
    provider = TracerProvider(sampler=CoralogixTransactionSampler())
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    outer = tracer.start_span("outer", kind=SpanKind.SERVER)
    start_new_transaction(outer, "outer-override")
    with trace.use_span(outer, end_on_exit=False):
        with tracer.start_as_current_span("promote-parent"):
            pass
        with tracer.start_as_current_span("nested", kind=SpanKind.SERVER):
            pass
    outer.end()
    provider.force_flush()

    nested = next(span for span in exporter.spans if span.name == "nested")
    assert nested.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "nested"
    provider.shutdown()  # type: ignore[no-untyped-call]
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_nested_server_ignores_sampler_name_from_untracked_parent() -> None:
    from coralogix_opentelemetry.trace.samplers import CoralogixTransactionSampler

    sampler = CoralogixTransactionSampler()
    parent_provider = TracerProvider(sampler=sampler)
    outer = parent_provider.get_tracer("parent").start_span(
        "outer", kind=SpanKind.SERVER
    )

    exporter = ListSpanExporter()
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("child")
    with trace.use_span(outer, end_on_exit=False):
        nested = tracer.start_span("nested", kind=SpanKind.SERVER)
    nested.update_name("nested-final")
    nested.end()
    provider.force_flush()

    assert exporter.spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == (
        "nested-final"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]
    outer.end()
    parent_provider.shutdown()  # type: ignore[no-untyped-call]


def test_257th_ended_span_flushes_trace_raw_and_enables_passthrough() -> None:
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            completion_holdback_millis=0,
            meter_provider=meter_provider,
        )
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("large-root", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    for index in range(257):
        tracer.start_span("large-{}".format(index), context=root_ctx).end()
    assert provider.force_flush() is True
    assert meter_provider.force_flush() is True
    assert len(exporter.spans) == 257
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER not in (span.attributes or {})
        and SELF_DURATION_ATTRIBUTE not in (span.attributes or {})
        for span in exporter.spans
    )
    assert not _self_duration_metric_span_names(reader)

    for index in range(257, 300):
        tracer.start_span("large-{}".format(index), context=root_ctx).end()
    root.end()
    assert provider.force_flush() is True
    assert meter_provider.force_flush() is True
    assert len(exporter.spans) == 301
    assert {span.name for span in exporter.spans} == {
        "large-root",
        *("large-{}".format(index) for index in range(300)),
    }
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER not in (span.attributes or {})
        and SELF_DURATION_ATTRIBUTE not in (span.attributes or {})
        for span in exporter.spans
    )
    assert not _self_duration_metric_span_names(reader)
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_256_ended_spans_remain_buffered_and_are_enriched() -> None:
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            completion_holdback_millis=0,
            meter_provider=meter_provider,
        )
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("small-root", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    for index in range(255):
        tracer.start_span("small-{}".format(index), context=root_ctx).end()
    assert exporter.spans == []
    root.end()
    assert provider.force_flush() is True
    assert meter_provider.force_flush() is True
    assert len(exporter.spans) == 256
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER in (span.attributes or {})
        and SELF_DURATION_ATTRIBUTE in (span.attributes or {})
        for span in exporter.spans
    )
    assert len(_self_duration_metric_span_names(reader)) == 256
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_side_tables_survive_until_last_inflight_batch() -> None:
    """Nested then outer batches on one TraceID must not drop membership early."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    outer = tracer.start_span("outer", kind=SpanKind.SERVER)
    outer_ctx = trace.set_span_in_context(outer)
    nested = tracer.start_span("nested", context=outer_ctx)
    start_new_transaction(nested, "fulfill")
    with trace.use_span(nested, end_on_exit=False):
        tracer.start_span("child").end()
    nested.end()
    provider.force_flush()
    # Outer still open; nested exported. Side tables for this TraceID must remain
    # so outer finalize can still resolve names/intervals.
    with processor._lock:
        outer_context = outer.get_span_context()
        assert (
            processor._membership.get((outer_context.trace_id, outer_context.span_id))
            is not None
        )
    outer.end()
    provider.force_flush()
    fulfill = [
        span
        for span in exporter.spans
        if (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_IDENTIFIER)
        == "fulfill"
    ]
    outer_spans = [span for span in exporter.spans if span.name == "outer"]
    assert fulfill
    assert outer_spans
    assert (
        outer_spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "outer"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_disjoint_child_intervals_use_bounded_covered_aggregate() -> None:
    """Sequential children under a live root must not grow an interval list."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    root = tracer.start_span("root", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    root_context = root.get_span_context()
    root_key = (root_context.trace_id, root_context.span_id)
    for index in range(40):
        child = tracer.start_span("child-{}".format(index), context=root_ctx)
        time.sleep(0.002)
        child.end()
    with processor._lock:
        residual = processor._child_intervals.get(root_key, [])
        covered = processor._child_covered_ns.get(root_key, 0)
        assert len(residual) <= 1
        assert covered > 0
        aggregate = covered + sum(end - start for start, end in residual)
        assert aggregate > 0
    root.end()
    provider.force_flush()
    roots = [span for span in exporter.spans if span.name == "root"]
    assert len(roots) == 1
    self_sec = roots[0].attributes[SELF_DURATION_ATTRIBUTE]
    wall_sec = (roots[0].end_time - roots[0].start_time) / 1_000_000_000.0
    # Exclusive self-duration must subtract folded child coverage (not ≈ wall).
    assert self_sec < wall_sec
    assert self_sec >= 0
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_metrics_use_final_root_name() -> None:
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    processor = TransactionSpanProcessor(
        exporter,
        completion_holdback_millis=0,
        meter_provider=meter_provider,
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    root = tracer.start_span("GET", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    for i in range(8):
        tracer.start_span("c{}".format(i), context=root_ctx).end()
    root.update_name("GET /route")
    root.end()
    provider.force_flush()
    meter_provider.force_flush()

    txn_names = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != METRIC_SELF_DURATION:
                    continue
                for point in metric.data.data_points:
                    txn_names.add(
                        dict(point.attributes).get(
                            CoralogixAttributes.TRANSACTION_IDENTIFIER
                        )
                    )
    assert "GET /route" in txn_names
    assert "GET" not in txn_names
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_preset_template_name_different_from_span_name_is_override() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span(
        "GET /users/123",
        kind=SpanKind.SERVER,
        attributes={CoralogixAttributes.TRANSACTION_IDENTIFIER: "GET /users/:id"},
    )
    root.end()
    provider.force_flush()

    assert (
        exporter.spans[0].attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "GET /users/:id"
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_accept_completed_failure_does_not_escape_on_end(
    monkeypatch: MonkeyPatch,
) -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    def boom(_spans: Sequence[ReadableSpan]) -> None:
        raise RuntimeError("annotate failed")

    monkeypatch.setattr(processor, "_accept_completed_trace", boom)

    span = tracer.start_span("root", kind=SpanKind.SERVER)
    span.end()  # must not raise
    provider.force_flush()
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_holdback_uses_single_scheduler_thread() -> None:
    exporter = ListSpanExporter()
    before = {
        t.ident
        for t in threading.enumerate()
        if t.name == "TransactionSpanProcessor-holdback"
    }
    processors = [
        TransactionSpanProcessor(exporter, completion_holdback_millis=200)
        for _ in range(3)
    ]
    provider = TracerProvider()
    for processor in processors:
        provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    for i in range(20):
        span = tracer.start_span("s{}".format(i), kind=SpanKind.SERVER)
        span.end()

    holdback_threads = {
        t.ident
        for t in threading.enumerate()
        if t.name == "TransactionSpanProcessor-holdback"
    }
    # One scheduler thread per processor instance (not per completed trace).
    assert len(holdback_threads - before) == 3
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_slow_export_does_not_block_holdback_deadlines() -> None:
    """Holdback worker must enqueue finalize work, not wait on exporters."""
    first_export_entered = threading.Event()
    release_first_export = threading.Event()
    dispatched_names: list = []
    dispatched_lock = threading.Lock()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            names = [span.name for span in spans]
            if "first" in names:
                first_export_entered.set()
                assert release_first_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=40)
    original_dispatch = processor._dispatch_accept_completed

    def tracking_dispatch(
        batches: Sequence[Sequence[ReadableSpan]],
        *,
        deadline: float | None = None,
    ) -> bool:
        with dispatched_lock:
            for batch in batches:
                if batch:
                    dispatched_names.append(batch[0].name)
        return original_dispatch(batches, deadline=deadline)

    processor._dispatch_accept_completed = tracking_dispatch  # type: ignore[method-assign]

    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    first = tracer.start_span("first", kind=SpanKind.SERVER)
    first.end()
    assert first_export_entered.wait(timeout=2.0), "first export should start"

    second = tracer.start_span("second", kind=SpanKind.SERVER)
    second.end()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with dispatched_lock:
            if "second" in dispatched_names:
                break
        time.sleep(0.02)
    with dispatched_lock:
        assert (
            "second" in dispatched_names
        ), "second holdback must dispatch while first export is still blocked: " + str(
            dispatched_names
        )

    release_first_export.set()
    provider.force_flush()
    provider.shutdown()  # type: ignore[no-untyped-call]
    assert {span.name for span in exporter.spans} == {"first", "second"}


def test_finalize_queue_defers_overflow_without_blocking(
    monkeypatch: MonkeyPatch,
) -> None:
    """Bounded finalize queue defers overflow without blocking callback threads."""
    monkeypatch.setattr(
        "coralogix_opentelemetry.trace.processors.transaction_span_processor."
        "DEFAULT_MAX_FINALIZE_QUEUE",
        1,
    )
    first_export_entered = threading.Event()
    release_exports = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            if not first_export_entered.is_set():
                first_export_entered.set()
                assert release_exports.wait(timeout=5.0)
            return super().export(spans)

    resource = Resource.create({"service.name": "test"})
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(
        exporter,
        completion_holdback_millis=30,
        meter_provider=meter_provider,
    )
    assert processor._finalize_queue.maxsize == 1
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    tracer.start_span("first", kind=SpanKind.SERVER).end()
    assert first_export_entered.wait(timeout=2.0), "first export should block worker"

    tracer.start_span("queued", kind=SpanKind.SERVER).end()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and processor._finalize_queue.qsize() < 1:
        time.sleep(0.01)
    assert processor._finalize_queue.qsize() == 1

    completed = threading.Event()

    def end_overflow() -> None:
        tracer.start_span("overflow", kind=SpanKind.SERVER).end()
        completed.set()

    thread = threading.Thread(target=end_overflow)
    thread.start()
    assert completed.wait(timeout=2.0), "overflow must not block Span.end"
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not processor._deferred_finalize:
        time.sleep(0.01)
    assert processor._deferred_finalize, "overflow must wait for force_flush"

    release_exports.set()
    thread.join(timeout=2.0)
    assert completed.is_set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and processor._deferred_finalize:
        time.sleep(0.01)
    assert not processor._deferred_finalize, "worker must retry deferred overflow"
    provider.force_flush()
    meter_provider.force_flush()
    span_names = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != METRIC_SELF_DURATION:
                    continue
                for point in metric.data.data_points:
                    span_names.add(dict(point.attributes).get("span.name"))
    assert (
        "overflow" in span_names
    ), "deferred overflow export must still record self-duration metrics"

    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()
    names = {span.name for span in exporter.spans}
    assert "first" in names
    assert "queued" in names
    assert "overflow" in names


def test_force_flush_returns_false_when_finalize_times_out() -> None:
    """force_flush must report failure if finalize work is still pending at deadline."""
    first_export_entered = threading.Event()
    release_export = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            first_export_entered.set()
            assert release_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=20)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    tracer.start_span("blocked", kind=SpanKind.SERVER).end()
    assert first_export_entered.wait(timeout=2.0)

    started = time.monotonic()
    assert processor.force_flush(timeout_millis=50) is False
    assert time.monotonic() - started < 1.0, "force_flush must not wait on exporter"

    release_export.set()
    assert processor.force_flush(timeout_millis=2000) is True
    provider.shutdown()  # type: ignore[no-untyped-call]
    assert {span.name for span in exporter.spans} == {"blocked"}


def test_force_flush_timeout_covers_extracted_holdback_batches() -> None:
    """Extracted holdback batches must use the finalize queue so timeout is honored."""
    export_entered = threading.Event()
    release_export = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            export_entered.set()
            assert release_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=60_000)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    held = tracer.start_span("held", kind=SpanKind.SERVER)
    held.end()
    assert processor._holdback.is_armed(("idle", held.get_span_context().trace_id))

    started = time.monotonic()
    # force_flush extracts the armed holdback batch and must not block forever
    # inside exporter.export when that work is queued to the finalize worker.
    assert processor.force_flush(timeout_millis=80) is False
    assert time.monotonic() - started < 1.5
    assert export_entered.wait(timeout=2.0)

    release_export.set()
    assert processor.force_flush(timeout_millis=2000) is True
    provider.shutdown()  # type: ignore[no-untyped-call]
    assert {span.name for span in exporter.spans} == {"held"}


def test_force_flush_waits_for_queue_capacity_instead_of_dropping(
    monkeypatch: MonkeyPatch,
) -> None:
    """force_flush must not drop-and-succeed when the finalize queue is temporarily full."""
    monkeypatch.setattr(
        "coralogix_opentelemetry.trace.processors.transaction_span_processor."
        "DEFAULT_MAX_FINALIZE_QUEUE",
        1,
    )
    export_entered = threading.Event()
    release_export = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            export_entered.set()
            assert release_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=60_000)
    assert processor._finalize_queue.maxsize == 1
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    for name in ("a", "b", "c"):
        tracer.start_span(name, kind=SpanKind.SERVER).end()

    # force_flush extracts three holdback batches; with queue size 1 and a blocked
    # exporter the second put must wait — and flush must report failure, not
    # abandon-and-return-True.
    started = time.monotonic()
    assert processor.force_flush(timeout_millis=100) is False
    assert time.monotonic() - started < 1.5
    assert export_entered.wait(timeout=2.0)
    assert processor._deferred_finalize, "unqueued batches must be retained for retry"

    release_export.set()
    # A later flush should export the retained batches after the exporter recovers.
    assert processor.force_flush(timeout_millis=2000) is True
    provider.shutdown()  # type: ignore[no-untyped-call]
    names = {span.name for span in exporter.spans}
    assert {"a", "b", "c"} <= names


def test_deferred_finalize_is_bounded(monkeypatch: MonkeyPatch) -> None:
    """Timed-out force_flush retention must not grow without a cap."""
    monkeypatch.setattr(
        "coralogix_opentelemetry.trace.processors.transaction_span_processor."
        "DEFAULT_MAX_FINALIZE_QUEUE",
        1,
    )
    monkeypatch.setattr(
        "coralogix_opentelemetry.trace.processors.transaction_span_processor."
        "DEFAULT_MAX_DEFERRED_FINALIZE",
        1,
    )
    release_export = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            assert release_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=60_000)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    for name in ("a", "b", "c", "d"):
        tracer.start_span(name, kind=SpanKind.SERVER).end()

    assert processor.force_flush(timeout_millis=50) is False
    assert len(processor._deferred_finalize) <= 1

    release_export.set()
    processor.force_flush(timeout_millis=2000)
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_force_flush_export_lock_respects_deadline() -> None:
    """force_flush must not block forever waiting on a held export lock."""
    lock_held = threading.Event()
    release_lock = threading.Event()

    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)

    def hold_lock() -> None:
        with processor._export_lock:
            lock_held.set()
            assert release_lock.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_held.wait(timeout=2.0)

    started = time.monotonic()
    assert processor.force_flush(timeout_millis=80) is False
    assert time.monotonic() - started < 1.5

    release_lock.set()
    holder.join(timeout=2.0)
    assert processor.force_flush(timeout_millis=2000) is True
    processor.shutdown()


def test_zero_holdback_end_does_not_block_on_slow_export() -> None:
    """With holdback=0, Span.end must still dispatch finalize instead of exporting inline."""
    first_export_entered = threading.Event()
    release_export = threading.Event()
    ended = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            first_export_entered.set()
            assert release_export.wait(timeout=5.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    def end_first() -> None:
        tracer.start_span("first", kind=SpanKind.SERVER).end()
        ended.set()

    t = threading.Thread(target=end_first)
    t.start()
    assert first_export_entered.wait(
        timeout=2.0
    ), "export should start on finalize worker"
    assert ended.wait(timeout=1.0), "Span.end must return while export is still blocked"
    t.join(timeout=1.0)

    tracer.start_span("second", kind=SpanKind.SERVER).end()
    release_export.set()
    assert processor.force_flush(timeout_millis=2000) is True
    provider.shutdown()  # type: ignore[no-untyped-call]
    assert {span.name for span in exporter.spans} == {"first", "second"}


def test_inherits_transaction_name_from_parent_attributes() -> None:
    """Untracked parent with cgx.transaction attrs must still supply the txn name."""
    from opentelemetry.trace import SpanContext, TraceFlags, set_span_in_context
    from opentelemetry.trace.span import NonRecordingSpan

    class ParentWithTxnAttrs(NonRecordingSpan):
        def __init__(self, context: SpanContext) -> None:
            super().__init__(context)
            self._attributes = {
                CoralogixAttributes.TRANSACTION_IDENTIFIER: "from-parent-attrs"
            }

        @property
        def attributes(self) -> dict:
            return self._attributes

    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    parent_sc = SpanContext(
        trace_id=0x2,
        span_id=0x2,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )
    parent_ctx = set_span_in_context(ParentWithTxnAttrs(parent_sc))
    with tracer.start_as_current_span(
        "internal-child", kind=SpanKind.INTERNAL, context=parent_ctx
    ):
        pass

    provider.force_flush()
    assert len(exporter.spans) == 1
    child = exporter.spans[0]
    assert (
        child.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        == "from-parent-attrs"
    )
    assert CoralogixAttributes.TRANSACTION_ROOT not in (child.attributes or {})
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_on_start_after_shutdown_does_not_grow_membership() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    provider.shutdown()  # type: ignore[no-untyped-call]
    before = len(processor._membership)
    tracer.start_span("after-shutdown", kind=SpanKind.SERVER).end()
    assert len(processor._membership) == before


def test_on_start_during_shutdown_rejects_new_trace_membership() -> None:
    """While `_stopped` but before exporter shutdown, new TraceIDs must not leak membership."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    processor._stopped = True
    before = len(processor._membership)
    tracer.start_span("during-shutdown", kind=SpanKind.SERVER).end()
    assert len(processor._membership) == before
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_force_flush_does_not_finalize_incomplete_traces() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    # Holdback only — every completed local tree is exported.
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
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

    assert processor.force_flush() is True
    assert {span.name for span in exporter.spans} == {"parent", "child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_shutdown_waits_for_in_flight_spans() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
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
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

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
    time.sleep(0.05)
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
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
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
    time.sleep(0.05)

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


def test_processor_records_self_duration_metrics_for_every_completed_trace() -> None:
    """Self-duration metrics fire for every completed local trace."""
    resource = Resource.create({"service.name": "test"})
    exporter = ListSpanExporter()
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            completion_holdback_millis=0,
            meter_provider=meter_provider,
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("fast", kind=SpanKind.SERVER):
        time.sleep(0.005)
    with tracer.start_as_current_span("slow", kind=SpanKind.SERVER):
        time.sleep(0.04)

    provider.force_flush()
    meter_provider.force_flush()
    span_names = set()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != METRIC_SELF_DURATION:
                    continue
                for point in metric.data.data_points:
                    span_names.add(dict(point.attributes).get("span.name"))
    assert "fast" in span_names
    assert "slow" in span_names
    assert {span.name for span in exporter.spans} >= {"fast", "slow"}
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_processor_exports_completed_traces_immediately() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            completion_holdback_millis=0,
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("now", kind=SpanKind.SERVER):
        pass

    provider.force_flush()
    assert any(span.name == "now" for span in exporter.spans)
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_processor_skips_all_enrichment_for_a_260_span_transaction() -> None:
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            meter_provider=meter_provider,
            completion_holdback_millis=0,
        )
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("root", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    for index in range(259):
        tracer.start_span("child-{}".format(index), context=root_ctx).end()
    root.end()
    provider.force_flush()
    meter_provider.force_flush()

    assert len(exporter.spans) == 260
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER not in (span.attributes or {})
        and SELF_DURATION_ATTRIBUTE not in (span.attributes or {})
        for span in exporter.spans
    )
    assert not _self_duration_metric_span_names(reader)
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_processor_uses_configured_span_limit() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter, completion_holdback_millis=0, max_transaction_spans=2
        )
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root", kind=SpanKind.SERVER) as root:
        root_ctx = trace.set_span_in_context(root)
        tracer.start_span("first", context=root_ctx).end()
        tracer.start_span("second", context=root_ctx).end()

    provider.force_flush()
    assert len(exporter.spans) == 3
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER not in (span.attributes or {})
        and SELF_DURATION_ATTRIBUTE not in (span.attributes or {})
        for span in exporter.spans
    )
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_processor_uses_configured_trace_limit() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0, max_traces=1)
    )
    tracer = provider.get_tracer("test")

    first = tracer.start_span("first", kind=SpanKind.SERVER)
    second = tracer.start_span("second", kind=SpanKind.SERVER)
    second.end()
    first.end()

    provider.force_flush()
    by_name = {span.name: span for span in exporter.spans}
    assert CoralogixAttributes.TRANSACTION_IDENTIFIER in (
        by_name["first"].attributes or {}
    )
    assert CoralogixAttributes.TRANSACTION_IDENTIFIER not in (
        by_name["second"].attributes or {}
    )
    assert SELF_DURATION_ATTRIBUTE not in (by_name["second"].attributes or {})

    tracer.start_span("third", kind=SpanKind.SERVER).end()
    provider.force_flush()
    third = next(span for span in exporter.spans if span.name == "third")
    assert CoralogixAttributes.TRANSACTION_IDENTIFIER in (third.attributes or {})
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_processor_records_self_duration_for_all_130_spans() -> None:
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(
            exporter,
            meter_provider=meter_provider,
            completion_holdback_millis=0,
        )
    )
    tracer = provider.get_tracer("test")

    root = tracer.start_span("root", kind=SpanKind.SERVER)
    root_ctx = trace.set_span_in_context(root)
    for index in range(129):
        tracer.start_span("child-{}".format(index), context=root_ctx).end()
    root.end()
    provider.force_flush()
    meter_provider.force_flush()

    assert len(exporter.spans) == 130
    assert all(
        CoralogixAttributes.TRANSACTION_IDENTIFIER in (span.attributes or {})
        for span in exporter.spans
    )
    assert (
        sum(
            (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT) is True
            for span in exporter.spans
        )
        == 1
    )
    assert all(
        SELF_DURATION_ATTRIBUTE in (span.attributes or {}) for span in exporter.spans
    )
    assert len(_self_duration_metric_span_names(reader)) == 130
    provider.shutdown()  # type: ignore[no-untyped-call]
    meter_provider.shutdown()


def test_shutdown_flushes_pending_completed_trace() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    processor = TransactionSpanProcessor(
        exporter,
        completion_holdback_millis=0,
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root", kind=SpanKind.SERVER):
        pass

    # Give the finalize worker a moment; shutdown must still flush if pending.
    time.sleep(0.05)
    processor.shutdown()
    assert any(span.name == "root" for span in exporter.spans)


def test_shutdown_drains_accepted_queue_before_stopping_exporter(
    monkeypatch: MonkeyPatch,
) -> None:
    export_started = threading.Event()
    release_export = threading.Event()

    class BlockingExporter(ListSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            export_started.set()
            release_export.wait(timeout=2.0)
            return super().export(spans)

    exporter = BlockingExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")
    tracer.start_span("first", kind=SpanKind.SERVER).end()
    assert export_started.wait(timeout=1.0)
    tracer.start_span("second", kind=SpanKind.SERVER).end()
    monkeypatch.setattr(processor, "_wait_for_idle_locked", lambda timeout_sec: None)
    shutdown = threading.Thread(target=processor.shutdown)
    shutdown.start()
    time.sleep(0.05)
    release_export.set()
    shutdown.join(timeout=2.0)
    assert not shutdown.is_alive()
    assert {span.name for span in exporter.spans} == {"first", "second"}


def test_processor_restarts_workers_after_fork() -> None:
    if not hasattr(os, "fork"):
        return
    read_fd, write_fd = os.pipe()
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        tracer.start_span("child", kind=SpanKind.SERVER).end()
        ok = provider.force_flush(timeout_millis=1000)
        os.write(write_fd, b"1" if ok and exporter.spans else b"0")
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    assert os.read(read_fd, 1) == b"1"
    os.close(read_fd)
    assert os.waitpid(pid, 0)[1] == 0
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_at_fork_callback_does_not_retain_shutdown_processor() -> None:
    if not hasattr(os, "register_at_fork"):
        return
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    processor_ref = weakref.ref(processor)
    processor.shutdown()
    del processor
    gc.collect()
    assert processor_ref() is None


def test_processor_server_under_local_parent_starts_new_transaction() -> None:
    exporter = ListSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
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
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    )
    tracer = provider.get_tracer("test")

    outer = tracer.start_span("outer", kind=SpanKind.SERVER)
    with trace.use_span(outer, end_on_exit=False):
        with tracer.start_as_current_span("inner", kind=SpanKind.SERVER):
            with tracer.start_as_current_span("db"):
                time.sleep(0.005)

        provider.force_flush()
        names = {span.name for span in exporter.spans}
        assert names == set(), "nested local transaction must wait for the outer trace"

    outer.end()
    provider.force_flush()
    names = {span.name for span in exporter.spans}
    assert {"outer", "inner", "db"} <= names
    provider.shutdown()  # type: ignore[no-untyped-call]


def test_completion_holdback_keeps_fire_and_forget_child() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=80)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    parent = tracer.start_span("parent", kind=SpanKind.SERVER)
    parent_ctx = trace.set_span_in_context(parent)
    parent.end()
    assert exporter.spans == [], "must not finalize while holdback is open"

    child = tracer.start_span("late-child", context=parent_ctx)
    child.end()
    time.sleep(0.12)
    provider.force_flush()
    assert {span.name for span in exporter.spans} == {"parent", "late-child"}
    provider.shutdown()  # type: ignore[no-untyped-call]


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


def test_folded_child_coverage_keeps_live_overlap_geometry() -> None:
    processor = TransactionSpanProcessor(ListSpanExporter())
    with processor._lock:
        # A child that began at 5 remains live while two siblings finish.
        processor._live_child_starts[(1, 1)] = {4: 5}
        processor._add_child_interval_locked(1, 1, 0, 10)
        processor._add_child_interval_locked(1, 1, 20, 30)
        assert processor._child_covered_ns[(1, 1)] == 5
        assert processor._child_intervals[(1, 1)] == [(5, 10), (20, 30)]

        processor._live_child_starts.pop((1, 1))
        processor._add_child_interval_locked(1, 1, 5, 40)
        assert processor._child_covered_ns[(1, 1)] == 40
        assert (1, 1) not in processor._child_intervals
    processor.shutdown()


def test_folded_child_coverage_clamps_to_ended_parent() -> None:
    processor = TransactionSpanProcessor(ListSpanExporter())
    with processor._lock:
        processor._buffers[1] = [_readable("root", span_id=1, root=True, end=100)]
        processor._add_child_interval_locked(1, 1, 0, 60)
        processor._add_child_interval_locked(1, 1, 80, 200)
        assert processor._child_covered_ns[(1, 1)] == 80
        assert (1, 1) not in processor._child_intervals
    processor.shutdown()


def test_side_tables_do_not_collide_for_same_span_id_in_different_traces() -> None:
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter)
    first = _readable("first", trace_id=1, span_id=1, root=True)
    second = _readable("second", trace_id=2, span_id=1, root=True)
    with processor._lock:
        processor._membership[(1, 1)] = TransactionMembership(
            root_span_id=1, is_root=True, override_name="first-transaction"
        )
        processor._membership[(2, 1)] = TransactionMembership(
            root_span_id=1, is_root=True, override_name="second-transaction"
        )

    processor._accept_completed_trace([first])
    processor._accept_completed_trace([second])

    assert [
        span.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
        for span in exporter.spans
    ] == [
        "first-transaction",
        "second-transaction",
    ]
    processor.shutdown()


def test_stale_cancelled_holdback_does_not_pop_replacement() -> None:
    """Cancelling an idle holdback must not drop a replacement arm for the same TraceID."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=60_000)
    trace_id = 0xABC
    outer = _readable("outer", span_id=1, root=True, end=10)

    with processor._lock:
        processor._buffers[trace_id] = [outer]
        processor._schedule_completion_locked(trace_id)
        stale = processor._pending_completions[trace_id]
        processor._cancel_pending_completion_locked(trace_id)
        processor._schedule_completion_locked(trace_id)
        replacement = processor._pending_completions[trace_id]
        assert replacement != stale
        assert processor._holdback.is_armed(("idle", trace_id))

    time.sleep(0.05)

    with processor._lock:
        assert processor._pending_completions.get(trace_id) == replacement
        assert processor._buffers.get(trace_id) == [outer]
    assert exporter.spans == []

    processor.shutdown()


def test_extract_completed_roots_deepest_first_excludes_extracted() -> None:
    """Nested SERVER root extracts before outer; outer must not re-export nested IDs."""
    exporter = ListSpanExporter()
    processor = TransactionSpanProcessor(exporter, completion_holdback_millis=0)
    trace_id = 0xDEF
    outer = _readable("outer", span_id=1, root=True, start=0, end=100)
    nested = _readable("inner", span_id=2, parent_id=1, root=True, start=10, end=60)
    child = _readable("db", span_id=3, parent_id=2, start=20, end=50)

    with processor._lock:
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
        TransactionSpanProcessor(exporter, completion_holdback_millis=0)
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
