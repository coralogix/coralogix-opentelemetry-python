"""Tests for the regular-trace harvest heap."""

from __future__ import annotations

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.harvest import (
    HarvestTrace,
    RegularTraceHeap,
    harvest_stub_spans,
    root_duration_ns,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags


def _span(
    name: str, *, span_id: int, start_ns: int, end_ns: int, root: bool = False
) -> ReadableSpan:
    attrs = {CoralogixAttributes.TRANSACTION_ROOT.value: True} if root else {}
    return ReadableSpan(
        name=name,
        context=SpanContext(
            trace_id=1,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        ),
        parent=None,
        resource=Resource.create({"service.name": "test"}),
        attributes=attrs,
        events=(),
        links=(),
        kind=SpanKind.SERVER,
        status=Status(StatusCode.UNSET),
        start_time=start_ns,
        end_time=end_ns,
    )


def test_heap_keeps_only_slowest_when_capacity_one() -> None:
    heap = RegularTraceHeap(max_traces=1)
    fast = HarvestTrace(
        duration_ns=100,
        spans=[_span("fast", span_id=1, start_ns=0, end_ns=100, root=True)],
    )
    slow = HarvestTrace(
        duration_ns=500,
        spans=[_span("slow", span_id=2, start_ns=0, end_ns=500, root=True)],
    )
    mid = HarvestTrace(
        duration_ns=200,
        spans=[_span("mid", span_id=3, start_ns=0, end_ns=200, root=True)],
    )

    assert heap.witness(fast) == []
    assert [s.name for s in heap.witness(mid)] == ["fast"]
    assert [s.name for s in heap.witness(slow)] == ["mid"]
    winners = heap.drain()
    assert len(winners) == 1
    assert winners[0].duration_ns == 500
    assert winners[0].spans[0].name == "slow"


def test_heap_restore_puts_drained_winners_back() -> None:
    heap = RegularTraceHeap(max_traces=1)
    winner = HarvestTrace(
        duration_ns=500,
        spans=[_span("slow", span_id=1, start_ns=0, end_ns=500, root=True)],
    )
    assert heap.witness(winner) == []
    drained = heap.drain()
    assert len(heap) == 0
    assert heap.restore(drained) == []
    assert len(heap) == 1
    assert heap.drain()[0].spans[0].name == "slow"


def test_heap_restore_respects_capacity_against_newer_candidates() -> None:
    heap = RegularTraceHeap(max_traces=1)
    old = HarvestTrace(
        duration_ns=100,
        spans=[_span("old", span_id=1, start_ns=0, end_ns=100, root=True)],
    )
    newer = HarvestTrace(
        duration_ns=500,
        spans=[_span("newer", span_id=2, start_ns=0, end_ns=500, root=True)],
    )
    assert heap.witness(old) == []
    drained = heap.drain()
    assert heap.witness(newer) == []
    stubs = heap.restore(drained)
    assert [s.name for s in stubs] == ["old"]
    assert len(heap) == 1
    assert heap.drain()[0].spans[0].name == "newer"


def test_heap_rejects_faster_than_current_winner() -> None:
    heap = RegularTraceHeap(max_traces=1)
    slow = HarvestTrace(
        duration_ns=500,
        spans=[_span("slow", span_id=1, start_ns=0, end_ns=500, root=True)],
    )
    fast = HarvestTrace(
        duration_ns=50,
        spans=[_span("fast", span_id=2, start_ns=0, end_ns=50, root=True)],
    )
    assert heap.witness(slow) == []
    assert [s.name for s in heap.witness(fast)] == ["fast"]
    assert heap.drain()[0].spans[0].name == "slow"


def test_zero_capacity_never_keeps() -> None:
    heap = RegularTraceHeap(max_traces=0)
    trace = HarvestTrace(
        duration_ns=500,
        spans=[_span("slow", span_id=1, start_ns=0, end_ns=500, root=True)],
    )
    assert heap.witness(trace) == []
    assert heap.drain() == []


def test_harvest_stub_spans_prefers_transaction_root() -> None:
    stubs = harvest_stub_spans(
        [
            _span("child", span_id=2, start_ns=0, end_ns=999),
            _span("root", span_id=1, start_ns=0, end_ns=100, root=True),
        ]
    )
    assert len(stubs) == 1
    assert stubs[0].name == "root"


def test_root_duration_prefers_transaction_root() -> None:
    spans = [
        _span("child", span_id=2, start_ns=0, end_ns=999),
        _span("root", span_id=1, start_ns=0, end_ns=100, root=True),
    ]
    assert root_duration_ns(spans) == 100
