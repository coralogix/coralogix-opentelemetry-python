"""Tests for the slowest-node trace heap."""

from __future__ import annotations

from typing import Dict, Optional

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.trace_heap import select_slowest_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags


def _ctx(trace_id: int, span_id: int) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )


def _span(
    name: str,
    *,
    span_id: int,
    start_ns: int,
    end_ns: int,
    parent_span_id: Optional[int] = None,
    trace_id: int = 1,
    attrs: Optional[Dict] = None,
) -> ReadableSpan:
    parent = None
    if parent_span_id is not None:
        parent = _ctx(trace_id, parent_span_id)
    return ReadableSpan(
        name=name,
        context=_ctx(trace_id, span_id),
        parent=parent,
        resource=Resource.create({"service.name": "test"}),
        attributes=attrs or {},
        events=(),
        links=(),
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.UNSET),
        start_time=start_ns,
        end_time=end_ns,
    )


def test_keeps_all_when_under_max_nodes() -> None:
    spans = [
        _span("root", span_id=1, start_ns=0, end_ns=100),
        _span("a", span_id=2, start_ns=10, end_ns=20, parent_span_id=1),
    ]
    kept = select_slowest_spans(
        spans, max_nodes=256, root_span_ids=["0000000000000001"]
    )
    assert len(kept) == 2


def test_keeps_longest_and_always_keeps_root() -> None:
    # Cap=3: root + 2 longest among children. Short cache/auth drop.
    root = _span(
        "root",
        span_id=1,
        start_ns=0,
        end_ns=200,
        attrs={CoralogixAttributes.TRANSACTION_ROOT.value: True},
    )
    auth = _span("auth", span_id=2, start_ns=1, end_ns=6, parent_span_id=1)  # 5
    cache = _span("cache", span_id=3, start_ns=10, end_ns=12, parent_span_id=1)  # 2
    db = _span("db", span_id=4, start_ns=20, end_ns=60, parent_span_id=1)  # 40
    http = _span("http", span_id=5, start_ns=70, end_ns=150, parent_span_id=1)  # 80
    render = _span(
        "render", span_id=6, start_ns=160, end_ns=170, parent_span_id=1
    )  # 10

    kept = select_slowest_spans(
        [root, auth, cache, db, http, render],
        max_nodes=3,
        root_span_ids=["0000000000000001"],
    )
    names = {span.name for span in kept}
    assert names == {"root", "db", "http"}


def test_same_name_different_durations_are_separate_nodes() -> None:
    root = _span("root", span_id=1, start_ns=0, end_ns=100)
    db_slow = _span(
        "db.select", span_id=2, start_ns=10, end_ns=50, parent_span_id=1
    )  # 40
    db_fast = _span(
        "db.select", span_id=3, start_ns=55, end_ns=58, parent_span_id=1
    )  # 3
    other = _span("other", span_id=4, start_ns=60, end_ns=70, parent_span_id=1)  # 10

    kept = select_slowest_spans(
        [root, db_slow, db_fast, other],
        max_nodes=3,
        root_span_ids=["0000000000000001"],
    )
    names = [span.name for span in kept]
    assert names.count("db.select") == 1  # only the slow one fits with root+other
    assert "other" in names
    assert "root" in names


def test_reparents_when_middle_parent_dropped() -> None:
    # root -> middleware(short) -> db(long). Cap=2 keeps root+db; db parent -> root.
    root = _span("root", span_id=1, start_ns=0, end_ns=100)
    mid = _span("middleware", span_id=2, start_ns=1, end_ns=2, parent_span_id=1)  # 1
    db = _span("db", span_id=3, start_ns=5, end_ns=90, parent_span_id=2)  # 85

    kept = select_slowest_spans(
        [root, mid, db],
        max_nodes=2,
        root_span_ids=["0000000000000001"],
    )
    assert {span.name for span in kept} == {"root", "db"}
    db_kept = next(span for span in kept if span.name == "db")
    assert db_kept.parent is not None
    assert db_kept.parent.span_id == 1


def test_no_root_span_id_uses_all_slots_for_heap() -> None:
    a = _span("a", span_id=1, start_ns=0, end_ns=10)  # 10
    b = _span("b", span_id=2, start_ns=0, end_ns=50)  # 50
    c = _span("c", span_id=3, start_ns=0, end_ns=30)  # 30

    kept = select_slowest_spans([a, b, c], max_nodes=2, root_span_ids=None)
    assert {span.name for span in kept} == {"b", "c"}


def test_zero_max_nodes_keeps_all_spans() -> None:
    spans = [
        _span("a", span_id=1, start_ns=0, end_ns=10),
        _span("b", span_id=2, start_ns=0, end_ns=50),
        _span("c", span_id=3, start_ns=0, end_ns=30),
    ]
    kept = select_slowest_spans(spans, max_nodes=0, root_span_ids=["0000000000000001"])
    assert [span.name for span in kept] == ["a", "b", "c"]


def test_preserves_parent_outside_trimmed_batch() -> None:
    """Remote / outer parents absent from the batch must not become None."""
    # Nested SERVER root (span 2) has parent 0x99 which is not in this batch.
    # Cap=2 keeps root+db; mid is dropped. Root must keep parent 0x99.
    remote_parent = SpanContext(
        trace_id=1,
        span_id=0x99,
        is_remote=True,
        trace_flags=TraceFlags(0x01),
    )
    root = ReadableSpan(
        name="nested-server",
        context=_ctx(1, 2),
        parent=remote_parent,
        resource=Resource.create({"service.name": "test"}),
        attributes={CoralogixAttributes.TRANSACTION_ROOT.value: True},
        events=(),
        links=(),
        kind=SpanKind.SERVER,
        status=Status(StatusCode.UNSET),
        start_time=0,
        end_time=100,
    )
    mid = _span("middleware", span_id=3, start_ns=1, end_ns=2, parent_span_id=2)
    db = _span("db", span_id=4, start_ns=5, end_ns=90, parent_span_id=3)

    kept = select_slowest_spans(
        [root, mid, db],
        max_nodes=2,
        root_span_ids=["0000000000000002"],
    )
    assert {span.name for span in kept} == {"nested-server", "db"}
    root_kept = next(span for span in kept if span.name == "nested-server")
    assert root_kept.parent is not None
    assert root_kept.parent.span_id == 0x99
    assert root_kept.parent.is_remote is True
    db_kept = next(span for span in kept if span.name == "db")
    assert db_kept.parent is not None
    assert db_kept.parent.span_id == 2
