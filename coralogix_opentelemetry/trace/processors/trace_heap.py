"""Longest-duration node heap for transaction traces.

Keep at most ``max_nodes`` spans, preferring longer durations. Every
``cgx.transaction.root`` is always retained. Dropped parents are re-linked to
the nearest kept ancestor (or become roots).

Uses ``heapq`` as a **min-heap by duration** among non-root candidates: when
full, the head is the shortest kept non-root (easiest to displace by a longer
span). There is no hand-rolled sift; ``heapq`` owns heap maintenance.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_MAX_TXN_TRACE_NODES,
)
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_parent
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, format_span_id

__all__ = [
    "DEFAULT_MAX_TXN_TRACE_NODES",
    "reparent_to_kept_ancestors",
    "select_slowest_spans",
    "span_duration_ns",
]


def span_duration_ns(span: ReadableSpan) -> int:
    if span.start_time is None or span.end_time is None:
        return 0
    return max(0, span.end_time - span.start_time)


def select_slowest_spans(
    spans: Sequence[ReadableSpan],
    *,
    max_nodes: int = DEFAULT_MAX_TXN_TRACE_NODES,
    root_span_ids: Optional[Sequence[str]] = None,
) -> List[ReadableSpan]:
    """Keep at most ``max_nodes`` spans, preferring longer durations.

    Every protected root span id is never evicted. If there are more protected
    roots than ``max_nodes``, all roots are kept. Remaining slots are filled
    with a duration min-heap.

    ``max_nodes <= 0`` disables trimming and returns every span (caller should
    normally pass a validated non-negative value; ``0`` means unlimited).

    ``root_span_ids`` must be a sequence of span-id hex strings (never a bare
    string — a string would be iterated as characters).
    """
    if max_nodes <= 0 or len(spans) <= max_nodes:
        return list(spans)

    protect: Set[str] = set(root_span_ids or ())

    roots: List[ReadableSpan] = []
    others: List[ReadableSpan] = []
    for span in spans:
        if span.context is None:
            continue
        span_id = format_span_id(span.context.span_id)
        if span_id in protect:
            roots.append(span)
        else:
            others.append(span)

    slots = max(0, max_nodes - len(roots))
    if slots == 0:
        return reparent_to_kept_ancestors(_order_kept(spans, roots), all_spans=spans)

    # Min-heap of (duration, tie_break, span). Head = shortest kept non-root.
    shortest_first: List[Tuple[int, int, ReadableSpan]] = []
    for index, span in enumerate(others):
        duration = span_duration_ns(span)
        item = (duration, index, span)
        if len(shortest_first) < slots:
            shortest_first.append(item)
            if len(shortest_first) == slots:
                heapq.heapify(shortest_first)
            continue
        if duration > shortest_first[0][0]:
            heapq.heapreplace(shortest_first, item)

    kept = [span for _duration, _index, span in shortest_first]
    kept.extend(roots)
    return reparent_to_kept_ancestors(_order_kept(spans, kept), all_spans=spans)


def _order_kept(
    original: Sequence[ReadableSpan], kept: Sequence[ReadableSpan]
) -> List[ReadableSpan]:
    kept_ids = {
        format_span_id(span.context.span_id)
        for span in kept
        if span.context is not None
    }
    return [
        span
        for span in original
        if span.context is not None and format_span_id(span.context.span_id) in kept_ids
    ]


def reparent_to_kept_ancestors(
    kept: Sequence[ReadableSpan],
    *,
    all_spans: Sequence[ReadableSpan],
) -> List[ReadableSpan]:
    """Point each kept span at the nearest kept ancestor.

    Parents outside ``all_spans`` (remote parents, or outer local transactions
    not in this batch) are preserved — only climb past ancestors known to have
    been trimmed from this batch.
    """
    by_id: Dict[str, ReadableSpan] = {
        format_span_id(span.context.span_id): span
        for span in all_spans
        if span.context is not None
    }
    kept_ids: Set[str] = {
        format_span_id(span.context.span_id)
        for span in kept
        if span.context is not None
    }

    result: List[ReadableSpan] = []
    for span in kept:
        if span.context is None:
            continue
        new_parent = _nearest_kept_parent_context(span, by_id=by_id, kept_ids=kept_ids)
        if new_parent is span.parent:
            result.append(span)
        else:
            result.append(copy_with_parent(span, new_parent))
    return result


def _nearest_kept_parent_context(
    span: ReadableSpan,
    *,
    by_id: Dict[str, ReadableSpan],
    kept_ids: Set[str],
) -> Optional[SpanContext]:
    parent = span.parent
    while parent is not None and parent.is_valid:
        parent_id = format_span_id(parent.span_id)
        if parent_id in kept_ids:
            kept_parent = by_id.get(parent_id)
            return kept_parent.context if kept_parent is not None else parent
        ancestor = by_id.get(parent_id)
        if ancestor is None:
            # Parent is outside this local batch (e.g. remote / outer txn).
            return parent
        parent = ancestor.parent
    return None
