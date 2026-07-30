"""Longest-duration node heap for transaction traces.

Keep at most ``max_nodes`` spans, preferring longer durations. Every
``cgx.transaction.root`` is always retained. Dropped parents are re-linked to
the nearest kept ancestor (or become roots).
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Status, StatusCode, format_span_id

# Default max spans kept in one local transaction waterfall.
DEFAULT_MAX_TXN_TRACE_NODES = 256


def span_duration_ns(span: ReadableSpan) -> int:
    if span.start_time is None or span.end_time is None:
        return 0
    return max(0, span.end_time - span.start_time)


def select_slowest_spans(
    spans: Sequence[ReadableSpan],
    *,
    max_nodes: int = DEFAULT_MAX_TXN_TRACE_NODES,
    root_span_ids: Optional[Union[str, Sequence[str]]] = None,
    root_span_id: Optional[str] = None,
) -> List[ReadableSpan]:
    """Keep at most ``max_nodes`` spans, preferring longer durations.

    Every protected root span id is never evicted. If there are more protected
    roots than ``max_nodes``, all roots are kept. Remaining slots are filled
    with a duration min-heap.
    """
    if max_nodes <= 0 or len(spans) <= max_nodes:
        return list(spans)

    protect: Set[str] = set()
    if root_span_ids is not None:
        if isinstance(root_span_ids, str):
            protect.add(root_span_ids)
        else:
            protect.update(root_span_ids)
    if root_span_id is not None:
        protect.add(root_span_id)

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

    # Min-heap of (duration, tie_break, span). When full, only longer durations
    # replace the current shortest - min-heap eviction.
    heap: List[Tuple[int, int, ReadableSpan]] = []
    for index, span in enumerate(others):
        duration = span_duration_ns(span)
        item = (duration, index, span)
        if len(heap) < slots:
            heap.append(item)
            if len(heap) == slots:
                heapq.heapify(heap)
            continue
        if duration > heap[0][0]:
            heapq.heapreplace(heap, item)

    kept = [span for _duration, _index, span in heap]
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
        if span.context is not None
        and format_span_id(span.context.span_id) in kept_ids
    ]


def reparent_to_kept_ancestors(
    kept: Sequence[ReadableSpan],
    *,
    all_spans: Sequence[ReadableSpan],
) -> List[ReadableSpan]:
    """Point each kept span at the nearest kept ancestor (or no parent)."""
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
            result.append(_copy_with_parent(span, new_parent))
    return result


def _nearest_kept_parent_context(
    span: ReadableSpan,
    *,
    by_id: Dict[str, ReadableSpan],
    kept_ids: Set[str],
):
    parent = span.parent
    while parent is not None and parent.is_valid:
        parent_id = format_span_id(parent.span_id)
        if parent_id in kept_ids:
            kept_parent = by_id.get(parent_id)
            return kept_parent.context if kept_parent is not None else parent
        ancestor = by_id.get(parent_id)
        if ancestor is None:
            break
        parent = ancestor.parent
    return None


def _copy_with_parent(span: ReadableSpan, parent) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=dict(span.attributes or {}),
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
