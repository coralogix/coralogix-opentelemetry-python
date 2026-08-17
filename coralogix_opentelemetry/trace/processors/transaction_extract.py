"""Extract completed local-transaction subtrees from a TraceID buffer."""

from __future__ import annotations

from typing import Dict, List, Set

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from opentelemetry.sdk.trace import ReadableSpan


def extract_completed_local_transactions(
    *,
    buffer: List[ReadableSpan],
    live: Dict[int, int],
    flush_leftover: bool,
) -> tuple[List[List[ReadableSpan]], List[ReadableSpan]]:
    """Return ``(batches, remaining_buffer)``.

    Nested ``cgx.transaction.root`` subtrees finalize when they have no live
    spans left, deepest root first. Leftover spans flush only when
    ``flush_leftover`` and nothing is live.
    """
    if not buffer:
        return [], []

    parent_of: Dict[int, int] = {}
    for span in buffer:
        if span.context is None:
            continue
        if span.parent is not None and span.parent.is_valid:
            parent_of[span.context.span_id] = span.parent.span_id
    for span_id, parent_id in live.items():
        if parent_id:
            parent_of[span_id] = parent_id

    def under_root(span_id: int, root_id: int) -> bool:
        cur = span_id
        seen: Set[int] = set()
        while cur and cur not in seen:
            if cur == root_id:
                return True
            seen.add(cur)
            cur = parent_of.get(cur, 0)
        return False

    def has_live_in_subtree(root_id: int) -> bool:
        if root_id in live:
            return True
        return any(under_root(live_id, root_id) for live_id in live)

    roots = [
        span
        for span in buffer
        if span.context is not None
        and (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT)
    ]

    def root_depth(root_id: int) -> int:
        depth = 0
        cur = root_id
        seen: Set[int] = set()
        while cur and cur not in seen:
            seen.add(cur)
            parent = parent_of.get(cur, 0)
            if not parent:
                break
            depth += 1
            cur = parent
        return depth

    roots.sort(
        key=lambda s: root_depth(s.context.span_id) if s.context else 0,
        reverse=True,
    )

    batches: List[List[ReadableSpan]] = []
    extracted: Set[int] = set()

    for root in roots:
        assert root.context is not None
        root_id = root.context.span_id
        if root_id in extracted or has_live_in_subtree(root_id):
            continue
        subtree = [
            span
            for span in buffer
            if span.context is not None
            and span.context.span_id not in extracted
            and under_root(span.context.span_id, root_id)
        ]
        if not subtree:
            continue
        for span in subtree:
            if span.context is not None:
                extracted.add(span.context.span_id)
        batches.append(subtree)

    remaining: List[ReadableSpan]
    if extracted:
        remaining = [
            span
            for span in buffer
            if span.context is None or span.context.span_id not in extracted
        ]
    else:
        remaining = list(buffer)

    if flush_leftover and not live and remaining:
        batches.append(remaining)
        remaining = []

    return batches, remaining
