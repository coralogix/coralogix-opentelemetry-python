"""Exclusive (self) wall-clock time for spans."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _SpanNode:
    span_id: str
    parent_span_id: str
    name: str
    start_ns: int
    end_ns: int
    children: list["_SpanNode"] = field(default_factory=list)

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def compute_self_time_ns(
    span_id: str,
    parent_span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    children: list[tuple[str, str, str, int, int]],
) -> dict[str, int]:
    """Build a tiny tree from flat rows and return self-time by span_id.

    ``children`` rows are ``(span_id, parent_span_id, name, start_ns, end_ns)``
    for every span in the local trace (including the root row's siblings).
    """
    rows = [(span_id, parent_span_id, name, start_ns, end_ns), *children]
    # Deduplicate by span_id (caller may pass full list as children only).
    by_id: dict[str, _SpanNode] = {}
    for sid, pid, n, s, e in rows:
        if sid not in by_id:
            by_id[sid] = _SpanNode(sid, pid, n, s, e)
    for node in by_id.values():
        if node.parent_span_id and node.parent_span_id in by_id:
            by_id[node.parent_span_id].children.append(node)
    return {sid: _self_time(node) for sid, node in by_id.items()}


def self_time_by_span_id(
    spans: list[tuple[str, str, str, int, int]],
) -> dict[str, int]:
    """Compute self-time for a full local trace.

    Each tuple is ``(span_id, parent_span_id, name, start_ns, end_ns)``.
    """
    if not spans:
        return {}
    by_id: dict[str, _SpanNode] = {}
    for sid, pid, name, start_ns, end_ns in spans:
        by_id[sid] = _SpanNode(sid, pid, name, start_ns, end_ns)
    for node in by_id.values():
        if node.parent_span_id and node.parent_span_id in by_id:
            by_id[node.parent_span_id].children.append(node)
    return {sid: _self_time(node) for sid, node in by_id.items()}


def _self_time(span: _SpanNode) -> int:
    duration = span.duration_ns
    if duration == 0 or not span.children:
        return duration
    child_intervals = [
        (child.start_ns, child.end_ns)
        for child in span.children
        if child.end_ns > child.start_ns
    ]
    covered = _covered_duration_ns(span.start_ns, span.end_ns, child_intervals)
    return max(0, duration - covered)


def _covered_duration_ns(
    parent_start: int,
    parent_end: int,
    intervals: list[tuple[int, int]],
) -> int:
    clamped: list[tuple[int, int]] = []
    for start, end in intervals:
        clipped_start = max(start, parent_start)
        clipped_end = min(end, parent_end)
        if clipped_end > clipped_start:
            clamped.append((clipped_start, clipped_end))
    if not clamped:
        return 0
    clamped.sort()
    merged_start, merged_end = clamped[0]
    covered = 0
    for start, end in clamped[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        covered += merged_end - merged_start
        merged_start, merged_end = start, end
    covered += merged_end - merged_start
    return covered
