"""Exclusive (self) duration for spans within a local transaction tree.

Self duration is a span's wall duration minus the time covered by its direct
children. Child intervals are clamped to the parent's ``[start, end)`` and
merged so overlapping / concurrent children are not double-counted.

Result unit is nanoseconds; callers that stamp ``cgx.transaction.self_duration``
convert to seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


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


def compute_self_duration_ns(
    span_id: str,
    parent_span_id: str,
    name: str,
    start_ns: int,
    end_ns: int,
    children: list[tuple[str, str, str, int, int]],
) -> dict[str, int]:
    """Build a tiny tree from flat rows and return self-duration by span_id.

    ``children`` rows are ``(span_id, parent_span_id, name, start_ns, end_ns)``
    for every span in the local trace (including the root row's siblings).
    """
    rows = [(span_id, parent_span_id, name, start_ns, end_ns), *children]
    by_id: dict[str, _SpanNode] = {}
    for sid, pid, n, s, e in rows:
        if sid not in by_id:
            by_id[sid] = _SpanNode(sid, pid, n, s, e)
    for node in by_id.values():
        if node.parent_span_id and node.parent_span_id in by_id:
            by_id[node.parent_span_id].children.append(node)
    return {sid: _self_duration(node) for sid, node in by_id.items()}


def self_duration_by_span_id(
    spans: list[tuple[str, str, str, int, int]],
) -> dict[str, int]:
    """Compute exclusive self-duration for a full local trace.

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
    return {sid: _self_duration(node) for sid, node in by_id.items()}


def _self_duration(span: _SpanNode) -> int:
    duration = span.duration_ns
    if duration == 0 or not span.children:
        return duration
    child_intervals = [
        (child.start_ns, child.end_ns)
        for child in span.children
        if child.end_ns > child.start_ns
    ]
    covered = covered_duration_ns(span.start_ns, span.end_ns, child_intervals)
    return max(0, duration - covered)


def clamp_intervals_to_parent(
    parent_start: int,
    parent_end: int,
    intervals: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Clip each interval to ``[parent_start, parent_end)``; drop empties."""
    clamped: List[Tuple[int, int]] = []
    for start, end in intervals:
        clipped_start = max(start, parent_start)
        clipped_end = min(end, parent_end)
        if clipped_end > clipped_start:
            clamped.append((clipped_start, clipped_end))
    return clamped


def merge_interval_duration_ns(intervals: Sequence[Tuple[int, int]]) -> int:
    """Sort once, merge overlaps, return total covered duration in ns."""
    if not intervals:
        return 0
    sorted_intervals = sorted(intervals)
    merged_start, merged_end = sorted_intervals[0]
    covered = 0
    for start, end in sorted_intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        covered += merged_end - merged_start
        merged_start, merged_end = start, end
    covered += merged_end - merged_start
    return covered


def covered_duration_ns(
    parent_start: int,
    parent_end: int,
    intervals: list[tuple[int, int]],
) -> int:
    """Exclusive-child coverage: clamp to parent, then merge overlapping ranges."""
    clamped = clamp_intervals_to_parent(parent_start, parent_end, intervals)
    return merge_interval_duration_ns(clamped)
