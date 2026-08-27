"""Harvest heap: keep only the slowest completed traces.

During a harvest window, completed local traces compete by root duration;
only the winners are exported when the harvest flushes (default capacity 1).

Implementation note: uses ``heapq`` as a **min-heap by duration** (index 0 /
head = shortest kept trace = easiest to displace when a slower candidate
arrives). There is no hand-rolled sift-up/sift-down; Python's ``heapq`` owns
those operations. Capacity ``0`` means “keep nothing” (caller exports another
way).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Sequence

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_HARVEST_PERIOD_MILLIS,
    DEFAULT_MAX_REGULAR_TRACES,
)
from opentelemetry.sdk.trace import ReadableSpan

__all__ = [
    "DEFAULT_HARVEST_PERIOD_MILLIS",
    "DEFAULT_MAX_REGULAR_TRACES",
    "HarvestTrace",
    "RegularTraceHeap",
    "harvest_stub_spans",
    "root_duration_ns",
]


@dataclass(order=True)
class HarvestTrace:
    """One completed, already-trimmed local trace competing for harvest export."""

    duration_ns: int
    spans: List[ReadableSpan] = field(compare=False)


class RegularTraceHeap:
    """Min-heap of harvest traces by duration; capacity = max regular traces.

    Head (``_heap[0]``) is the shortest kept winner — the first to be displaced
    when a longer completed local trace arrives.
    """

    def __init__(self, max_traces: int = DEFAULT_MAX_REGULAR_TRACES) -> None:
        if max_traces < 0:
            raise ValueError("max_traces must be >= 0")
        self._max_traces = max_traces
        # Min-heap: shortest duration at head.
        self._shortest_first: List[HarvestTrace] = []

    @property
    def max_traces(self) -> int:
        return self._max_traces

    def __len__(self) -> int:
        return len(self._shortest_first)

    def is_keeper(self, duration_ns: int) -> bool:
        if self._max_traces <= 0:
            return False
        if len(self._shortest_first) < self._max_traces:
            return True
        return duration_ns >= self._shortest_first[0].duration_ns

    def witness(self, trace: HarvestTrace) -> List[ReadableSpan]:
        """Offer a completed trace. Returns root stubs for any loser (reject/displace)."""
        if self._max_traces <= 0:
            return []
        if len(self._shortest_first) < self._max_traces:
            heapq.heappush(self._shortest_first, trace)
            return []
        if trace.duration_ns <= self._shortest_first[0].duration_ns:
            return harvest_stub_spans(trace.spans)
        displaced = heapq.heapreplace(self._shortest_first, trace)
        return harvest_stub_spans(displaced.spans)

    def drain(self) -> List[HarvestTrace]:
        """Remove and return all kept traces (order not significant)."""
        traces = list(self._shortest_first)
        self._shortest_first.clear()
        return traces

    def restore(self, traces: Sequence[HarvestTrace]) -> List[ReadableSpan]:
        """Re-admit drained winners with capacity-aware eviction.

        Returns root stubs for any trace that loses to the current heap (same
        policy as ``witness``), so restore cannot grow the heap past
        ``max_regular_traces``.
        """
        stubs: List[ReadableSpan] = []
        for trace in traces:
            stubs.extend(self.witness(trace))
        return stubs


def harvest_stub_spans(spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
    """Root-only spans for APM presence when a completed tree loses harvest."""
    if not spans:
        return []
    stubs = [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(CoralogixAttributes.TRANSACTION_ROOT) is True
    ]
    if stubs:
        return stubs
    best = spans[0]
    best_dur = 0
    if best.start_time is not None and best.end_time is not None:
        best_dur = max(0, best.end_time - best.start_time)
    for span in spans[1:]:
        if span.start_time is None or span.end_time is None:
            continue
        duration = max(0, span.end_time - span.start_time)
        if duration > best_dur:
            best = span
            best_dur = duration
    return [best]


def root_duration_ns(spans: Sequence[ReadableSpan]) -> int:
    """Max duration among cgx.transaction.root spans, else max span duration."""
    max_root_duration = 0
    found_root = False
    max_duration = 0
    for span in spans:
        if span.start_time is None or span.end_time is None:
            continue
        duration = max(0, span.end_time - span.start_time)
        max_duration = max(max_duration, duration)
        attrs = span.attributes or {}
        if attrs.get(CoralogixAttributes.TRANSACTION_ROOT):
            found_root = True
            max_root_duration = max(max_root_duration, duration)
    if found_root:
        return max_root_duration
    return max_duration
