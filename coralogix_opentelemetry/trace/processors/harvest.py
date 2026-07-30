"""Harvest heap: keep only the slowest completed traces.

During a harvest window, completed local traces compete by root duration;
only the winners are exported when the harvest flushes (default capacity 1).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Sequence

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from opentelemetry.sdk.trace import ReadableSpan

# Default harvest capacity and period.
DEFAULT_MAX_REGULAR_TRACES = 1
DEFAULT_HARVEST_PERIOD_MILLIS = 60_000


@dataclass(order=True)
class HarvestTrace:
    """One completed, already-trimmed local trace competing for harvest export."""

    duration_ns: int
    spans: List[ReadableSpan] = field(compare=False)


class RegularTraceHeap:
    """Min-heap of harvest traces by duration; capacity = max regular traces."""

    def __init__(self, max_traces: int = DEFAULT_MAX_REGULAR_TRACES) -> None:
        if max_traces < 0:
            raise ValueError("max_traces must be >= 0")
        self._max_traces = max_traces
        self._heap: List[HarvestTrace] = []

    @property
    def max_traces(self) -> int:
        return self._max_traces

    def __len__(self) -> int:
        return len(self._heap)

    def is_keeper(self, duration_ns: int) -> bool:
        if self._max_traces <= 0:
            return False
        if len(self._heap) < self._max_traces:
            return True
        return duration_ns >= self._heap[0].duration_ns

    def witness(self, trace: HarvestTrace) -> bool:
        """Offer a completed trace. Returns True if it was kept."""
        if self._max_traces <= 0:
            return False
        if len(self._heap) < self._max_traces:
            heapq.heappush(self._heap, trace)
            return True
        if trace.duration_ns <= self._heap[0].duration_ns:
            return False
        heapq.heapreplace(self._heap, trace)
        return True

    def drain(self) -> List[HarvestTrace]:
        """Remove and return all kept traces (order not significant)."""
        traces = list(self._heap)
        self._heap.clear()
        return traces


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
