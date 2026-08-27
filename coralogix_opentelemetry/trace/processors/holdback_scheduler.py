"""Single-thread holdback scheduler (replaces per-trace ``threading.Timer``).

High-throughput services can complete many traces per second. Creating a
``threading.Timer`` per completion (and another for nested holdback) scales
threads one-for-one with in-flight holdbacks. This module keeps one daemon
worker and a min-heap of deadlines instead.

Callbacks must return quickly (extract / enqueue only). Do not call exporters
from a deadline callback — that would stall every other holdback.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)

_Callback = Callable[[], None]
_HeapItem = Tuple[float, int, Any, _Callback]


class HoldbackScheduler:
    """Schedule named delayed callbacks on one shared worker thread."""

    def __init__(self, *, name: str = "TransactionSpanProcessor-holdback") -> None:
        self._cv = threading.Condition()
        self._seq = 0
        self._heap: List[_HeapItem] = []
        # key -> seq of the currently armed callback (stale heap entries ignored)
        self._active: Dict[Any, int] = {}
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def schedule(self, key: Any, delay_sec: float, callback: _Callback) -> int:
        """Arm ``callback`` after ``delay_sec``, replacing any prior arm for ``key``.

        Returns a generation token for the armed callback (useful in tests).
        """
        if delay_sec < 0:
            delay_sec = 0.0
        with self._cv:
            if self._stopped:
                return 0
            self._seq += 1
            seq = self._seq
            deadline = time.monotonic() + delay_sec
            self._active[key] = seq
            heapq.heappush(self._heap, (deadline, seq, key, callback))
            # Drop cancelled entries before they accumulate under cancel/reschedule.
            if len(self._heap) > max(64, 4 * max(1, len(self._active))):
                self._compact_locked()
            self._cv.notify()
            return seq

    def cancel(self, key: Any) -> None:
        with self._cv:
            if self._active.pop(key, None) is not None:
                self._cv.notify()

    def cancel_all(self) -> None:
        with self._cv:
            if self._active:
                self._active.clear()
                self._cv.notify()

    def is_armed(self, key: Any) -> bool:
        with self._cv:
            return key in self._active

    def shutdown(self, timeout: Optional[float] = 5.0) -> None:
        with self._cv:
            self._stopped = True
            self._active.clear()
            self._heap.clear()
            self._cv.notify_all()
        self._thread.join(timeout=timeout)

    def _compact_locked(self) -> None:
        live: List[_HeapItem] = [
            item for item in self._heap if self._active.get(item[2]) == item[1]
        ]
        heapq.heapify(live)
        self._heap = live

    def _run(self) -> None:
        while True:
            callback: Optional[_Callback] = None
            with self._cv:
                while not self._stopped and not self._heap:
                    self._cv.wait()
                if self._stopped:
                    return

                while self._heap:
                    deadline, seq, key, cb = self._heap[0]
                    if self._active.get(key) != seq:
                        heapq.heappop(self._heap)
                        continue
                    now = time.monotonic()
                    wait = deadline - now
                    if wait > 0:
                        self._cv.wait(timeout=wait)
                        # Re-evaluate after wake (cancel, reschedule, or shutdown).
                        break
                    heapq.heappop(self._heap)
                    if self._active.pop(key, None) != seq:
                        continue
                    callback = cb
                    break
                else:
                    continue

            if callback is None:
                continue
            try:
                callback()
            except Exception:
                _LOG.exception("HoldbackScheduler callback failed")
