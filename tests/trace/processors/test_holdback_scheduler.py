"""Tests for the shared holdback scheduler."""

from __future__ import annotations

import threading
import time

from coralogix_opentelemetry.trace.processors.holdback_scheduler import HoldbackScheduler


def test_scheduler_fires_once_and_cancel_prevents_fire() -> None:
    scheduler = HoldbackScheduler(name="test-holdback")
    fired: list = []
    lock = threading.Lock()

    def mark(key: str) -> None:
        with lock:
            fired.append(key)

    scheduler.schedule("a", 0.05, lambda: mark("a"))
    scheduler.schedule("b", 0.05, lambda: mark("b"))
    scheduler.cancel("b")
    time.sleep(0.12)
    with lock:
        assert fired == ["a"]
    scheduler.shutdown()


def test_reschedule_replaces_prior_arm() -> None:
    scheduler = HoldbackScheduler(name="test-holdback-reschedule")
    fired: list = []

    scheduler.schedule("k", 0.2, lambda: fired.append("old"))
    time.sleep(0.02)
    scheduler.schedule("k", 0.05, lambda: fired.append("new"))
    time.sleep(0.12)
    assert fired == ["new"]
    scheduler.shutdown()
