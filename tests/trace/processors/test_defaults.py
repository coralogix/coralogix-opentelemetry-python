"""Tests for processor env defaults."""

from __future__ import annotations

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_COMPLETION_HOLDBACK_MILLIS,
    env_int,
    resolve_completion_holdback_millis,
)
from pytest import MonkeyPatch


def test_env_int_accepts_zero_and_rejects_negatives(monkeypatch: MonkeyPatch) -> None:
    name = "TEST_INT"
    monkeypatch.setenv(name, "0")
    assert env_int(name, DEFAULT_COMPLETION_HOLDBACK_MILLIS) == 0

    monkeypatch.setenv(name, "-1")
    assert (
        env_int(name, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
        == DEFAULT_COMPLETION_HOLDBACK_MILLIS
    )


def test_resolve_completion_holdback(monkeypatch: MonkeyPatch) -> None:
    assert resolve_completion_holdback_millis(0) == 0
    assert resolve_completion_holdback_millis(-10) == DEFAULT_COMPLETION_HOLDBACK_MILLIS
    monkeypatch.setenv("OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS", "25")
    assert resolve_completion_holdback_millis() == 25
