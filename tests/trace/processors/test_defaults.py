"""Tests for processor env defaults."""

from __future__ import annotations

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_COMPLETION_HOLDBACK_MILLIS,
    DEFAULT_MAX_TRACES,
    DEFAULT_MAX_TRANSACTION_SPANS,
    env_int,
    resolve_completion_holdback_millis,
    resolve_max_traces,
    resolve_max_transaction_spans,
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


def test_resolve_transaction_limits(monkeypatch: MonkeyPatch) -> None:
    assert resolve_max_transaction_spans(0) == 0
    assert resolve_max_traces(-1) == DEFAULT_MAX_TRACES
    monkeypatch.setenv("CORALOGIX_MAX_SPANS_PER_TRACE", "12")
    monkeypatch.setenv("CORALOGIX_MAX_TRANSACTION_TRACES", "34")
    assert resolve_max_transaction_spans() == 12
    assert resolve_max_traces() == 34
    monkeypatch.setenv("CORALOGIX_MAX_SPANS_PER_TRACE", "bad")
    assert resolve_max_transaction_spans() == DEFAULT_MAX_TRANSACTION_SPANS
