"""Tests for processor env defaults."""

from __future__ import annotations

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_COMPLETION_HOLDBACK_MILLIS,
    DEFAULT_MAX_TXN_TRACE_NODES,
    ENV_MAX_NODES,
    env_int,
    resolve_completion_holdback_millis,
    resolve_max_nodes,
)
from pytest import MonkeyPatch


def test_env_int_accepts_zero_and_rejects_negatives(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_NODES, "0")
    assert env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES) == 0

    monkeypatch.setenv(ENV_MAX_NODES, "-1")
    assert (
        env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES)
        == DEFAULT_MAX_TXN_TRACE_NODES
    )


def test_resolve_max_nodes_constructor_and_env(monkeypatch: MonkeyPatch) -> None:
    assert resolve_max_nodes(12) == 12
    assert resolve_max_nodes(0) == 0
    assert resolve_max_nodes(-3) == DEFAULT_MAX_TXN_TRACE_NODES

    monkeypatch.setenv(ENV_MAX_NODES, "42")
    assert resolve_max_nodes() == 42


def test_resolve_completion_holdback(monkeypatch: MonkeyPatch) -> None:
    assert resolve_completion_holdback_millis(0) == 0
    assert resolve_completion_holdback_millis(-10) == DEFAULT_COMPLETION_HOLDBACK_MILLIS
    monkeypatch.setenv("OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS", "25")
    assert resolve_completion_holdback_millis() == 25
