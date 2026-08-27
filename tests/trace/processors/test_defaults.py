"""Tests for processor env defaults."""

from __future__ import annotations

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_COMPLETION_HOLDBACK_MILLIS,
    DEFAULT_HARVEST_PERIOD_MILLIS,
    DEFAULT_MAX_REGULAR_TRACES,
    DEFAULT_MAX_TXN_TRACE_NODES,
    ENV_MAX_NODES,
    env_int,
    resolve_completion_holdback_millis,
    resolve_harvest_period_millis,
    resolve_max_nodes,
    resolve_max_regular_traces,
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

    monkeypatch.setenv(ENV_MAX_NODES, "-42")
    assert (
        env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES)
        == DEFAULT_MAX_TXN_TRACE_NODES
    )


def test_env_int_falls_back_on_invalid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_MAX_NODES, raising=False)
    assert env_int(ENV_MAX_NODES, 7) == 7

    monkeypatch.setenv(ENV_MAX_NODES, "")
    assert env_int(ENV_MAX_NODES, 7) == 7

    monkeypatch.setenv(ENV_MAX_NODES, "  ")
    assert env_int(ENV_MAX_NODES, 7) == 7

    monkeypatch.setenv(ENV_MAX_NODES, "abc")
    assert env_int(ENV_MAX_NODES, 7) == 7

    monkeypatch.setenv(ENV_MAX_NODES, "42")
    assert env_int(ENV_MAX_NODES, 7) == 42


def test_resolve_constructor_negatives_fall_back_to_defaults() -> None:
    assert resolve_max_nodes(-1) == DEFAULT_MAX_TXN_TRACE_NODES
    assert resolve_max_regular_traces(-3) == DEFAULT_MAX_REGULAR_TRACES
    assert resolve_harvest_period_millis(-10) == DEFAULT_HARVEST_PERIOD_MILLIS
    assert resolve_completion_holdback_millis(-5) == DEFAULT_COMPLETION_HOLDBACK_MILLIS


def test_resolve_constructor_zero_is_preserved() -> None:
    assert resolve_max_nodes(0) == 0
    assert resolve_max_regular_traces(0) == 0
    assert resolve_harvest_period_millis(0) == 0
    assert resolve_completion_holdback_millis(0) == 0
