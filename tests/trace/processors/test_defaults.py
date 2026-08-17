"""Tests for processor env defaults."""

from __future__ import annotations

from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_MAX_TXN_TRACE_NODES,
    ENV_MAX_NODES,
    env_int,
)


def test_env_int_accepts_zero_and_rejects_negatives(monkeypatch) -> None:
    monkeypatch.setenv(ENV_MAX_NODES, "0")
    assert env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES) == 0

    monkeypatch.setenv(ENV_MAX_NODES, "-1")
    assert env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES) == DEFAULT_MAX_TXN_TRACE_NODES

    monkeypatch.setenv(ENV_MAX_NODES, "-42")
    assert env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES) == DEFAULT_MAX_TXN_TRACE_NODES


def test_env_int_falls_back_on_invalid(monkeypatch) -> None:
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
