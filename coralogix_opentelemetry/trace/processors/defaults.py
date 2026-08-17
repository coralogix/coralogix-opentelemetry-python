"""Shared defaults and env overrides for TransactionSpanProcessor options.

Constructor keyword arguments win over environment variables. When a constructor
option is omitted (``None``), the matching ``OTEL_CX_TRANSACTION_*`` env var is
read; invalid or empty values fall back to the constant default.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_MAX_TXN_TRACE_NODES = 256
DEFAULT_MAX_REGULAR_TRACES = 1
DEFAULT_HARVEST_PERIOD_MILLIS = 60_000
DEFAULT_COMPLETION_HOLDBACK_MILLIS = 100

ENV_MAX_NODES = "OTEL_CX_TRANSACTION_MAX_NODES"
ENV_MAX_REGULAR_TRACES = "OTEL_CX_TRANSACTION_MAX_REGULAR_TRACES"
ENV_HARVEST_PERIOD_MILLIS = "OTEL_CX_TRANSACTION_HARVEST_PERIOD_MILLIS"
ENV_COMPLETION_HOLDBACK_MILLIS = "OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS"


def env_int(name: str, default: int) -> int:
    """Parse ``name`` as int; return ``default`` when missing, invalid, or negative.

    Zero is valid (e.g. disable harvest / holdback).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def resolve_max_nodes(value: Optional[int] = None) -> int:
    if value is not None:
        return value
    return env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES)


def resolve_max_regular_traces(value: Optional[int] = None) -> int:
    if value is not None:
        return value
    return env_int(ENV_MAX_REGULAR_TRACES, DEFAULT_MAX_REGULAR_TRACES)


def resolve_harvest_period_millis(value: Optional[int] = None) -> int:
    if value is not None:
        return value
    return env_int(ENV_HARVEST_PERIOD_MILLIS, DEFAULT_HARVEST_PERIOD_MILLIS)


def resolve_completion_holdback_millis(value: Optional[int] = None) -> int:
    if value is not None:
        return value
    return env_int(ENV_COMPLETION_HOLDBACK_MILLIS, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
