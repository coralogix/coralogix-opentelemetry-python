"""Shared defaults and env overrides for TransactionSpanProcessor options.

Constructor keyword arguments win over environment variables. When a constructor
option is omitted (``None``), the matching ``OTEL_CX_TRANSACTION_*`` env var is
read; invalid or empty values fall back to the constant default.

Negative constructor or env values fall back to the default. Zero is valid and
has option-specific meaning (see each ``resolve_*`` docstring / README).
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_MAX_TXN_TRACE_NODES = 256
DEFAULT_COMPLETION_HOLDBACK_MILLIS = 100

ENV_MAX_NODES = "OTEL_CX_TRANSACTION_MAX_NODES"
ENV_COMPLETION_HOLDBACK_MILLIS = "OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS"


def env_int(name: str, default: int) -> int:
    """Parse ``name`` as int; return ``default`` when missing, invalid, or negative.

    Zero is valid (e.g. disable holdback / trimming).
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


def _non_negative_or_default(value: Optional[int], default: int) -> int:
    """Resolve an optional constructor int: ``None`` → env handled by caller.

    Negative values fall back to ``default``. Zero is preserved.
    """
    if value is None:
        return default
    if value < 0:
        return default
    return value


def resolve_max_nodes(value: Optional[int] = None) -> int:
    """Max spans kept per completed local trace.

    ``0`` disables trimming (keep every span). Negative → default.
    """
    if value is not None:
        return _non_negative_or_default(value, DEFAULT_MAX_TXN_TRACE_NODES)
    return env_int(ENV_MAX_NODES, DEFAULT_MAX_TXN_TRACE_NODES)


def resolve_completion_holdback_millis(value: Optional[int] = None) -> int:
    """Post-idle delay before finalizing a local trace.

    ``0`` finalizes immediately. Negative → default.
    """
    if value is not None:
        return _non_negative_or_default(value, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
    return env_int(ENV_COMPLETION_HOLDBACK_MILLIS, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
