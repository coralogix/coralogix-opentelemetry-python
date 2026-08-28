"""Shared transaction processor defaults and environment overrides."""

from __future__ import annotations

import os
from typing import Optional

MAX_ENRICHED_SPANS = 256
DEFAULT_COMPLETION_HOLDBACK_MILLIS = 100

ENV_COMPLETION_HOLDBACK_MILLIS = "OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS"


def env_int(name: str, default: int) -> int:
    """Parse ``name`` as int; return ``default`` when missing, invalid, or negative.

    Zero is valid (e.g. disable completion holdback).
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


def resolve_completion_holdback_millis(value: Optional[int] = None) -> int:
    """Post-idle delay before finalizing a local trace.

    ``0`` finalizes immediately. Negative → default.
    """
    if value is not None:
        return _non_negative_or_default(value, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
    return env_int(ENV_COMPLETION_HOLDBACK_MILLIS, DEFAULT_COMPLETION_HOLDBACK_MILLIS)
