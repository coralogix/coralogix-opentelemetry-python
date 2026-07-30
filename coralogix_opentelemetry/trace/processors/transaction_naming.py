"""Transaction naming rules applied by TransactionSpanProcessor.on_start."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, get_current_span
from opentelemetry.util.types import AttributeValue


def resolve_transaction(
    *,
    span_name: str,
    span_kind: SpanKind,
    parent_context: Optional[Context],
) -> Tuple[str, bool]:
    """Return ``(transaction, starts_transaction)``.

    Starts a new local transaction when there is no parent transaction name,
    the parent is remote, or the span kind is SERVER/CONSUMER.
    """
    parent_span = get_current_span(parent_context)
    parent_sc = parent_span.get_span_context() if parent_span is not None else None
    existing = _parent_transaction_name(parent_span)
    parent_is_remote = bool(
        parent_sc is not None and parent_sc.is_valid and parent_sc.is_remote
    )
    starts = (
        existing is None
        or parent_is_remote
        or span_kind in (SpanKind.SERVER, SpanKind.CONSUMER)
    )
    transaction = span_name if starts else (existing or span_name)
    return transaction, starts


def _attr_get(
    attrs: Optional[Mapping[Any, AttributeValue]], key: str
) -> Optional[AttributeValue]:
    if not attrs:
        return None
    value = attrs.get(key)
    if value is None:
        for k, v in attrs.items():
            if str(k) == key:
                return v
    return value


def _parent_transaction_name(parent_span: Optional[object]) -> Optional[str]:
    attrs = (
        getattr(parent_span, "attributes", None) if parent_span is not None else None
    )
    if not attrs:
        return None
    value = _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value)
    return str(value) if value is not None else None
