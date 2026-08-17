"""Transaction start vs inherit decisions and export-time name stamping.

**on_start (track only):** decide whether this span starts a new local
transaction or inherits. Set ``cgx.transaction.root`` for starters. Record
membership in a side table. Do **not** freeze ``cgx.transaction`` from the
early span name (frameworks may ``update_name`` later, e.g. ``GET`` →
``GET /myroute``).

**export finalize:** resolve ``override_name ?? root.final_name`` and stamp
``cgx.transaction`` onto every span in the completed local-transaction batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_attributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.trace import SpanKind, get_current_span
from opentelemetry.util.types import AttributeValue


@dataclass
class TransactionMembership:
    """Side-table entry for a live/ended span in a local transaction."""

    root_span_id: int
    is_root: bool
    override_name: Optional[str] = None


def starts_new_transaction(
    *,
    span_kind: SpanKind,
    parent_context: Optional[Context],
    parent_has_local_transaction: bool,
) -> bool:
    """Return True when this span opens a new local transaction.

    Starts when there is no parent local transaction, the parent is remote, or
    the span kind is SERVER/CONSUMER.
    """
    parent_span = get_current_span(parent_context)
    parent_sc = parent_span.get_span_context() if parent_span is not None else None
    parent_is_remote = bool(
        parent_sc is not None and parent_sc.is_valid and parent_sc.is_remote
    )
    return (
        not parent_has_local_transaction
        or parent_is_remote
        or span_kind in (SpanKind.SERVER, SpanKind.CONSUMER)
    )


def parent_has_transaction_attrs(parent_span: Optional[object]) -> bool:
    """True when parent already carries local-txn attrs (sampler / override)."""
    attrs = (
        getattr(parent_span, "attributes", None) if parent_span is not None else None
    )
    if not attrs:
        return False
    if _attr_get(attrs, CoralogixAttributes.TRANSACTION_ROOT.value) is True:
        return True
    return _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value) is not None


def preset_transaction_name(span: Span) -> Optional[str]:
    """Return an explicit ``cgx.transaction`` already on the span, if any."""
    attrs = getattr(span, "attributes", None)
    value = _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value)
    return str(value) if value is not None else None


def apply_on_start_root_flag(span: Span, starts: bool) -> None:
    """Set root flag only; never freeze the transaction name from span.name."""
    if starts:
        span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)


def resolve_batch_transaction_name(
    spans: Sequence[ReadableSpan],
    membership: Mapping[int, TransactionMembership],
) -> str:
    """Final transaction name for a completed local-transaction batch."""
    root = _batch_root(spans)
    if root is None or root.context is None:
        return spans[0].name if spans else ""

    member = membership.get(root.context.span_id)
    if member is not None and member.override_name:
        return member.override_name

    attrs = root.attributes or {}
    preset = attrs.get(CoralogixAttributes.TRANSACTION_IDENTIFIER)
    if preset is not None:
        return str(preset)
    return root.name


def stamp_transaction_attributes(
    spans: Sequence[ReadableSpan],
    transaction_name: str,
) -> list[ReadableSpan]:
    """Stamp ``cgx.transaction`` (and keep root) on every span in the batch."""
    stamped: list[ReadableSpan] = []
    for span in spans:
        attrs: Dict[str, AttributeValue] = dict(span.attributes or {})
        attrs[CoralogixAttributes.TRANSACTION_IDENTIFIER] = transaction_name
        if attrs.get(CoralogixAttributes.TRANSACTION_ROOT):
            attrs[CoralogixAttributes.TRANSACTION_ROOT] = True
        stamped.append(copy_with_attributes(span, attrs))
    return stamped


def _batch_root(spans: Sequence[ReadableSpan]) -> Optional[ReadableSpan]:
    for span in spans:
        if (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT):
            return span
    return spans[0] if spans else None


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


# Kept for callers/tests that still import the old helper name.
def resolve_transaction(
    *,
    span_name: str,
    span_kind: SpanKind,
    parent_context: Optional[Context],
) -> Tuple[str, bool]:
    """Deprecated shape: returns ``(placeholder_name, starts)``.

    Name is not frozen at start anymore; ``span_name`` is only a placeholder for
    callers that still unpack the old tuple. Prefer ``starts_new_transaction``.
    """
    parent_span = get_current_span(parent_context)
    parent_has = parent_has_transaction_attrs(parent_span)
    starts = starts_new_transaction(
        span_kind=span_kind,
        parent_context=parent_context,
        parent_has_local_transaction=parent_has,
    )
    return (span_name, starts)
