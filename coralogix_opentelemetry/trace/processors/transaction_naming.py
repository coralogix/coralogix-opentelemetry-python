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
import threading
import weakref
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from coralogix_opentelemetry.trace.common import (
    CoralogixAttributes,
    CoralogixTraceState,
)
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_attributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.trace import SpanKind, get_current_span
from opentelemetry.util.types import AttributeValue

_processor_root_markers: set[Tuple[int, int]] = set()
_processor_root_markers_lock = threading.Lock()


def _span_key(span: object) -> Optional[Tuple[int, int]]:
    context = getattr(span, "context", None)
    if context is None:
        context = getattr(span, "get_span_context", lambda: None)()
    if context is None or not getattr(context, "is_valid", False):
        return None
    return (int(context.trace_id), int(context.span_id))


def mark_processor_root(span: Span) -> None:
    key = _span_key(span)
    if key is None:
        return
    with _processor_root_markers_lock:
        _processor_root_markers.add(key)
    try:
        weakref.finalize(span, _processor_root_markers.discard, key)
    except TypeError:
        pass


def has_processor_root_marker(span: Span) -> bool:
    key = _span_key(span)
    if key is None:
        return False
    with _processor_root_markers_lock:
        return key in _processor_root_markers


@dataclass
class TransactionMembership:
    """Side-table entry for a live/ended span in a local transaction."""

    root_span_id: int
    is_root: bool
    override_name: Optional[str] = None
    inherited_name: Optional[str] = None
    # Span name observed at on_start (before framework update_name).
    start_name: Optional[str] = None
    root_flag_added: bool = False
    raw_attribute_limit: Optional[int] = None
    # start_new_transaction() stores its name outside span attributes; retain
    # that provenance through delayed raw export.
    helper_added: bool = False
    helper_previous_attributes: Optional[
        Dict[CoralogixAttributes, AttributeValue]
    ] = None


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
    return (
        _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value) is not None
    )


def preset_transaction_name(span: Span) -> Optional[str]:
    """Return a ``cgx.transaction`` already on the span, if any."""
    attrs = getattr(span, "attributes", None)
    value = _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value)
    return str(value) if value is not None else None


def explicit_transaction_override(span: object) -> bool:
    """True when ``start_new_transaction`` marked this span's name as explicit."""
    attrs = getattr(span, "attributes", None)
    return _attr_get(attrs, CoralogixAttributes.TRANSACTION_EXPLICIT.value) is True


def parent_transaction_from_tracestate(parent_span: Optional[object]) -> Optional[str]:
    """Return ``cgx_transaction`` from parent SpanContext TraceState, if any."""
    if parent_span is None:
        return None
    sc = getattr(parent_span, "get_span_context", lambda: None)()
    if sc is None or not getattr(sc, "is_valid", False):
        return None
    trace_state = getattr(sc, "trace_state", None)
    if trace_state is None:
        return None
    value = trace_state.get(CoralogixTraceState.TRANSACTION_IDENTIFIER)
    if value is None or value == "":
        return None
    return str(value)


def parent_transaction_from_attrs(parent_span: Optional[object]) -> Optional[str]:
    """Return ``cgx.transaction`` from parent span attributes, if any."""
    attrs = (
        getattr(parent_span, "attributes", None) if parent_span is not None else None
    )
    value = _attr_get(attrs, CoralogixAttributes.TRANSACTION_IDENTIFIER.value)
    return str(value) if value is not None else None


def resolve_batch_transaction_name(
    spans: Sequence[ReadableSpan],
    membership: Mapping[int, TransactionMembership],
) -> str:
    """Final transaction name for a completed local-transaction batch.

    Prefer an explicit ``start_new_transaction`` / route-template override.
    Sampler-injected ``cgx.transaction`` that merely echoed the on_start span
    name is ignored so ``update_name`` can still supply the final name.
    """
    root = _batch_root(spans, membership)
    if root is not None and root.context is not None:
        if explicit_transaction_override(root):
            preset = (root.attributes or {}).get(
                CoralogixAttributes.TRANSACTION_IDENTIFIER
            )
            if preset is not None:
                return str(preset)

        member = membership.get(root.context.span_id)
        if member is not None and member.override_name:
            return member.override_name

        attrs = root.attributes or {}
        if attrs.get(CoralogixAttributes.TRANSACTION_ROOT) or (
            member is not None and member.is_root
        ):
            return root.name

    # Leftover without ROOT: prefer TraceState-inherited name on any member.
    for span in spans:
        if span.context is None:
            continue
        member = membership.get(span.context.span_id)
        if member is not None and member.inherited_name:
            return member.inherited_name
        if member is not None and member.override_name:
            return member.override_name

    if root is not None:
        return root.name
    return spans[0].name if spans else ""


def stamp_transaction_attributes(
    spans: Sequence[ReadableSpan],
    transaction_name: str,
    membership: Mapping[int, TransactionMembership],
) -> list[ReadableSpan]:
    """Stamp ``cgx.transaction`` (and keep root) on every span in the batch."""
    stamped: list[ReadableSpan] = []
    for span in spans:
        attrs: Dict[str, AttributeValue] = dict(span.attributes or {})
        is_root = bool(attrs.get(CoralogixAttributes.TRANSACTION_ROOT)) or (
            span.context is not None
            and membership.get(span.context.span_id) is not None
            and membership[span.context.span_id].is_root
        )
        for key in (
            CoralogixAttributes.TRANSACTION_IDENTIFIER,
            CoralogixAttributes.TRANSACTION_ROOT,
            CoralogixAttributes.TRANSACTION_EXPLICIT,
            CoralogixAttributes.TRANSACTION_EXPLICIT.value,
        ):
            attrs.pop(key, None)
        prioritized: Dict[str, AttributeValue] = dict(attrs)
        if is_root:
            prioritized[CoralogixAttributes.TRANSACTION_ROOT] = True
        prioritized[CoralogixAttributes.TRANSACTION_IDENTIFIER] = transaction_name
        stamped.append(copy_with_attributes(span, prioritized))
    return stamped


def _batch_root(
    spans: Sequence[ReadableSpan], membership: Mapping[int, TransactionMembership]
) -> Optional[ReadableSpan]:
    for span in spans:
        if (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT):
            return span
        member = (
            membership.get(span.context.span_id) if span.context is not None else None
        )
        if member is not None and member.is_root:
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
