"""Override the local transaction name on an in-flight span (processor path).

Sets ``cgx.transaction`` + ``cgx.transaction.root`` and marks the name as an
explicit override (``cgx.transaction.explicit``). Sampler echoes of the start
span name are not treated as overrides, so frameworks can still ``update_name``
before export finalize.
"""

from __future__ import annotations

import threading
import weakref
from typing import Dict, Optional, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.transaction_naming import (
    has_processor_root_marker,
    processor_root_attribute_limit,
)
from opentelemetry.sdk.trace import BoundedAttributes
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue


_explicit_names: Dict[
    Tuple[int, int],
    Tuple[str, Dict[CoralogixAttributes, AttributeValue], Optional[int]],
] = {}
_explicit_names_lock = threading.Lock()


def _span_key(span: object) -> Optional[Tuple[int, int]]:
    context = getattr(span, "context", None)
    if context is None:
        context = getattr(span, "get_span_context", lambda: None)()
    if context is None or not getattr(context, "is_valid", False):
        return None
    return (int(context.trace_id), int(context.span_id))


def _clear_explicit_name(key: Tuple[int, int]) -> None:
    with _explicit_names_lock:
        _explicit_names.pop(key, None)


def explicit_transaction_name(span: object) -> Optional[str]:
    """Return the explicit name retained outside bounded span attributes."""
    key = _span_key(span)
    if key is None:
        return None
    with _explicit_names_lock:
        state = _explicit_names.get(key)
        return state[0] if state is not None else None


def explicit_transaction_previous_attributes(
    span: object,
) -> Dict[CoralogixAttributes, AttributeValue]:
    key = _span_key(span)
    if key is None:
        return {}
    with _explicit_names_lock:
        state = _explicit_names.get(key)
        return dict(state[1]) if state is not None else {}


def explicit_transaction_attribute_limit(span: object) -> Optional[int]:
    key = _span_key(span)
    if key is None:
        return None
    with _explicit_names_lock:
        state = _explicit_names.get(key)
        return state[2] if state is not None else None


def start_new_transaction(span: Span, name: str) -> Span:
    attrs = getattr(span, "attributes", None) or {}
    previous = {
        key: attrs[key]
        for key in (
            CoralogixAttributes.TRANSACTION_IDENTIFIER,
            CoralogixAttributes.TRANSACTION_ROOT,
            CoralogixAttributes.TRANSACTION_EXPLICIT,
        )
        if key in attrs
    }
    if has_processor_root_marker(span):
        previous.pop(CoralogixAttributes.TRANSACTION_ROOT, None)
    key = _span_key(span)
    prior = None
    if key is not None:
        with _explicit_names_lock:
            prior = _explicit_names.get(key)
    if prior is not None:
        previous = prior[1]
        raw_attribute_limit = prior[2]
    else:
        raw_attribute_limit = processor_root_attribute_limit(span)
        bounded = getattr(span, "_attributes", None)
        if isinstance(bounded, BoundedAttributes) and bounded.maxlen is not None:
            if raw_attribute_limit is None:
                raw_attribute_limit = bounded.maxlen
            bounded.maxlen += 3
    if key is not None:
        with _explicit_names_lock:
            _explicit_names[key] = (name, previous, raw_attribute_limit)
        try:
            weakref.finalize(span, _clear_explicit_name, key)
        except TypeError:
            pass
    span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)
    span.set_attribute(CoralogixAttributes.TRANSACTION_EXPLICIT, True)
    span.set_attribute(CoralogixAttributes.TRANSACTION_IDENTIFIER, name)
    return span
