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
from opentelemetry.trace import Span


_explicit_names: Dict[Tuple[int, int], str] = {}
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
        return _explicit_names.get(key)


def start_new_transaction(span: Span, name: str) -> Span:
    key = _span_key(span)
    if key is not None:
        with _explicit_names_lock:
            _explicit_names[key] = name
        try:
            weakref.finalize(span, _clear_explicit_name, key)
        except TypeError:
            pass
    span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)
    span.set_attribute(CoralogixAttributes.TRANSACTION_EXPLICIT, True)
    span.set_attribute(CoralogixAttributes.TRANSACTION_IDENTIFIER, name)
    return span
