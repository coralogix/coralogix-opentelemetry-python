"""Override the local transaction name on an in-flight span (processor path).

Sets ``cgx.transaction`` + ``cgx.transaction.root`` and marks the name as an
explicit override (``cgx.transaction.explicit``). Sampler echoes of the start
span name are not treated as overrides, so frameworks can still ``update_name``
before export finalize.
"""

from __future__ import annotations

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from opentelemetry.trace import Span


def start_new_transaction(span: Span, name: str) -> Span:
    span.set_attribute(CoralogixAttributes.TRANSACTION_IDENTIFIER, name)
    span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)
    span.set_attribute(CoralogixAttributes.TRANSACTION_EXPLICIT, True)
    return span
