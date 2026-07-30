"""Override the local transaction name on an in-flight span (processor path)."""

from __future__ import annotations

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from opentelemetry.trace import Span


def start_new_transaction(span: Span, name: str) -> Span:
    span.set_attribute(CoralogixAttributes.TRANSACTION_IDENTIFIER, name)
    span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)
    return span
