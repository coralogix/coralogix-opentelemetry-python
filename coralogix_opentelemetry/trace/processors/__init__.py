from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_COMPLETION_HOLDBACK_MILLIS,
    DEFAULT_MAX_TXN_TRACE_NODES,
)
from coralogix_opentelemetry.trace.processors.start_new_transaction import (
    start_new_transaction,
)
from coralogix_opentelemetry.trace.processors.transaction_finalize import (
    METRIC_SELF_DURATION,
    SELF_DURATION_ATTRIBUTE,
)
from coralogix_opentelemetry.trace.processors.transaction_span_processor import (
    TransactionSpanProcessor,
)

DEFAULT_MAX_NODES = DEFAULT_MAX_TXN_TRACE_NODES

__all__ = [
    "DEFAULT_COMPLETION_HOLDBACK_MILLIS",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_TXN_TRACE_NODES",
    "METRIC_SELF_DURATION",
    "SELF_DURATION_ATTRIBUTE",
    "TransactionSpanProcessor",
    "start_new_transaction",
]
