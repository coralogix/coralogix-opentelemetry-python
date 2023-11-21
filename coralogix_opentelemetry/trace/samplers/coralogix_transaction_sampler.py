import logging
from typing import Optional, Sequence

from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, ALWAYS_ON
from opentelemetry.trace import (
    SpanKind,
    Link,
    TraceState,
    get_current_span,
    SpanContext,
)
from opentelemetry.util.types import Attributes

from coralogix_opentelemetry.trace.common import (
    CoralogixAttributes,
    CoralogixTraceState,
)

logger = logging.getLogger(__name__)


class CoralogixTransactionSampler(Sampler):
    def __init__(self, base_sampler: Optional[Sampler] = None):
        if base_sampler is not None:
            self._base_sampler = base_sampler
            return
        self._base_sampler = base_sampler or ALWAYS_ON

    def should_sample(
        self,
        parent_context: Optional[Context],
        trace_id: int,
        name: str,
        kind: SpanKind = None,  # type: ignore
        attributes: Attributes = None,
        links: Sequence[Link] = None,  # type: ignore
        trace_state: TraceState = None,  # type: ignore
    ) -> SamplingResult:
        result = self._base_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )
        try:
            span_context = CoralogixTransactionSampler._get_span_context(parent_context)
            trace_state = CoralogixTransactionSampler._get_trace_state(
                span_context, trace_state
            )

            # if distributed transaction exists, use it,
            # if not this is the first span and thus the root of the distributed transaction
            distributed_transaction = (
                trace_state.get(CoralogixTraceState.DISTRIBUTED_TRANSACTION_IDENTIFIER)
                or name
            )

            # if span is remote, then start a new transaction, else try to use existing transaction
            transaction = (
                name
                if span_context and span_context.is_remote
                else (
                    trace_state.get(CoralogixTraceState.TRANSACTION_IDENTIFIER) or name
                )
            )

            trace_state = (
                (result.trace_state or TraceState())
                .add(CoralogixTraceState.TRANSACTION_IDENTIFIER, transaction)
                .add(
                    CoralogixTraceState.DISTRIBUTED_TRANSACTION_IDENTIFIER,
                    distributed_transaction,
                )
            )

            new_attributes = dict(result.attributes or {})  # for mypy
            new_attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] = transaction
            new_attributes[
                CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER
            ] = distributed_transaction
            return SamplingResult(result.decision, new_attributes, trace_state)
        except Exception:
            logger.exception(
                "CoralogixTransactionSampler failed, returning original sampler result"
            )
            return result

    @staticmethod
    def _get_trace_state(
        span_context: Optional[SpanContext], trace_state: Optional[TraceState] = None
    ) -> TraceState:
        if trace_state is not None:
            return trace_state
        elif span_context:
            trace_state = span_context.trace_state
        return trace_state or TraceState()

    @staticmethod
    def _get_span_context(parent_context: Optional[Context]) -> Optional[SpanContext]:
        if parent_context:
            current_span = get_current_span(parent_context)
            if current_span:
                return current_span.get_span_context()
        return None

    def get_description(self) -> str:
        return "CoralogixTransactionSampler"
