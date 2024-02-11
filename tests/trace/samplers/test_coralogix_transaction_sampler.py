from typing import Mapping, Optional, Sequence

import opentelemetry.trace
import pytest
from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.samplers import CoralogixTransactionSampler
from opentelemetry.context.context import Context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    TraceState,
)
from opentelemetry.util.types import Attributes
from pytest_mock import MockerFixture
from opentelemetry import trace

test_parent_context: Optional[Context] = None
test_trace_id: int = 1
test_name: str = "span-name"
test_kind: SpanKind = SpanKind.SERVER
test_attributes: Attributes = {}
test_links: Sequence[Link] = []
test_trace_state: TraceState = TraceState()

should_sample_args = (
    test_parent_context,
    test_trace_id,
    test_name,
    test_kind,
    test_attributes,
    test_links,
    test_trace_state,
)


def mapping_contains(subset: Mapping, superset: Mapping) -> bool:
    return set(subset.items()).issubset(set(superset.items()))


def get_remote_context(context: Context) -> Context:
    span = opentelemetry.trace.get_current_span(context)
    if not span.is_recording():
        return context
    span_context = span.get_span_context()
    new_span_context = SpanContext(
        trace_id=span_context.trace_id,
        span_id=span_context.span_id,
        is_remote=True,
        trace_flags=span_context.trace_flags,
        trace_state=span_context.trace_state,
    )
    return opentelemetry.trace.set_span_in_context(
        NonRecordingSpan(new_span_context), context
    )


NON_SAMPLED_ATTRIBUTE_NAME = "non_sampled"


class TestAttributeSamplingSampler(Sampler):
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
        return SamplingResult(
            decision=Decision.DROP
            if attributes and NON_SAMPLED_ATTRIBUTE_NAME in attributes
            else Decision.RECORD_AND_SAMPLE
        )

    def get_description(self) -> str:
        return "TestAttributeSamplingSampler"


@pytest.mark.parametrize("decision", Decision)
def test_respect_base_sampler_result(mocker: MockerFixture, decision: Decision) -> None:
    base_sampler = mocker.Mock()
    base_sampler.should_sample.return_value = SamplingResult(decision=decision)
    sampler = CoralogixTransactionSampler(base_sampler=base_sampler)
    result = sampler.should_sample(
        test_parent_context,
        test_trace_id,
        test_name,
        test_kind,
        test_attributes,
        test_links,
        test_trace_state,
    )
    assert (
        base_sampler.should_sample.call_count == 1
    ), "internal sampler should have been called once"
    assert (
        base_sampler.should_sample.call_args.args == should_sample_args
    ), "internal sampler should have been called with same args as CoralogixAttributeSampler"
    assert (
        result.decision == decision
    ), "decision from CoralogixTransactionProcessor should be the same as internal sampler"


def test_return_same_attributes_from_base_sampler(mocker: MockerFixture) -> None:
    return_attributes = {"a": "b"}
    base_sampler = mocker.Mock()
    base_sampler.should_sample.return_value = SamplingResult(
        decision=Decision.RECORD_AND_SAMPLE, attributes=return_attributes
    )
    sampler = CoralogixTransactionSampler(base_sampler=base_sampler)
    result = sampler.should_sample(
        test_parent_context,
        test_trace_id,
        test_name,
        test_kind,
        test_attributes,
        test_links,
        test_trace_state,
    )
    assert (
        base_sampler.should_sample.call_count == 1
    ), "internal sampler should have been called once"
    assert (
        base_sampler.should_sample.call_args.args == should_sample_args
    ), "internal sampler should have been called with same args as CoralogixAttributeSampler"
    assert mapping_contains(
        return_attributes, result.attributes
    ), "result attributes must contain all attributes from internal sampler result"


def test_return_same_trace_state_from_base_sampler(mocker: MockerFixture) -> None:
    return_trace_state = TraceState(entries=[("a", "b")])
    base_sampler = mocker.Mock()
    base_sampler.should_sample.return_value = SamplingResult(
        decision=Decision.RECORD_AND_SAMPLE, trace_state=return_trace_state
    )
    sampler = CoralogixTransactionSampler(base_sampler=base_sampler)
    result = sampler.should_sample(
        test_parent_context,
        test_trace_id,
        test_name,
        test_kind,
        test_attributes,
        test_links,
        test_trace_state,
    )
    assert (
        base_sampler.should_sample.call_count == 1
    ), "internal sampler should have been called once"
    assert (
        base_sampler.should_sample.call_args.args == should_sample_args
    ), "internal sampler should have been called with same args as CoralogixAttributeSampler"
    assert mapping_contains(
        return_trace_state, result.trace_state
    ), "result attributes must contain all attributes from internal sampler result"


def test_transaction_attribute_propagated() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    span3 = tracer.start_span("three", context=context)
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span1 must created a transaction attribute"
        assert (
            span2.attributes
            and span2.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span2 must have transaction attribute from parent"
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span3 must have transaction attribute from parent"
    else:
        assert span1.is_recording(), "span1 must be recording"
        assert span2.is_recording(), "span2 must be recording"
        assert span3.is_recording(), "span3 must be recording"
    span3.end()
    span2.end()
    span1.end()


def test_transaction_attribute_propagated_when_not_passing_context() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    with tracer.start_as_current_span("one"):
        span1 = trace.get_current_span()
        with tracer.start_as_current_span("two"):
            span2 = trace.get_current_span()
            with tracer.start_as_current_span("three"):
                span3 = trace.get_current_span()
                if (
                    isinstance(span1, ReadableSpan)
                    and isinstance(span2, ReadableSpan)
                    and isinstance(span3, ReadableSpan)
                ):
                    assert (
                        span1.attributes
                        and span1.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
                        == "one"
                    ), "span1 must created a transaction attribute"
                    assert (
                        span2.attributes
                        and span2.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
                        == "one"
                    ), "span2 must have transaction attribute from parent"
                    assert (
                        span3.attributes
                        and span3.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER]
                        == "one"
                    ), "span3 must have transaction attribute from parent"
                else:
                    assert span1.is_recording(), "span1 must be recording"
                    assert span2.is_recording(), "span2 must be recording"
                    assert span3.is_recording(), "span3 must be recording"


def test_transaction_attribute_propagated_even_when_not_sampling() -> None:
    sampler = CoralogixTransactionSampler(TestAttributeSamplingSampler())
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span(
        "one", context=context, attributes={NON_SAMPLED_ATTRIBUTE_NAME: True}
    )
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span(
        "two", context=context, attributes={NON_SAMPLED_ATTRIBUTE_NAME: True}
    )
    context = opentelemetry.trace.set_span_in_context(span2, context)
    span3 = tracer.start_span("three", context=context)
    if (
        not span1.is_recording()
        and not span2.is_recording()
        and isinstance(span3, ReadableSpan)
    ):
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span3 must have transaction attribute from parent"
    else:
        assert not span1.is_recording(), "span1 must not be recording"
        assert not span2.is_recording(), "span2 must not be recording"
        assert span3.is_recording(), "span3 must be recording"
    span3.end()
    span2.end()
    span1.end()


def test_new_transaction_after_remote_span_context() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    context = get_remote_context(context)
    span3 = tracer.start_span("three", context=context)
    context = opentelemetry.trace.set_span_in_context(span3, context)
    span4 = tracer.start_span("four", context=context)
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
        and isinstance(span4, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span1 must created a transaction attribute"
        assert (
            span2.attributes
            and span2.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "one"
        ), "span2 must have transaction attribute from parent"
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "three"
        ), "span3 must created a transaction attribute"
        assert (
            span4.attributes
            and span4.attributes[CoralogixAttributes.TRANSACTION_IDENTIFIER] == "three"
        ), "span4 must have transaction attribute from parent"
    span4.end()
    span3.end()
    span2.end()
    span1.end()


def test_distributed_transaction_attribute_propagated() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    span3 = tracer.start_span("three", context=context)
    span3.end()
    span2.end()
    span1.end()
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span1 must created a distributed transaction attribute"
        assert (
            span2.attributes
            and span2.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span2 must have distributed transaction attribute from parent"
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span3 must have distributed transaction attribute from parent"
    else:
        assert span1.is_recording(), "span1 must be recording"
        assert span2.is_recording(), "span2 must be recording"
        assert span3.is_recording(), "span3 must be recording"


def test_distributed_transaction_attribute_propagated_even_when_not_sampling() -> None:
    sampler = CoralogixTransactionSampler(TestAttributeSamplingSampler())
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span(
        "one", context=context, attributes={NON_SAMPLED_ATTRIBUTE_NAME: True}
    )
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span(
        "two", context=context, attributes={NON_SAMPLED_ATTRIBUTE_NAME: True}
    )
    context = opentelemetry.trace.set_span_in_context(span2, context)
    span3 = tracer.start_span("three", context=context)
    if (
        not span1.is_recording()
        and not span2.is_recording()
        and isinstance(span3, ReadableSpan)
    ):
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span3 must have distributed transaction attribute from parent"
    else:
        assert not span1.is_recording(), "span1 must not be recording"
        assert not span2.is_recording(), "span2 must not be recording"
        assert span3.is_recording(), "span3 must be recording"
    span3.end()
    span2.end()
    span1.end()


def test_distributed_transaction_is_the_same_after_remote_span_context() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    context = get_remote_context(context)
    span3 = tracer.start_span("three", context=context)
    context = opentelemetry.trace.set_span_in_context(span3, context)
    span4 = tracer.start_span("four", context=context)
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
        and isinstance(span4, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span1 must created a distributed transaction attribute"
        assert (
            span2.attributes
            and span2.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span2 must have distributed transaction attribute from parent"
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span3 must have distributed transaction attribute from parent"
        assert (
            span4.attributes
            and span4.attributes[CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER]
            == "one"
        ), "span4 must have distributed transaction attribute from parent"
    span4.end()
    span3.end()
    span2.end()
    span1.end()


def test_transaction_root_attribute_add_to_creator() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    span3 = tracer.start_span("three", context=context)
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
        ), "span1 must have transaction root attribute"
        assert (
            span2.attributes is None
            or CoralogixAttributes.TRANSACTION_ROOT not in span2.attributes
        ), "span2 must not have transaction root attribute"
        assert (
            span3.attributes is None
            or CoralogixAttributes.TRANSACTION_ROOT not in span3.attributes
        ), "span3 must not have transaction root attribute"
    else:
        assert isinstance(span1, ReadableSpan), "span1 must be instance of Span"
        assert isinstance(span2, ReadableSpan), "span2 must be instance of Span"
        assert isinstance(span3, ReadableSpan), "span3 must be instance of Span"
    span3.end()
    span2.end()
    span1.end()


def test_transaction_root_attribute_add_after_remote() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("two", context=context)
    context = opentelemetry.trace.set_span_in_context(span2, context)
    context = get_remote_context(context)
    span3 = tracer.start_span("three", context=context)
    context = opentelemetry.trace.set_span_in_context(span3, context)
    span4 = tracer.start_span("four", context=context)
    if (
        isinstance(span1, ReadableSpan)
        and isinstance(span2, ReadableSpan)
        and isinstance(span3, ReadableSpan)
        and isinstance(span4, ReadableSpan)
    ):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
        ), "span1 must have transaction root attribute"
        assert (
            span2.attributes is None
            or CoralogixAttributes.TRANSACTION_ROOT not in span2.attributes
        ), "span2 must not have transaction root attribute"
        assert (
            span3.attributes
            and span3.attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
        ), "span3 must have transaction root attribute"
        assert (
            span4.attributes is None
            or CoralogixAttributes.TRANSACTION_ROOT not in span4.attributes
        ), "span4 must not have transaction root attribute"
    span4.end()
    span3.end()
    span2.end()
    span1.end()


def test_transaction_root_span_with_same_name_should_not_be_transaction_root() -> None:
    sampler = CoralogixTransactionSampler()
    tracer_provider = TracerProvider(sampler=sampler)
    tracer = tracer_provider.get_tracer("default")
    context = None
    span1 = tracer.start_span("one", context=context)
    context = opentelemetry.trace.set_span_in_context(span1, context)
    span2 = tracer.start_span("one", context=context)
    if isinstance(span1, ReadableSpan) and isinstance(span2, ReadableSpan):
        assert (
            span1.attributes
            and span1.attributes[CoralogixAttributes.TRANSACTION_ROOT] is True
        ), "span1 must have transaction root attribute"
        assert (
            span2.attributes is None
            or CoralogixAttributes.TRANSACTION_ROOT not in span2.attributes
        ), "span2 must not have transaction root attribute"
    span2.end()
    span1.end()
