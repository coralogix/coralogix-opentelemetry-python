"""ReadableSpan copy helpers used when mutating immutable ended spans."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import BoundedAttributes, ReadableSpan
from opentelemetry.trace import SpanContext, Status, StatusCode
from opentelemetry.util.types import AttributeValue


def _passthrough_attributes(span: ReadableSpan) -> Any:
    # Prefer the SDK's BoundedAttributes so dropped_attributes survives the copy.
    internal = getattr(span, "_attributes", None)
    if internal is not None:
        return internal
    return span.attributes


def _passthrough_events(span: ReadableSpan) -> Any:
    # span.events returns a tuple; _events keeps BoundedList.dropped.
    internal = getattr(span, "_events", None)
    if internal is not None:
        return internal
    return span.events


def _passthrough_links(span: ReadableSpan) -> Any:
    internal = getattr(span, "_links", None)
    if internal is not None:
        return internal
    return span.links


def _attributes_preserving_dropped(
    span: ReadableSpan,
    attributes: Mapping[str, AttributeValue],
    maxlen: Optional[int] = None,
) -> Any:
    """Rebuild attributes while keeping the original dropped-attribute count."""
    original = getattr(span, "_attributes", None)
    if not isinstance(original, BoundedAttributes):
        return dict(attributes)
    rebuilt = BoundedAttributes(
        maxlen=original.maxlen if maxlen is None else maxlen,
        attributes=None,
        immutable=False,
        max_value_len=original.max_value_len,
    )
    rebuilt.dropped = original.dropped
    for key, value in attributes.items():
        rebuilt[key] = value
    rebuilt._immutable = True
    return rebuilt


def copy_with_parent(span: ReadableSpan, parent: Optional[SpanContext]) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=_passthrough_attributes(span),
        events=_passthrough_events(span),
        links=_passthrough_links(span),
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def copy_with_attributes(
    span: ReadableSpan,
    attributes: Mapping[str, AttributeValue],
    *,
    max_attributes: Optional[int] = None,
) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=_attributes_preserving_dropped(span, attributes, max_attributes),
        events=_passthrough_events(span),
        links=_passthrough_links(span),
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
