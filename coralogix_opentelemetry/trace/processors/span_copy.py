"""ReadableSpan copy helpers used when mutating immutable ended spans."""

from __future__ import annotations

from typing import Mapping, Optional

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, Status, StatusCode
from opentelemetry.util.types import AttributeValue


def copy_with_parent(span: ReadableSpan, parent: Optional[SpanContext]) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=dict(span.attributes or {}),
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def copy_with_attributes(
    span: ReadableSpan, attributes: Mapping[str, AttributeValue]
) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=dict(attributes),
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
