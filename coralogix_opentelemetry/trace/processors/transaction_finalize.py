"""Export-time annotation: transaction name + exclusive self-duration + metrics."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.self_duration import (
    self_duration_by_span_id,
)
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_attributes
from coralogix_opentelemetry.trace.processors.transaction_naming import (
    TransactionMembership,
    resolve_batch_transaction_name,
    stamp_transaction_attributes,
)
from opentelemetry.metrics import Histogram, MeterProvider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import format_span_id
from opentelemetry.util.types import AttributeValue

METRIC_SELF_DURATION = "cgx.transaction.self_duration"
SELF_DURATION_ATTRIBUTE = "cgx.transaction.self_duration"
METRIC_ATTR_SPAN_NAME = "span.name"
_LOG = logging.getLogger(__name__)


def annotate_completed_batch(
    spans: Sequence[ReadableSpan],
    *,
    child_intervals: Dict[int, List[Tuple[int, int]]],
    membership: Dict[int, TransactionMembership],
    self_duration_hist: Histogram,
    transaction_name: Optional[str] = None,
    max_enriched_spans: Optional[int] = None,
) -> List[ReadableSpan]:
    """Enrich a bounded batch, or export a larger batch without enrichment.

    Order matters: transaction attrs are stamped first so metric labels see the
    final ``cgx.transaction`` value (not an early start-time name).

    ``transaction_name`` overrides name resolution (used when recording metrics
    for live-buffer evictions that do not include the transaction root span).
    """
    if max_enriched_spans is not None and len(spans) > max_enriched_spans:
        return [strip_transaction_enrichment(span) for span in spans]

    txn_name = transaction_name or resolve_batch_transaction_name(spans, membership)
    named = stamp_transaction_attributes(spans, txn_name, membership)
    return _annotate_with_self_duration_and_metrics(
        named,
        child_intervals,
        self_duration_hist,
    )


def _annotate_with_self_duration_and_metrics(
    spans: Sequence[ReadableSpan],
    child_intervals: Dict[int, List[Tuple[int, int]]],
    self_duration_hist: Histogram,
) -> List[ReadableSpan]:
    direct_child_intervals: Dict[int, Set[Tuple[int, int]]] = {}
    for span in spans:
        if (
            span.parent is None
            or not span.parent.is_valid
            or span.start_time is None
            or span.end_time is None
        ):
            continue
        direct_child_intervals.setdefault(span.parent.span_id, set()).add(
            (span.start_time, span.end_time)
        )

    rows: List[tuple] = []
    for span in spans:
        if span.context is None or span.start_time is None or span.end_time is None:
            continue
        parent_id = ""
        if span.parent is not None and span.parent.is_valid:
            parent_id = format_span_id(span.parent.span_id)
        sid = format_span_id(span.context.span_id)
        rows.append((sid, parent_id, span.name, span.start_time, span.end_time))
        batch_child_keys = direct_child_intervals.get(span.context.span_id)
        for index, (start_ns, end_ns) in enumerate(
            child_intervals.get(span.context.span_id, [])
        ):
            if batch_child_keys is not None and (start_ns, end_ns) in batch_child_keys:
                continue
            child_sid = "{}:prior:{}".format(sid, index)
            rows.append((child_sid, sid, "_prior_child", start_ns, end_ns))

    self_durations = self_duration_by_span_id(rows)
    annotated = [_copy_with_self_duration(span, self_durations) for span in spans]
    for span in annotated:
        if span.context is None:
            continue
        self_duration_ns = self_durations.get(format_span_id(span.context.span_id))
        if self_duration_ns is None:
            continue
        attrs = dict(span.attributes or {})
        metric_attrs: Dict[str, AttributeValue] = {METRIC_ATTR_SPAN_NAME: span.name}
        txn = attrs.get(CoralogixAttributes.TRANSACTION_IDENTIFIER)
        if txn is not None:
            metric_attrs[CoralogixAttributes.TRANSACTION_IDENTIFIER] = str(txn)
        if attrs.get(CoralogixAttributes.TRANSACTION_ROOT):
            metric_attrs[CoralogixAttributes.TRANSACTION_ROOT] = True
        try:
            self_duration_hist.record(self_duration_ns / 1_000_000_000.0, metric_attrs)
        except Exception:
            _LOG.exception("Failed to record transaction self-duration metric")
    return annotated


def _copy_with_self_duration(
    span: ReadableSpan,
    self_durations: Dict[str, int],
) -> ReadableSpan:
    attrs = dict(span.attributes or {})
    if span.context is not None:
        sid = format_span_id(span.context.span_id)
        if sid in self_durations:
            max_attributes = getattr(getattr(span, "_attributes", None), "maxlen", None)
            if (
                isinstance(max_attributes, int)
                and len(attrs) >= max_attributes
                and CoralogixAttributes.TRANSACTION_IDENTIFIER in attrs
            ):
                return span
            attrs[SELF_DURATION_ATTRIBUTE] = self_durations[sid] / 1_000_000_000.0
    return copy_with_attributes(span, attrs)


def strip_transaction_enrichment(span: ReadableSpan) -> ReadableSpan:
    attrs = dict(span.attributes or {})
    for attribute in (
        CoralogixAttributes.TRANSACTION_IDENTIFIER,
        CoralogixAttributes.TRANSACTION_ROOT,
        CoralogixAttributes.TRANSACTION_EXPLICIT,
        SELF_DURATION_ATTRIBUTE,
    ):
        attrs.pop(attribute, None)
    return copy_with_attributes(span, attrs)


def create_self_duration_histogram(
    meter_provider: Optional[MeterProvider],
) -> Histogram:
    from opentelemetry import metrics

    if meter_provider is not None:
        meter = meter_provider.get_meter("coralogix.opentelemetry.transaction", "0.1.3")
    else:
        meter = metrics.get_meter("coralogix.opentelemetry.transaction", "0.1.3")
    return cast(
        Histogram,
        meter.create_histogram(
            name=METRIC_SELF_DURATION,
            unit="s",
            description=(
                "Exclusive (self) wall duration per span within a Coralogix transaction"
            ),
        ),
    )
