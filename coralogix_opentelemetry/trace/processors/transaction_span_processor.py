"""Transaction SpanProcessor: naming, self-time, self-time metric, and harvest.

This processor is the supported path going forward. The legacy
``CoralogixTransactionSampler`` remains for backward compatibility only.

On start it stamps ``cgx.transaction`` / ``cgx.transaction.root`` using the
Go-aligned boundary rules (new local transaction when parent attrs are missing,
parent is remote, or kind is SERVER/CONSUMER). Each process owns its own local
transaction; ``cgx.transaction.distributed`` is not used.

On end it buffers by ``trace_id``. As soon as a local transaction subtree has
no live spans left (the ``cgx.transaction.root`` and all of its descendants
have ended), that subtree is finalized even if an outer ancestor is still
open. Remaining spans on the same ``trace_id`` stay buffered until their own
local transaction completes. After the last live span on a TraceID ends, a
short completion holdback (default 100ms) waits so fire-and-forget children
can still join before leftover spans are finalized.

For each completed local transaction the pipeline is:

1. Compute exclusive self-time on the *full* tree, stamp
   ``cgx.transaction.self_time`` (seconds) on every span, and record the
   ``cgx.transaction.self_time`` histogram (unit ``s``, always on) for every
   span in the full, untrimmed tree.
2. Trim the tree to the ``max_nodes`` slowest spans (default 256). The
   transaction root is always kept; spans whose kept ancestor was dropped
   are re-parented to the nearest surviving ancestor.
3. If ``max_regular_traces <= 0``, export the trimmed trace immediately.
   Otherwise the trimmed trace competes in a harvest heap keyed by the
   transaction root duration (``cgx.transaction.root``); only the slowest
   ``max_regular_traces`` local traces per harvest window are exported
   (default 1).
4. A daemon harvester thread flushes harvest winners every
   ``harvest_period_millis`` (default 60000).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.harvest import (
    DEFAULT_HARVEST_PERIOD_MILLIS,
    DEFAULT_MAX_REGULAR_TRACES,
    HarvestTrace,
    RegularTraceHeap,
    root_duration_ns,
)
from coralogix_opentelemetry.trace.processors.self_time import self_time_by_span_id
from coralogix_opentelemetry.trace.processors.trace_heap import (
    DEFAULT_MAX_TXN_TRACE_NODES,
    select_slowest_spans,
)
from coralogix_opentelemetry.trace.processors.transaction_naming import (
    resolve_transaction,
)
from opentelemetry import metrics
from opentelemetry.context import Context
from opentelemetry.metrics import Histogram, MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode, format_span_id

_LOG = logging.getLogger(__name__)

METRIC_SELF_TIME = "cgx.transaction.self_time"
SELF_TIME_ATTRIBUTE = "cgx.transaction.self_time"
METRIC_ATTR_SPAN_NAME = "span.name"

DEFAULT_MAX_NODES = DEFAULT_MAX_TXN_TRACE_NODES
DEFAULT_COMPLETION_HOLDBACK_MILLIS = 100


class TransactionSpanProcessor(SpanProcessor):
    """Full transaction tagging + self-time + trim + harvest.

    * Self-time is stamped as span attribute and metric ``cgx.transaction.self_time``
      (unit ``s``) computed on the full, untrimmed local trace.
    * ``max_nodes`` (default 256): keep the longest spans inside one local trace
      (default 256); the transaction root is never evicted.
    * ``max_regular_traces`` (default 1): keep only the slowest completed local
      trace(s) per harvest window. Set ``0`` to export every completed trace
      immediately (no harvest sampling).
    * ``harvest_period_millis`` (default 60000): how often the harvest winners
      are flushed to the wrapped exporter.
    * ``completion_holdback_millis`` (default 100): after the last live span on
      a TraceID ends, wait this long before finalizing leftovers so
      fire-and-forget children can still join. Set ``0`` to finalize immediately.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_regular_traces: int = DEFAULT_MAX_REGULAR_TRACES,
        harvest_period_millis: int = DEFAULT_HARVEST_PERIOD_MILLIS,
        completion_holdback_millis: int = DEFAULT_COMPLETION_HOLDBACK_MILLIS,
        meter_provider: Optional[MeterProvider] = None,
    ) -> None:
        self._exporter = span_exporter
        self._max_nodes = max_nodes
        self._max_regular_traces = max_regular_traces
        self._harvest_period_millis = harvest_period_millis
        self._completion_holdback_millis = completion_holdback_millis
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._buffers: Dict[int, List[ReadableSpan]] = {}
        self._live_parents: Dict[int, Dict[int, int]] = {}
        self._child_intervals: Dict[int, List[Tuple[int, int]]] = {}
        self._pending_completions: Dict[int, threading.Timer] = {}
        self._stopped = False
        self._exporter_shutdown = False
        self._shutdown_started = False
        self._shutdown_done = threading.Event()
        self._pending_finalize = 0
        self._idle = threading.Condition(self._lock)
        self._self_time_hist: Histogram = _create_self_time_histogram(meter_provider)

        self._harvest = RegularTraceHeap(max_regular_traces)
        self._harvest_stop = threading.Event()
        self._harvester: Optional[threading.Thread] = None
        if max_regular_traces > 0 and harvest_period_millis > 0:
            self._harvester = threading.Thread(
                target=self._harvest_loop,
                name="TransactionSpanProcessor-harvester",
                daemon=True,
            )
            self._harvester.start()

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        if span.context is None or not span.context.is_valid:
            return

        transaction, starts = resolve_transaction(
            span_name=span.name,
            span_kind=span.kind,
            parent_context=parent_context,
        )
        span.set_attribute(CoralogixAttributes.TRANSACTION_IDENTIFIER, transaction)
        if starts:
            span.set_attribute(CoralogixAttributes.TRANSACTION_ROOT, True)

        trace_id = span.context.trace_id
        parent_id = 0
        if span.parent is not None and span.parent.is_valid:
            parent_id = span.parent.span_id

        with self._lock:
            if self._exporter_shutdown:
                return
            self._cancel_pending_completion_locked(trace_id)
            if self._stopped:
                if trace_id not in self._live_parents and trace_id not in self._buffers:
                    return
                live = self._live_parents.setdefault(trace_id, {})
                live[span.context.span_id] = parent_id
                return
            live = self._live_parents.setdefault(trace_id, {})
            live[span.context.span_id] = parent_id

    def on_end(self, span: ReadableSpan) -> None:
        if span.context is None or not span.context.is_valid:
            return
        trace_id = span.context.trace_id
        completed_batches: List[List[ReadableSpan]] = []
        with self._lock:
            if self._exporter_shutdown:
                return
            tracked = trace_id in self._live_parents or trace_id in self._buffers
            if self._stopped and not tracked:
                return
            if (
                span.parent is not None
                and span.parent.is_valid
                and span.start_time is not None
                and span.end_time is not None
            ):
                self._child_intervals.setdefault(span.parent.span_id, []).append(
                    (span.start_time, span.end_time)
                )
            self._buffers.setdefault(trace_id, []).append(span)
            live = self._live_parents.get(trace_id)
            if live is not None:
                live.pop(span.context.span_id, None)
                if not live:
                    self._live_parents.pop(trace_id, None)

            still_live = bool(self._live_parents.get(trace_id))
            if still_live:
                completed_batches = self._extract_completed_local_transactions_locked(
                    trace_id, flush_leftover=False
                )
            elif self._buffers.get(trace_id):
                completed_batches = self._schedule_completion_locked(trace_id)
            else:
                completed_batches = []

            self._pending_finalize += len(completed_batches)
            if self._total_live_locked() == 0 and self._pending_finalize == 0:
                self._idle.notify_all()
        for batch in completed_batches:
            try:
                self._accept_completed_trace(batch)
            finally:
                with self._lock:
                    self._pending_finalize -= 1
                    self._idle.notify_all()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:
            if self._exporter_shutdown:
                return True
            batches = self._flush_pending_completions_locked()
            self._pending_finalize += len(batches)
        for batch in batches:
            try:
                self._accept_completed_trace(batch)
            finally:
                with self._lock:
                    self._pending_finalize -= 1
                    self._idle.notify_all()
        with self._lock:
            deadline = time.monotonic() + max(0.001, timeout_millis / 1000.0)
            while self._pending_finalize > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._idle.wait(timeout=remaining)
        self._flush_harvest()
        with self._export_lock:
            if self._exporter_shutdown:
                return True
            force_flush_fn = getattr(self._exporter, "force_flush", None)
            if force_flush_fn is None:
                return True
            try:
                result = force_flush_fn(timeout_millis=timeout_millis)
            except TypeError:
                result = force_flush_fn(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_started:
                already = True
            else:
                already = False
                self._shutdown_started = True
                self._stopped = True
        if already:
            self._shutdown_done.wait(timeout=60.0)
            return

        try:
            self._stop_harvester()
            self._flush_harvest()

            with self._lock:
                self._wait_for_idle_locked(timeout_sec=30.0)
                holdback_batches = self._flush_pending_completions_locked()
                batches: List[List[ReadableSpan]] = list(holdback_batches)
                for trace_id in list(self._buffers.keys()):
                    if self._live_parents.get(trace_id):
                        dropped = self._buffers.pop(trace_id, None) or []
                        self._live_parents.pop(trace_id, None)
                        for span in dropped:
                            if span.context is not None:
                                self._child_intervals.pop(span.context.span_id, None)
                        continue
                    extracted = self._extract_completed_local_transactions_locked(
                        trace_id, flush_leftover=True
                    )
                    batches.extend(extracted)
                self._pending_finalize += len(batches)
                self._buffers.clear()
                self._live_parents.clear()

            for batch in batches:
                try:
                    self._accept_completed_trace(batch)
                finally:
                    with self._lock:
                        self._pending_finalize -= 1
                        self._idle.notify_all()

            with self._lock:
                deadline = time.monotonic() + 30.0
                while self._pending_finalize > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._idle.wait(timeout=remaining)

            self._flush_harvest()

            with self._lock:
                self._exporter_shutdown = True
                self._child_intervals.clear()
            with self._export_lock:
                self._exporter.shutdown()
        finally:
            self._shutdown_done.set()

    def _total_live_locked(self) -> int:
        return sum(len(live) for live in self._live_parents.values())

    def _wait_for_idle_locked(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while self._total_live_locked() > 0 or self._pending_finalize > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._idle.wait(timeout=remaining)

    def _stop_harvester(self) -> None:
        self._harvest_stop.set()
        if self._harvester is not None:
            self._harvester.join(timeout=max(30.0, self._harvest_period_millis / 1000.0 + 5.0))

    def _harvest_loop(self) -> None:
        period_s = self._harvest_period_millis / 1000.0
        while not self._harvest_stop.is_set():
            if self._harvest_stop.wait(timeout=period_s):
                break
            self._flush_harvest()

    def _cancel_pending_completion_locked(self, trace_id: int) -> None:
        timer = self._pending_completions.pop(trace_id, None)
        if timer is not None:
            timer.cancel()

    def _flush_pending_completions_locked(self) -> List[List[ReadableSpan]]:
        batches: List[List[ReadableSpan]] = []
        for trace_id in list(self._pending_completions.keys()):
            self._cancel_pending_completion_locked(trace_id)
            if self._live_parents.get(trace_id):
                continue
            batches.extend(
                self._extract_completed_local_transactions_locked(
                    trace_id, flush_leftover=True
                )
            )
        for trace_id in list(self._buffers.keys()):
            if self._live_parents.get(trace_id):
                continue
            if trace_id in self._pending_completions:
                continue
            batches.extend(
                self._extract_completed_local_transactions_locked(
                    trace_id, flush_leftover=True
                )
            )
        return batches

    def _schedule_completion_locked(self, trace_id: int) -> List[List[ReadableSpan]]:
        self._cancel_pending_completion_locked(trace_id)
        if self._completion_holdback_millis <= 0:
            return self._extract_completed_local_transactions_locked(
                trace_id, flush_leftover=True
            )

        timer: Optional[threading.Timer] = None

        def _fire() -> None:
            batches: List[List[ReadableSpan]] = []
            with self._lock:
                if self._pending_completions.get(trace_id) is not timer:
                    return
                self._pending_completions.pop(trace_id, None)
                if self._exporter_shutdown:
                    return
                if self._live_parents.get(trace_id):
                    return
                batches = self._extract_completed_local_transactions_locked(
                    trace_id, flush_leftover=True
                )
                self._pending_finalize += len(batches)
            for batch in batches:
                try:
                    self._accept_completed_trace(batch)
                finally:
                    with self._lock:
                        self._pending_finalize -= 1
                        self._idle.notify_all()

        timer = threading.Timer(self._completion_holdback_millis / 1000.0, _fire)
        timer.daemon = True
        self._pending_completions[trace_id] = timer
        timer.start()
        return []

    def _extract_completed_local_transactions_locked(
        self, trace_id: int, *, flush_leftover: bool = True
    ) -> List[List[ReadableSpan]]:
        buffer = self._buffers.get(trace_id)
        if not buffer:
            return []

        live = self._live_parents.get(trace_id, {})
        parent_of: Dict[int, int] = {}
        for span in buffer:
            if span.context is None:
                continue
            if span.parent is not None and span.parent.is_valid:
                parent_of[span.context.span_id] = span.parent.span_id
        for span_id, parent_id in live.items():
            if parent_id:
                parent_of[span_id] = parent_id

        def under_root(span_id: int, root_id: int) -> bool:
            cur = span_id
            seen = set()
            while cur and cur not in seen:
                if cur == root_id:
                    return True
                seen.add(cur)
                cur = parent_of.get(cur, 0)
            return False

        def has_live_in_subtree(root_id: int) -> bool:
            if root_id in live:
                return True
            return any(under_root(live_id, root_id) for live_id in live)

        roots = [
            span
            for span in buffer
            if span.context is not None
            and (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT)
        ]

        def root_depth(root_id: int) -> int:
            depth = 0
            cur = root_id
            seen = set()
            while cur and cur not in seen:
                seen.add(cur)
                parent = parent_of.get(cur, 0)
                if not parent:
                    break
                depth += 1
                cur = parent
            return depth

        roots.sort(
            key=lambda s: root_depth(s.context.span_id) if s.context else 0,
            reverse=True,
        )

        batches: List[List[ReadableSpan]] = []
        extracted: set = set()

        for root in roots:
            assert root.context is not None
            root_id = root.context.span_id
            if root_id in extracted or has_live_in_subtree(root_id):
                continue
            subtree = [
                span
                for span in buffer
                if span.context is not None
                and span.context.span_id not in extracted
                and under_root(span.context.span_id, root_id)
            ]
            if not subtree:
                continue
            for span in subtree:
                if span.context is not None:
                    extracted.add(span.context.span_id)
            batches.append(subtree)

        if extracted:
            remaining = [
                span
                for span in buffer
                if span.context is None or span.context.span_id not in extracted
            ]
            if remaining:
                self._buffers[trace_id] = remaining
            else:
                self._buffers.pop(trace_id, None)

        if flush_leftover and not live and self._buffers.get(trace_id):
            batches.append(self._buffers.pop(trace_id))

        return batches

    def _accept_completed_trace(self, spans: Sequence[ReadableSpan]) -> None:
        with self._lock:
            interval_snapshot: Dict[int, List[Tuple[int, int]]] = {
                span.context.span_id: list(
                    self._child_intervals.get(span.context.span_id, [])
                )
                for span in spans
                if span.context is not None
            }
        annotated = self._annotate_with_self_time_and_metrics(spans, interval_snapshot)

        root_span_ids = [
            format_span_id(span.context.span_id)
            for span in annotated
            if span.context is not None
            and (span.attributes or {}).get(CoralogixAttributes.TRANSACTION_ROOT)
        ]

        trimmed = select_slowest_spans(
            annotated,
            max_nodes=self._max_nodes,
            root_span_ids=root_span_ids,
        )
        if not trimmed:
            with self._lock:
                for span in annotated:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)
            return

        try:
            if self._max_regular_traces <= 0 or self._harvest_period_millis <= 0:
                self._export_spans(trimmed)
                return

            candidate = HarvestTrace(
                duration_ns=root_duration_ns(trimmed),
                spans=list(trimmed),
            )
            with self._lock:
                if self._shutdown_started or self._exporter_shutdown:
                    pass
                else:
                    self._harvest.witness(candidate)
                    return
            self._export_spans(trimmed)
        finally:
            with self._lock:
                for span in annotated:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)

    def _annotate_with_self_time_and_metrics(
        self,
        spans: Sequence[ReadableSpan],
        child_intervals: Dict[int, List[Tuple[int, int]]],
    ) -> List[ReadableSpan]:
        rows: List[tuple] = []
        for span in spans:
            if span.context is None or span.start_time is None or span.end_time is None:
                continue
            parent_id = ""
            if span.parent is not None and span.parent.is_valid:
                parent_id = format_span_id(span.parent.span_id)
            sid = format_span_id(span.context.span_id)
            rows.append((sid, parent_id, span.name, span.start_time, span.end_time))
            for index, (start_ns, end_ns) in enumerate(
                child_intervals.get(span.context.span_id, [])
            ):
                child_sid = f"{sid}:prior:{index}"
                if any(
                    other.context is not None
                    and other.parent is not None
                    and other.parent.is_valid
                    and other.parent.span_id == span.context.span_id
                    and other.start_time == start_ns
                    and other.end_time == end_ns
                    for other in spans
                ):
                    continue
                rows.append((child_sid, sid, "_prior_child", start_ns, end_ns))

        self_times = self_time_by_span_id(rows)
        annotated = [_copy_with_self_time(span, self_times) for span in spans]
        for span in annotated:
            attrs = dict(span.attributes or {})
            self_time_sec = attrs.get(SELF_TIME_ATTRIBUTE)
            if self_time_sec is None:
                continue
            metric_attrs: Dict[str, object] = {METRIC_ATTR_SPAN_NAME: span.name}
            txn = attrs.get(CoralogixAttributes.TRANSACTION_IDENTIFIER)
            if txn is not None:
                metric_attrs[CoralogixAttributes.TRANSACTION_IDENTIFIER] = str(txn)
            if attrs.get(CoralogixAttributes.TRANSACTION_ROOT):
                metric_attrs[CoralogixAttributes.TRANSACTION_ROOT] = True
            self._self_time_hist.record(float(self_time_sec), metric_attrs)
        return annotated

    def _flush_harvest(self) -> None:
        with self._export_lock:
            if self._exporter_shutdown:
                return
            with self._lock:
                winners = self._harvest.drain()
            for winner in winners:
                try:
                    result = self._exporter.export(list(winner.spans))
                except Exception:
                    _LOG.exception("TransactionSpanProcessor failed to export spans")
                    continue
                if result is SpanExportResult.FAILURE:
                    _LOG.warning("TransactionSpanProcessor exporter returned FAILURE")

    def _export_spans(self, spans: Sequence[ReadableSpan]) -> None:
        with self._export_lock:
            if self._exporter_shutdown:
                return
            try:
                result = self._exporter.export(list(spans))
            except Exception:
                _LOG.exception("TransactionSpanProcessor failed to export spans")
                return
            if result is SpanExportResult.FAILURE:
                _LOG.warning("TransactionSpanProcessor exporter returned FAILURE")


def _copy_with_self_time(
    span: ReadableSpan, self_times: Dict[str, int]
) -> ReadableSpan:
    attrs = dict(span.attributes or {})
    if span.context is not None:
        sid = format_span_id(span.context.span_id)
        if sid in self_times:
            attrs[SELF_TIME_ATTRIBUTE] = self_times[sid] / 1_000_000_000.0
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource if span.resource is not None else Resource.create({}),
        attributes=attrs,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status if span.status is not None else Status(StatusCode.UNSET),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _create_self_time_histogram(
    meter_provider: Optional[MeterProvider],
) -> Histogram:
    if meter_provider is not None:
        meter = meter_provider.get_meter("coralogix.opentelemetry.transaction", "0.1.3")
    else:
        meter = metrics.get_meter("coralogix.opentelemetry.transaction", "0.1.3")
    return meter.create_histogram(
        name=METRIC_SELF_TIME,
        unit="s",
        description="Exclusive (self) wall time per span within a Coralogix transaction",
    )
