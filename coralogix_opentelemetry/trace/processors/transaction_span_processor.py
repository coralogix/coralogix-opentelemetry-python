"""Transaction SpanProcessor: naming, self-duration, metric, and trim.

This processor is the supported path going forward. The legacy
``CoralogixTransactionSampler`` remains for backward compatibility only.

Flow
----
**on_start** — track only. Decide new vs inherit (no parent local txn / remote /
SERVER / CONSUMER). Set ``cgx.transaction.root`` for starters. Record side-table
membership. Do **not** freeze ``cgx.transaction`` from the early span name
(frameworks may ``update_name`` later). Explicit ``start_new_transaction`` / route-template overrides (name ≠ start
span name) are stored as ``override_name``. Sampler echoes of the start name
are not treated as overrides so ``update_name`` can still win.

**on_end / finalize** — buffer by ``trace_id``. When a local-transaction subtree
has no live spans left, finalize that subtree even if an outer ancestor is
still open. After the last live span on a TraceID ends, a short completion
holdback (default 100ms) waits so fire-and-forget children can still join.

For each completed local transaction the export pipeline is:

1. Stamp ``cgx.transaction`` from ``override_name ?? root.final_name`` onto
   every span in the batch.
2. Compute exclusive self-duration on the *full* tree, stamp
   ``cgx.transaction.self_duration`` (seconds), and record the matching
   histogram (unit ``s``, always on).
3. Trim to the ``max_nodes`` slowest spans (default 256); root always kept.
4. Export the trimmed batch immediately (no client-side harvest sampling).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_MAX_TXN_TRACE_NODES,
    resolve_completion_holdback_millis,
    resolve_max_nodes,
)
from coralogix_opentelemetry.trace.processors.holdback_scheduler import (
    HoldbackScheduler,
)
from coralogix_opentelemetry.trace.processors.trace_heap import select_slowest_spans
from coralogix_opentelemetry.trace.processors.transaction_extract import (
    extract_completed_local_transactions,
    has_extractable_nested_transaction,
)
from coralogix_opentelemetry.trace.processors.transaction_finalize import (
    annotate_completed_batch,
    create_self_duration_histogram,
)
from coralogix_opentelemetry.trace.processors.transaction_naming import (
    TransactionMembership,
    apply_on_start_root_flag,
    explicit_transaction_override,
    parent_has_transaction_attrs,
    parent_transaction_from_attrs,
    parent_transaction_from_tracestate,
    preset_transaction_name,
    starts_new_transaction,
)
from opentelemetry.context import Context
from opentelemetry.metrics import Histogram, MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import format_span_id, get_current_span

_LOG = logging.getLogger(__name__)

DEFAULT_MAX_NODES = DEFAULT_MAX_TXN_TRACE_NODES

# HoldbackScheduler keys: distinguish idle vs nested arms for the same TraceID.
_HOLD_IDLE = "idle"
_HOLD_NESTED = "nested"
# Cap queued completed batches so a stalled exporter cannot grow memory without bound.
DEFAULT_MAX_FINALIZE_QUEUE = 256
# Cap batches retained across timed-out force_flush calls (same order as the queue).
DEFAULT_MAX_DEFERRED_FINALIZE = DEFAULT_MAX_FINALIZE_QUEUE


class TransactionSpanProcessor(SpanProcessor):
    """Full transaction tagging + self-duration + trim + export.

    Constructor options override env vars. Omitted options read
    ``OTEL_CX_TRANSACTION_*`` (see README), then fall back to defaults.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        *,
        max_nodes: Optional[int] = None,
        completion_holdback_millis: Optional[int] = None,
        meter_provider: Optional[MeterProvider] = None,
    ) -> None:
        self._exporter = span_exporter
        self._max_nodes = resolve_max_nodes(max_nodes)
        self._completion_holdback_millis = resolve_completion_holdback_millis(
            completion_holdback_millis
        )
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._buffers: Dict[int, List[ReadableSpan]] = {}
        self._live_parents: Dict[int, Dict[int, int]] = {}
        self._membership: Dict[int, TransactionMembership] = {}
        self._child_intervals: Dict[int, List[Tuple[int, int]]] = {}
        self._pending_completions: Dict[int, int] = {}
        self._pending_nested_completions: Dict[int, int] = {}
        self._holdback = HoldbackScheduler()
        # Holdback deadlines must not wait on exporters; finalize/export runs here.
        # Bound the queue so a stalled exporter drops overflow instead of OOM.
        self._finalize_queue: queue.Queue = queue.Queue(
            maxsize=DEFAULT_MAX_FINALIZE_QUEUE
        )
        self._finalize_stop = threading.Event()
        self._finalize_worker = threading.Thread(
            target=self._finalize_loop,
            name="TransactionSpanProcessor-finalize",
            daemon=True,
        )
        self._finalize_worker.start()
        self._stopped = False
        self._exporter_shutdown = False
        self._shutdown_started = False
        self._shutdown_done = threading.Event()
        self._pending_finalize = 0
        self._idle = threading.Condition(self._lock)
        # Batches extracted by force_flush that could not be queued before the
        # deadline — retained for a later flush instead of being abandoned.
        self._deferred_finalize: List[List[ReadableSpan]] = []
        self._self_duration_hist: Histogram = create_self_duration_histogram(
            meter_provider
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Track membership + root flag only; do not freeze transaction name."""
        try:
            self._on_start_impl(span, parent_context)
        except Exception:
            _LOG.exception("TransactionSpanProcessor.on_start failed")

    def _on_start_impl(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        if span.context is None or not span.context.is_valid:
            return

        parent_span = get_current_span(parent_context)
        parent_id = 0
        if span.parent is not None and span.parent.is_valid:
            parent_id = span.parent.span_id

        with self._lock:
            parent_member = self._membership.get(parent_id) if parent_id else None
            inherited_from_ts = parent_transaction_from_tracestate(parent_span)
            parent_has_local = (
                parent_member is not None
                or parent_has_transaction_attrs(parent_span)
                or inherited_from_ts is not None
            )

        starts = starts_new_transaction(
            span_kind=span.kind,
            parent_context=parent_context,
            parent_has_local_transaction=parent_has_local,
        )
        apply_on_start_root_flag(span, starts)
        start_name = span.name
        preset = preset_transaction_name(span)
        # Sampler echoes start name into cgx.transaction; only treat a pre-set
        # value as override when it differs (route template / explicit rename).
        override = preset if (preset is not None and preset != start_name) else None

        trace_id = span.context.trace_id
        span_id = span.context.span_id

        with self._lock:
            # After exporter shutdown, do not grow membership — on_end will also
            # no-op and would never clean these entries up.
            if self._exporter_shutdown:
                return
            # During shutdown, only continue tracking in-flight TraceIDs. Do not
            # insert membership for brand-new traces (on_end would never clear it).
            if (
                self._stopped
                and trace_id not in self._live_parents
                and trace_id not in self._buffers
            ):
                return
            if starts:
                self._membership[span_id] = TransactionMembership(
                    root_span_id=span_id,
                    is_root=True,
                    override_name=override,
                    start_name=start_name,
                )
            else:
                root_span_id = (
                    parent_member.root_span_id
                    if parent_member is not None
                    else (parent_id or span_id)
                )
                inherited_name = None
                # Name from TraceState / attrs without a locally tracked root.
                if parent_member is None:
                    inherited_name = inherited_from_ts or parent_transaction_from_attrs(
                        parent_span
                    )
                self._membership[span_id] = TransactionMembership(
                    root_span_id=root_span_id,
                    is_root=False,
                    override_name=None,
                    inherited_name=inherited_name,
                    start_name=start_name,
                )

            self._cancel_pending_completion_locked(trace_id)
            if self._stopped:
                live = self._live_parents.setdefault(trace_id, {})
                live[span_id] = parent_id
                return
            live = self._live_parents.setdefault(trace_id, {})
            live[span_id] = parent_id

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._on_end_impl(span)
        except Exception:
            _LOG.exception("TransactionSpanProcessor.on_end failed")

    def _on_end_impl(self, span: ReadableSpan) -> None:
        if span.context is None or not span.context.is_valid:
            return

        # start_new_transaction may set an explicit override after on_start.
        attrs = span.attributes or {}
        preset = attrs.get(CoralogixAttributes.TRANSACTION_IDENTIFIER)
        with self._lock:
            member = self._membership.get(span.context.span_id)
            if member is not None and member.is_root and preset is not None:
                if explicit_transaction_override(span):
                    member.override_name = str(preset)
                elif (
                    not member.override_name
                    and member.start_name is not None
                    and str(preset) != member.start_name
                ):
                    # Template / late attr different from the on_start name.
                    member.override_name = str(preset)

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
                and self._is_local_parent_locked(trace_id, span.parent.span_id)
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
                completed_batches = self._schedule_nested_completion_locked(trace_id)
            elif self._buffers.get(trace_id):
                self._cancel_pending_nested_completion_locked(trace_id)
                completed_batches = self._schedule_completion_locked(trace_id)
            else:
                completed_batches = []

            self._pending_finalize += len(completed_batches)
            if self._total_live_locked() == 0 and self._pending_finalize == 0:
                self._idle.notify_all()
        # Always queue finalize/export (including zero-holdback) so Span.end
        # never blocks on a slow exporter and the bounded backlog applies.
        self._dispatch_accept_completed(completed_batches)

    def _run_accept_completed(self, batch: Sequence[ReadableSpan]) -> None:
        """Run finalize/export off the caller's contract: never raise to Span.end."""
        try:
            self._accept_completed_trace(batch)
        except Exception:
            _LOG.exception(
                "TransactionSpanProcessor failed while accepting a completed trace"
            )
        finally:
            with self._lock:
                self._pending_finalize -= 1
                self._idle.notify_all()

    def _abandon_completed_batch(
        self, batch: Sequence[ReadableSpan], *, adjust_pending: bool = True
    ) -> None:
        """Drop export for an unqueued batch; still record self-duration metrics."""
        try:
            with self._lock:
                interval_snapshot: Dict[int, List[Tuple[int, int]]] = {
                    span.context.span_id: list(
                        self._child_intervals.get(span.context.span_id, [])
                    )
                    for span in batch
                    if span.context is not None
                }
                membership_snapshot = {
                    span.context.span_id: self._membership[span.context.span_id]
                    for span in batch
                    if span.context is not None
                    and span.context.span_id in self._membership
                }
            annotate_completed_batch(
                batch,
                child_intervals=interval_snapshot,
                membership=membership_snapshot,
                self_duration_hist=self._self_duration_hist,
            )
        except Exception:
            _LOG.exception(
                "TransactionSpanProcessor failed while recording metrics for "
                "a dropped finalize batch"
            )
        finally:
            with self._lock:
                if adjust_pending:
                    self._pending_finalize -= 1
                for span in batch:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)
                        self._membership.pop(span.context.span_id, None)
                self._idle.notify_all()

    def _retain_deferred_batches(
        self, batches: Sequence[Sequence[ReadableSpan]]
    ) -> None:
        """Keep unqueued force_flush batches for retry, within a bounded cap."""
        overflow: List[List[ReadableSpan]] = []
        with self._lock:
            for batch in batches:
                payload = list(batch)
                if len(self._deferred_finalize) >= DEFAULT_MAX_DEFERRED_FINALIZE:
                    overflow.append(payload)
                    continue
                self._deferred_finalize.append(payload)
            self._pending_finalize -= len(batches)
            self._idle.notify_all()
        for batch in overflow:
            _LOG.error(
                "TransactionSpanProcessor deferred finalize full "
                "(max=%d); dropping retained batch of %d span(s)",
                DEFAULT_MAX_DEFERRED_FINALIZE,
                len(batch),
            )
            # pending already adjusted above for these batches.
            self._abandon_completed_batch(batch, adjust_pending=False)

    def _enqueue_finalize_item(
        self, item: object, *, deadline: Optional[float] = None
    ) -> bool:
        """Put one finalize item on the worker queue.

        When ``deadline`` is None (Span.end / holdback path), drop immediately on
        Full. When a deadline is set (force_flush), wait for capacity until then.
        """
        if deadline is None:
            try:
                self._finalize_queue.put_nowait(item)
                return True
            except queue.Full:
                return False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                self._finalize_queue.put(item, timeout=remaining)
                return True
            except queue.Full:
                return False

    def _dispatch_accept_completed(
        self,
        batches: Sequence[Sequence[ReadableSpan]],
        *,
        deadline: Optional[float] = None,
    ) -> bool:
        """Queue finalize/export so callers never block on exporters.

        Hot path (no deadline): if the queue is full, record metrics then drop.
        force_flush (deadline set): wait for capacity until the deadline; return
        False if any batch could not be queued in time. Unqueued force_flush
        batches are retained in ``_deferred_finalize`` for a later retry.
        """
        for index, batch in enumerate(batches):
            payload = list(batch)
            if self._enqueue_finalize_item(payload, deadline=deadline):
                continue
            if deadline is None:
                _LOG.error(
                    "TransactionSpanProcessor finalize queue full "
                    "(max=%d); dropping completed batch of %d span(s)",
                    DEFAULT_MAX_FINALIZE_QUEUE,
                    len(payload),
                )
                self._abandon_completed_batch(payload)
                continue
            unqueued = [payload] + [list(rest) for rest in batches[index + 1 :]]
            self._retain_deferred_batches(unqueued)
            _LOG.error(
                "TransactionSpanProcessor finalize queue full "
                "(max=%d); retaining up to %d batch(es) for a later force_flush",
                DEFAULT_MAX_FINALIZE_QUEUE,
                DEFAULT_MAX_DEFERRED_FINALIZE,
            )
            return False
        return True

    def _finalize_loop(self) -> None:
        while True:
            try:
                item = self._finalize_queue.get(timeout=0.25)
            except queue.Empty:
                if self._finalize_stop.is_set():
                    return
                continue
            try:
                self._run_accept_completed(item)
            finally:
                self._finalize_queue.task_done()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        deadline = time.monotonic() + max(0.001, timeout_millis / 1000.0)
        with self._lock:
            if self._exporter_shutdown:
                return True
            batches = list(self._deferred_finalize)
            self._deferred_finalize = []
            batches.extend(self._flush_pending_completions_locked())
            self._pending_finalize += len(batches)
        # Wait for queue capacity within the deadline — retain on timeout.
        if not self._dispatch_accept_completed(batches, deadline=deadline):
            return False
        with self._lock:
            while self._pending_finalize > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._idle.wait(timeout=remaining)
            timed_out = self._pending_finalize > 0
        if timed_out:
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        acquired = self._export_lock.acquire(timeout=remaining)
        if not acquired:
            return False
        try:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if self._exporter_shutdown:
                return True
            force_flush_fn = getattr(self._exporter, "force_flush", None)
            if force_flush_fn is None:
                return True
            try:
                result = force_flush_fn(timeout_millis=remaining_ms)
            except TypeError:
                result = force_flush_fn(remaining_ms)
            return True if result is None else bool(result)
        finally:
            self._export_lock.release()

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
            with self._lock:
                self._wait_for_idle_locked(timeout_sec=30.0)
                holdback_batches = self._flush_pending_completions_locked()
                batches: List[List[ReadableSpan]] = list(self._deferred_finalize)
                self._deferred_finalize = []
                batches.extend(holdback_batches)
                for trace_id in list(self._buffers.keys()):
                    if self._live_parents.get(trace_id):
                        dropped = self._buffers.pop(trace_id, None) or []
                        self._live_parents.pop(trace_id, None)
                        for span in dropped:
                            if span.context is not None:
                                self._child_intervals.pop(span.context.span_id, None)
                                self._membership.pop(span.context.span_id, None)
                        continue
                    extracted = self._extract_completed_local_transactions_locked(
                        trace_id, flush_leftover=True
                    )
                    batches.extend(extracted)
                self._pending_finalize += len(batches)
                self._buffers.clear()
                self._live_parents.clear()

            for batch in batches:
                self._run_accept_completed(batch)

            with self._lock:
                deadline = time.monotonic() + 30.0
                while self._pending_finalize > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._idle.wait(timeout=remaining)

            with self._lock:
                self._exporter_shutdown = True
                self._child_intervals.clear()
                self._membership.clear()
            with self._export_lock:
                self._exporter.shutdown()
        finally:
            self._holdback.shutdown()
            self._finalize_stop.set()
            self._finalize_worker.join(timeout=30.0)
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

    def _cancel_pending_completion_locked(self, trace_id: int) -> None:
        if self._pending_completions.pop(trace_id, None) is not None:
            self._holdback.cancel((_HOLD_IDLE, trace_id))

    def _cancel_pending_nested_completion_locked(self, trace_id: int) -> None:
        if self._pending_nested_completions.pop(trace_id, None) is not None:
            self._holdback.cancel((_HOLD_NESTED, trace_id))

    def _is_local_parent_locked(self, trace_id: int, parent_id: int) -> bool:
        live = self._live_parents.get(trace_id)
        if live is not None and parent_id in live:
            return True
        for span in self._buffers.get(trace_id, []):
            if span.context is not None and span.context.span_id == parent_id:
                return True
        return False

    def _flush_pending_completions_locked(self) -> List[List[ReadableSpan]]:
        batches: List[List[ReadableSpan]] = []
        for trace_id in list(self._pending_completions.keys()):
            self._cancel_pending_completion_locked(trace_id)
        for trace_id in list(self._pending_nested_completions.keys()):
            self._cancel_pending_nested_completion_locked(trace_id)
        for trace_id in list(self._buffers.keys()):
            if self._live_parents.get(trace_id):
                batches.extend(
                    self._extract_completed_local_transactions_locked(
                        trace_id, flush_leftover=False
                    )
                )
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

        token = 0

        def _fire() -> None:
            batches: List[List[ReadableSpan]] = []
            with self._lock:
                if self._pending_completions.get(trace_id) != token:
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
            # Return quickly: export runs on the finalize worker, not this deadline thread.
            self._dispatch_accept_completed(batches)

        token = self._holdback.schedule(
            (_HOLD_IDLE, trace_id),
            self._completion_holdback_millis / 1000.0,
            _fire,
        )
        self._pending_completions[trace_id] = token
        return []

    def _schedule_nested_completion_locked(
        self, trace_id: int
    ) -> List[List[ReadableSpan]]:
        buffer = self._buffers.get(trace_id, [])
        live = self._live_parents.get(trace_id, {})
        if not has_extractable_nested_transaction(buffer=buffer, live=live):
            self._cancel_pending_nested_completion_locked(trace_id)
            return []
        if trace_id in self._pending_nested_completions:
            return []
        if self._completion_holdback_millis <= 0:
            return self._extract_completed_local_transactions_locked(
                trace_id, flush_leftover=False
            )

        token = 0

        def _fire() -> None:
            batches: List[List[ReadableSpan]] = []
            with self._lock:
                if self._pending_nested_completions.get(trace_id) != token:
                    return
                self._pending_nested_completions.pop(trace_id, None)
                if self._exporter_shutdown:
                    return
                if self._live_parents.get(trace_id):
                    batches = self._extract_completed_local_transactions_locked(
                        trace_id, flush_leftover=False
                    )
                self._pending_finalize += len(batches)
                if self._live_parents.get(
                    trace_id
                ) and has_extractable_nested_transaction(
                    buffer=self._buffers.get(trace_id, []),
                    live=self._live_parents.get(trace_id, {}),
                ):
                    self._schedule_nested_completion_locked(trace_id)
            self._dispatch_accept_completed(batches)

        token = self._holdback.schedule(
            (_HOLD_NESTED, trace_id),
            self._completion_holdback_millis / 1000.0,
            _fire,
        )
        self._pending_nested_completions[trace_id] = token
        return []

    def _extract_completed_local_transactions_locked(
        self, trace_id: int, *, flush_leftover: bool = True
    ) -> List[List[ReadableSpan]]:
        buffer = self._buffers.get(trace_id)
        if not buffer:
            return []
        live = self._live_parents.get(trace_id, {})
        batches, remaining = extract_completed_local_transactions(
            buffer=buffer, live=live, flush_leftover=flush_leftover
        )
        if remaining:
            self._buffers[trace_id] = remaining
        else:
            self._buffers.pop(trace_id, None)
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
            membership_snapshot = {
                span.context.span_id: self._membership[span.context.span_id]
                for span in spans
                if span.context is not None and span.context.span_id in self._membership
            }
        annotated = annotate_completed_batch(
            spans,
            child_intervals=interval_snapshot,
            membership=membership_snapshot,
            self_duration_hist=self._self_duration_hist,
        )

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
                        self._membership.pop(span.context.span_id, None)
            return

        try:
            self._export_spans(trimmed)
        finally:
            with self._lock:
                for span in annotated:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)
                        self._membership.pop(span.context.span_id, None)

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
