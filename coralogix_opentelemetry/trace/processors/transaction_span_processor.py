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
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.defaults import (
    DEFAULT_MAX_TXN_TRACE_NODES,
    resolve_completion_holdback_millis,
    resolve_max_nodes,
)
from coralogix_opentelemetry.trace.processors.holdback_scheduler import (
    HoldbackScheduler,
)
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_parent
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
from opentelemetry.trace import SpanContext, format_span_id, get_current_span

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
        # Compact parent links / contexts so live-buffer eviction can rebind
        # descendants without retaining every ended ancestor ReadableSpan.
        self._span_contexts: Dict[int, SpanContext] = {}
        self._span_parent_ids: Dict[int, int] = {}
        self._parent_rebind: Dict[int, int] = {}
        # Span IDs observed on each TraceID — used to drop compact side tables
        # once the TraceID has no live or buffered spans left.
        self._trace_span_ids: Dict[int, Set[int]] = {}
        # Live-trim evictions waiting on live descendants before self-duration
        # metrics can include those children's intervals.
        self._pending_drop_metrics: Dict[int, ReadableSpan] = {}
        self._pending_drop_waiters: Dict[int, Set[int]] = {}
        # Batches extracted but not yet accept/abandon-finished, per TraceID.
        # Side tables must outlive the first queued batch when several share a
        # TraceID (nested + outer extracted together).
        self._inflight_batches_by_trace: Dict[int, int] = {}
        # Span IDs removed from `_buffers` by live-trim (compact links retained).
        self._evicted_from_buffer: Dict[int, Set[int]] = {}
        # Final root span names captured at root on_end (after update_name).
        self._root_final_names: Dict[int, str] = {}
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
        self._start_finalize_worker()
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
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._restart_after_fork)

    def _start_finalize_worker(self) -> None:
        self._finalize_worker = threading.Thread(
            target=self._finalize_loop,
            name="TransactionSpanProcessor-finalize",
            daemon=True,
        )
        self._finalize_worker.start()

    def _restart_after_fork(self) -> None:
        """Discard parent-only pending work and recreate child worker state."""
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._holdback.restart_after_fork()
        self._finalize_queue = queue.Queue(maxsize=DEFAULT_MAX_FINALIZE_QUEUE)
        self._finalize_stop = threading.Event()
        self._pending_finalize = 0
        self._deferred_finalize = []
        self._buffers.clear()
        self._live_parents.clear()
        self._membership.clear()
        self._span_contexts.clear()
        self._span_parent_ids.clear()
        self._parent_rebind.clear()
        self._trace_span_ids.clear()
        self._pending_completions.clear()
        self._pending_nested_completions.clear()
        self._pending_drop_metrics.clear()
        self._pending_drop_waiters.clear()
        self._inflight_batches_by_trace.clear()
        self._evicted_from_buffer.clear()
        self._root_final_names.clear()
        self._child_intervals.clear()
        self._shutdown_started = False
        self._shutdown_done = threading.Event()
        if not self._exporter_shutdown:
            self._stopped = False
            self._start_finalize_worker()

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
            trace_already_tracked = (
                span.context.trace_id in self._live_parents
                or span.context.trace_id in self._buffers
            )
            parent_has_local = (
                parent_member is not None
                or parent_has_transaction_attrs(parent_span)
                or inherited_from_ts is not None
                # Parent may have been live-trimmed out of the buffer while a
                # caller still holds its context; keep inheriting on this TraceID.
                or (bool(parent_id) and trace_already_tracked)
            )

        starts = starts_new_transaction(
            span_kind=span.kind,
            parent_context=parent_context,
            parent_has_local_transaction=parent_has_local,
        )
        apply_on_start_root_flag(span, starts)
        start_name = span.name
        preset = preset_transaction_name(span)

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

            # Sampler echoes start name or copies the outer txn onto nested
            # SERVER/CONSUMER roots; only a differing non-inherited preset is
            # an override (route template / explicit rename).
            override = None
            if preset is not None and preset != start_name:
                parent_txn = None
                if parent_member is not None:
                    parent_txn = (
                        parent_member.override_name
                        or parent_member.inherited_name
                        or parent_member.start_name
                    )
                elif inherited_from_ts is not None:
                    parent_txn = inherited_from_ts
                else:
                    parent_txn = parent_transaction_from_attrs(parent_span)
                if not (starts and parent_txn is not None and preset == parent_txn):
                    override = preset

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
                # start_new_transaction() may mark the parent as a root after
                # the parent's on_start recorded is_root=False. Promote so
                # children join the nested transaction and live-trim protects it.
                if (
                    parent_member is not None
                    and parent_id
                    and (
                        (getattr(parent_span, "attributes", None) or {}).get(
                            CoralogixAttributes.TRANSACTION_ROOT
                        )
                        is True
                    )
                ):
                    parent_member.is_root = True
                    parent_member.root_span_id = parent_id
                    if explicit_transaction_override(parent_span):
                        parent_preset = parent_transaction_from_attrs(parent_span)
                        if parent_preset is not None:
                            parent_member.override_name = parent_preset
                    root_span_id = parent_id
                elif parent_member is None and parent_id:
                    # Walk compact links to the nearest known transaction root.
                    cur = parent_id
                    seen_walk: Set[int] = set()
                    while cur and cur not in seen_walk:
                        seen_walk.add(cur)
                        walked = self._membership.get(cur)
                        if walked is not None:
                            root_span_id = walked.root_span_id
                            break
                        cur = self._span_parent_ids.get(cur, 0)
                    else:
                        open_roots = self._open_transaction_root_ids_locked(trace_id)
                        if open_roots:
                            root_span_id = next(iter(open_roots))
                self._membership[span_id] = TransactionMembership(
                    root_span_id=root_span_id,
                    is_root=False,
                    override_name=None,
                    inherited_name=inherited_name,
                    start_name=start_name,
                )

            self._span_contexts[span_id] = span.context
            self._trace_span_ids.setdefault(trace_id, set()).add(span_id)

            self._cancel_pending_completion_locked(trace_id)
            # Parent may already have been live-trimmed; rebind new children onto
            # the nearest retained ancestor so export does not keep a dangling id.
            effective_parent_id = parent_id
            if parent_id and parent_id in self._evicted_from_buffer.get(trace_id, ()):
                kept = self._nearest_retained_ancestor_locked(trace_id, parent_id)
                if kept and kept != parent_id:
                    effective_parent_id = kept
                    self._parent_rebind[span_id] = kept
            if effective_parent_id:
                self._span_parent_ids[span_id] = effective_parent_id
            else:
                self._span_parent_ids.pop(span_id, None)
            live = self._live_parents.setdefault(trace_id, {})
            live[span_id] = effective_parent_id
            if self._stopped:
                return

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
            if member is not None and attrs.get(CoralogixAttributes.TRANSACTION_ROOT):
                # Dynamic root (e.g. start_new_transaction on an INTERNAL child).
                member.is_root = True
                member.root_span_id = span.context.span_id
            if member is not None and member.is_root and preset is not None:
                if explicit_transaction_override(span):
                    member.override_name = str(preset)
                elif (
                    not member.override_name
                    and member.start_name is not None
                    and str(preset) != member.start_name
                ):
                    parent_txn = None
                    if span.parent is not None and span.parent.is_valid:
                        parent_member = self._membership.get(span.parent.span_id)
                        if parent_member is not None:
                            parent_txn = (
                                parent_member.override_name
                                or parent_member.inherited_name
                                or parent_member.start_name
                            )
                    # Ignore sampler copy of the outer txn onto a nested root.
                    if parent_txn is None or str(preset) != parent_txn:
                        member.override_name = str(preset)
            if member is not None and member.is_root:
                self._root_final_names[span.context.span_id] = span.name

        # Record intervals against the SDK parent before any live-trim rebind so
        # deferred eviction metrics still see async children.
        original_parent_id = 0
        if span.parent is not None and span.parent.is_valid:
            original_parent_id = span.parent.span_id

        trace_id = span.context.trace_id
        completed_batches: List[List[ReadableSpan]] = []
        dropped_for_metrics: List[ReadableSpan] = []
        with self._lock:
            if self._exporter_shutdown:
                return
            tracked = trace_id in self._live_parents or trace_id in self._buffers
            if self._stopped and not tracked:
                return
            if (
                original_parent_id
                and span.start_time is not None
                and span.end_time is not None
                and self._is_local_parent_locked(trace_id, original_parent_id)
            ):
                self._child_intervals.setdefault(original_parent_id, []).append(
                    (span.start_time, span.end_time)
                )

            # Live-buffer eviction may have rebound this span's parent past
            # dropped ancestors; apply before buffering so export does not keep
            # a dangling id.
            rebind_parent_id = self._parent_rebind.pop(span.context.span_id, None)
            rebind_ctx = (
                self._span_contexts.get(rebind_parent_id) if rebind_parent_id else None
            )
            if rebind_ctx is not None:
                span = copy_with_parent(span, rebind_ctx)

            self._buffers.setdefault(trace_id, []).append(span)
            live = self._live_parents.get(trace_id)
            if live is not None:
                live.pop(span.context.span_id, None)
                if not live:
                    self._live_parents.pop(trace_id, None)

            # Ending a waiter may unblock deferred eviction metrics.
            dropped_for_metrics.extend(
                self._release_drop_metrics_for_ended_locked(span.context.span_id)
            )

            still_live = bool(self._live_parents.get(trace_id))
            if still_live:
                completed_batches = self._schedule_nested_completion_locked(trace_id)
                # Bound ended spans under live roots so one long-lived root
                # cannot grow `_buffers` without limit before max_nodes trim.
                dropped_for_metrics.extend(self._trim_live_buffer_locked(trace_id))
            elif self._buffers.get(trace_id):
                self._cancel_pending_nested_completion_locked(trace_id)
                completed_batches = self._schedule_completion_locked(trace_id)
            else:
                completed_batches = []

            self._pending_finalize += len(completed_batches)
            self._note_inflight_batches_locked(completed_batches)
            if self._total_live_locked() == 0 and self._pending_finalize == 0:
                self._idle.notify_all()
        if dropped_for_metrics:
            self._record_metrics_for_dropped_spans(
                dropped_for_metrics, trace_id=trace_id
            )
        # Always queue finalize/export (including zero-holdback) so Span.end
        # never blocks on a slow exporter and the bounded backlog applies.
        self._dispatch_accept_completed(completed_batches)

    def _run_accept_completed(self, batch: Sequence[ReadableSpan]) -> None:
        """Run finalize/export off the caller's contract: never raise to Span.end."""
        stranded_metrics: List[ReadableSpan] = []
        try:
            self._accept_completed_trace(batch)
        except Exception:
            _LOG.exception(
                "TransactionSpanProcessor failed while accepting a completed trace"
            )
        finally:
            with self._lock:
                self._pending_finalize -= 1
                self._finish_inflight_batch_locked(batch)
                # After this batch is no longer inflight, idle TraceIDs can drop
                # side tables (nested+outer may share a TraceID across batches).
                stranded_metrics = self._release_completed_traces_locked(batch)
                self._idle.notify_all()
        self._record_stranded_drop_metrics(stranded_metrics)

    def _record_metrics_for_dropped_spans(
        self, spans: Sequence[ReadableSpan], *, trace_id: int
    ) -> None:
        """Record self-duration metrics for spans evicted before export."""
        recorded_ids: Set[int] = set()
        try:
            with self._lock:
                interval_snapshot: Dict[int, List[Tuple[int, int]]] = {
                    span.context.span_id: list(
                        self._child_intervals.get(span.context.span_id, [])
                    )
                    for span in spans
                    if span.context is not None
                }
                membership_snapshot: Dict[int, TransactionMembership] = {}
                for span in spans:
                    if span.context is None:
                        continue
                    member = self._membership.get(span.context.span_id)
                    if member is None:
                        continue
                    membership_snapshot[span.context.span_id] = member
                    root_member = self._membership.get(member.root_span_id)
                    if root_member is not None:
                        membership_snapshot[member.root_span_id] = root_member

                groups: Dict[int, List[ReadableSpan]] = {}
                for span in spans:
                    if span.context is None:
                        continue
                    member = membership_snapshot.get(span.context.span_id)
                    root_id = (
                        member.root_span_id
                        if member is not None
                        else span.context.span_id
                    )
                    groups.setdefault(root_id, []).append(span)

                group_names: Dict[int, str] = {}
                annotate_groups: Dict[int, List[ReadableSpan]] = {}
                for root_id, group in groups.items():
                    name = self._metrics_transaction_name_locked(trace_id, root_id)
                    if name is None:
                        # Root still open without a final/override name — wait.
                        for span in group:
                            if span.context is None:
                                continue
                            sid = span.context.span_id
                            self._pending_drop_metrics[sid] = span
                            self._pending_drop_waiters.setdefault(sid, set()).add(
                                root_id
                            )
                        continue
                    group_names[root_id] = name
                    annotate_groups[root_id] = group

            for root_id, group in annotate_groups.items():
                annotate_completed_batch(
                    group,
                    child_intervals=interval_snapshot,
                    membership=membership_snapshot,
                    self_duration_hist=self._self_duration_hist,
                    transaction_name=group_names.get(root_id),
                )
            # Spans deferred for root rename keep their child_intervals.
            recorded_ids = {
                span.context.span_id
                for group in annotate_groups.values()
                for span in group
                if span.context is not None
            }
        except Exception:
            _LOG.exception(
                "TransactionSpanProcessor failed while recording metrics for "
                "live-buffer overflow spans"
            )
        finally:
            with self._lock:
                for span in spans:
                    if span.context is None:
                        continue
                    sid = span.context.span_id
                    if sid in recorded_ids and sid not in self._pending_drop_metrics:
                        self._child_intervals.pop(sid, None)

    def _forget_span_locked(self, span_id: int) -> None:
        """Drop side-table rows for a span that will not be exported."""
        self._membership.pop(span_id, None)
        self._span_contexts.pop(span_id, None)
        self._span_parent_ids.pop(span_id, None)
        self._parent_rebind.pop(span_id, None)
        self._pending_drop_metrics.pop(span_id, None)
        self._pending_drop_waiters.pop(span_id, None)
        self._root_final_names.pop(span_id, None)

    def _batch_trace_id(self, batch: Sequence[ReadableSpan]) -> Optional[int]:
        for span in batch:
            if span.context is not None:
                return int(span.context.trace_id)
        return None

    def _note_inflight_batches_locked(
        self, batches: Sequence[Sequence[ReadableSpan]]
    ) -> None:
        for batch in batches:
            trace_id = self._batch_trace_id(batch)
            if trace_id is None:
                continue
            self._inflight_batches_by_trace[trace_id] = (
                self._inflight_batches_by_trace.get(trace_id, 0) + 1
            )

    def _finish_inflight_batch_locked(self, batch: Sequence[ReadableSpan]) -> None:
        trace_id = self._batch_trace_id(batch)
        if trace_id is None:
            return
        left = self._inflight_batches_by_trace.get(trace_id, 0) - 1
        if left <= 0:
            self._inflight_batches_by_trace.pop(trace_id, None)
        else:
            self._inflight_batches_by_trace[trace_id] = left

    def _nearest_retained_ancestor_locked(
        self, trace_id: int, span_id: int
    ) -> Optional[int]:
        """Walk compact parents past live-trim evictions to a retained ancestor."""
        evicted = self._evicted_from_buffer.get(trace_id) or set()
        live = self._live_parents.get(trace_id) or {}
        buffered_ids = {
            span.context.span_id
            for span in self._buffers.get(trace_id) or []
            if span.context is not None
        }
        cur = self._span_parent_ids.get(span_id, 0)
        seen: Set[int] = set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur not in evicted and (
                cur in live or cur in buffered_ids or cur in self._span_contexts
            ):
                return cur
            cur = self._span_parent_ids.get(cur, 0)
        return None

    def _metrics_transaction_name_locked(
        self, trace_id: int, root_id: int
    ) -> Optional[str]:
        """Resolve eviction-metric txn name; None means defer until root is final."""
        root_member = self._membership.get(root_id)
        if root_member is not None:
            if root_member.override_name:
                return root_member.override_name
            if root_member.inherited_name:
                return root_member.inherited_name
        live = self._live_parents.get(trace_id) or {}
        if root_id in live:
            # Root still open: start_name is not final (frameworks may update_name).
            return None
        final_name = self._root_final_names.get(root_id)
        if final_name:
            return final_name
        for span in self._buffers.get(trace_id) or []:
            if span.context is not None and span.context.span_id == root_id:
                return span.name
        if root_member is not None and root_member.start_name:
            return root_member.start_name
        return None

    def _release_idle_trace_side_tables_locked(self, trace_id: int) -> None:
        """Forget compact ancestry once a TraceID has nothing live, buffered, or queued."""
        if (
            self._live_parents.get(trace_id)
            or self._buffers.get(trace_id)
            or self._inflight_batches_by_trace.get(trace_id, 0) > 0
        ):
            return
        self._evicted_from_buffer.pop(trace_id, None)
        for span_id in self._trace_span_ids.pop(trace_id, set()):
            self._child_intervals.pop(span_id, None)
            self._forget_span_locked(span_id)

    def _flush_pending_drop_metrics_locked(self, trace_id: int) -> List[ReadableSpan]:
        """Return and clear every deferred eviction still pending on ``trace_id``."""
        ready: List[ReadableSpan] = []
        pending_ids = [
            span_id
            for span_id, span in self._pending_drop_metrics.items()
            if span.context is not None and span.context.trace_id == trace_id
        ]
        for span_id in pending_ids:
            span = self._pending_drop_metrics.pop(span_id, None)
            self._pending_drop_waiters.pop(span_id, None)
            if span is not None:
                ready.append(span)
        return ready

    def _release_drop_metrics_for_ended_locked(
        self, ended_span_id: int
    ) -> List[ReadableSpan]:
        """Unblock deferred eviction metrics once a waiting live child ends."""
        ready: List[ReadableSpan] = []
        for dropped_id, waiters in list(self._pending_drop_waiters.items()):
            if ended_span_id not in waiters:
                continue
            waiters.discard(ended_span_id)
            if waiters:
                continue
            self._pending_drop_waiters.pop(dropped_id, None)
            span = self._pending_drop_metrics.pop(dropped_id, None)
            if span is not None:
                ready.append(span)
        return ready

    def _open_transaction_root_ids_locked(self, trace_id: int) -> Set[int]:
        """Roots that still have at least one live member on this TraceID."""
        live = self._live_parents.get(trace_id) or {}
        open_roots: Set[int] = set()
        for live_id in live:
            member = self._membership.get(live_id)
            if member is not None:
                open_roots.add(member.root_span_id)
            else:
                open_roots.add(live_id)
        return open_roots

    def _trim_live_buffer_locked(self, trace_id: int) -> List[ReadableSpan]:
        """Evict surplus ended spans under open transactions; return for metrics.

        Each open local transaction gets its own ``max_nodes`` budget. Transaction
        roots are protected; ended ancestors of live spans are not retained — live
        descendants are rebound to the nearest kept ancestor instead. Caller must
        hold ``_lock``.
        """
        if self._max_nodes <= 0:
            return []
        buf = self._buffers.get(trace_id)
        if buf is None:
            return []

        live = self._live_parents.get(trace_id) or {}
        open_root_ids = self._open_transaction_root_ids_locked(trace_id)
        frozen: List[ReadableSpan] = []
        by_root: Dict[int, List[ReadableSpan]] = {}
        for span in buf:
            if span.context is None:
                continue
            member = self._membership.get(span.context.span_id)
            root_id = (
                member.root_span_id if member is not None else span.context.span_id
            )
            if root_id in open_root_ids:
                by_root.setdefault(root_id, []).append(span)
            else:
                frozen.append(span)

        needs_trim = False
        for root_id, spans in by_root.items():
            reserve = 1 if root_id in live else 0
            if len(spans) > max(0, self._max_nodes - reserve):
                needs_trim = True
                break
        if not needs_trim:
            return []

        kept_open: List[ReadableSpan] = []
        dropped: List[ReadableSpan] = []
        for root_id, spans in by_root.items():
            reserve = 1 if root_id in live else 0
            cap = max(0, self._max_nodes - reserve)
            if len(spans) <= cap:
                kept_open.extend(spans)
                continue

            protect: List[str] = []
            for span in spans:
                if span.context is None:
                    continue
                sid = span.context.span_id
                member = self._membership.get(sid)
                attrs = span.attributes or {}
                if (member is not None and member.is_root) or attrs.get(
                    CoralogixAttributes.TRANSACTION_ROOT
                ):
                    protect.append(format_span_id(sid))

            # Strict per-transaction cap. Do not expand for ancestry — rebind
            # live descendants below instead of retaining every ended parent.
            trimmed = (
                []
                if cap == 0
                else select_slowest_spans(spans, max_nodes=cap, root_span_ids=protect)
            )
            kept_ids = {
                span.context.span_id for span in trimmed if span.context is not None
            }
            dropped.extend(
                span
                for span in spans
                if span.context is not None and span.context.span_id not in kept_ids
            )
            kept_open.extend(trimmed)

        dropped_ids = {
            span.context.span_id for span in dropped if span.context is not None
        }
        parent_of: Dict[int, int] = dict(self._span_parent_ids)
        parent_of.update(live)
        for span in buf:
            if (
                span.context is not None
                and span.parent is not None
                and span.parent.is_valid
            ):
                parent_of[span.context.span_id] = span.parent.span_id

        # Defer metrics for dropped ancestors of still-live spans so child
        # intervals can still be attributed before self-duration is recorded.
        ready: List[ReadableSpan] = []
        if dropped_ids and live:
            live_ancestor_ids: Set[int] = set()
            for live_id in live:
                cur = live_id
                seen: Set[int] = set()
                while cur and cur not in seen:
                    live_ancestor_ids.add(cur)
                    seen.add(cur)
                    cur = parent_of.get(cur, 0)
            for span in dropped:
                if span.context is None:
                    continue
                sid = span.context.span_id
                if sid not in live_ancestor_ids:
                    ready.append(span)
                    continue
                self._pending_drop_metrics[sid] = span
                waiters = self._pending_drop_waiters.setdefault(sid, set())
                for live_id in live:
                    cur = live_id
                    seen = set()
                    while cur and cur not in seen:
                        if cur == sid:
                            waiters.add(live_id)
                            break
                        seen.add(cur)
                        cur = parent_of.get(cur, 0)
        else:
            ready = list(dropped)

        if dropped_ids and live:
            for live_id in list(live.keys()):
                parent_id = live.get(live_id, 0)
                seen = set()
                while parent_id and parent_id in dropped_ids and parent_id not in seen:
                    seen.add(parent_id)
                    parent_id = parent_of.get(parent_id, 0)
                if parent_id != live.get(live_id, 0):
                    live[live_id] = parent_id
                    if parent_id:
                        self._parent_rebind[live_id] = parent_id
                        self._span_parent_ids[live_id] = parent_id
                    else:
                        self._parent_rebind.pop(live_id, None)
                        self._span_parent_ids.pop(live_id, None)

        self._buffers[trace_id] = frozen + kept_open
        if dropped_ids:
            self._evicted_from_buffer.setdefault(trace_id, set()).update(dropped_ids)
        if dropped:
            _LOG.error(
                "TransactionSpanProcessor live buffer over max_nodes=%d; "
                "dropping %d ended span(s) under open transaction(s)",
                self._max_nodes,
                len(dropped),
            )
        return ready

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
                self._finish_inflight_batch_locked(batch)
                for span in batch:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)
                        self._forget_span_locked(span.context.span_id)
                stranded = self._release_completed_traces_locked(batch)
                self._idle.notify_all()
            self._record_stranded_drop_metrics(stranded)

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
            deferred = list(self._deferred_finalize)
            self._deferred_finalize = []
            new_batches = self._flush_pending_completions_locked()
            self._note_inflight_batches_locked(new_batches)
            batches = deferred + new_batches
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
                deferred = list(self._deferred_finalize)
                self._deferred_finalize = []
                extracted: List[List[ReadableSpan]] = []
                for trace_id in list(self._buffers.keys()):
                    if self._live_parents.get(trace_id):
                        dropped = self._buffers.pop(trace_id, None) or []
                        self._live_parents.pop(trace_id, None)
                        for span in dropped:
                            if span.context is not None:
                                self._child_intervals.pop(span.context.span_id, None)
                                self._forget_span_locked(span.context.span_id)
                        continue
                    extracted.extend(
                        self._extract_completed_local_transactions_locked(
                            trace_id, flush_leftover=True
                        )
                    )
                self._note_inflight_batches_locked(holdback_batches + extracted)
                batches = deferred + holdback_batches + extracted
                self._pending_finalize += len(batches)
                self._buffers.clear()
                self._live_parents.clear()

            for batch in batches:
                self._run_accept_completed(batch)

            with self._lock:
                while self._pending_finalize > 0:
                    self._idle.wait()

            with self._lock:
                self._exporter_shutdown = True
                self._child_intervals.clear()
                self._membership.clear()
                self._span_contexts.clear()
                self._span_parent_ids.clear()
                self._parent_rebind.clear()
                self._trace_span_ids.clear()
                self._pending_drop_metrics.clear()
                self._pending_drop_waiters.clear()
                self._inflight_batches_by_trace.clear()
                self._evicted_from_buffer.clear()
                self._root_final_names.clear()
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
        # Evicted parents awaiting deferred metrics still need child intervals.
        if parent_id in self._pending_drop_metrics:
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
                self._note_inflight_batches_locked(batches)
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
                self._note_inflight_batches_locked(batches)
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
                        self._forget_span_locked(span.context.span_id)
            return

        try:
            self._export_spans(trimmed)
        finally:
            with self._lock:
                for span in annotated:
                    if span.context is not None:
                        self._child_intervals.pop(span.context.span_id, None)
                        self._forget_span_locked(span.context.span_id)

    def _record_stranded_drop_metrics(self, spans: Sequence[ReadableSpan]) -> None:
        by_trace: Dict[int, List[ReadableSpan]] = {}
        for span in spans:
            if span.context is None:
                continue
            by_trace.setdefault(span.context.trace_id, []).append(span)
        for trace_id, group in by_trace.items():
            self._record_metrics_for_dropped_spans(group, trace_id=trace_id)

    def _release_completed_traces_locked(
        self, spans: Sequence[ReadableSpan]
    ) -> List[ReadableSpan]:
        """After a batch is done, drop idle TraceID side tables; return stranded metrics."""
        stranded: List[ReadableSpan] = []
        trace_ids = {
            span.context.trace_id for span in spans if span.context is not None
        }
        for trace_id in trace_ids:
            if self._live_parents.get(trace_id) or self._buffers.get(trace_id):
                continue
            stranded.extend(self._flush_pending_drop_metrics_locked(trace_id))
            self._release_idle_trace_side_tables_locked(trace_id)
        return stranded

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
