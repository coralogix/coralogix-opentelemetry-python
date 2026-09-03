"""Transaction SpanProcessor: naming, self-duration metrics, and export.

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

1. Until the configured span cap on a TraceID, stamp ``cgx.transaction`` from
   ``override_name ?? root.final_name``, compute exclusive self-duration, and
   record the matching histogram (unit ``s``).
2. On that 257th span, flush ended buffered spans raw and proxy later spans
   unchanged, without transaction tags or self-duration metrics.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import weakref
from typing import Dict, List, Optional, Sequence, Set, Tuple

from coralogix_opentelemetry.trace.common import CoralogixAttributes
from coralogix_opentelemetry.trace.processors.defaults import (
    resolve_completion_holdback_millis,
    resolve_max_traces,
    resolve_max_transaction_spans,
)
from coralogix_opentelemetry.trace.processors.holdback_scheduler import (
    HoldbackScheduler,
)
from coralogix_opentelemetry.trace.processors.span_copy import copy_with_attributes
from coralogix_opentelemetry.trace.processors.start_new_transaction import (
    explicit_transaction_name,
)
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
from opentelemetry.context import (
    Context,
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
)
from opentelemetry.metrics import Histogram, MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import get_current_span

_LOG = logging.getLogger(__name__)

# HoldbackScheduler keys: distinguish idle vs nested arms for the same TraceID.
_HOLD_IDLE = "idle"
_HOLD_NESTED = "nested"
_HOLD_PASSTHROUGH = "passthrough"
# Cap queued completed batches so a stalled exporter cannot grow memory without bound.
DEFAULT_MAX_FINALIZE_QUEUE = 256
# Cap batches retained across timed-out force_flush calls (same order as the queue).
DEFAULT_MAX_DEFERRED_FINALIZE = DEFAULT_MAX_FINALIZE_QUEUE


def _restart_processor_after_fork(
    processor_ref: "weakref.ReferenceType[TransactionSpanProcessor]",
) -> None:
    processor = processor_ref()
    if processor is not None:
        processor._restart_after_fork()


class TransactionSpanProcessor(SpanProcessor):
    """Full transaction tagging + conditional self-duration + full export."""

    def __init__(
        self,
        span_exporter: SpanExporter,
        *,
        completion_holdback_millis: Optional[int] = None,
        meter_provider: Optional[MeterProvider] = None,
        max_transaction_spans: Optional[int] = None,
        max_traces: Optional[int] = None,
    ) -> None:
        self._exporter = span_exporter
        self._completion_holdback_millis = resolve_completion_holdback_millis(
            completion_holdback_millis
        )
        self._max_transaction_spans = resolve_max_transaction_spans(
            max_transaction_spans
        )
        self._max_traces = resolve_max_traces(max_traces)
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._buffers: Dict[int, List[ReadableSpan]] = {}
        # The 257th started span switches this TraceID to raw passthrough.
        self._passthrough_traces: Set[int] = set()
        self._pending_passthrough_cleanup: Dict[int, int] = {}
        self._live_parents: Dict[int, Dict[int, int]] = {}
        # Raw traces need only a count to know when their tombstone can expire.
        self._passthrough_live_counts: Dict[int, int] = {}
        # Coalesce rapid raw ends per trace so they do not fill the worker queue.
        # A batch is capped at the queue capacity; further batches go through the
        # existing bounded queue/deferred/drop path.
        self._raw_pending_by_trace: Dict[int, List[ReadableSpan]] = {}
        self._raw_exporting_traces: Dict[int, int] = {}
        self._membership: Dict[Tuple[int, int], TransactionMembership] = {}
        self._root_memberships_by_trace: Dict[int, Set[int]] = {}
        # Batches extracted but not yet accept/abandon-finished, per TraceID.
        # Side tables must outlive the first queued batch when several share a
        # TraceID (nested + outer extracted together).
        self._inflight_batches_by_trace: Dict[int, int] = {}
        # Merged child intervals retain exact geometry for backdated siblings.
        # Transaction span caps bound this per-parent state.
        self._child_intervals: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._live_child_starts: Dict[Tuple[int, int], Dict[int, int]] = {}
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
        self._deferred_raw_finalize: List[List[ReadableSpan]] = []
        self._self_duration_hist: Histogram = create_self_duration_histogram(
            meter_provider
        )
        if hasattr(os, "register_at_fork"):
            processor_ref = weakref.ref(self)
            os.register_at_fork(
                after_in_child=lambda: _restart_processor_after_fork(processor_ref)
            )

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
        self._finalize_queue = queue.Queue(maxsize=DEFAULT_MAX_FINALIZE_QUEUE)
        self._finalize_stop = threading.Event()
        self._pending_finalize = 0
        self._deferred_finalize = []
        self._deferred_raw_finalize = []
        self._buffers.clear()
        self._passthrough_traces.clear()
        self._pending_passthrough_cleanup.clear()
        self._live_parents.clear()
        self._passthrough_live_counts.clear()
        self._raw_pending_by_trace.clear()
        self._raw_exporting_traces.clear()
        self._membership.clear()
        self._root_memberships_by_trace.clear()
        self._pending_completions.clear()
        self._pending_nested_completions.clear()
        self._inflight_batches_by_trace.clear()
        self._child_intervals.clear()
        self._live_child_starts.clear()
        self._shutdown_started = False
        self._shutdown_done = threading.Event()
        if not self._exporter_shutdown:
            self._stopped = False
            self._holdback.restart_after_fork()
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

        trace_id = span.context.trace_id
        span_id = span.context.span_id
        raw_batches: List[List[ReadableSpan]] = []

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
                and trace_id not in self._passthrough_live_counts
            ):
                return
            if trace_id in self._passthrough_traces:
                self._cancel_passthrough_cleanup_locked(trace_id)
                self._passthrough_live_counts[trace_id] = (
                    self._passthrough_live_counts.get(trace_id, 0) + 1
                )
                return

            # Keep deciding inheritance and registering this live child in one
            # critical section. Otherwise its parent can end between the lookup
            # and registration and extract a separate transaction batch.
            parent_member = (
                self._membership.get((trace_id, parent_id)) if parent_id else None
            )
            inherited_from_ts = parent_transaction_from_tracestate(parent_span)
            trace_already_tracked = (
                trace_id in self._live_parents
                or trace_id in self._buffers
                or trace_id in self._inflight_batches_by_trace
            )
            if (
                self._max_traces > 0
                and not trace_already_tracked
                and len(
                    (
                        set(self._live_parents)
                        | set(self._buffers)
                        | set(self._inflight_batches_by_trace)
                    )
                    - self._passthrough_traces
                )
                >= self._max_traces
            ):
                self._passthrough_traces.add(trace_id)
                self._passthrough_live_counts[trace_id] = 1
                return
            parent_has_local = (
                parent_member is not None
                or parent_has_transaction_attrs(parent_span)
                or inherited_from_ts is not None
                or (bool(parent_id) and trace_already_tracked)
            )
            starts = starts_new_transaction(
                span_kind=span.kind,
                parent_context=parent_context,
                parent_has_local_transaction=parent_has_local,
            )
            original_attributes = dict(span.attributes or {})
            root_flag_added = apply_on_start_root_flag(span, starts)
            start_name = span.name
            preset = preset_transaction_name(span)

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
                if not (
                    starts
                    and parent_txn is not None
                    and (
                        preset == parent_txn
                        or (
                            parent_member is not None
                            and preset == parent_member.start_name
                        )
                    )
                ):
                    override = preset

            if starts:
                self._membership[(trace_id, span_id)] = TransactionMembership(
                    root_span_id=span_id,
                    is_root=True,
                    override_name=override,
                    inherited_name=inherited_from_ts
                    or parent_transaction_from_attrs(parent_span),
                    start_name=start_name,
                    root_flag_added=root_flag_added,
                    raw_attributes=original_attributes if root_flag_added else None,
                )
                self._root_memberships_by_trace.setdefault(trace_id, set()).add(span_id)
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
                # the parent's on_start recorded is_root=False. Promote it so
                # children join the nested transaction.
                parent_explicit_name = explicit_transaction_name(parent_span)
                if (
                    parent_member is not None
                    and parent_id
                    and (
                        (getattr(parent_span, "attributes", None) or {}).get(
                            CoralogixAttributes.TRANSACTION_ROOT
                        )
                        is True
                        or (
                            explicit_transaction_override(parent_span)
                            and parent_transaction_from_attrs(parent_span) is not None
                        )
                        or parent_explicit_name is not None
                    )
                ):
                    parent_member.is_root = True
                    parent_member.root_span_id = parent_id
                    self._root_memberships_by_trace.setdefault(trace_id, set()).add(
                        parent_id
                    )
                    parent_preset = (
                        parent_explicit_name
                        or parent_transaction_from_attrs(parent_span)
                    )
                    if parent_preset is not None:
                        parent_member.override_name = parent_preset
                    root_span_id = parent_id
                self._membership[(trace_id, span_id)] = TransactionMembership(
                    root_span_id=root_span_id,
                    is_root=False,
                    override_name=None,
                    inherited_name=inherited_name,
                    start_name=start_name,
                    raw_attributes=original_attributes if root_flag_added else None,
                )

            self._cancel_pending_completion_locked(trace_id)
            live = self._live_parents.setdefault(trace_id, {})
            live[span_id] = parent_id
            if parent_id and self._is_local_parent_locked(trace_id, parent_id):
                self._live_child_starts.setdefault((trace_id, parent_id), {})[
                    span_id
                ] = int(span.start_time or 0)
            if (
                len(self._buffers.get(trace_id, [])) + len(live)
                > self._max_transaction_spans
            ):
                self._cancel_pending_completion_locked(trace_id)
                self._cancel_pending_nested_completion_locked(trace_id)
                self._passthrough_traces.add(trace_id)
                self._passthrough_live_counts[trace_id] = len(live)
                self._live_parents.pop(trace_id, None)
                raw = self._buffers.pop(trace_id, [])
                if raw:
                    self._schedule_raw_batch_locked(raw)
                    raw_batches.append(raw)
        self._dispatch_raw_exports(raw_batches)

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
        explicit_name = explicit_transaction_name(span)
        trace_id = span.context.trace_id
        with self._lock:
            member = self._membership.get((trace_id, span.context.span_id))
            if member is not None and (
                attrs.get(CoralogixAttributes.TRANSACTION_ROOT)
                or (explicit_transaction_override(span) and preset is not None)
                or explicit_name is not None
            ):
                # Dynamic root (e.g. start_new_transaction on an INTERNAL child).
                member.is_root = True
                member.root_span_id = span.context.span_id
                self._root_memberships_by_trace.setdefault(trace_id, set()).add(
                    span.context.span_id
                )
            if member is not None and explicit_name is not None:
                member.helper_added = True
            if (
                member is not None
                and member.is_root
                and (preset is not None or explicit_name is not None)
            ):
                if explicit_name is not None:
                    member.override_name = explicit_name
                elif explicit_transaction_override(span):
                    member.override_name = str(preset)
                elif (
                    not member.override_name
                    and member.start_name is not None
                    and str(preset) != member.start_name
                ):
                    parent_txn = member.inherited_name
                    parent_member = None
                    if span.parent is not None and span.parent.is_valid:
                        parent_member = self._membership.get(
                            (trace_id, span.parent.span_id)
                        )
                        if parent_member is not None:
                            parent_txn = (
                                parent_member.override_name
                                or parent_member.inherited_name
                                or parent_member.start_name
                            )
                    # Ignore sampler copy of the outer txn onto a nested root.
                    if parent_txn is None or (
                        str(preset) != parent_txn
                        and (
                            parent_member is None
                            or str(preset) != parent_member.start_name
                        )
                    ):
                        member.override_name = str(preset)
        original_parent_id = 0
        if span.parent is not None and span.parent.is_valid:
            original_parent_id = span.parent.span_id

        completed_batches: List[List[ReadableSpan]] = []
        raw_batches: List[List[ReadableSpan]] = []
        with self._lock:
            if self._exporter_shutdown:
                return
            tracked = (
                trace_id in self._live_parents
                or trace_id in self._buffers
                or trace_id in self._passthrough_live_counts
            )
            if self._stopped and not tracked:
                return
            if trace_id in self._passthrough_traces:
                remaining = self._passthrough_live_counts.get(trace_id, 0) - 1
                if remaining <= 0:
                    self._passthrough_live_counts.pop(trace_id, None)
                else:
                    self._passthrough_live_counts[trace_id] = remaining
                pending = self._raw_pending_by_trace.setdefault(trace_id, [])
                pending.append(span)
                if (
                    not self._raw_exporting_traces.get(trace_id)
                    or len(pending) >= DEFAULT_MAX_FINALIZE_QUEUE
                ):
                    raw_batches = [self._raw_pending_by_trace.pop(trace_id)]
                    self._schedule_raw_batch_locked(raw_batches[0])
                if not self._passthrough_live_counts.get(trace_id):
                    self._schedule_passthrough_cleanup_locked(trace_id)
            else:
                if (
                    original_parent_id
                    and span.start_time is not None
                    and span.end_time is not None
                    and self._is_local_parent_locked(trace_id, original_parent_id)
                ):
                    parent_key = (trace_id, original_parent_id)
                    live_children = self._live_child_starts.get(parent_key)
                    if live_children is not None:
                        live_children.pop(span.context.span_id, None)
                        if not live_children:
                            self._live_child_starts.pop(parent_key, None)
                    self._add_child_interval_locked(
                        trace_id, original_parent_id, span.start_time, span.end_time
                    )

                self._buffers.setdefault(trace_id, []).append(span)
                live = self._live_parents.get(trace_id)
                if live is not None:
                    live.pop(span.context.span_id, None)
                    if not live:
                        self._live_parents.pop(trace_id, None)

                if len(self._buffers[trace_id]) > self._max_transaction_spans:
                    self._cancel_pending_completion_locked(trace_id)
                    self._cancel_pending_nested_completion_locked(trace_id)
                    self._passthrough_traces.add(trace_id)
                    raw_batches = [self._buffers.pop(trace_id)]
                    self._schedule_raw_batch_locked(raw_batches[0])
                    self._passthrough_live_counts[trace_id] = len(
                        self._live_parents.pop(trace_id, {})
                    )
                    if not self._passthrough_live_counts.get(trace_id):
                        self._schedule_passthrough_cleanup_locked(trace_id)
                elif self._live_parents.get(trace_id):
                    # Wait for the complete trace so a later overflow cannot
                    # leave an already-exported nested batch enriched.
                    pass
                elif self._buffers.get(trace_id):
                    self._cancel_pending_nested_completion_locked(trace_id)
                    completed_batches = self._schedule_completion_locked(trace_id)

                self._pending_finalize += len(completed_batches)
                self._note_inflight_batches_locked(completed_batches)
            if self._total_live_locked() == 0 and self._pending_finalize == 0:
                self._idle.notify_all()
        # Both enriched and raw batches use the worker so Span.end never waits
        # for a stalled exporter.
        self._dispatch_accept_completed(completed_batches)
        self._dispatch_raw_exports(raw_batches)

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
                self._finish_inflight_batch_locked(batch)
                # After this batch is no longer inflight, idle TraceIDs can drop
                # side tables (nested+outer may share a TraceID across batches).
                self._idle.notify_all()

    def _run_raw_export(self, batch: Sequence[ReadableSpan]) -> None:
        """Export a trace that crossed the enrichment limit without blocking Span.end."""
        next_raw_batches: List[List[ReadableSpan]] = []
        try:
            with self._lock:
                raw = self._strip_processor_root_flags_locked(batch)
            self._export_spans(raw)
        finally:
            with self._lock:
                self._pending_finalize -= 1
                self._finish_inflight_batch_locked(batch)
                for span in batch:
                    if span.context is not None:
                        self._forget_span_locked(
                            span.context.trace_id, span.context.span_id
                        )
                trace_id = self._batch_trace_id(batch)
                next_raw_batches = self._advance_raw_export_chain_locked(batch)
                if (
                    trace_id is not None
                    and not self._live_parents.get(trace_id)
                    and not self._passthrough_live_counts.get(trace_id)
                    and trace_id not in self._inflight_batches_by_trace
                ):
                    self._schedule_passthrough_cleanup_locked(trace_id)
                self._idle.notify_all()
            self._dispatch_raw_exports(next_raw_batches)

    def _schedule_raw_batch_locked(self, batch: Sequence[ReadableSpan]) -> None:
        """Account for a raw batch before handing it to the bounded worker path."""
        trace_id = self._batch_trace_id(batch)
        if trace_id is not None:
            self._raw_exporting_traces[trace_id] = (
                self._raw_exporting_traces.get(trace_id, 0) + 1
            )
        self._pending_finalize += 1
        self._note_inflight_batches_locked([batch])

    def _advance_raw_export_chain_locked(
        self, batch: Sequence[ReadableSpan]
    ) -> List[List[ReadableSpan]]:
        """Finish one raw batch and schedule its bounded coalesced successor."""
        trace_id = self._batch_trace_id(batch)
        if trace_id is None:
            return []
        active = self._raw_exporting_traces.get(trace_id, 0)
        if active <= 1:
            self._raw_exporting_traces.pop(trace_id, None)
        else:
            self._raw_exporting_traces[trace_id] = active - 1
        pending = self._raw_pending_by_trace.pop(trace_id, [])
        if not pending:
            return []
        self._schedule_raw_batch_locked(pending)
        return [pending]

    def _strip_processor_root_flags_locked(
        self, spans: Sequence[ReadableSpan]
    ) -> List[ReadableSpan]:
        """Remove transaction metadata added by the processor before passthrough."""
        raw: List[ReadableSpan] = []
        for span in spans:
            if span.context is None:
                raw.append(span)
                continue
            member = self._membership.get((span.context.trace_id, span.context.span_id))
            helper_added = bool(
                (member is not None and member.helper_added)
                or explicit_transaction_name(span) is not None
            )
            if (member is None or not member.root_flag_added) and not helper_added:
                raw.append(span)
                continue
            attrs = (
                dict(member.raw_attributes)
                if member is not None
                and member.root_flag_added
                and member.raw_attributes is not None
                else dict(span.attributes or {})
            )
            if member is not None and member.root_flag_added:
                attrs.pop(CoralogixAttributes.TRANSACTION_ROOT, None)
            if helper_added:
                for key in (
                    CoralogixAttributes.TRANSACTION_IDENTIFIER,
                    CoralogixAttributes.TRANSACTION_ROOT,
                    CoralogixAttributes.TRANSACTION_EXPLICIT,
                ):
                    attrs.pop(key, None)
            raw.append(copy_with_attributes(span, attrs))
        return raw

    def _add_child_interval_locked(
        self, trace_id: int, parent_id: int, start: int, end: int
    ) -> None:
        """Accumulate exact merged child coverage for a parent."""
        parents = list(self._buffers.get(trace_id, []))
        for parent in parents:
            if parent.context is None or parent.context.span_id != parent_id:
                continue
            if parent.start_time is not None and parent.end_time is not None:
                start = max(start, parent.start_time)
                end = min(end, parent.end_time)
            break
        if end <= start:
            return
        parent_key = (trace_id, parent_id)
        prior = list(self._child_intervals.get(parent_key, []))
        prior.append((start, end))
        prior.sort()
        merged: List[Tuple[int, int]] = []
        for interval_start, interval_end in prior:
            if merged and interval_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
            else:
                merged.append((interval_start, interval_end))
        self._child_intervals[parent_key] = merged

    def _child_coverage_snapshot_locked(
        self, spans: Sequence[ReadableSpan]
    ) -> Dict[int, List[Tuple[int, int]]]:
        intervals = {
            span.context.span_id: list(
                self._child_intervals.get(
                    (span.context.trace_id, span.context.span_id), []
                )
            )
            for span in spans
            if span.context is not None
        }
        return intervals

    def _forget_span_locked(self, trace_id: int, span_id: int) -> None:
        """Drop side-table rows for a span that will not be exported."""
        span_key = (trace_id, span_id)
        member = self._membership.pop(span_key, None)
        if member is not None and member.is_root:
            roots = self._root_memberships_by_trace.get(trace_id)
            if roots is not None:
                roots.discard(span_id)
                if not roots:
                    self._root_memberships_by_trace.pop(trace_id, None)
        self._child_intervals.pop(span_key, None)
        self._live_child_starts.pop(span_key, None)

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

    def _abandon_completed_batch(
        self, batch: Sequence[ReadableSpan], *, adjust_pending: bool = True
    ) -> None:
        """Drop export for an unqueued batch; still record self-duration metrics."""
        try:
            with self._lock:
                interval_snapshot = self._child_coverage_snapshot_locked(batch)
                membership_snapshot = {
                    span.context.span_id: self._membership[
                        (span.context.trace_id, span.context.span_id)
                    ]
                    for span in batch
                    if span.context is not None
                    and (span.context.trace_id, span.context.span_id)
                    in self._membership
                }
            annotate_completed_batch(
                batch,
                child_intervals=interval_snapshot,
                membership=membership_snapshot,
                self_duration_hist=self._self_duration_hist,
                max_enriched_spans=self._max_transaction_spans,
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
                        self._forget_span_locked(
                            span.context.trace_id, span.context.span_id
                        )
                self._idle.notify_all()

    def _retain_deferred_batches(
        self, batches: Sequence[Sequence[ReadableSpan]]
    ) -> None:
        """Keep unqueued force_flush batches for retry, within a bounded cap."""
        overflow: List[List[ReadableSpan]] = []
        with self._lock:
            for batch in batches:
                payload = list(batch)
                if (
                    len(self._deferred_finalize) + len(self._deferred_raw_finalize)
                    >= DEFAULT_MAX_DEFERRED_FINALIZE
                ):
                    overflow.append(payload)
                    continue
                self._deferred_finalize.append(payload)
            self._idle.notify_all()
        self._retry_deferred_batches()
        for batch in overflow:
            _LOG.error(
                "TransactionSpanProcessor deferred finalize full "
                "(max=%d); dropping retained batch of %d span(s)",
                DEFAULT_MAX_DEFERRED_FINALIZE,
                len(batch),
            )
            self._abandon_completed_batch(batch)

    def _abandon_raw_batch(self, batch: Sequence[ReadableSpan]) -> None:
        """Drop an over-cap raw batch once all bounded retention is exhausted."""
        next_raw_batches: List[List[ReadableSpan]] = []
        with self._lock:
            self._pending_finalize -= 1
            self._finish_inflight_batch_locked(batch)
            for span in batch:
                if span.context is not None:
                    self._forget_span_locked(
                        span.context.trace_id, span.context.span_id
                    )
            next_raw_batches = self._advance_raw_export_chain_locked(batch)
            trace_id = self._batch_trace_id(batch)
            if (
                trace_id is not None
                and not self._live_parents.get(trace_id)
                and not self._passthrough_live_counts.get(trace_id)
                and trace_id not in self._inflight_batches_by_trace
            ):
                self._schedule_passthrough_cleanup_locked(trace_id)
            self._idle.notify_all()
        self._dispatch_raw_exports(next_raw_batches)

    def _retain_deferred_raw_batches(
        self, batches: Sequence[Sequence[ReadableSpan]]
    ) -> None:
        """Keep unqueued raw batches for retry within the same bounded cap."""
        overflow: List[List[ReadableSpan]] = []
        with self._lock:
            for batch in batches:
                payload = list(batch)
                if (
                    len(self._deferred_finalize) + len(self._deferred_raw_finalize)
                    >= DEFAULT_MAX_DEFERRED_FINALIZE
                ):
                    overflow.append(payload)
                    continue
                self._deferred_raw_finalize.append(payload)
            self._idle.notify_all()
        self._retry_deferred_batches()
        for batch in overflow:
            _LOG.error(
                "TransactionSpanProcessor deferred finalize full "
                "(max=%d); dropping raw batch of %d span(s)",
                DEFAULT_MAX_DEFERRED_FINALIZE,
                len(batch),
            )
            self._abandon_raw_batch(batch)

    def _retry_deferred_batches(self) -> None:
        """Move deferred batches back to the worker whenever capacity returns."""
        while True:
            with self._lock:
                if self._deferred_finalize:
                    item: object = self._deferred_finalize[0]
                elif self._deferred_raw_finalize:
                    item = ("raw", self._deferred_raw_finalize[0])
                else:
                    return
                try:
                    self._finalize_queue.put_nowait(item)
                except queue.Full:
                    return
                if self._deferred_finalize:
                    self._deferred_finalize.pop(0)
                else:
                    self._deferred_raw_finalize.pop(0)

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

        Hot path (no deadline): if the queue is full, retain the batch for a
        later force_flush rather than blocking callback threads.
        force_flush (deadline set): wait for capacity until the deadline; return
        False if any batch could not be queued in time. Unqueued force_flush
        batches are retained in ``_deferred_finalize`` for a later retry.
        """
        for index, batch in enumerate(batches):
            payload = list(batch)
            if self._enqueue_finalize_item(payload, deadline=deadline):
                continue
            if deadline is None:
                _LOG.debug(
                    "TransactionSpanProcessor finalize queue full "
                    "(max=%d); retaining completed batch of %d span(s) for force_flush",
                    DEFAULT_MAX_FINALIZE_QUEUE,
                    len(payload),
                )
                self._retain_deferred_batches([payload])
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

    def _dispatch_raw_exports(
        self,
        batches: Sequence[Sequence[ReadableSpan]],
        *,
        deadline: Optional[float] = None,
    ) -> bool:
        """Queue raw exports without blocking the ending application thread."""
        for index, batch in enumerate(batches):
            payload = list(batch)
            if self._enqueue_finalize_item(("raw", payload), deadline=deadline):
                continue
            if deadline is None:
                self._retain_deferred_raw_batches([payload])
                continue
            unqueued = [payload] + [list(rest) for rest in batches[index + 1 :]]
            self._retain_deferred_raw_batches(unqueued)
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
                if isinstance(item, tuple) and item[0] == "raw":
                    self._run_raw_export(item[1])
                else:
                    self._run_accept_completed(item)
            finally:
                self._finalize_queue.task_done()
                self._retry_deferred_batches()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        deadline = time.monotonic() + max(0.001, timeout_millis / 1000.0)
        with self._lock:
            if self._exporter_shutdown:
                return True
            deferred = list(self._deferred_finalize)
            self._deferred_finalize = []
            deferred_raw = list(self._deferred_raw_finalize)
            self._deferred_raw_finalize = []
            new_batches = self._flush_pending_completions_locked()
            self._note_inflight_batches_locked(new_batches)
            batches = deferred + new_batches
            self._pending_finalize += len(new_batches)
        # Wait for queue capacity within the deadline — retain on timeout.
        if not self._dispatch_accept_completed(batches, deadline=deadline):
            self._retain_deferred_raw_batches(deferred_raw)
            return False
        if not self._dispatch_raw_exports(deferred_raw, deadline=deadline):
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
                deferred_raw = list(self._deferred_raw_finalize)
                self._deferred_raw_finalize = []
                extracted: List[List[ReadableSpan]] = []
                raw_batches: List[List[ReadableSpan]] = []
                for trace_id in list(self._buffers.keys()):
                    if self._live_parents.get(trace_id):
                        raw_batches.append(self._buffers.pop(trace_id, []))
                        self._live_parents.pop(trace_id, None)
                        continue
                    extracted.extend(
                        self._extract_completed_local_transactions_locked(
                            trace_id, flush_leftover=True
                        )
                    )
                raw_batches.extend(self._raw_pending_by_trace.values())
                self._note_inflight_batches_locked(
                    holdback_batches + extracted + raw_batches
                )
                batches = deferred + holdback_batches + extracted
                self._pending_finalize += (
                    len(holdback_batches) + len(extracted) + len(raw_batches)
                )
                self._buffers.clear()
                self._live_parents.clear()
                self._passthrough_live_counts.clear()
                self._raw_pending_by_trace.clear()
                self._raw_exporting_traces.clear()

            for batch in batches:
                self._run_accept_completed(batch)
            for batch in deferred_raw + raw_batches:
                self._run_raw_export(batch)

            with self._lock:
                while self._pending_finalize > 0:
                    self._idle.wait()

            with self._lock:
                self._exporter_shutdown = True
                self._child_intervals.clear()
                self._live_child_starts.clear()
                self._membership.clear()
                self._root_memberships_by_trace.clear()
                self._inflight_batches_by_trace.clear()
                self._passthrough_traces.clear()
                self._pending_passthrough_cleanup.clear()
            with self._export_lock:
                self._exporter.shutdown()
        finally:
            self._holdback.shutdown()
            self._finalize_stop.set()
            self._finalize_worker.join(timeout=30.0)
            self._shutdown_done.set()

    def _total_live_locked(self) -> int:
        return sum(len(live) for live in self._live_parents.values()) + sum(
            self._passthrough_live_counts.values()
        )

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

    def _cancel_passthrough_cleanup_locked(self, trace_id: int) -> None:
        if self._pending_passthrough_cleanup.pop(trace_id, None) is not None:
            self._holdback.cancel((_HOLD_PASSTHROUGH, trace_id))

    def _schedule_passthrough_cleanup_locked(self, trace_id: int) -> None:
        self._cancel_passthrough_cleanup_locked(trace_id)

        token = 0

        def _fire() -> None:
            with self._lock:
                if self._pending_passthrough_cleanup.get(trace_id) != token:
                    return
                self._pending_passthrough_cleanup.pop(trace_id, None)
                if (
                    not self._live_parents.get(trace_id)
                    and not self._passthrough_live_counts.get(trace_id)
                    and trace_id not in self._inflight_batches_by_trace
                ):
                    self._passthrough_traces.discard(trace_id)

        token = self._holdback.schedule(
            (_HOLD_PASSTHROUGH, trace_id),
            self._completion_holdback_millis / 1000.0,
            _fire,
        )
        self._pending_passthrough_cleanup[trace_id] = token

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
        root_span_ids = self._root_memberships_by_trace.get(trace_id, set())
        batches, remaining = extract_completed_local_transactions(
            buffer=buffer,
            live=live,
            flush_leftover=flush_leftover,
            root_span_ids=root_span_ids,
        )
        if remaining:
            self._buffers[trace_id] = remaining
        else:
            self._buffers.pop(trace_id, None)
        return batches

    def _accept_completed_trace(self, spans: Sequence[ReadableSpan]) -> None:
        with self._lock:
            interval_snapshot = self._child_coverage_snapshot_locked(spans)
            membership_snapshot = {
                span.context.span_id: self._membership[
                    (span.context.trace_id, span.context.span_id)
                ]
                for span in spans
                if span.context is not None
                and (span.context.trace_id, span.context.span_id) in self._membership
            }
        annotated = annotate_completed_batch(
            spans,
            child_intervals=interval_snapshot,
            membership=membership_snapshot,
            self_duration_hist=self._self_duration_hist,
            max_enriched_spans=self._max_transaction_spans,
        )

        try:
            self._export_spans(annotated)
        finally:
            with self._lock:
                for span in annotated:
                    if span.context is not None:
                        self._forget_span_locked(
                            span.context.trace_id, span.context.span_id
                        )

    def _export_spans(self, spans: Sequence[ReadableSpan]) -> None:
        with self._export_lock:
            if self._exporter_shutdown:
                return
            token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
            try:
                try:
                    result = self._exporter.export(list(spans))
                except Exception:
                    _LOG.exception("TransactionSpanProcessor failed to export spans")
                    return
            finally:
                detach(token)
            if result is SpanExportResult.FAILURE:
                _LOG.warning("TransactionSpanProcessor exporter returned FAILURE")
