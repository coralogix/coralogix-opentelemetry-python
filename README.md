# coralogix-opentelemetry-python

Coralogix extensions for OpenTelemetry Python.

## Transaction processor

Use `TransactionSpanProcessor` for transaction tagging, exclusive self-duration,
and the self-duration metric. The legacy `CoralogixTransactionSampler` remains
available for backward compatibility only.

### Defaults

The processor **exports every completed local trace in full**. Transactions of
at most **256** spans receive transaction tags, self-duration attributes, and
metrics. On the next ended span, larger transactions flush the buffered spans
raw and proxy later spans without processor-added tags or self-duration metrics.
By default, concurrently buffered traces are unlimited; set
`CORALOGIX_MAX_TRANSACTION_TRACES` to a positive value to bound them.
Constructor keyword arguments override environment variables. When a keyword is
omitted, the matching env var is read; invalid values fall back to the default.

```python
from coralogix_opentelemetry.trace.processors import TransactionSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(TransactionSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

### Options

| Option | Type | Default | Env var | Meaning |
|---|---|---|---|---|
| `completion_holdback_millis` | int | `100` | `OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS` | After the last live span on a TraceID ends, wait so fire-and-forget children can join. `0` = finalize immediately. Negative → default |
| `max_transaction_spans` | int | `256` | `CORALOGIX_MAX_SPANS_PER_TRACE` | Maximum spans to buffer and enrich per trace. On the next span, export the whole trace raw. `0` = unlimited |
| `max_traces` | int | `0` | `CORALOGIX_MAX_TRANSACTION_TRACES` | Maximum transactions retained in memory while their spans are still live or awaiting completion. Once full, newly seen transactions pass through raw until buffered transactions finish. `0` = unlimited |
| `meter_provider` | MeterProvider | global | — | MeterProvider for the self-duration histogram |

Requires OpenTelemetry API/SDK **1.21+** (metrics API and `ReadableSpan.instrumentation_scope`).

### Attributes

| Key | Meaning |
|---|---|
| `cgx.transaction` | Local transaction name (stamped at export from the root’s final name, or an explicit override) |
| `cgx.transaction.root` | `true` on transaction starters (set at start) |
| `cgx.transaction.self_duration` | Exclusive wall duration (seconds) |

### Metric

Histogram `cgx.transaction.self_duration` (unit `s`) is recorded for every span
in completed local transactions of at most 256 spans.

### Transaction boundaries

A span starts a new local transaction when there is no parent local transaction,
the parent is remote, or the span kind is `SERVER` / `CONSUMER`. Each process
owns its own local transaction; `cgx.transaction.distributed` is not used.

`cgx.transaction` is **not** frozen on start. Frameworks may rename the root
(for example `GET` → `GET /myroute`); the final name is stamped onto the batch
at export. `start_new_transaction(span, name)` sets an explicit override that
wins over the root’s final span name.

### Exclusive self-duration

Self-duration is the span’s wall duration minus time covered by direct children.
Child intervals are clamped to the parent and merged so overlapping children are
not double-subtracted.
