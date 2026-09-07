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

## Benchmark

To help users estimate resource usage, we ran this benchmark. The first table processes 10,000 traces at each depth from 8 to 2,048 spans to show the impact of increasingly deep transactions. The second table processes traces with a depth of 1,000 spans at increasing trace counts to show the effect of transaction volume.

### 10000 traces by depth

| Depth | Traces | RSS base MiB | RSS peak MiB | RSS delta MiB | Spans/s |
|---:|---:|---:|---:|---:|---:|
|  8  |  10000  | 24.80 | 34.60 | 9.80 | 27013.20 |
|  16  |  10000  | 24.80 | 36.70 | 11.90 | 28127.70 |
|  32  |  10000  | 24.50 | 36.00 | 11.50 | 25983.00 |
|  64  |  10000  | 25.00 | 31.60 | 6.70 | 21062.30 |
|  128  |  10000  | 24.70 | 30.20 | 5.50 | 14107.20 |
|  256  |  10000  | 24.60 | 33.50 | 8.90 | 8738.00 |
|  512  |  10000  | 24.60 | 35.10 | 10.50 | 72827.70 |
|  1024  |  10000  | 24.60 | 34.70 | 10.10 | 80321.10 |
|  2048  |  10000  | 24.80 | 40.10 | 15.30 | 86220.70 |

### Depth 1000 by trace count

| Depth | Traces | RSS base MiB | RSS peak MiB | RSS delta MiB | Spans/s |
|---:|---:|---:|---:|---:|---:|
|  1000  |  100  | 24.50 | 32.50 | 8.00 | 81080.20 |
|  1000  |  1000  | 24.40 | 33.80 | 9.40 | 81081.70 |
|  1000  |  10000  | 24.70 | 36.30 | 11.60 | 81337.00 |
|  1000  |  100000  | 24.40 | 38.00 | 13.60 | 82765.80 |
