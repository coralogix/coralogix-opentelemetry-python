# coralogix-opentelemetry-python

Coralogix extensions for OpenTelemetry Python.

## Transaction processor

Use `TransactionSpanProcessor` for transaction tagging, exclusive self-duration,
and the self-duration metric. The legacy `CoralogixTransactionSampler` remains
available for backward compatibility only.

### Defaults and harvest (read carefully)

By default the processor keeps at most **256** slowest spans per local trace
(`max_nodes`) and exports only the **slowest** completed local trace every
**60s** (`max_regular_traces=1`). That is intentional New Relic-style harvest
sampling: most completed local traces are **not** exported as full waterfalls.

Harvest losers still:

- record `cgx.transaction.self_duration` histogram points for every span in the
  completed local tree
- export a **root stub** span so APM still sees the transaction name

Set `max_regular_traces=0` (or `harvest_period_millis=0`) to export every
completed (trimmed) local trace immediately with no harvest sampling.

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
| `max_nodes` | int | `256` | `OTEL_CX_TRANSACTION_MAX_NODES` | Max spans kept per completed local trace (slowest first; txn root always kept) |
| `max_regular_traces` | int | `1` | `OTEL_CX_TRANSACTION_MAX_REGULAR_TRACES` | Slowest completed local traces kept per harvest window. `0` = export every completed trimmed trace immediately |
| `harvest_period_millis` | int | `60000` | `OTEL_CX_TRANSACTION_HARVEST_PERIOD_MILLIS` | Harvest flush interval. `<= 0` exports every completed trace immediately (no heap) |
| `completion_holdback_millis` | int | `100` | `OTEL_CX_TRANSACTION_COMPLETION_HOLDBACK_MILLIS` | After the last live span on a TraceID ends, wait so fire-and-forget children can join. `0` = finalize immediately |
| `meter_provider` | MeterProvider | global | — | MeterProvider for the self-duration histogram |

### Attributes

| Key | Meaning |
|---|---|
| `cgx.transaction` | Local transaction name (stamped at export from the root’s final name, or an explicit override) |
| `cgx.transaction.root` | `true` on transaction starters (set at start) |
| `cgx.transaction.self_duration` | Exclusive wall duration (seconds) |

### Metric

Histogram `cgx.transaction.self_duration` (unit `s`) is always recorded for every
span in every completed local trace, including harvest losers.

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
not double-counted.
