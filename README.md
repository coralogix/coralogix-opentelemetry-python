# coralogix-opentelemetry-python

Coralogix extensions for OpenTelemetry Python.

## Transaction processor

Use `TransactionSpanProcessor` for transaction tagging, exclusive self-time, and
the self-time metric. The legacy `CoralogixTransactionSampler` remains available
for backward compatibility only.

By default: keep at most **256**
slowest spans per local trace (`max_nodes`), and export only the **slowest**
completed local trace every **60s** (`max_regular_traces=1`). Self-time metrics
are still recorded for every completed local trace. Pass `max_regular_traces=0`
to export every completed (trimmed) trace immediately.

```python
from coralogix_opentelemetry.trace.processors import TransactionSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(TransactionSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

### Attributes

| Key | Meaning |
|---|---|
| `cgx.transaction` | Local transaction name |
| `cgx.transaction.root` | `true` on transaction starters |
| `cgx.transaction.self_time` | Exclusive wall time (seconds) |

### Metric

Histogram `cgx.transaction.self_time` (unit `s`) is always recorded.

### Transaction boundaries

A span starts a new local transaction when there is no parent transaction, the
parent is remote, or the span kind is `SERVER` / `CONSUMER`. Each process owns
its own local transaction; `cgx.transaction.distributed` is not used.
