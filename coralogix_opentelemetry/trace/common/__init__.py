from enum import Enum


def _attribute_to_trace_state(attribute_name: str) -> str:
    return attribute_name.replace(".", "_")


class CoralogixAttributes(str, Enum):
    TRANSACTION_IDENTIFIER = "cgx.transaction"
    DISTRIBUTED_TRANSACTION_IDENTIFIER = "cgx.transaction.distributed"
    TRANSACTION_ROOT = "cgx.transaction.root"
    # Marks an application start_new_transaction() override (not a sampler echo).
    TRANSACTION_EXPLICIT = "cgx.transaction.explicit"


class CoralogixTraceState:
    TRANSACTION_IDENTIFIER = _attribute_to_trace_state(
        CoralogixAttributes.TRANSACTION_IDENTIFIER
    )
    DISTRIBUTED_TRANSACTION_IDENTIFIER = _attribute_to_trace_state(
        CoralogixAttributes.DISTRIBUTED_TRANSACTION_IDENTIFIER
    )
