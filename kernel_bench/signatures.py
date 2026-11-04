"""Op signatures, as the harness sees them.

The signature objects themselves live in `backend.inference.op_signatures` —
one class, one registry, read by both the harness and the cost model, because
the two used to keep separate axis tables and disagreed about ten of nineteen
signatures. Re-exported here so `kernel_bench.signatures` stays the harness's
name for them.

What is genuinely this side's: `op_to_signature`, the harness's policy on an
op nobody mapped.
"""
from __future__ import annotations

from typing import Optional

from backend.inference.op_signatures import (      # noqa: F401  (re-export)
    OpSignature,
    all_signatures,
    get_signature,
    register_signature,
)


def op_to_signature(op: str) -> Optional[str]:
    """The signature this op is measured as, or None if it is not measurable.

    The map itself is the catalog's `binds_ops` (`op_signatures.op_bindings`) — the
    same declaration the envelope key is built from — because a copy here is exactly how
    the two came to disagree about nine ops, `linear` among them. This side
    adds only its own contract: None for an op nobody mapped, so the planner
    refuses it rather than inventing a signature no provider serves.
    """
    from backend.inference.op_signatures import op_bindings
    return op_bindings().get(op)
