"""The per-rank weight ledger must build.

Every page that draws a model prices its weights off one ledger:
`capacity.weights` (parameters, their scales, buffers, per node), turned into
per-copy bytes with the rank's tensor rows and the walk's copy counts
(`local_tensors`, `instances`). The layout builds that ledger from each stored tensor's
declared layout, so a stored weight with no `weight_axes`, or a tensor with no
format to price it at, cannot be priced — and the Inspector, which draws
every block, fails on it.

The byte anchor does not see this: it reads the checkpoint join, which needs
no layout. So the gate builds the same ledger the drawing reads, and a
failure comes back as an error the spec writer can act on instead of a
published spec whose page does not load.
"""
from __future__ import annotations

from typing import Any, List

from backend.arch.eval import capacity as capacity_mod

from ._finding import Finding

#: What a spec declares so the layout can place a stored weight.
_DECLARE_AXES = (
    "A stored weight with no declared layout needs `weight_axes` on the node "
    "that stores it: one contract per "
    "stored axis, `{ replicated: true }` for an axis no parallelism splits "
    "or `{ world: ..., parts: ..., fit: ... }` for one that is sharded.")

#: What a spec declares so a stored tensor has a width.
_DECLARE_FORMAT = (
    "Give it a dtype through `quant_context` (`per_role`, `overrides` or "
    "`default_dtype`).")


def run_weight_ledger_check(analysis: Any) -> List[Finding]:
    """No findings when the rank's weight ledger builds; else one, saying
    which tensor could not be priced and what to declare."""
    try:
        capacity_mod.weights(analysis)
        for _row in analysis.local_tensors():
            pass
        _ = analysis.instances
    except capacity_mod.UnpricedTensor as exc:
        return [Finding(kind="weight_ledger_unpriced",
                        detail=f"weight ledger: {exc} {_DECLARE_FORMAT}")]
    except ValueError as exc:
        return [Finding(kind="weight_ledger_unbuilt",
                        detail=f"weight ledger: {exc}. {_DECLARE_AXES}")]
    return []
