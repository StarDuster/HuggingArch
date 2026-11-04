"""Dtype attachment for spec-graph rendering, from `Analysis.tensors()`.

The index is `Analysis.tensor_index()` — a flat `{aggregation_id → TensorRow}`
map owned by the one join. Renderer-dict nodes carry
`n["id"] = ir.aggregation_id`, so `attach_dtypes` does a direct equality
lookup. Build the index once per analyze request from the same Analysis the
rest of the request already has.
"""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from backend.arch.eval.dim import value_of

from ._consts import DTYPE_BYTES

if TYPE_CHECKING:
    from backend.arch.eval.analysis import TensorRow


def attach_dtypes(
    nodes_list: List[Dict[str, Any]],
    index: Dict[str, "TensorRow"],
) -> int:
    """Walk block_graph nodes; for each leaf, attach `weight_dtype` /
    `weight_dtype_bytes` from the analysis index. Returns the count
    of nodes successfully stamped.
    """
    if not index:
        return 0
    return _stamp_recursive(nodes_list, index)


def _stamp_recursive(nodes, index) -> int:
    """Recursive walk; returns number of stamps applied. A wrapper node
    that absorbed its primary's tensor (via `_node_build`'s primary
    semantics — `mlp.gate` standing in for the inner `gate` linear)
    matches by id even though it has children, so we try the lookup
    BEFORE recursing into kids."""
    stamped = 0
    for n in nodes:
        nid = n.get("id")
        mt = index.get(nid) if nid else None
        if mt is not None:
            n["weight_dtype"] = mt.logical_dtype or mt.storage_dtype
            if mt.elements:
                n["weight_dtype_bytes"] = float(
                    value_of(mt.storage_bytes)
                ) / float(value_of(mt.elements))
            elif n["weight_dtype"] in DTYPE_BYTES:
                n["weight_dtype_bytes"] = DTYPE_BYTES[n["weight_dtype"]]
            n["_matched_tensor"] = mt.name
            stamped += 1
        if n.get("children"):
            stamped += _stamp_recursive(n["children"], index)
    return stamped
