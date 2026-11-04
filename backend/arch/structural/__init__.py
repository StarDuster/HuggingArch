"""Structural check: AST-based verification that a spec's block graph
matches the model's actual `forward()` source.

Use `locate_layer_forward(files, config, auto_class)` once per model to
find the decoder layer's `forward`, then `check(layer_forward, block)` per
block — returns a list of warning dicts (empty = clean). Doesn't gate validation; intended to surface structural
bugs that pure param-count checks can't catch (missing residual adds,
swapped norm positions, ignored conditional branches, etc.).

The package is strictly layered:
    types.py         — shared StructOp dataclass (no deps)
    forward_ast.py   — AST → List[StructOp]            (depends on types)
    spec_graph.py    — block graph → List[StructOp]    (depends on types)
    comparator.py    — compare two List[StructOp]      (depends on types)
    __init__.py      — public facade (below)

Callers should only import from this module; the sub-modules are
implementation detail.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .comparator import compare
from .forward_ast import extract_forward_ops, locate_layer_forward
from .spec_graph import extract_spec_ops
from .types import StructOp


def check(layer_forward: Any, block: Any) -> List[Dict[str, Any]]:
    """Run the full structural check of `block` against the located
    layer `forward`. Returns [] when the spec block is None — this is a
    best-effort diagnostic, not a gating check."""
    if layer_forward is None or block is None:
        return []
    fwd_ops = extract_forward_ops(layer_forward)
    if not fwd_ops:
        return []
    spec_ops = extract_spec_ops(block)
    if not spec_ops:
        return []
    return compare(fwd_ops, spec_ops)


__all__ = ["StructOp", "check", "compare", "extract_forward_ops", "extract_spec_ops",
           "locate_layer_forward"]
