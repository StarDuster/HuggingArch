"""Compare two StructOp sequences and surface structural discrepancies.

Pure function over the shared `StructOp` schema — doesn't know or care
whether the ops came from AST parsing or spec graph extraction.
"""
from __future__ import annotations

from backend.arch.validate._finding import Finding

from typing import Any, Dict, List

from .types import StructOp


def _is_structural(op: StructOp) -> bool:
    return op.kind in ("call", "add")


def _count_by_kind(ops: List[StructOp], kind: str) -> int:
    return sum(1 for o in ops if o.kind == kind)


def compare(fwd_ops: List[StructOp], spec_ops: List[StructOp]) -> List[Dict[str, Any]]:
    """Return a list of warning dicts; empty = no mismatches detected.

    Each warning has at least `kind` and `detail`. Additional fields
    (`forward_ref`, `spec_ref`, `forward_order`, `spec_order`, …) vary
    by warning kind — see inline.

    False positives are possible (forward may have ops the spec
    intentionally collapses into a component); callers should surface
    these as warnings, not errors.
    """
    warnings: List[Dict[str, Any]] = []

    fwd_struct = [o for o in fwd_ops if _is_structural(o)]
    spec_struct = [o for o in spec_ops if _is_structural(o)]

    # Residual add count
    fwd_adds = _count_by_kind(fwd_struct, "add")
    spec_adds = _count_by_kind(spec_struct, "add")
    if fwd_adds != spec_adds:
        warnings.append(Finding(
            kind="add_count_mismatch",
            detail=f"forward has {fwd_adds} residual adds, spec has {spec_adds}",
            extra={"forward_adds": fwd_adds, "spec_adds": spec_adds},
        ))

    # Submodule name set
    fwd_call_names = [o.name for o in fwd_struct if o.kind == "call"]
    spec_call_names = [o.name for o in spec_struct if o.kind == "call"]
    fwd_set = set(fwd_call_names)
    spec_set = set(spec_call_names)

    warnings.extend(Finding(
        kind="forward_call_missing_in_spec",
        detail=f"forward calls self.{name}(...) but spec has no node named {name!r}",
        extra={"forward_ref": name},
    ) for name in fwd_set - spec_set)

    warnings.extend(Finding(
        kind="spec_call_not_in_forward",
        detail=f"spec node {name!r} has no matching self.{name}(...) call in forward",
        extra={"spec_ref": name},
    ) for name in spec_set - fwd_set)

    # Relative order of the names present on BOTH sides
    common = fwd_set & spec_set
    fwd_order = [n for n in fwd_call_names if n in common]
    spec_order = [n for n in spec_call_names if n in common]
    # Spec may repeat a name across expansion — dedupe by first-occurrence
    def _dedupe_first(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    if _dedupe_first(fwd_order) != _dedupe_first(spec_order):
        warnings.append(Finding(
            kind="call_order_mismatch",
            detail="spec calls common submodules in a different order than forward",
            extra={"forward_order": _dedupe_first(fwd_order),
                   "spec_order": _dedupe_first(spec_order)},
        ))

    # Branches → heterogeneity cue. Only fire when forward has IF
    # branches AND spec expressed no per-template differentiation for
    # this block — i.e. a single-template spec running against a
    # forward with conditional module dispatch. Multi-template specs
    # (assignments:) are assumed to cover heterogeneity already.
    branches = [o for o in fwd_ops if o.kind == "branch_detected"]
    if branches:
        warnings.append(Finding(
            kind="forward_has_branches",
            detail=(
                f"forward has {', '.join(o.name for o in branches)} with self-submodule calls; "
                "if module selection varies across layers the spec must use `assignments:` or "
                "multiple `blocks:` — verify coverage per branch"
            ),
            extra={"branch_kinds": [o.name for o in branches]},
        ))

    return warnings
