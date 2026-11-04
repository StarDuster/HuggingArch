"""Did the harness build the cell it was asked to measure?

A row says what it measured — a shape and a dtype per operand slot — and
the walk bills it for exactly that. Nothing checked that the tensors
handed to the kernel were that cell, and three different bugs lived in
the gap:

  * Gemma-2's KV cache is fp32 (it is the one model in the corpus
    published in float32) and the harness allocated bf16, because no
    builder read the `kv` slot. The row was billed four bytes an element
    for a cache that moved two, and 8,227 rows reported twice the
    bandwidth they moved.
  * A MoE cell whose per-rank shape floored to 0 rows could not be built,
    so the builders fell back to the identity scalars and measured a
    geometry the row does not bill — one row per expert against a
    numerator for the whole dispatch.
  * DeepSeek-V4's compressor billed its fp32 state and built bf16, the
    same error as Gemma's in a different component.

Every one of them was found by hand, after the fact, by multiplying out a
shape and comparing it with `achieved_bw × p50`. This is that arithmetic
as a gate, done the only way that needs no second traffic model: compare
the operands' SHAPE and DTYPE against the contract the cell already
declares (`OpSignature.inputs`, `TensorLayout.role`), not their bytes
against a model of what the kernel ought to move.

What it cannot see: whether the kernel computes the right function (the
cross-check's job, when an oracle exists), and whether a provider casts
away from the inputs it was handed after this ran (`BenchCase` carries no
handle on its operands yet).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .invocation import Invocation


def complaints(inv: Invocation, inputs: Mapping[str, Any]) -> List[str]:
    """What the built operands get wrong about `inv`, one line each.

    Empty when the cell was built as declared, or when the contract
    cannot be bound at these dims (a variant geometry the signature
    describes with axes this cell does not carry — nothing to compare).
    """
    from backend.inference.op_signatures import get_signature

    sig = get_signature(inv.signature)
    if sig is None:
        return []
    dims = _dims_for(inv)
    out: List[str] = []
    for layout in sig.layouts_for(dims):
        built = inputs.get(layout.name)
        if built is None or not hasattr(built, "shape"):
            continue                  # an operand this provider builds itself
        try:
            want = layout.extents(dims)
        except (KeyError, TypeError, ValueError):
            continue                  # not bindable at these dims
        if any(extent is None for extent in want):
            continue
        got = tuple(int(x) for x in built.shape)
        if tuple(int(x) for x in want) != got:
            out.append(
                f"{layout.name}: cell says {tuple(int(x) for x in want)}, "
                f"built {got}")
        wrong = _dtype_complaint(inv, layout, built)
        if wrong:
            out.append(wrong)
    return out


def _dtype_complaint(inv: Invocation, layout: Any, built: Any) -> Optional[str]:
    """The operand's dtype against the slot its role names.

    A format too narrow to hold a reference value is kept lossless on
    purpose (`refs._store_dtype`: an fp8 operand is carried as bf16 and
    each provider quantizes), so a WIDER tensor than the slot is the
    convention and not a complaint. A narrower one is the Gemma bug.
    """
    from .refs import torch_dtype

    role = layout.role or "activation"
    declared = inv.dtype_for(role)
    try:
        want = torch_dtype(declared)
    except Exception:
        return None                   # a format with no torch dtype (packed)
    got = getattr(built, "dtype", None)
    if got is None or got == want:
        return None
    want_size = getattr(want, "itemsize", 0)
    got_size = getattr(got, "itemsize", 0)
    if got_size >= want_size:
        return None                   # lossless carrier for a narrow format
    return (f"{layout.name}: cell says {declared} ({want_size}B), built "
            f"{got} ({got_size}B) — the row is billed for {declared}")


def _dims_for(inv: Invocation) -> Dict[str, int]:
    """The cell's dims as the contract reads them — the carried engine
    shape where it has one, as `refs._shape` does, so an operand built
    from the graph's own layout is compared against the same numbers."""
    from .refs import _dims

    return _dims(inv)
