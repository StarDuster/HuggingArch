"""Arithmetic intensity, stated once.

FLOPs per byte of HBM access — of one per-op row, of a sum of rows (a module
subtree, a declaration's groups) or of a whole step. Every publisher of an
`arithmetic_intensity` field divides here, so a row, its parent and the phase
headline answer on one basis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def arithmetic_intensity(flops: float, access: float) -> Optional[float]:
    """`flops / access`; None for what moves no bytes (there is no ratio,
    and 0 would read as a pure-memory op)."""
    access = float(access or 0)
    return float(flops or 0) / access if access > 0 else None


def intensity_formula(flops: float, access: float) -> Optional[Dict[str, Any]]:
    """The popover beside an `arithmetic_intensity` field, off the same two
    operands; None where there is no ratio."""
    ai = arithmetic_intensity(flops, access)
    if ai is None:
        return None
    return {
        "symbolic": "AI = FLOPs / access_bytes",
        "substituted": f"{int(flops or 0):,d} / {int(access):,d}",
        "result": f"{ai:.4g} FLOPs/B",
    }
