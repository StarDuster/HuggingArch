"""The precision string of the model-config panel.

It states the analysis' weight formats (`Analysis.weight_bytes_by_dtype`,
the `tensors()` join) and its scale dtype (`dominant_weight_scale_dtype`),
which the summary carries as `weight_formats` / `quant_metadata_scale_dtype`.
It reads no safetensors header: a header names the container a packed
weight is stored in (I8 / U8 / I32), not the format it holds (FP4 / INT4),
and a narrowest-dtype race over it is a second reading of the checkpoint
that disagreed with the composition table on every packed model.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def resolve_precision(summary: Mapping[str, Any]) -> Optional[str]:
    """`FP4 91.0% (scale: ue8m0) · F8_E4M3 6.6% · BF16 2.3%`: each weight
    format and its share of the weight bytes, largest first, the scale
    dtype beside the largest. A single format is its name alone. None when
    the summary walked no spec."""
    formats = [row for row in summary.get("weight_formats") or ()
               if row.get("bytes")]
    if not formats:
        return None
    total = sum(row["bytes"] for row in formats)
    # A format holding under 0.05% of the bytes (an F32 bias, an I64
    # buffer) is not a precision of the model; it stays in the
    # composition table's per-dtype bytes.
    shown = [(row["dtype"], row["bytes"] / total * 100) for row in formats
             if round(row["bytes"] / total * 100, 1) > 0]
    scale = summary.get("quant_metadata_scale_dtype")
    parts = []
    for i, (name, share) in enumerate(shown):
        part = name if len(shown) == 1 else f"{name} {share:.1f}%"
        if i == 0 and scale:
            part += f" (scale: {scale})"
        parts.append(part)
    return " · ".join(parts)
