"""The analysis's parameter counts against the counts the model card states.

The checkpoint anchors say every stored tensor is accounted for; they say
nothing about how a token uses them. DeepSeek-V4.1-Flash's generated spec
matched its checkpoint byte for byte and reported 214B activated parameters
where the card states 8B in prefill and 16B in decode: its 196B of n-gram
tables were counted as touched whole. The card is the one outside statement
of how the model runs, so the figures are held against it.

Per figure family — the stored total, and the activated count — the check
asks whether ANY count of that kind the card states (`hf.model_card`) lies
within `TOLERANCE` of ANY of the analysis's readings of that family (with and without the auxiliary stacks,
with and without lookup tables; activated per decoded and per prompt token).
A card states its siblings' counts beside its own, rounds to the digits it
writes, and words its counts freely, so a miss of every pairing is what is
reported, and only as a warning. A dense model's activated count is its
total and is not held apart; a card that states nothing, or that the hub
cannot serve, says nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.hf.model_card import model_card, stated_counts

from ._finding import Finding

TOLERANCE = 0.15


def readings(report: Any) -> Dict[str, Dict[str, int]]:
    """The analysis's readings of each figure family (keyed by the card's
    kind), by name."""
    aux = sum(report.aux_params.values())
    aux_active = sum(report.aux_active_params.values())
    aux_prefill = sum(report.aux_prefill_active_params.values())
    lookup = report.lookup_table_params
    return {
        "total": {
            "total": report.total_params,
            "main stack": report.total_params - aux,
            "main stack without lookup tables":
                report.total_params - aux - report.main_lookup_table_params,
            "total without lookup tables": report.total_params - lookup,
        },
        "active": {
            "per decoded token": report.total_decode_active_params,
            "main stack, per decoded token": report.total_active_params - aux_active,
            "per prompt token": report.total_prefill_active_params,
            "main stack, per prompt token":
                report.total_prefill_active_params - aux_prefill,
        },
    }


def _near(value: int, stated: Any) -> bool:
    """Within `TOLERANCE` of the stated count, beyond the rounding of the
    digits the card wrote ("3B" stands for 2.5B-3.5B)."""
    return (stated.count > 0 and abs(value - stated.count)
            <= TOLERANCE * stated.count + stated.rounding)


def run_model_card_check(model_id: str, report: Any) -> List[Finding]:
    card, _why = model_card(model_id)
    if card is None:
        return []
    stated = stated_counts(card)
    if not stated:
        return []
    out: List[Finding] = []
    for family, ours in readings(report).items():
        said_here = [s for s in stated if s.kind == family]
        if not said_here or any(_near(value, s)
                                for value in ours.values() if value
                                for s in said_here):
            continue
        if family == "active" and all(
                value >= (1 - TOLERANCE) * report.total_params
                for value in ours.values() if value):
            # Dense: what a token activates is the model; an activated count
            # the card states belongs to a sibling it lists.
            continue
        shown = ", ".join(f"{name} {value / 1e9:.2f}B"
                          for name, value in ours.items() if value)
        said = ", ".join(sorted({f"{s.count / 1e9:g}B" for s in said_here}))
        out.append(Finding(
            kind="model_card_disagrees",
            detail=(
                f"no {family} parameter count the model card states ({said}) is "
                f"within {TOLERANCE:.0%} of the analysis's ({shown}). Check "
                f"what the card counts against what the spec makes a token "
                f"touch: a table read a few rows per token, layers a prompt "
                f"token skips, a stack the card leaves out."),
            extra={"family": family,
                   "stated": sorted({s.count for s in said_here}),
                   "analysis": ours}))
    return out
