"""The named counts a weight leaf is repeated by.

A weight leaf's `multiplicity` is a product of counts, and every one of them
was DECLARED by something: a block template's invocation says how many layers
it runs, each enclosing `repeat:` says how many copies of its node exist. The
storage tooltip prints them — `x 61 layers x 256 experts` — so somebody has
to say which count is which, and until this layer the consumer guessed:

    unit = "experts" if "expert" in raw else "copies"

A substring test on a dim symbol's name. Seven symbols appear on a `repeat:`
across the corpus and the guess answers for all seven from their spelling: it
is right about `n_expert_copies` by accident, right about `n_shared_experts`
for the wrong reason, and wrong about `n_layers`, `n_mtp_layers`,
`n_vis_layers`, `n_audio_layers` and `n_speech_banks`, each of which it calls
"copies".

The classification exists — `dims.yaml` gives every symbol a `kind` — but
`kind` cannot carry the display noun: `expert_copies` drags
`shardable_by: [moe_copies]` with it, and expert parallelism does not shard
`n_shared_experts`, so spelling the shared experts' kind to fix a word would
break their sharding. The noun is therefore a missing DECLARATION, not a fact
anything downstream can derive, and it is now `dims.yaml`'s `unit:`.

What this module does is read the declarations and hand over the factors.
It decomposes nothing: the product is checked against the multiplicity the
tree already carries, and a product that does not account for it publishes
one honest factor rather than a remainder.
"""
from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Tuple

import backend.arch.dims as dim_registry



class CopyFactor(NamedTuple):
    """One declared count a leaf is repeated by."""

    #: The symbol that named the count, in the MODEL's vocabulary. None when
    #: the declaration wrote a literal (`repeat: "2"`), when the count came
    #: from an assignment's layer indices, or when the expression names more
    #: than one symbol — in all three there is no single declaration to cite.
    symbol: Optional[str]
    #: Its value in THIS walk's binding.
    count: int
    #: `dims.yaml`'s `kind`, verbatim. `layer_count` for a block
    #: invocation's count that no symbol named, which is what that count IS.
    kind: str
    #: `dims.yaml`'s `unit` — the display noun — or None, in which case a
    #: reader shows the symbol's own name. Never defaulted to a generic
    #: word: that is the bug this layer replaces.
    unit: Optional[str]


#: What a block invocation's count is when no symbol named it. A stack's
#: count is the number of layers of that template, which is a fact about
#: where the factor came from rather than a guess about a word — the symbol
#: is absent, the provenance is not.
_BLOCK_KIND = "layer_count"
_BLOCK_UNIT = "layers"


def _named(raw: Any) -> Optional[str]:
    """The one symbol `raw` names; None when the declaration wrote a literal
    or an expression over several symbols.

    As written, not substituted outward through the enclosing components'
    `dims:` bindings. That is deliberate and it is what `ir_count_symbols`
    already does for the same declarations, which the breakdown's own
    divisor half reads: a count is cited by the name its declaration used,
    and the two halves of one tooltip must not disagree about it.
    """
    if raw is None:
        return None
    name = str(raw).strip()
    # A bare identifier IS a symbol; a literal (`repeat: "2"`) and an
    # expression (`repeat: "a * b"`) are not identifiers, so the same test
    # answers both — the way a shape slot is told from an extent
    # (`axis_propagation.signature_for`).
    return name if name.isidentifier() else None


def _factor(raw: Any, count: int) -> CopyFactor:
    """One factor, its noun read off the declaration that named it."""
    symbol = _named(raw)
    declared = dim_registry.get(symbol) if symbol else None
    if declared is None:
        return CopyFactor(symbol, count, _BLOCK_KIND if symbol is None
                          else "whole", None)
    return CopyFactor(symbol, count, declared.kind,
                      declared.unit or None)


def _block_factors(program: Any, node: Any) -> List[CopyFactor]:
    """The block invocation's counts, outermost first.

    Two, not one, because a group's count is a product of two separate
    declarations — `BlockMeta.outer_repeat` times the stack's own — and
    collapsing them would publish a number with no declaration behind it
    (Nemotron-3-Ultra's MTP: `repeat: n_mtp_layers` around a two-variant
    per-layer stack). A model-level leaf belongs to no invocation and gets
    none of these.
    """
    ordinal = program.block_of(node.uid)
    if ordinal is None:
        return []
    meta = program.blocks[ordinal].meta
    outer = int(meta.outer_repeat or 1)
    own, remainder = divmod(int(meta.count), outer)
    if remainder:
        raise ValueError(
            f"block invocation {meta.template_name!r} counts {meta.count} "
            f"layers, which its enclosing group's {outer} does not divide")
    out: List[CopyFactor] = []
    if outer > 1:
        out.append(_as_layers(_factor(meta.outer_repeat_raw, outer)))
    if own > 1:
        out.append(_as_layers(_factor(meta.count_raw, own)))
    return out


def _as_layers(factor: CopyFactor) -> CopyFactor:
    """A block invocation's factor, whose noun is `layers` when the
    declaration named no symbol to read one off.

    Not a fallback for a symbol that declares no unit — that case publishes
    None on purpose, and a reader shows the symbol's own name. This is the
    case where there is no symbol at all, because the count came from an
    assignment's layer indices or from a literal, and what it counts is
    still layers of a stack.
    """
    if factor.symbol is None:
        return factor._replace(kind=_BLOCK_KIND, unit=_BLOCK_UNIT)
    return factor


def copy_factors_of(program: Any, node: Any) -> Tuple[CopyFactor, ...]:
    """`node`'s declared copy factors, outermost to innermost.

    `()` for a leaf nothing repeats. A product that does not account for the
    tree's own `multiplicity` publishes ONE factor, `(None, multiplicity,
    "whole", None)`: a consumer must not be handed a remainder to reconcile,
    and a count whose declarations do not multiply to it is a count with no
    declaration this layer can cite.
    """
    total = int(node.multiplicity or 1)
    repeats: List[CopyFactor] = []
    walking = node
    while walking is not None:
        own = walking.repeat
        if own is not None and int(own) > 1:
            repeats.append(_factor(walking.repeat_raw, int(own)))
        walking = walking.parent
    factors = [*_block_factors(program, node), *reversed(repeats)]
    product = 1
    for factor in factors:
        product *= factor.count
    if product != total:
        return () if total == 1 else (
            CopyFactor(None, total, "whole", None),)
    return tuple(factors)
