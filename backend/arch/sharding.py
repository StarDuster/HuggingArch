"""Phase names and the one fact two layers must not disagree about.

`PREFILL` / `DECODE` live here because the lowest layer that branches
on the serving phase should own the word. `degree_in_phase` is the
single statement of "this world splits nothing in this phase" —
`Parallelism.policy` sizes axes from it.
"""
from __future__ import annotations

import backend.arch.dims as dims

#: An encoder stack over its own media inputs (`eval.program.Program.encoding`).
ENCODE = "encode"
PREFILL = "prefill"
DECODE = "decode"


def degree_in_phase(world: str, degree: int, phase: str) -> int:
    """`degree`, or 1 when `world` splits nothing in `phase`.

    The restriction itself is not written here: it is `dims.yaml`'s
    `phases:` on the world. An unregistered name is unrestricted, which
    is the same answer omitting `phases:` gives. Today `sequence` is
    the only restricted world.
    """
    entry = dims.world_by_label().get(world)
    if entry is not None and not entry.active_in(phase):
        return 1
    return max(1, int(degree or 1))
