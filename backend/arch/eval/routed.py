"""`RoutedLaunch` — one rank's MoE dispatch boundary, as a grouped kernel
launches it.

A dispatch (`permute`) lays the rank's tokens out per expert,
`[n_expert_copies, M, d]`, and everything up to its `combine` runs on that
buffer. The walk keeps `M` the MEAN rows one expert receives — exact, often
fractional (`Evaluator._tokens_per_expert`) — and the layout keeps the
busiest rank's share of it (`ExpectedRoutedLoad`). A kernel runs whole rows,
so the launch is a padded geometry, and every reader of that geometry — the
benchmark identity, the Kernel Bench plan and its providers, the overlay's
numerators at the launched geometry — reads it here, once, rather than
re-deriving it from a shape:

    source_rows      T  the tokens the rank holds (the dispatch's token
                        operand rows)
    routed_rows      R  the rows dispatched onto the rank's experts, exact
                        (the busiest rank's, under expert parallelism)
    local_experts    E  the experts the rank holds
    touched_experts     the experts that receive rows, `min(ceil(R), E)`
    rows_per_expert     each touched expert's rows, padded:
                        `ceil(R / touched_experts)`
    launch_rows         `touched_experts x rows_per_expert`

Only the experts that receive rows are padded: a grouped kernel launches no
blocks for an empty expert (vLLM's `moe_align_block_size` skips them,
`csrc/moe/moe_align_sum_kernels.cu`), so it reads no weights for one either.
`touched_experts` is the least number of experts R rows can reach when they
spread evenly — the SOL reading, a lower bound on what a real routing
reaches. The walk's weight law reads `Min(R, E)` experts' worth
(`row_roofline.weight_demand`), which is `touched_experts` only where R is
whole; a launch reads whole experts, so its numerator re-reads that law at
the launched rows (`calibration_overlay._at_geometry`). The frameworks pad
each expert to a block of rows (`BLOCK_SIZE_M`); this pads to a row.

The record is per dispatch, keyed by the dispatch's uid
(`Analysis.routed`); which dispatch a node's buffer came from is the graph's
(`Program.dispatch_of`). Its fields are extents of the walk, so a symbolic
walk's record is a law read at any point (`Point.read`); the padded readings
are `ceiling` / `Min` of them, numbers at a number and laws at a law.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Mapping, Sequence, Tuple

import sympy as sp

from .dim import as_number, exact, value_of


def _whole(value: sp.Basic) -> Any:
    """A reading as an `int` where it is a number, else the law it is."""
    return int(value) if value.is_Integer else value


@dataclass(frozen=True)
class RoutedLaunch:
    """One rank's dispatch boundary (module docstring)."""

    #: The uid of the dispatch (`permute`) this boundary is.
    dispatch: int
    #: T: the rows of the dispatch's token operand on this rank.
    source_rows: Any
    #: R: the rows dispatched onto this rank's experts — exact, a mean.
    routed_rows: Any
    #: E: the experts this rank holds.
    local_experts: int

    @classmethod
    def of(cls, dispatch: int, token: Sequence[Any],
           buffer: Sequence[Any]) -> "RoutedLaunch":
        """The boundary read off the dispatch's own tensors on one rank: its
        token operand `[B, s, d]` and the buffer it writes `[E, M, d]` —
        exactly those, the layouts `permute` declares (`shape_in` /
        `shape_out`), so its experts are axis 0 of the buffer and its rows
        axis 1, as every reader of the record takes them."""
        if len(token) != 3 or len(buffer) != 3:
            raise ValueError(
                f"dispatch {dispatch}: a token operand [B, s, d] and a buffer "
                f"[E, M, d] are its tensors, not {list(token)!r} and "
                f"{list(buffer)!r}")
        return cls(dispatch=dispatch, source_rows=token[0] * token[1],
                   routed_rows=buffer[0] * buffer[1],
                   local_experts=int(value_of(buffer[0])))

    def wire(self) -> Dict[str, Any]:
        """The record as a per-op row carries it: each extent exact and
        JSON-native (`dim.as_number`: an `int`, or a fraction's exact
        string)."""
        return {"dispatch": self.dispatch,
                "source_rows": as_number(self.source_rows),
                "routed_rows": as_number(self.routed_rows),
                "local_experts": self.local_experts}

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "RoutedLaunch":
        """The record a row carries (`wire`), back as itself, its extents
        exact."""
        return cls(dispatch=int(wire["dispatch"]),
                   source_rows=Fraction(str(wire["source_rows"])),
                   routed_rows=Fraction(str(wire["routed_rows"])),
                   local_experts=int(wire["local_experts"]))

    @property
    def touched_experts(self) -> Any:
        """The experts that receive rows: `min(ceil(R), E)`."""
        return _whole(sp.Min(sp.ceiling(exact(self.routed_rows)),
                             sp.Integer(self.local_experts)))

    @property
    def rows_per_expert(self) -> Any:
        """Each touched expert's rows, padded to a whole row:
        `ceil(R / touched)` (0 where nothing is routed)."""
        touched = exact(self.touched_experts)
        return _whole(sp.ceiling(exact(self.routed_rows)
                                 / sp.Max(touched, sp.Integer(1))))

    @property
    def launch_rows(self) -> Any:
        """The rows one launch runs: every touched expert at its padded
        rows."""
        return _whole(exact(self.touched_experts)
                      * exact(self.rows_per_expert))

    @property
    def expert_rows(self) -> Tuple[Any, ...]:
        """The rows each launched expert runs, expert by expert: every
        touched expert at `rows_per_expert` — the token counts a grouped
        kernel is handed for this launch."""
        return (self.rows_per_expert,) * int(self.touched_experts)
