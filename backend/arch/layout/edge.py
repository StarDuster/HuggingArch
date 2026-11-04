"""`EdgeState`: per-axis layout plus optional pending reduction and routed load."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .layout import AxisLayout, Unknown, tensor_layout
from .value import Partial, ValueProvenance
from .workload import ExpectedRoutedLoad


@dataclass(frozen=True)
class EdgeState:
    """One edge's distributed state.

    `layout` is one `AxisLayout` per tensor axis, aligned with the producer's
    output axes. `value` is the value side: None is complete, a
    `value.Partial` is a pending reduction. The walk's
    `Redistribute.resolves_a_partial` asks the state, so `carries_partial`
    is part of the layout-rule contract.
    """

    layout: Tuple[AxisLayout, ...]
    value: Optional[Partial] = None
    #: Runtime load evidence for the single routed-rows axis. It is deliberately
    #: excluded from equality: communication/type equality is layout + value,
    #: while the estimate travels beside them until combine consumes it.
    workload: Optional[ExpectedRoutedLoad] = field(default=None, compare=False)
    #: Semantic graph evidence for handlers that compose named operand roles.
    #: Like workload, this travels beside the layout algebra and cannot create
    #: a redistribution by making two otherwise-identical states unequal.
    provenance: Optional[ValueProvenance] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", tuple(self.layout))
        for axis in self.layout:
            if isinstance(axis, Unknown):
                raise ValueError(
                    f"Unknown layout reached an EdgeState: "
                    f"{axis.reason or 'no reason recorded'}")
        # Whole-tensor legality is TensorLayout's — scope uniformity and the
        # uniform-fiber check — reused rather than restated, so a state one
        # family cuts along two axes (diagonal coverage) is rejected at
        # construction and `lower_diff` only ever prices legal tensors.
        # (`Unknown` is checked above: it is invisible to the fiber check.)
        tensor_layout(self.layout)
        if self.value is not None and not isinstance(self.value, Partial):
            raise ValueError(
                f"EdgeState.value is None (complete) or a value.Partial; "
                f"got {self.value!r} — the vocabulary is closed")
        if self.workload is not None and not isinstance(
                self.workload, ExpectedRoutedLoad):
            raise ValueError(
                "EdgeState.workload is None or ExpectedRoutedLoad; got "
                f"{self.workload!r}")
        if self.provenance is not None and not isinstance(
                self.provenance, ValueProvenance):
            raise ValueError(
                "EdgeState.provenance is None or ValueProvenance; got "
                f"{self.provenance!r}")

    def carries_partial(self) -> bool:
        return self.value is not None

    def finalized(self) -> "EdgeState":
        """The same layout with the pending reduction resolved."""
        return (EdgeState(self.layout, None, self.workload, self.provenance)
                if self.value is not None else self)
