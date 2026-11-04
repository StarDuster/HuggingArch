"""`ShapeNode` — the dim-map node view.

The shape/dim propagation family (`axis_propagation.infer_shape` / `dim_map`)
reads a small, fixed set of fields off a "node". A real walk passes a
`SpecIRNode` (which carries all of them); probes and validate's rank checks pass
a `ShapeProbe`. Naming the contract here — a low-level module that `spec_tree`
may import but never the reverse — lets those reads become declared field
accesses instead of `getattr` defaults, without coupling these modules to
`spec_tree`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


class ShapeRef(Protocol):
    """One input edge of a `ShapeNode`: the producing node view and its port."""

    src: "Optional[ShapeNode]"
    port: str


class ShapeNode(Protocol):
    """The dim-map view of a node. `SpecIRNode` conforms structurally; the
    non-IR path uses `ShapeProbe`.

    Five fields only — no `shape_out` (output shapes live in `Analysis.shapes`,
    not on the node) and none of the ~50 IR / layout fields
    (uid / op / display_path / weight_axes / parent / expert_weight / …), which
    belong to `eval.node.Evaluable`, not here.
    """

    #: The OUTPUT shape the node declares, or None. Only a source op declares
    #: one: its `shape:` is the tensor it stores, and its output IS that
    #: tensor (`parameter_tensor`). A weight-bearing op's `shape:` is its
    #: weight's, which says nothing about its output. Optional[Tuple[Any,
    #: ...]] — IR shapes may be dim expressions, not ints, so `Any`.
    out_shape: Optional[Tuple[Any, ...]]
    #: Declared factor names per axis; `{}` means "labels unknown", not
    #: "declared empty" (validate's rank views carry no factor names).
    dims_raw: Mapping[str, str]
    axis: Optional[int]
    axes: Optional[Sequence[int]]
    inputs_obj: Tuple["ShapeRef", ...]


@dataclass(frozen=True)
class _Ref:
    """Concrete `ShapeRef` for probe / validate input edges."""

    src: "Optional[ShapeNode]" = None
    port: str = ""


@dataclass(frozen=True)
class ShapeProbe:
    """The non-IR shape view fed to `infer_shape` / `dim_map`: probes and
    validate's rank checks. The walk passes a real `SpecIRNode`, which conforms
    structurally. Carries only the five dim-map fields — not a walkable node."""

    out_shape: Optional[Tuple[Any, ...]] = None
    dims_raw: Mapping[str, str] = field(default_factory=dict)  # {} = labels unknown
    axis: Optional[int] = None
    axes: Optional[Sequence[int]] = None
    inputs_obj: Tuple[_Ref, ...] = ()

    @classmethod
    def probe(cls, *, axis: Optional[int] = None,
              axes: Optional[Sequence[int]] = None,
              out_shape: Optional[Sequence[Any]] = None) -> "ShapeProbe":
        """A stand-in node from ALREADY-resolved `axis` / `axes` / `out_shape`.

        Deliberately does not take `op`: the `axis_locked` / `permute_to`
        fallback stays at the caller, because `op` may be `str | OpSpec |
        CustomOpSpec` and `CustomOpSpec` has no `permute_to` — a direct read
        would raise. Callers that want the fallback apply it themselves.
        """
        return cls(axis=axis, axes=axes,
                   out_shape=tuple(out_shape) if out_shape else None)
