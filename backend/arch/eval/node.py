"""What this package requires of a node, written down as two contracts.

The package walks `SpecIRNode` and, after a fusion, holds a graph containing
`FusedNode` as well. Neither fact was stated anywhere: every function took
`node: Any` and reached for fields with `getattr(node, "repeat_raw", None)`, so
the contract lived in the union of a hundred defensive accesses and nothing said
which fields a node MUST have or what a missing one meant.

Two contracts and not one, because the two node kinds are asked different
questions at different times:

    Evaluable   what the WALK reads — dims, counts, input refs, zone flag.
                Only `SpecIRNode` satisfies it, and only it ever needs to: a
                fused node is created after the walk, out of nodes that were
                already evaluated.

    Identified  what a POST-WALK consumer reads — identity, topology and origin.
                `FusedNode` satisfies this and not the other, which is exactly
                the distinction `Rewrite` encodes: a fusion produces a
                correspondence between identities, so an identity and its
                boundary are all the new node needs to carry.

`Evaluable` extends `Identified`, so a walk-time node is usable everywhere a
post-walk one is and the ordering is a subtype relation rather than a
convention.

These are `typing.Protocol`s — structural, checked statically, not inherited.
`SpecIRNode` lives in `spec_tree`, the shared IR substrate; making it
inherit from anything here would be this package reaching into a module it
has deliberately kept at arm's length.

A field that is genuinely optional is typed `Optional[...]` and read directly.
Writing the contract down settled one question by itself: three call sites read
`getattr(node, "tensor_name", None) or node.name`, and `SpecIRNode` has no such
field — it carries `weight_name`. The left branch could never fire, so those
sites had always used the local name and the fallback was the whole behaviour.
They now say so.
"""
from __future__ import annotations

from typing import (
    Any, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable,
)


@runtime_checkable
class InputRef(Protocol):
    """One declared input edge.

    `src` is None for a ref that names no sibling: an enclosing scope's
    input (`x`, `residual`, `x_hc`), a model-wide stream, a stage's
    `inputs:` name, or a model input (`model_inputs:`). Those read
    `external_sid` against the frame instead, which is why `Scope` carries
    the aliases and streams it binds.
    """

    #: the producing node, or None for a ref that names no sibling
    src: Optional["Identified"]
    #: "" for the main output, named for a multi-output producer
    port: str
    #: the nested-form string id an unresolved ref kept, "" otherwise
    external_sid: str
    #: whether `external_sid` is a component formal the tree grafted onto
    #: its caller's name, rather than a name the body reads as written
    grafted: bool


@runtime_checkable
class Identified(Protocol):
    """A node with an identity, a role and a place in the tree.

    The contract every layer keys on. `uid` is tree-unique and allocated by
    `SpecIRTree`, including for nodes created during mutation, which is what
    lets `Layer` join on it instead of on composed name strings.
    """

    uid: int
    name: str
    op: str
    role: Optional[str]
    inputs_obj: Tuple[InputRef, ...]
    provenance: str

    @property
    def children(self) -> Sequence["Identified"]: ...


@runtime_checkable
class Evaluable(Identified, Protocol):
    """A node the walk can evaluate.

    Every field the evaluator reads, and the reason each is here:

    `dims` / `dims_raw` / `dim` / `dim_raw`
        the node's own bindings. Numeric consumers read the RAW form and
        re-resolve — see `Evaluator._node_dims_override` for why the resolved
        twin is the stale one.
    `repeat` / `active` / `repeat_raw` / `active_raw`
        its own copy counts, resolved and as written. Re-resolved per frame,
        because a per-rank binding divides the expert-copies symbol and a baked
        integer cannot see that.
    `shape`
        an explicit n-D shape, for source ops that declare rather than derive.
    `axis` / `axes`
        which axis an axis-aware op acts on, and the full permutation for
        `transpose` / `permute`.
    `inputs_obj`
        the dataflow, as object refs.
    `in_dispatch_zone`
        whether this node sits between a `permute` and its `combine`, i.e.
        inside a MoE's per-expert workload.
    `is_component`
        whether it has a body of its own rather than a cost of its own.
    `enclosing_components`
        the component invocations it sits inside, which a `component:`-scoped
        quant override selects on.

    An on-disk name is NOT among them. A node carries its local `name`; the
    checkpoint path is composed by `storage.tensor_name` from the node, its op
    and the stack's prefix, and composing it is the only way to get one.
    """

    is_component: bool
    in_dispatch_zone: bool
    dims: Mapping[str, Any]
    dims_raw: Mapping[str, str]
    dim: Optional[Any]
    dim_raw: Optional[str]
    repeat: Optional[int]
    active: Optional[int]
    repeat_raw: Optional[str]
    active_raw: Optional[str]
    shape: Optional[Tuple[Any, ...]]
    weight_axes: Optional[Tuple[Any, ...]]
    expert_weight: Optional[str]
    stored_axes: Optional[Tuple[int, ...]]
    stored_split: Optional[Tuple[Any, int]]
    stored_without_graph_reader: Optional[Any]
    host_offload: Optional[str]
    axis: Optional[int]
    axes: Optional[List[int]]
    owns_weight: bool
    stores_checkpoint_copy: bool
    boundary_role: Optional[str]

    @property
    def children(self) -> Sequence["Evaluable"]: ...

    @property
    def enclosing_components(self) -> Tuple[str, ...]:
        """The components the node sits inside — the scope a quant
        override's `component:` selector is resolved against."""
        ...
