"""`Layer[T]` — one derived fact about every node, and `Rewrite` — what a graph
transform did, in a form layers can follow.

Derived facts belong to distinct concerns: identity, cost, shape, axis labels,
symbolic expressions, display strings and resolved dimensions. Keeping those
concerns in one tree would make every consumer depend on every fact and force
graph rewrites to reconstruct them together.

A `Layer` is one of those concerns, on its own, keyed by node uid:

    shapes:   Layer[Shape]
    cost:     Layer[Cost]
    formulas: Layer[Expr]
    display:  formulas.zip(shapes).map(render)      # derived, not stored

`map` / `zip` make the derived ones derived rather than persisted, which is the
point: `display` costs nothing until someone renders, instead of ten strings
built per node per walk.

Keying on `uid` rather than on the composed `aggregation_id` string is the other
half. Several node identities are useful at boundaries — local name, uid,
tensor name, and the composed `f"{template}|{display_path}"` — and crossing
between representations means string surgery. `SpecIRTree` already allocates a
tree-unique uid per node and hands out fresh ones for nodes created during
mutation, so the identity exists; it just isn't what anything joins on.

Layers do not all come from the walk
------------------------------------
`Shapes` / `Cost` / `Formulas` / `Placement` propagate along edges. `Formats`
(quant) does not: a weight's dtype follows from its role and the model's
`QuantContext`, not from what feeds it. `Measured` does not either — it is
external corpus joined in by identity. Those two are sideband layers, aligned by
uid and never derived by traversal, which is the structural reading of "measured
sits beside the theory, it does not overwrite it".

`Bytes = Shapes.zip(Formats)` is where a propagated layer and a sideband layer
meet, and it is deliberately ONE place. Every consumer that re-derives this
join by name instead is a fork of the same fact — the failure mode is a read
path assuming a uniform fp8 while the storage path uses the native bf16 for
the same tensor, with nothing loud in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Generic, Iterator, Mapping, Optional,
    Sequence, Tuple, TypeVar,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .node import Identified
    from .program import Program

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
K = TypeVar("K")

#: Distinct from a key whose value is legitimately `None`, which is a bucket.
_MISSING = object()


class MissingValue(KeyError):
    """A layer was asked for a node it has no value for.

    Distinct from a value that is legitimately `None`: a layer may hold None for
    a node whose shape could not be inferred, and that is an answer. Absence is
    not, so it raises rather than returning None and letting the caller mistake
    one for the other.
    """


@dataclass(frozen=True)
class Rewrite:
    """What a `GraphTransform` did, as a mapping layers can be carried across.

    A graph transform is not only a new graph — it is also the correspondence
    between the old nodes and the new ones, and without that correspondence
    every existing layer silently becomes garbage: its keys refer to nodes the
    new graph no longer contains, and a lookup that should fail instead misses
    into whatever uid happens to be reused.

    `origins` maps each NEW node to the old nodes it came from: three
    projections fused into one QKV node map that node to those three. A node
    the transform left alone does not appear (identity is implied), and a node
    it deleted appears nowhere.

    Storage is deliberately out of scope. Fusion rewrites the COMPUTE graph —
    a fused QKV still reads three weight tensors, each with its own dtype and
    its own bytes on disk — so the sideband `Formats` layer keys on storage
    nodes that no compute-graph rewrite touches, and needs no rebase at all.
    Giving `Rewrite` a "preserve" mode for it would invent a second, subtly
    different meaning for the same operation.
    """

    origins: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    dropped: frozenset = frozenset()
    #: Each input of a NEW node, as the input of the old node it carries:
    #: `(new uid, input) -> (old uid, input)`. A fused node reads, at each
    #: input, what the member whose edge it took over read there.
    inputs: Mapping[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)

    @property
    def added(self) -> frozenset:
        return frozenset(self.origins)

    def sources(self, uid: int) -> Tuple[int, ...]:
        """The old nodes `uid` was built from; empty for an untouched node."""
        return tuple(self.origins.get(uid, ()))

    def is_identity(self) -> bool:
        return not self.origins and not self.dropped and not self.inputs

    def then(self, other: "Rewrite") -> "Rewrite":
        """Compose two rewrites applied in order, flattening origins through the
        intermediate graph so a layer can cross both in one `rebase`."""
        origins: Dict[int, Tuple[int, ...]] = {}
        for new_uid, mid_uids in other.origins.items():
            flat: list = []
            for mid in mid_uids:
                first = self.origins.get(mid)
                flat.extend(first if first is not None else (mid,))
            origins[new_uid] = tuple(flat)
        for new_uid, srcs in self.origins.items():
            if new_uid not in origins and new_uid not in other.dropped:
                origins[new_uid] = srcs
        inputs: Dict[Tuple[int, int], Tuple[int, int]] = {
            key: self.inputs.get(mid, mid)
            for key, mid in other.inputs.items()}
        for key, old in self.inputs.items():
            if key not in inputs and key[0] not in other.dropped:
                inputs[key] = old
        return Rewrite(origins=origins, dropped=self.dropped | other.dropped,
                       inputs=inputs)


@dataclass(frozen=True)
class Layer(Generic[T]):
    """One derived fact per node, keyed by node uid.

    `program` is the graph the keys refer to, and it is optional for one reason:
    a layer built mid-walk has values before it has anywhere to look a uid up.
    Everything a finished `Analysis` hands out carries it, so a derivation that
    needs the node — an op name, a role — asks the layer rather than being
    handed the program a second time alongside it.
    """

    program: Optional["Program"] = None
    values: Mapping[int, T] = field(default_factory=dict)

    # -- reading ---------------------------------------------------------

    def __getitem__(self, node: Any) -> T:
        uid = _uid_of(node)
        try:
            return self.values[uid]
        except KeyError:
            raise MissingValue(
                f"layer has no value for uid {uid}"
                f"{f' ({node.name!r})' if hasattr(node, 'name') else ''}"
            ) from None

    def get(self, node: Any, default: Any = None) -> Any:
        return self.values.get(_uid_of(node), default)

    def __contains__(self, node: Any) -> bool:
        return _uid_of(node) in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    # -- deriving --------------------------------------------------------

    def map(self, fn: Callable[[T], U]) -> "Layer[U]":
        """Apply `fn` to every value. The derived-not-stored path: `display` is
        `formulas.zip(shapes).map(render)`, computed when read."""
        return Layer(self.program, {u: fn(v) for u, v in self.values.items()})

    def map_with_node(self, fn: Callable[["Identified", T], U]) -> "Layer[U]":
        """`map`, for derivations that also need the node (an op name, a role)."""
        if self.program is None:
            raise ValueError("map_with_node needs a program to resolve uids")
        by_uid = self.program.by_uid
        return Layer(self.program,
                     {u: fn(by_uid(u), v) for u, v in self.values.items()})

    def filter(self, pred: Callable[[T], bool]) -> "Layer[T]":
        return Layer(self.program,
                     {u: v for u, v in self.values.items() if pred(v)})

    def leaves(self) -> "Layer[T]":
        """Only the nodes with nothing beneath them.

        The restriction every aggregation over a TREE needs and none of them
        should spell out: a component's value already includes its children, so
        summing both counts the subtree twice. Stated once here rather than as a
        `if node.children: continue` inside each consumer's loop, which is where
        it was and which is a filter wearing a traversal's clothes.
        """
        if self.program is None:
            raise ValueError("leaves needs a program to know what has children")
        by_uid = self.program.by_uid
        return Layer(self.program, {u: v for u, v in self.values.items()
                                    if not by_uid(u).children})

    def group_by(self, keys: "Layer[K]",
                 combine: Callable[[Sequence[T]], T]) -> Dict[K, T]:
        """Fold this layer's values into buckets named by another layer.

        Attribution — cost by role, cost by stack — is a join and a group-by
        once both sides are layers. The engine reaches the same answer with a
        recursive classifier that re-descends the result tree deciding per node
        which bucket it belongs to, and a node it cannot classify falls through
        a chain of tests into "neither"; here a node with no key is simply not
        in the key layer, which is the same statement without the fall-through.

        The intersection of the two domains, for the reason `zip` gives: a key
        layer is partial by construction — only a roled node has a role.
        """
        buckets: Dict[K, list] = {}
        for uid, value in self.values.items():
            key = keys.values.get(uid, _MISSING)
            if key is _MISSING:
                continue
            buckets.setdefault(key, []).append(value)  # type: ignore[arg-type]
        return {key: combine(parts) for key, parts in buckets.items()}

    def zip(self, other: "Layer[U]") -> "Layer[Tuple[T, U]]":
        """Pair values on the nodes BOTH layers cover.

        The intersection is the honest join: a propagated layer covers every
        node while a sideband layer covers only the ones it has data for
        (`Measured` over a partial corpus, `Formats` over weight-bearing nodes),
        so requiring equal domains would make the common case an error.
        """
        return Layer(self.program or other.program,
                     {u: (v, other.values[u])
                      for u, v in self.values.items() if u in other.values})

    def zip_with(self, other: "Layer[U]",
                 fn: Callable[[T, U], V]) -> "Layer[V]":
        return self.zip(other).map(lambda pair: fn(pair[0], pair[1]))

    # -- following a graph transform -------------------------------------

    def rebase(self, rw: Rewrite, program: Optional["Program"],
               combine: Callable[[Tuple[T, ...]], T]) -> "Layer[T]":
        """Carry this layer across a graph transform.

        A node the transform left alone keeps its value. A node it created takes
        `combine(values of its origins)` — for cost that is a sum, for shape it
        is the surviving endpoint's. A node it dropped is gone.

        `combine` is required rather than defaulted because there is no correct
        default: silently taking the first origin's value would make a fused
        QKV report one projection's FLOPs, which is exactly the kind of
        three-times-under-count that reads as plausible.
        """
        out: Dict[int, T] = {u: v for u, v in self.values.items()
                             if u not in rw.dropped and u not in rw.origins}
        for new_uid, srcs in rw.origins.items():
            present = tuple(self.values[s] for s in srcs if s in self.values)
            if present:
                out[new_uid] = combine(present)
        return Layer(program if program is not None else self.program, out)

    # -- building --------------------------------------------------------

    @classmethod
    def build(cls, program: Optional["Program"], pairs) -> "Layer[T]":
        """From `(node_or_uid, value)` pairs."""
        return cls(program, {_uid_of(k): v for k, v in pairs})

    def with_value(self, node: Any, value: T) -> "Layer[T]":
        merged = dict(self.values)
        merged[_uid_of(node)] = value
        return Layer(self.program, merged)

    def __repr__(self) -> str:
        return f"Layer(n={len(self.values)})"


def _uid_of(node: Any) -> int:
    """Accept a node or a bare uid. Layers are keyed by uid, but callers holding
    a node should not have to reach for `.uid` at every call site."""
    if isinstance(node, int):
        return node
    uid = getattr(node, "uid", None)
    if uid is None:
        raise TypeError(f"not a node and not a uid: {node!r}")
    return uid
