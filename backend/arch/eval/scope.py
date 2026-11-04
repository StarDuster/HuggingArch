"""`Scope` — one frame of the new walk.

The engine's frame carries four dimensions. This one carries two, and the
difference is entirely which graph is being walked.

The engine walks `BlockGraph`: the authored syntax, with `op: <param>` still
unsubstituted, the dispatch zone still to be computed per graph, and input refs
still local-name strings resolved against a per-graph table. So it needs a
callee-params frame, a zone set, and a sibling table.

This walks the IR that `iter_tree` already produced, where all three are settled:
`op` is substituted, `in_dispatch_zone` is a flag on the node, and `inputs_obj`
holds object refs whose targets carry globally unique uids — so a single flat
`{uid: Shape}` map serves the whole walk, with no per-graph scoping at all.

What is left:

  `binding`   the dim environment (chained, immutable)
  `aliases`   the enclosing graph's `inputs:` bound by the caller
  `streams`   the model-wide streams this stage can see (`ModelLayer.streams`
              without `carries`, and those its components `reads`), each
              the live producer tensor. A stack's carried stream is not
              here: it arrives on its block's formal 0, like any input.

`aliases` exists for the refs `iter_tree` could not resolve to an object: 228 of
6374 across the shipped corpus, and every one of them names an enclosing scope's
input (`x`, `residual`, `x_hc`) rather than a sibling. They cannot be uid-keyed
because the value depends on the call site, not on the node — the same component
body sees a different `x` per invocation.

`streams` is deliberately NOT an alias: an alias is re-bound by the caller at
every component boundary, while a stream belongs to the layer stack and must
survive `entering()` so a nested component (pre_norm → mla_dsa_attention_shared)
still sees it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple,
)

from .binding import Binding

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .values import Shape

_NO_ALIASES: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class Carried:
    """What crosses one edge: a tensor, and where it lives.

    One value on one edge. The walk already threads a `Shape` along
    every edge and resolves the refs `iter_tree` could not — component inputs,
    a model-level `inputs:` edge, a stream — and a distributed-tensor STATE
    travels the same edges by the same rules. Carrying it beside the shape
    rather than in a parallel map is what keeps "which tensor arrives here" a
    question with one answer: a second traversal answering it independently
    is free to disagree, resolving a component's formal `x` to the previous
    sibling instead of to the tensor its caller bound.
    """

    shape: Optional["Shape"] = None
    #: layout `EdgeState`, or None when the walk carries no rule
    state: Any = None
    #: `KVLength` when this edge is a stack stream whose producer cached
    #: the tensor. Readers adopt it as `s_kv`; they do not declare the cache.
    kv_length: Any = None
    #: `(uid, port)` of the node that produced this tensor. A ref that crosses
    #: a scope boundary resolves to the caller's `Carried`, not to a `src`;
    #: this is how a redistribution on that edge still names its producer.
    producer: Optional[Tuple[int, str]] = field(default=None, compare=False)
    #: Which occurrences produced this value, per iteration of the stage
    #: that reads it (`occurrence.Reference`): bound where the walk binds
    #: the value, and carried with it into every component body.
    origin: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class Scope:
    binding: Binding
    aliases: Mapping[str, "Carried"] = _NO_ALIASES
    # The model-wide streams the enclosing stage can see, keyed by name.
    # Unlike `aliases`, this survives `entering()` — see the module
    # docstring.
    streams: Mapping[str, "Carried"] = _NO_ALIASES
    # Whether this is a component body's frame (`calling`): its aliases are
    # what one call handed in, so a name its body reads that the tree did
    # not graft onto them is a stream, whatever aliases share its spelling.
    in_call: bool = False
    # What fraction of an expert axis below this point a token runs.
    #
    # 1 everywhere except inside a MoE dispatch zone, where a token is routed to
    # `n_experts_active` of `n_route_targets` destinations, so a weight whose
    # leading axis is the experts (`n_expert_copies`, `kind: expert_copies`)
    # has that fraction of its elements touched per token. The denominator
    # counts DESTINATIONS, not experts that hold weights: on LongCat-Flash 256
    # of the 768 are `nn.Identity()`, and dividing by the 512 weighted ones
    # would run every token through 12 experts instead of 8. Decided once,
    # where the zone is entered, rather than rediscovered at every leaf.
    # None outside every dispatch zone.
    active_ratio: Optional[Tuple[Any, Any]] = None
    # How many physical copies of everything below this point the model holds,
    # and how many of them run per token.
    #
    # Cumulative, and inherited exactly like `binding` is: a block template's
    # layer count seeds it, each enclosing `repeat:` multiplies it. It rides on
    # the frame rather than being read back off the IR node because the count is
    # RESOLVED — `repeat: n_expert_copies` against this frame's binding, which is
    # per-rank when the frame is — and a cumulative int baked at build time
    # cannot be re-resolved.
    copies: Tuple[Any, Any] = (1, 1)

    # -- dim access ------------------------------------------------------

    def flat_dims(self) -> Dict[str, Any]:
        """The binding as a plain dict, which is what an op's `flops_fn` /
        `params_fn` takes.

        Recomputed per call rather than cached on the frame: a frame is created
        per component invocation, not per node, so this runs once per subtree
        and caching it would trade a real allocation for a hypothetical one.
        Node-local dims are merged by the caller, not folded in here — they
        belong to the node, and a frame outlives any single node.
        """
        return self.binding.flat()

    def child(self, **binds: Any) -> "Scope":
        return replace(self, binding=self.binding.child(**binds))

    def child_map(self, binds: Mapping[str, Any]) -> "Scope":
        if not binds:
            return self
        return replace(self, binding=self.binding.child_map(binds))

    def frame(self, binds: Mapping[str, Any],
              written: Optional[Mapping[str, Any]] = None) -> "Scope":
        """The frame a component call or a block assignment opens: its
        `dims:` over this one, with the `derived:` names they feed derived
        again (`Binding.frame`; `written` is the map as the spec wrote it)."""
        if not binds:
            return self
        return replace(self, binding=self.binding.frame(binds, written))

    # -- component boundaries --------------------------------------------

    def entering(self, aliases: Mapping[str, "Carried"]) -> "Scope":
        """Frame for a component body: the caller's bound inputs REPLACE the
        outer ones. A callee must not see its caller's inputs — that would
        make an unresolved alias silently pick up an outer tensor's shape
        instead of surfacing as missing. `streams` is NOT replaced: a stream
        belongs to the layer stack, not to the call site."""
        return replace(self, aliases=dict(aliases))

    def calling(self, aliases: Mapping[str, "Carried"]) -> "Scope":
        """Frame for a component call's body (`entering`), marked as one
        (`in_call`)."""
        return replace(self, aliases=dict(aliases), in_call=True)

    def alias(self, name: str) -> Optional["Carried"]:
        return self.aliases.get(name)

    def stream(self, name: str) -> Optional["Carried"]:
        return self.streams.get(name)

    def read(self, ref: Any) -> Optional["Carried"]:
        """What an unresolved `in:` ref reads here. In a component body a
        name the tree did not graft (`InputRef.grafted`) is a stream the
        body reads: the call's aliases carry the caller's names, and one of
        them may be spelled like it (an outer formal `s` handed to a
        renamed formal, beside a stream `s`). Anywhere else, `resolve`."""
        if self.in_call and not ref.grafted:
            return self.streams.get(ref.external_sid)
        return self.resolve(ref.external_sid)

    def resolve(self, name: str) -> Optional["Carried"]:
        """A name: an input the caller bound, else a model-wide stream the
        stage can see. Within one graph a formal and a stream of one name
        are refused at load (`BlockGraph._check_stream_targets`); across a
        call they may meet in one frame, which is why a component body's
        refs go through `read`."""
        found = self.aliases.get(name)
        if found is None:
            found = self.streams.get(name)
        return found

    def repeated(self, storage: Any, active: Any) -> "Scope":
        """Frame for the inside of `storage` copies of something.

        A block template used by 28 layers, an expert bank of 256 — the same
        operation, and the difference between them is only where the count came
        from. Multiplying into the frame is what lets a value DERIVED after the
        walk reach a model total: cost gets its multiplier applied at the point
        of production, and nothing else did.
        """
        if storage == 1 and active == 1:
            return self
        return replace(self, copies=(self.copies[0] * storage,
                                     self.copies[1] * active))

    def routed(self, active: Any, total: Any) -> "Scope":
        """Frame for the inside of a dispatch zone: a token runs `active` of
        the `total` experts every expert axis below here holds.

        Kept as the pair rather than a computed ratio so the multiplication
        stays exact — `n_experts_active / n_route_targets` of a weight's
        elements, an integer when it divides and the `Rational` average
        otherwise (`Evaluator._active_params`) — instead of routing an expert
        count through a float.
        """
        return replace(self, active_ratio=(active, total))

    def __repr__(self) -> str:
        return f"Scope({self.binding!r}, aliases={sorted(self.aliases)})"
