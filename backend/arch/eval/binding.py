"""`Binding` — the dim environment, as a chained immutable scope.

Every extent a frame binds is the GLOBAL one. What one rank holds is
`arch.layout`'s answer, taken from the tensor's layout rather than by
dividing a symbol on the way out of the environment — so a binding means the
same thing to every reader and there is no ordering question about when a
divisor fires.

A non-divisible cut is refused when the layout is constructed, unless the
`Block` declared `pad_to`. There is no silent "largest even divisor" on
the way out of a lookup: that rewrite had no replica-cell picture, and
the walk and the placement were free to disagree.

What a binding holds
--------------------
Every number the evaluator produces comes from resolving a spec expression
(`out="n_heads*(d_qk_nope+d_qk_rope)"`) against `{symbol: value}`. That map is
layered, and the layers have different lifetimes: spec-declared dims live for the
whole run, the workload (`B`, `s_query`, `s_kv`, `s_pairs`) for one
analysis, a per-template override (`n_kv_heads` rebound for a hybrid model's SWA
layers) for one block, a component call site's `dims:` for one subtree, and the
MoE dispatch narrowing `(B, s_query) → (M, 1)` for the nodes inside one zone.
Op templates still write the residual sequence as `s_ctx`; a lookup of that
name reads `s_query`. The phase does not bind `s_ctx` as a second number.

The engine implements those lifetimes by mutating one dict and restoring a copy
afterwards, at five separate sites. `Binding` implements them as a parent chain:
`child()` returns a new frame, and nothing is restored because nothing was
overwritten. The frame a line sees is the one it was handed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterator, Mapping, Optional, Tuple

import sympy as sp

import backend.arch.dims as dim_registry

from backend.arch.law_text import read_law

from .dim import Dim, eval_dim_expr, exact, expr_of, names_of, value_of

_EMPTY: Mapping[str, Any] = MappingProxyType({})

# Op templates name the residual sequence `s_ctx` because an op has no
# prefill/decode. The phase binds that length as `s_query` (decode: the
# identity is `n_verify`). One number; this alias is the lookup, not a
# second bind.
_S_CTX = "s_ctx"
_S_QUERY = "s_query"


@dataclass(frozen=True)
class Binding:
    """`{dim symbol: value}` as an immutable frame with a parent chain."""

    own: Mapping[str, Any] = _EMPTY
    parent: Optional["Binding"] = None
    # Symbols whose BOUND value outranks whatever extent the dataflow produced.
    # Declared by whoever binds them, because knowing that an axis is not in the
    # dataflow is knowing something about the workload, not about the graph: in
    # decode the K/V leaves emit only the new token while attention reads a
    # cache no node produces. Empty unless something says otherwise.
    authoritative: FrozenSet[str] = frozenset()
    # Whether this frame carries identities as well as values, i.e. came from
    # `tracked()`. A flag rather than an inspection of the values, because every
    # `child` would otherwise pay a scan to find out.
    #
    # What it BUYS is that a value bound into a tracked frame keeps an identity
    # even when it did not come through the expression funnel. Two sites bind
    # bare integers: a block template's `dim_overrides` — MiMo-V2.5's SWA layers
    # rebinding `n_kv_heads` — and the MoE dispatch narrowing. Both dropped the
    # symbol's name, and it did not stop there: `v_split_heads` came out with an
    # anonymous head axis and every slice, rope and concat downstream of it
    # inherited one. One lost identity, a whole attention block's worth of
    # consequence.
    _tracked: bool = False

    # -- reading ---------------------------------------------------------

    def __getitem__(self, symbol: str) -> Any:
        frame: Optional[Binding] = self
        while frame is not None:
            if symbol in frame.own:
                return frame.own[symbol]
            frame = frame.parent
        if symbol == _S_CTX:
            return self[_S_QUERY]
        raise KeyError(symbol)

    def get(self, symbol: str, default: Any = None) -> Any:
        try:
            return self[symbol]
        except KeyError:
            return default

    def __contains__(self, symbol: str) -> bool:
        frame: Optional[Binding] = self
        while frame is not None:
            if symbol in frame.own:
                return True
            frame = frame.parent
        if symbol == _S_CTX:
            return _S_QUERY in self
        return False

    def raw(self, symbol: str) -> Any:
        """The bound value. Extents are global; this is the same as `[]`."""
        return self[symbol]

    def _owns(self, symbol: str) -> bool:
        frame: Optional[Binding] = self
        while frame is not None:
            if symbol in frame.own:
                return True
            frame = frame.parent
        return False

    def symbols(self) -> FrozenSet[str]:
        out: set = set()
        frame: Optional[Binding] = self
        while frame is not None:
            out |= set(frame.own)
            frame = frame.parent
        return frozenset(out)

    def flat(self) -> Dict[str, Any]:
        """Every visible symbol. For handing to an expression evaluator that
        wants a plain mapping; inner frames win. `s_ctx` is filled from
        `s_query` when the phase did not bind it, so YAML `s_ctx // k`
        still evaluates."""
        out = {s: self[s] for s in self.symbols()}
        if _S_QUERY in out and _S_CTX not in out:
            out[_S_CTX] = out[_S_QUERY]
        return out

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols())

    def __len__(self) -> int:
        return len(self.symbols())

    # -- scoping ---------------------------------------------------------

    def child(self, **binds: Any) -> "Binding":
        """A new frame binding `binds`, inheriting everything else. Replaces
        `saved = dict(dims)` … `dims = saved`: there is nothing to restore
        because the parent frame was never written to."""
        return self.child_map(binds)

    def child_map(self, binds: Mapping[str, Any]) -> "Binding":
        """`child` for a computed mapping (a component's `dims:` block).

        A bare value bound into a TRACKED frame is given its symbol's identity
        here — see `_tracked`. Anything that already carries one passes through
        untouched, so a value that came via the expression funnel keeps the
        composed expression the funnel built rather than being flattened to its
        own name.
        """
        return Binding(own=self._identify(binds), parent=self,
                       authoritative=self.authoritative,
                       _tracked=self._tracked)

    def frame(self, binds: Mapping[str, Any],
              written: Optional[Mapping[str, Any]] = None) -> "Binding":
        """The frame a component call or a block assignment opens: `binds`
        over this binding, and every `derived:` name (`dims.yaml`) that reads
        — directly or through another — a name `binds` gives another value,
        derived again in the new frame.

        A `derived:` name is never written by a spec, so its value is always
        its expression, and in a frame that rebinds `n_experts` that is the
        frame's own: a draft pool's call carries its own `n_expert_copies`
        and `n_route_targets`. Against the WHOLE frame, so an inherited
        `n_zero_experts` still counts. A `default:` does not follow: the frame
        cannot tell a defaulted value from one the model bound. A name `binds`
        states itself is what that map says. `written` is the map as the
        spec wrote it, when the caller has it: an entry written as its own
        name (`n_experts: n_experts`, LongCat's block) rebinds nothing, so
        nothing derived from it moves (`rebound_names`).

        Each value is bound under its own identity — `Dim.symbol(name, …)`,
        never the source's `Dim` passed through — so the expert axis is the
        expert axis whatever count it came to, and expert parallelism divides
        it by identity while the denominator stays whole. A primitive's local
        `dims:` is not a frame and uses `child_map`: it has no subtree for a
        derived value to reach."""
        child = self.child_map(binds)
        rederived = rederived_values(binds, written, self.flat(),
                                     tracked=self._tracked)
        if len(rederived) == 0:
            return child
        return replace(child, own={**child.own, **rederived})

    def _identify(self, binds: Mapping[str, Any]) -> Dict[str, Any]:
        if not self._tracked:
            return dict(binds)
        return {name: value if isinstance(value, Dim)
                else Dim.symbol(name, value)
                for name, value in binds.items()}

    def with_authoritative(self, *symbols: str) -> "Binding":
        """Declare that these symbols' bound values outrank the dataflow."""
        return replace(self, authoritative=self.authoritative | frozenset(symbols))

    def tracked(self) -> "Binding":
        """The same binding with every dim carrying its own name alongside its
        value — `Dim`, both halves — so one walk answers what an axis IS and
        what it comes to. Every dim is wrapped, those bound to 0 or 1 too: in
        decode `s_query` is bound to 1, and a query axis with no identity
        could not be told apart from another axis of 1 (`bool` of a `Dim`
        resolves to its value, so a flag tested with `if d.get("bias")`
        still reads its number).
        """
        tracked = {name: Dim.symbol(name, value)
                   for name, value in self.flat().items()}
        return Binding(own=tracked, parent=None,
                       authoritative=self.authoritative, _tracked=True)

    def rebased(self) -> "Binding":
        """Collapse the parent chain into one frame. Chains are short (block →
        component → node) so lookup depth is not a concern, but a long-lived
        binding handed to many walks is worth flattening once."""
        return Binding(own=self.flat(), parent=None,
                       authoritative=self.authoritative)

    def __repr__(self) -> str:
        depth = 0
        frame: Optional[Binding] = self
        while frame is not None:
            depth += 1
            frame = frame.parent
        return f"Binding(depth={depth}, symbols={len(self)})"


@dataclass(frozen=True)
class Definition:
    """A derived name as its definition: the law in the names it reads
    (`law`), what each of those names was bound to where it was formed
    (`bindings`, `{name: value}`), and its value there. The law is what a
    reader of names follows (`frame_names`) and what a formula's where
    column prints, with the bindings put in beside it."""

    law: Any
    bindings: Mapping[str, Any]
    value: Any

    @classmethod
    def of(cls, law: Any, values: Mapping[str, Any]) -> "Definition":
        """A declared `law` (text in names) over the names `values` binds,
        evaluated exactly there: a whole number is an `int`."""
        expr = read_law(law)
        bindings = {symbol.name: value_of(values[symbol.name])
                    for symbol in expr.free_symbols}
        value = expr.xreplace({
            symbol: exact(bindings[symbol.name])
            for symbol in expr.free_symbols})
        return cls(expr, bindings,
                   int(value) if isinstance(value, sp.Integer) else value)

    @classmethod
    def of_extent(cls, extent: Any, frame: Mapping[str, Any]) -> "Definition":
        """An op's law evaluated in `frame`: its identity half is the law in
        the frame's names, its value half the value, and each name the law
        reads is bound to what the frame binds it to."""
        law = exact(expr_of(extent))
        return cls(law, {symbol.name: value_of(frame[symbol.name])
                         for symbol in law.free_symbols
                         if symbol.name in frame},
                   value_of(extent))

    def __add__(self, other: "Definition") -> "Definition":
        return Definition(self.law + other.law,
                          {**self.bindings, **other.bindings},
                          self.value + other.value)


#: The phase's derived step extents, each as the law it is declared to be
#: (`Phase.resolve` reads them here). `s_past`: the keys before the step's
#: first query. `s_pairs`: the step's live (query, key) pairs of the dense
#: causal matrix — the query at offset `i` sees `s_past + i + 1` keys.
STEP_LAWS: Mapping[str, str] = MappingProxyType({
    "s_past": "s_kv - s_query",
    "s_pairs": "s_query*s_past + s_query*(s_query + 1)/2",
})


def rebound_names(binds: Mapping[str, Any],
                  written: Optional[Mapping[str, Any]] = None) -> set:
    """The names a frame's `dims:` gives another value: every entry, except
    one the spec wrote as its own name (`n_experts: n_experts`), which is a
    declared identity. Read off the text, not the values — whether two
    values are one is a magnitude question, and the spec already said."""
    written = written or {}
    return {name for name in binds
            if str(written.get(name, "")).strip() != name}


def rederivations(binds: Mapping[str, Any],
                  written: Optional[Mapping[str, Any]],
                  visible: Any) -> Tuple[Dict[str, str], ...]:
    """The `derived:` names a frame opening `binds` over the names `visible`
    derives again (`Binding.frame`), each with its expression, round by
    round: a round's names read only names known before it — `visible`,
    `binds`, the rounds before — and at least one name moved — rebound by
    `binds` (`rebound_names`) or derived again by an earlier round."""
    moved = rebound_names(binds, written)
    known = set(visible) | set(binds)
    pending = {name: sym for name, sym in _derived_rows().items()
               if name not in binds and name in visible}
    rounds = []
    while True:
        ready = {name: sym.derived for name, sym in pending.items()
                 if any(r in moved for r in sym.reads)
                 and all(r in known for r in sym.reads)}
        if len(ready) == 0:
            return tuple(rounds)
        for name in ready:
            del pending[name]
        rounds.append(ready)
        moved.update(ready)
        known.update(ready)
    return tuple(rounds)


def rederived_values(binds: Mapping[str, Any],
                     written: Optional[Mapping[str, Any]],
                     visible: Mapping[str, Any], *,
                     tracked: bool = False) -> Dict[str, Any]:
    """The `derived:` names a plain dim mapping reopens, as plain values.

    `Binding.frame` is the engine's scoped form. The spec-tree builder keeps a
    stack of mappings while it expands calls; this is the same re-derivation for
    that stack, so an IR leaf and the runtime placement see one frame."""
    values = {**visible, **binds}
    out: Dict[str, Any] = {}
    for ready in rederivations(binds, written, visible):
        derived = {
            name: _derived_in(name, expression, values, tracked)
            for name, expression in ready.items()}
        values.update(derived)
        out.update(derived)
    return out


def declared_names(names: Any, binds: Mapping[str, Any],
                   written: Optional[Mapping[str, Any]],
                   visible: Optional[Any] = None,
                   rederived: Optional[Mapping[str, str]] = None) -> set:
    """`names`, read inside the frame `binds` opens, in the names of the
    frame around it — by what the frame declares, not by what it came to.

    A name `binds` gives is the names of its expression as the spec wrote
    it (`written`; the bound value where nothing was written). A name the
    frame derives again (`rederivations`, when the frame opens over the
    names `visible` — a component call's or a block assignment's;
    `visible` None is a primitive's own `dims:`, which derives nothing) is
    the names its `derived:` expression reads, themselves read inside the
    frame; `rederived` adds names the frame reads as laws of its own
    (`{name: text}`: the phase's `STEP_LAWS`). Any other name is the
    enclosing frame's own."""
    rederived = dict(rederived or {})
    if visible is not None:
        for ready in rederivations(binds, written, visible):
            rederived.update(ready)
    written = written or {}
    out: set = set()
    pending = list(names)
    seen: set = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in rederived:
            pending.extend(names_of(read_law(rederived[name])))
        elif name in binds:
            text = written.get(name, binds[name])
            out |= (names_of(read_law(text)) if isinstance(text, str)
                    else set(names_of(text)))
        else:
            out.add(name)
    return out


def _derived_in(name: str, expression: str, values: Mapping[str, Any],
                tracked: bool) -> Any:
    """A `derived:` name's value in a frame, bound under its own identity
    (`Dim.symbol`) in a tracked binding — never the source's `Dim`."""
    value = value_of(eval_dim_expr(expression, dict(values)))
    if isinstance(value, sp.Integer):
        value = int(value)
    return Dim.symbol(name, value) if tracked else value


def _derived_rows() -> Mapping[str, Any]:
    """The registry's `derived:` rows, base and spec-local, re-read from the
    merged table each call (a spec's `dim_symbols:` layer may come and go)."""
    return {name: sym for name, sym in dim_registry.symbols().items()
            if sym.derived}
