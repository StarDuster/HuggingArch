"""Block label derivation — the one short name a block template goes by.
It is computed once, when the block invocations are enumerated, and kept
on the invocation (`BlockMeta.short_name`): the architecture diagram's
block cards and group-sublayer cards (MTP / vision / etc.), the Inspector's
composition and attention cards, and the inference per-op and memory tables'
layer headers (`Program.block_label`) all read it there, so a block is named
one way.

Single rule, over the wrapper a block calls (`wrapper_of`: the template
the loader synthesized it from, else the wrapper a hand-written one-node
adapter binds, `BlockGraph.adapted_wrapper`):
  1. It binds an attention and an FFN  →  "attn + ffn"
  2. Else, what it calls, each by what it names (`contents_label`):
     Nemotron-H's mamba layer "Mamba2", Kimi-K3's "KDA + MoE FFN",
     DeepSeek-V3's MTP block "MLA + MoE FFN"
  3. Else (it calls nothing that has a name): the template name.
Only facts already declared are read — an attention's `attn_display`, a
role's label, what a block calls — so a name has no second statement to
drift from.

Every bound argument is shown by what it names (`slot_label`): an attention
component by its `attn_display` ("MLA", "CSA", …), an FFN by its role
("MoE FFN" / "Dense FFN"), a block that is itself a synthesized or adapted
wrapper by its own label — so DeepSeek-V3's MTP block reads
`mtp_block_with_head(inner=MLA + MoE FFN)`, not the inner block's
generated template id. Names are looked up in the spec's namespace
(`resolve`, blocks then components), the one the loader expanded them in.

Why both branches exist — they read two different shapes, and neither can
answer for the other:

  * A SYNTHESIZED block is the expansion of inline shorthand: six nodes
    with the `<param>` placeholders already substituted. Nothing in that
    graph still says "pre_norm", so the loader records the template it
    expanded. Derived from the graph it would not be.
  * A HAND-WRITTEN adapter (`qwen35_full_moe_block`, `chatglm_block`, and
    seven others) is authored as the one node it is: `op: pre_norm` with
    `attn`/`ffn` params right there. The template isn't recorded because
    it was never expanded — it IS the graph, so `adapted_wrapper`
    reads it off directly rather than the loader restating it.

A short name is a name: two templates of one stack must not share it. When
the labels collapse two (DeepSeek-V4's hash-routed and score-routed MoE
are both "CSA + MoE FFN"), `distinct_block_labels` tells them apart by
where each runs (`· layer 2` / `· layers 4, 6, …, 60 (every 2)`); only
where the layers cannot (no layers known, or the same ones) does it show
the arguments that differ by their component names.
"""
from __future__ import annotations

from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Mapping,
                    Optional, Sequence, Tuple, TYPE_CHECKING)

from backend.arch import roles
from backend.arch.layer_span import layer_span

if TYPE_CHECKING:
    from backend.arch.schema.graph import BlockGraph

Resolve = Callable[[str], Optional["BlockGraph"]]

def wrapper_of(block: Optional[Any],
               resolve: Resolve) -> Optional[Tuple[str, Dict[str, str]]]:
    """`(wrapper, bindings)` a block calls — the template the loader
    expanded it from, else the wrapper a one-node adapter binds — or None."""
    if block is None:
        return None
    if block.synth_template:
        return block.synth_template, dict(block.synth_params or {})
    return block.adapted_wrapper(resolve)


def slot_label(name: str, resolve: Resolve,
               keep: FrozenSet[str] = frozenset(),
               _seen: FrozenSet[str] = frozenset()) -> str:
    """What a bound argument `name` is shown as (module docstring). A name
    in `keep` is shown as itself — or, when it is a wrapper the loader
    generated a name for, as its label with the kept arguments it binds
    (`CSA[indexer=csa2_indexer]`); so is a name the namespace does not
    hold."""
    if not name or name in _seen:
        return name
    graph = resolve(name)
    if graph is None:
        return name
    if name in keep:
        wrapped = wrapper_of(graph, resolve)
        if wrapped is None:
            return name
        kept = [(k, str(v)) for k, v in wrapped[1].items() if str(v) in keep]
        if not kept:
            return name
        own = slot_label(name, resolve, frozenset(), _seen)
        args = ", ".join(f"{k}={slot_label(v, resolve, keep, _seen | {name})}"
                         for k, v in kept)
        return f"{own}[{args}]"
    if graph.attn_display:
        return graph.attn_display
    role = roles.get(graph.role) if graph.role else None
    if role is not None and role.label:
        return role.label
    if wrapper_of(graph, resolve) is not None:
        return derive_block_label(name, graph, resolve, keep,
                                  _seen | {name})
    # A component that names nothing itself reads as what it calls
    # (ChatGLM's attention shim around a GQA attention reads "GQA").
    return contents_label(graph, resolve, keep, _seen | {name}) or name


def derive_block_label(
    tmpl: str,
    block: Optional[Any],
    resolve: Resolve,
    keep: FrozenSet[str] = frozenset(),
    _seen: FrozenSet[str] = frozenset(),
) -> str:
    """The human-readable label of block template `tmpl` (module docstring).
    `block` is its resolved BlockGraph (None when it can't be resolved);
    `resolve` is the spec's namespace lookup; `keep` names components shown
    by name rather than by label, the block's own wrapper among them when
    the wrapper is what tells it apart."""
    wrapped = wrapper_of(block, resolve)
    seen = _seen | {tmpl}
    if wrapped is None:
        return contents_label(block, resolve, keep, seen) or tmpl
    wrapper, params = wrapped

    def label(value: str) -> str:
        return slot_label(str(value), resolve, keep, seen)

    attn_b = label(params.get("attn", ""))
    ffn_b = label(params.get("ffn", ""))
    if attn_b and ffn_b and wrapper not in keep:
        return f"{attn_b} + {ffn_b}"
    if wrapper not in keep:
        own = contents_label(block, resolve, keep, seen)
        if own:
            return own
    if block.synth_template is None and wrapper not in keep:
        return tmpl
    args = ", ".join(f"{k}={label(v)}" for k, v in params.items())
    return f"{wrapper}({args})" if args else wrapper


def contents_label(block: Optional[Any], resolve: Resolve,
                   keep: FrozenSet[str] = frozenset(),
                   _seen: FrozenSet[str] = frozenset()) -> str:
    """What a block that binds no attention-and-FFN pair reads as: the
    components it calls, each by what it names (`slot_label`: an attention
    by its `attn_display`, an FFN by its role, a wrapped block by its own
    label), in the order it calls them, each once — Nemotron-H's mamba
    layer reads "Mamba2", Kimi-K3's layers "KDA + MoE FFN", DeepSeek-V3's
    MTP block "MLA + MoE FFN" (the stack it runs in says it is the MTP).
    A component that names nothing itself is read through for what it
    calls. A hand-written adapter's one node is its wrapper, so it reads
    through its bindings. Empty when nothing it calls has a name."""
    if block is None:
        return ""
    wrapped = wrapper_of(block, resolve)
    if wrapped is not None and block.synth_template is None:
        called = [str(v) for v in wrapped[1].values()]
    else:
        called = [node.op for node in (block.graph or {}).values()]
    parts: List[str] = []
    for name in called:
        if name in _seen:
            continue
        graph = resolve(name)
        if graph is None:
            continue  # an op, not a component
        shown = slot_label(name, resolve, keep, _seen)
        found = ([shown] if shown != name else
                 [p for p in contents_label(graph, resolve, keep,
                                            _seen | {name}).split(" + ") if p])
        for part in found:
            if part not in parts:
                parts.append(part)
    return " + ".join(parts)


def slot_bindings(name: str, resolve: Resolve,
                  _prefix: str = "") -> Dict[str, str]:
    """Every slot the block `name` binds, through the wrappers it binds in
    turn, by its dotted path: `{"attn": "csa2_attention__…",
    "attn.compressor": "csa2_compressor_gated", "ffn": …}`."""
    wrapped = wrapper_of(resolve(name), resolve)
    if wrapped is None:
        return {}
    out: Dict[str, str] = {}
    for slot, value in wrapped[1].items():
        path = f"{_prefix}{slot}"
        out[path] = str(value)
        if str(value) != name:
            out.update(slot_bindings(str(value), resolve, path + "."))
    return out


def differing_slots(templates: Iterable[str], resolve: Resolve
                    ) -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """Per template of `templates` (blocks of one kind), the innermost
    slots whose binding is not the same in all of them, with what each
    binds there: what tells the template from its kind-mates."""
    templates = list(dict.fromkeys(templates))
    bound = {t: slot_bindings(t, resolve) for t in templates}
    slots = set().union(*bound.values()) if bound else set()
    differ = {s for s in slots
              if len({bound[t].get(s) for t in templates}) > 1}
    leaves = sorted(s for s in differ
                    if not any(o.startswith(s + ".") for o in differ))
    return {t: tuple((s, bound[t][s]) for s in leaves if s in bound[t])
            for t in templates}


def _bound_names(block: Optional[Any], resolve: Resolve,
                 _seen: FrozenSet[str] = frozenset()) -> FrozenSet[str]:
    """The wrapper `block` calls and every component name it binds,
    through nested wrappers."""
    wrapped = wrapper_of(block, resolve)
    if wrapped is None:
        return frozenset()
    out = {wrapped[0]}
    for value in wrapped[1].values():
        value = str(value)
        out.add(value)
        if value not in _seen:
            out |= _bound_names(resolve(value), resolve, _seen | {value})
    return frozenset(out)


#: Past this many characters a run of layer spans is read as a count: a
#: name is for telling two blocks apart at a glance, and the layers
#: themselves are listed beside it where a page shows a block.
_SPAN_CHARS = 32


def _where(layers: Sequence[int]) -> str:
    """Where a block runs in its stack, as a short name says it: `layer 2`,
    `layers 0–2`, `layers 4, 6, …, 60 (every 2)`, or `66 layers from 1`."""
    span = layer_span(layers)
    if len(layers) == 1:
        return f"layer {span}"
    if len(span) > _SPAN_CHARS:
        return f"{len(layers)} layers from {layers[0]}"
    return f"layers {span}"


def distinct_block_labels(stacks: Iterable[Iterable[str]],
                          resolve: Resolve, *,
                          wrappers_only: bool = False,
                          layers: Optional[Sequence[Mapping[str, Sequence[int]]]] = None,
                          ) -> Dict[str, str]:
    """`derive_block_label` for each template of one model, given as the
    templates of each stack it runs in, with any two of one stack that
    would read the same told apart: the wrappers and components called by
    some but not all of a colliding set are shown by name (`slot_label`).

    `wrappers_only` tells them apart only by the wrapper each calls itself,
    leaving the components bound into its slots by label: the KIND of
    block — two templates that differ in what fills an attention slot are
    one kind, one whose wrapper carries a module the other's does not
    (DeepSeek-V4.1's `dsv41_hc_block_engram`) is another."""
    stacks = [list(dict.fromkeys(names)) for names in stacks]
    labels = {t: derive_block_label(t, resolve(t), resolve)
              for names in stacks for t in names}
    keep: Dict[str, FrozenSet[str]] = {}
    spanned: Dict[str, str] = {}
    for at, names in enumerate(stacks):
        by_label: Dict[str, List[str]] = {}
        for t in names:
            by_label.setdefault(labels[t], []).append(t)
        own_layers = layers[at] if layers is not None else {}
        for clash in by_label.values():
            if len(clash) < 2:
                continue
            # Where each runs tells them apart in words a reader has
            # (`layers 0–2` / `layers 3–60`), where the components they
            # bind would only repeat ids.
            spans = [_where(sorted(own_layers[t])) if own_layers.get(t)
                     else None for t in clash]
            if None not in spans and len(set(spans)) == len(spans):
                for t, span in zip(clash, spans, strict=True):
                    spanned[t] = f"{labels[t]} · {span}"
                continue
            bound = [_bound_names(resolve(t), resolve) for t in clash]
            differ = frozenset().union(*bound) - frozenset.intersection(*bound)
            if wrappers_only:
                own = {w[0] for w in (wrapper_of(resolve(t), resolve)
                                      for t in clash) if w is not None}
                differ &= own
            for t in clash:
                keep[t] = keep.get(t, frozenset()) | differ
    for t, names in keep.items():
        labels[t] = derive_block_label(t, resolve(t), resolve, names)
    labels.update(spanned)
    return labels
