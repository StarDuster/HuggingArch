"""Cross-layer stream checks — anchor the `streams:` topology.

A `streams:` on a repeated layer group declares the stack's streams. The
one it carries across its block boundary (`carries: true`, `hidden`) is
checked by the schema (`ArchSpec._check_carried_streams`): at most one per
stack, never named inside a component. This module checks the others,
which attach at inner nodes: state threaded across layers such as DSA
shared top-k indices,
YOCO/CLA shared KV, feature-accumulators — declared by the stack, produced
and consumed through each component's `stream_out:` / `reads:` interface. This module gives that
declaration a ground-truth anchor:

  1. Truthfulness (hard signal): every declared stream that cites a `src`
     anchor must have that snippet occur in the forward() source recorded
     for the model (its snapshot, or a custom model's own source), and
     every declared inner stream must have both a producer and a consumer
     among the components its stack binds. A stream is a real claim about cross-layer
     dataflow — an unmatched anchor means the spec asserts a stream the
     source does not have.
     Same treatment as an op's source_anchor. With no recorded source the
     anchors cannot be checked; `cross_layer_anchor_notes` says so beside
     the forward-source warnings.

  2. Shape, when `shape` is authored: it must equal the shape the model's
     walk gives the node each layer writes the stream from
     (`SpecIRTree.stream_writes`), in every block invocation that writes
     it. The declaration resolves in that invocation's frame, so a dim
     symbol overridden by `per_layer.assign[].dims` gives the stream a
     per-variant width. Eval does not seed the live edge from this list;
     omit `shape` when the producer tensor is the only shape that should
     exist.

Producers and consumers are the components the declaring stack binds
(`ArchSpec.bound_templates`), never the spec's whole component table, so a
component no declaration of the stack binds neither produces nor consumes;
shapes are the model's walk. Every finding is promoted to an error by the
orchestrator.
"""
from __future__ import annotations

import re
from typing import List, Optional

from backend.arch.eval.dim import eval_dim_expr
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment

from ._finding import Finding
from ._shapes import _equal_dim

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _resolves(anchor: str, source: str) -> bool:
    """True if the anchor snippet occurs in source (whitespace-normalized).

    Tries the whole anchor, then the longest backtick-quoted code span inside
    it (anchors are usually `Class.forward `<code>`` prose)."""
    a = _norm(anchor)
    if not a:
        return True
    src = _norm(source)
    if a in src:
        return True
    for quoted in re.findall(r"`([^`]+)`", anchor):
        if _norm(quoted) and _norm(quoted) in src:
            return True
    return False


class _StreamShapeCheck:
    """A stream's authored `shape` against the shape the model's walk gives
    the node each layer writes it from, in the frame of the block
    invocation that writes it.

    The walk runs at B=1, S=4096 (the axiom harness's reference point, see
    `_axioms._resolve_expect`), NOT S=1: a selection-bound axis
    (`Min(n_topk_keys, s_query)`) collapses to 1 at S=1, so shapes are
    compared at steady state where the bound dominates. A model that does
    not walk is one finding carrying the walk's refusal.
    """

    def __init__(self, resolved, program: Program):
        self.resolved = resolved
        self.program = program
        spec = resolved.spec
        #: `(stack position, stream) -> declaration`, for authored shapes.
        self.shaped = {
            (spec.layer_position(owner), name): decl
            for (owner, name), decl in spec.stream_names().inner.items()
            if decl.shape}

    def run(self) -> List[dict]:
        if not self.shaped:
            return []
        try:
            analysis = Deployment(self.program).at(
                batch=1, input_tokens=4096).analyze()
        except Exception as exc:
            return [Finding(
                kind="stream_shape_unjudged", template="<model>",
                detail=(f"the model's walk raised {type(exc).__name__}: "
                        f"{exc} — declared stream shapes are compared "
                        f"against it, so none were"))]
        out: List[dict] = []
        seen = set()
        # The walked program's writes: a phase may run a component in another
        # form (`transforms.phase_form`), and the shapes are that form's.
        for (stack, _layer, name), uid in (
                analysis.program.ir.stream_writes.items()):
            decl = self.shaped.get((stack, name))
            if decl is None or uid in seen:
                continue
            seen.add(uid)
            finding = self.check_write(analysis, name, decl, uid)
            if finding is not None:
                out.append(finding)
        return out

    def check_write(self, analysis, name, decl, uid) -> Optional[dict]:
        """One writing node against the declaration, resolved in the frame
        of the block invocation it belongs to."""
        program = analysis.program
        node = program.by_uid(uid)
        block = program.block_of(uid)
        if block is None:
            # A stream's `seed`: a model-level op, in the model's frame.
            where = node.name
            env = analysis.binding.flat()
        else:
            template = program.blocks[block]
            where = template.name
            # The frame this block's nodes ran in, as the walk bound it —
            # this reader does not open the frame a second time.
            env = analysis.block_frame(template)
        try:
            want = [eval_dim_expr(str(axis), env) for axis in decl.shape]
        except Exception as exc:
            return Finding(
                kind="stream_shape_unresolved", stream=name,
                template=where,
                detail=f"{where!r}: declared shape "
                       f"{list(decl.shape)} did not resolve: "
                       f"{type(exc).__name__}: {exc}")
        shape = analysis.shapes.values.get(uid)
        actual = shape.out if shape is not None else None
        if actual is None:
            return None
        if (len(want) == len(actual)
                and all(_equal_dim(w, a)
                        for w, a in zip(want, actual, strict=True))):
            return None
        return Finding(
            kind="cross_layer_stream_shape_mismatch",
            stream=name, template=where,
            detail=(
                f"stream {name!r} declares shape "
                f"{list(decl.shape)} (resolves to {want}) "
                f"but producer node {node.display_path!r} in "
                f"{where!r} evaluates to {list(actual)} — "
                f"per-variant dims come from `per_layer.assign[].dims`; "
                f"align the declaration or the override."),
        )


def _recorded_source(model_id: str) -> Optional[str]:
    from backend.hf.forward_source import recorded_forward_source
    return recorded_forward_source(model_id)


def _anchored_streams(resolved) -> List[tuple]:
    """(stream, `src` anchor, what it is) of every stream that cites one:
    the stack's inner streams and its appended streams."""
    from backend.arch.spec_tree import iter_tree
    names = resolved.spec.stream_names()
    ir = iter_tree(resolved)
    out = [(name, decl.src, "stream")
           for (_owner, name), decl in names.inner.items() if decl.src]
    out += [(name, names.produced[name][1].src, "appended stream")
            for name in ir.streams
            if name in names.produced and names.produced[name][1].src]
    return out


def run_cross_layer_checks(model_id: str, resolved) -> List[dict]:
    """Return a list of cross-layer stream warnings."""
    warnings: List[dict] = []
    source = _recorded_source(model_id)
    spec = getattr(resolved, "spec", resolved)

    # Streams declared on the layer stack, and who writes and reads them.
    # An inner stream is its declaring stack's (`ArchSpec.stream_names`); a
    # carried stream is local to its stack and has no producer or consumer
    # to check here. An appended stream is produced by its stack's `from:`
    # declaration, not by a component, and who reads it is
    # `Program.stream_pairings`' to pair; only its anchor is checked here.
    # Which layer a reader reads is its `reads_from`, resolved where the IR
    # is built (`spec_tree.build._StreamTable`).
    #
    # Whether a stack wires a writer and a reader is a question about the
    # stack's declarations: the components it binds, in every assignment
    # (`ArchSpec.bound_templates`, followed through `reachable_graphs` as
    # the schema's carried-stream check follows them). A component no
    # declaration of that stack binds is not its writer or reader; an
    # assignment this config selects no layer of still declares one (a
    # shared spec's reader layers may be absent from one model).
    program = Program.of(resolved)
    names = spec.stream_names()
    appended = {name for name in program.ir.streams if name in names.produced}
    layers = dict(spec.layer_paths())

    # 1. stream well-formedness — a declared stream needs a writer AND a
    # reader among the components its stack binds (a stream nobody writes
    # is dangling; one nobody reads is dead), and its source anchor must
    # resolve.
    for owner, sname in names.inner:
        layer = layers[owner]
        stacks = list(layer.layers or ()) if layer.group else [layer]
        bound = [graph for _name, graph in spec.reachable_graphs(
            spec.bound_templates(stacks))]
        if not any(sname in (graph.stream_out or {}) for graph in bound):
            warnings.append(Finding(
                kind="cross_layer_stream_no_producer", stream=sname,
                detail=(f"stream {sname!r} is declared but no component "
                        f"its stack binds `stream_out`s it — dangling "
                        f"stream."),
            ))
        if not any(sname in (graph.reads or ()) for graph in bound):
            warnings.append(Finding(
                kind="cross_layer_stream_no_consumer", stream=sname,
                detail=(f"stream {sname!r} is declared but no component "
                        f"its stack binds `reads` it — dead stream."),
            ))

    if source is not None:
        warnings.extend(
            Finding(
                kind="cross_layer_src_unresolved", stream=sname,
                detail=(f"{what} {sname!r} cites a forward-source anchor not "
                        f"found in the model's recorded source: {str(src)[:100]!r}"),
            )
            for sname, src, what in _anchored_streams(resolved)
            if not _resolves(src, source))

    # 2. a component the model binds READS a stream no stack declares —
    # the consumer expects a stream nobody set up. A `stream_out` for an
    # undeclared stream is benign: the component merely CAN produce it, but
    # this model doesn't wire that stream (e.g. an all-full DSA model reuses
    # the full-indexer attention component without any shared reader).
    consumed = {stream for _name, graph in spec.reachable_graphs()
                for stream in (graph.reads or ())}
    warnings.extend(
        Finding(
            kind="cross_layer_stream_undeclared", stream=s,
            detail=(f"a component `reads` stream {s!r} but no layer group "
                    f"declares it under `streams:` — dangling consumer."),
        )
        for s in sorted(consumed)
        if s not in names.inner_names() and s not in appended
    )

    # 3. declared shape vs the walked shape of each writing node.
    warnings.extend(_StreamShapeCheck(resolved, program).run())
    return warnings


def cross_layer_anchor_notes(model_id: str, resolved) -> List[dict]:
    """The stream anchors that could not be checked: no forward source is
    recorded for this model. A note beside the forward-source warnings,
    never an error — the spec cannot supply the missing source."""
    if _recorded_source(model_id) is not None:
        return []
    anchored = sorted({name for name, _src, _what in _anchored_streams(resolved)})
    if not anchored:
        return []
    return [Finding(
        kind="cross_layer_src_unchecked",
        detail=(f"stream anchors of {', '.join(anchored)} were not checked: "
                f"no forward source is recorded for {model_id} (no snapshot "
                f"source, not a custom model)"),
    )]
