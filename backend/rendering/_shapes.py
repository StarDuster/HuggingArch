"""Analysis-report decoration and edge construction for block-graph render.

Three responsibilities:
  1. **Virtual-input shape inference** — find the resolved input shape for
     each block input by probing the first consumer's `shape_in`. Lets the
     UI label edges from blocks whose input is not `[B, S, D]` (vision
     `[B,C,H,W]`) without a per-shape special case.
  2. **Report → renderer-dict decoration** — `build_report_index` builds
     a flat `{aggregation_id: NodeReport}` index over the read-only analysis
     report; `apply_report_fields` copies its numeric and symbolic fields onto
     a renderer node. This is presentation only: inference never consumes the
     report as an evaluation model.
  3. **Edge build** — `build_edges` walks the renderer-dict tree emitting
     one edge per `(src, dst, port_idx)` triple. Reads `tensor_out` /
     `tensor_outs` written by `apply_report_fields`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.arch import dims as dims_mod
from backend.arch.eval.dim import algebra, value_of



# ---- Virtual-input shape inference ---------------------------------------


def infer_input_shape(inp_name: str, br_nodes: List[Any]
                      ) -> Optional[Tuple[List[int], Optional[List[str]]]]:
    """Find the first block-graph node that reads `inp_name` and return
    that argument's resolved shape, with its per-axis expressions (None
    where the walk bound none).

    Relies on the model analysis having threaded the upstream result
    into the block stage — the first consumer's
    `shape_in` is the real upstream tensor shape, not an inferred
    template. No fallback descent into compound children: that
    heuristic mislabels multi-input compounds where the first
    sub_graph child consumes a different alias than the wrapper's
    first input (V4-Pro hc_weighted_reduce). When the engine couldn't
    resolve the shape, return None — better than guessing wrong."""
    def _walk(nodes):
        for nr in nodes:
            yield nr
            yield from _walk(nr.children or ())

    def _arg(nr: Any, idx: int):
        si = nr.shape_in
        axes = nr.shape_in_axes or []
        arg_axes = list(axes[idx]) if idx < len(axes) else None
        if isinstance(si[0], list):
            return (list(si[idx]), arg_axes) if idx < len(si) else None
        # Single flat shape — engine attributes it to slot 0 only.
        return (list(si), arg_axes) if idx == 0 else None

    # Eval projects expanded IR roots rather than recreating the legacy
    # component wrapper. Match the actual unresolved external reference first;
    # this follows the graph identity and works for both representations.
    for nr in _walk(br_nodes):
        ir = nr.ir_node
        refs = tuple(getattr(ir, "inputs_obj", ()) or ())
        for idx, ref in enumerate(refs):
            if ref.src is not None or ref.external_sid != inp_name:
                continue
            if not nr.shape_in:
                continue
            found = _arg(nr, idx)
            if found is not None:
                return found
    return None


# ---- Presentation-report index ------------------------------------------


def build_report_index(
    reports: List[Any], index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build `{aggregation_id: NodeReport}` depth-first,
    first-write-wins).

    First-write-wins is a guard, not a live case: a wrapper and its
    no-module child (a router and its gate `linear`) carry distinct
    aggregation_ids, so each keeps its own report. Were two ever to share
    a key, the wrapper — visited first — would keep its component-level
    shape (e.g. `router_noaux_tc`'s topk-scaled `[B,S,K]`) rather than the
    linear's local `(B,S,in)→(B,S,out)`.

    An elided block-template wrapper can have `ir_node=None`; skip the wrapper
    itself but retain its projected children.

    Pass an existing dict via `index` to merge several block results
    (one model_result → many block_results) into the same flat dict.
    """
    if index is None:
        index = {}

    def _walk(report) -> None:
        ir = report.ir_node
        if ir is None:
            for child in (report.children or []):
                _walk(child)
            return
        sid = ir.aggregation_id
        if sid and sid not in index:
            index[sid] = report
        for child in (report.children or []):
            _walk(child)

    for report in reports:
        _walk(report)
    return index


# ---- Presentation report → renderer-dict decoration ----------------------


def _fmt_shape(s: Any) -> str:
    """Format a shape for display. Multi-input ops carry shape as a list
    of input shapes; join with `,` rather than `×` to avoid misreading
    as multiplication of dims. An empty/missing shape renders as ""."""
    if not s:
        return ""
    if isinstance(s[0], list):
        return ", ".join("[" + ",".join(str(x) for x in sub) + "]" for sub in s)
    return "[" + ",".join(str(x) for x in s) + "]"


def dim_glossary(binding: Any) -> Dict[str, str]:
    """Binding.this as a `{dim: int-string}` map for the fused-axis
    factorizer. Tolerates python int, Dim, and sympy Integer; silently
    drops non-integer / unconvertible entries. Empty dict when nothing
    is bound."""
    out: Dict[str, str] = {}
    items = binding.flat().items() if binding is not None and hasattr(binding, "flat") else ()
    for k, v in items:
        try:
            out[str(k)] = str(int(value_of(v)))
        except (TypeError, ValueError):  # per-item: one bad item must not abort the loop
            continue
    return out


def _fmt_shape_factored(shape: Any, axes: Any,
                        name_to_value: Optional[Dict[str, str]]) -> str:
    """Format a single-tensor numeric shape, expanding fused *feature*
    axes into their factor product (`8192` → `128·64`) so a packed
    head×head_dim axis is legible without hovering the symbolic tooltip.

    Computed at RENDER time from two things that survive the whole
    inference pipeline (fusion + sharding re-thread both, since the
    per-axis symbolic form is load-bearing for the TP-divisor pass):
      • `axes` — the per-axis symbolic form (`["B","S","Ih*In"]`), i.e.
        the node's `shape_out_axes` / `shape_outs_axes[port]`.
      • `name_to_value` — the model's `{dim: int}` glossary
        (Binding.this), used to substitute each product
        axis textually (`Ih*In` → `128*64`).

    An axis expands only when its symbolic form is a product of >=2
    dims that ALL resolve to ints AND whose product equals the numeric
    axis size. The equality guard keeps this correct under TP: a sharded
    axis (numeric = full/tp) won't match the full `128*64` product, so it
    falls back to the plain sharded int. No per-node values field is
    threaded — the substitution happens here from surviving data."""
    if not shape or isinstance(shape[0], list):
        return _fmt_shape(shape)
    ok = bool(axes) and bool(name_to_value) and len(axes) == len(shape)
    return "[" + ",".join(
        _factored_axis(x, str(axes[i]).replace(" ", "") if ok else "",
                       name_to_value)
        for i, x in enumerate(shape)) + "]"


def _factored_axis(x: Any, sym: str,
                   name_to_value: Optional[Dict[str, str]]) -> str:
    """One numeric axis, as a factor product when its symbolic form is a
    product of resolved dims equal to it (`_fmt_shape_factored`)."""
    if sym and "*" in sym and name_to_value:
        parts = re.split(r"([^A-Za-z_0-9])", sym)
        val = "".join(name_to_value.get(p, p) for p in parts)
        if re.fullmatch(r"[0-9]+(\*[0-9]+)+", val):
            factors = [int(t) for t in val.split("*")]
            prod = 1
            for f in factors:
                prod *= f
            if prod == x:
                return "·".join(str(f) for f in factors)
    return str(x)


def _is_workload_dim(name: str) -> bool:
    """A per-request workload dim — one the engine supplies (`dims.yaml`
    `engine_supplied`: batch, context and cache lengths) — as opposed to a
    dim the model fixes."""
    d = dims_mod.get(name)
    return d is not None and d.engine_supplied


def _workload_axis(sym: str, name_to_value: Optional[Dict[str, str]]
                   ) -> Optional[str]:
    """An axis expression that names a workload dim, with every other dim
    replaced by its resolved value (`B*s_ctx*n_experts_active/n_routed` →
    `B*s_ctx/32`); None when it names none."""
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", sym))
    if not any(_is_workload_dim(n) for n in names):
        return None
    expr = algebra(sym)
    fixed = {s: int(name_to_value[str(s)]) for s in expr.free_symbols
             if name_to_value and str(s) in name_to_value
             and not _is_workload_dim(str(s))}
    return _pretty_axis(str(expr.subs(fixed)))


def fmt_shape_workload(shape: Any, axes: Any,
                        name_to_value: Optional[Dict[str, str]]) -> str:
    """A shape with its workload axes left symbolic (`[B,s_ctx,4096]`): an
    axis whose expression names a workload dim shows that expression, every
    other axis its resolved size (factored as `_fmt_shape_factored` does).
    The label of a graph drawn for the model rather than for a run — the
    Inspector's — so no batch or length is ever printed as a number. Falls
    back to the numeric label where the walk bound no axis expressions."""
    if not shape:
        return ""
    if isinstance(shape[0], list):
        axes_list = axes if axes and len(axes) == len(shape) else [None] * len(shape)
        return ", ".join(fmt_shape_workload(sub, a, name_to_value)
                         for sub, a in zip(shape, axes_list, strict=True))
    if axes and len(axes) == 1 and isinstance(axes[0], (list, tuple)):
        axes = axes[0]              # a single input's axes, as one list
    ok = bool(axes) and len(axes) == len(shape)
    toks: List[str] = []
    for i, x in enumerate(shape):
        sym = _pretty_axis(axes[i]) if ok else ""
        workload = _workload_axis(sym, name_to_value) if sym else None
        toks.append(workload if workload is not None
                    else _factored_axis(x, sym, name_to_value))
    return "[" + ",".join(toks) + "]"


def _pretty_axis(axis: str) -> str:
    """Tidy a sympy.sstr axis like `N*D` for the tooltip: just strip
    whitespace. Multiplication keeps `*` so the rendered axis label
    can be copied back into Python / sympy verbatim."""
    return str(axis).replace(" ", "")


def _fmt_axes(axes_lists) -> str:
    """Format the per-input axes list (list-of-lists) for the tooltip.
    Mirrors `_fmt_shape`'s comma-join convention so multi-input ops
    stay readable: `[B,S,H], [B,S,N·D]`."""
    if not axes_lists:
        return ""
    if isinstance(axes_lists[0], (list, tuple)):
        return ", ".join("[" + ",".join(_pretty_axis(a) for a in sub) + "]" for sub in axes_lists)
    return "[" + ",".join(_pretty_axis(a) for a in axes_lists) + "]"


def apply_report_fields(
    n: dict,
    report: Any,
    name_to_value: Optional[Dict[str, str]] = None,
    workload_symbolic: bool = False,
) -> None:
    """Decorate renderer node `n` from a read-only `NodeReport`.

    Independent emission per field — zero-input ops
    (parameter_vector / _tensor) still get a `tensor_out` even when
    `shape_in` is empty; symbolic / values rows are emitted only when
    analysis supplied them. Stashes raw shapes as `_si` / `_so` for
    the downstream memory pass.

    `name_to_value` is the model's `{dim: int}` glossary
    (Binding.this via `dim_glossary`); when supplied, fused feature axes
    in the numeric shape labels are expanded to their factor product
    (`8192` → `128·64`). None → plain folded ints (all callers safe).

    `workload_symbolic` labels the shapes with their workload axes left
    symbolic (`fmt_shape_workload`) — the form of a graph drawn for the
    model, not for one batch and length."""
    si = report.shape_in
    so = report.shape_out
    so_axes = report.shape_out_axes
    if workload_symbolic:
        if si:
            n["tensor_in"] = fmt_shape_workload(
                si, report.shape_in_axes, name_to_value)
        if so:
            n["tensor_out"] = fmt_shape_workload(so, so_axes, name_to_value)
    else:
        if si:
            n["tensor_in"] = _fmt_shape(si)
        if so:
            # Expand fused feature axes (`8192` → `128·64`) so a packed
            # head×head_dim axis reads structurally, not as an opaque product.
            n["tensor_out"] = _fmt_shape_factored(so, so_axes, name_to_value)

    # Per-port shapes for multi-output producers (hc_pre, hc_split_sinkhorn).
    # Used by `build_edges` so that an edge sourced from a producer's
    # secondary port gets THAT port's shape label, not the main `tensor_out`.
    shape_outs = report.shape_outs or {}
    if shape_outs:
        outs_axes = report.shape_outs_axes or {}
        fmt = fmt_shape_workload if workload_symbolic else _fmt_shape_factored
        n["tensor_outs"] = {
            port: fmt(s, outs_axes.get(port), name_to_value)
            for port, s in shape_outs.items() if s
        }

    # Payload boundary: a dispatch-zone leaf's exact per-copy flops is a
    # Fraction; a reader of the graph wants a number, and JSON wants one too.
    n["flops"] = (
        report.flops if isinstance(report.flops, int) else round(report.flops)
    )
    n["params"] = report.params
    # Independent stored-copy bytes (checkpoint ledger), not the
    # `weight_bytes` compute_memory later derives from children.
    n["storage_bytes"] = int(getattr(report, "storage_bytes", 0) or 0)
    n["_si"] = si  # temp, used for access_bytes / compute_memory
    n["_so"] = so

    # Symbolic / substituted formula sides are independent fields (no
    # shared `=` sentinel). Frontend renders both rows when present.
    if report.flops_symbolic:
        n["flops_symbolic"] = report.flops_symbolic
        # The same law in the block's names (`NodeReport.flops_law`): what a
        # sum over several nodes reads (`views.abstraction`'s L1 boxes).
        n["flops_law"] = str(report.flops_law)
    if report.flops_values:
        n["flops_values"] = report.flops_values
    if report.params_symbolic:
        n["params_symbolic"] = report.params_symbolic
    if report.params_values:
        n["params_values"] = report.params_values

    # Per-axis symbolic forms for the tooltip. The numeric tensor_in /
    # tensor_out fields above carry the substituted shape ([1,2048,8192]);
    # these carry the underlying dim expressions ([B,S,N·D]) so the user
    # can see *which* dims composed each axis without having to back-
    # derive from the raw numbers.
    si_axes = report.shape_in_axes
    so_axes = report.shape_out_axes
    if si_axes:
        n["tensor_in_symbolic"] = _fmt_axes(si_axes)
    if so_axes:
        n["tensor_out_symbolic"] = "[" + ",".join(_pretty_axis(a) for a in so_axes) + "]"


# ---- Edge construction ---------------------------------------------------


def _port_label(dst_node: dict, input_idx: int, n_inputs: int) -> str:
    """The i-th input's role label, from what `_node_build` projected off
    the leaf's declaration (`port_labels`): a list is indexed by input
    position; a str is a flat template's one role, shown only when the node
    keeps more than one visible input (`n_inputs`, counted by the edge
    builder after absorption), where a lone input's role would be redundant."""
    labels = dst_node.get("port_labels")
    if not labels:
        return ""
    if isinstance(labels, str):
        return labels if n_inputs > 1 else ""
    return labels[input_idx] if input_idx < len(labels) else ""


def build_edges(nodes_list: List[dict]) -> List[dict]:
    """Walk the UI tree, emitting one edge per (src, dst, input_idx)
    triple. Recurses into components, attaching `child_edges` on each
    parent.

    Every edge is drawn in ONE scope: the reader's. What stands for the
    producer there is the producer itself when they are siblings, and
    otherwise the region of the reader's scope the producer sits in —
    `_anchor` below decides which.

    A dependency on an OUTER-scope node — where a child references an
    outer node via `resolved_inputs` (e.g. `hc_attn_pre.flat.inputs=['x']`
    pointing at the block-scope `x`) — has no such region, and is NOT
    emitted here. It is already covered by:
      • the wrapper's own `inputs:` declaration (root-scope edge
        `x → hc_attn_pre` exists at the parent scope)
      • the frontend's recursive `expandDst(wrapper, src)` fan-out,
        which routes the wrapper-level edge to the SPECIFIC inner
        consumer when the wrapper is expanded — recursing into nested
        expanded compounds as needed.

    Emitting such an edge directly was wrong both ways:
      • placed at dst's parent compound → src is OUT of that compound's
        scope; ELK silently drops the edge (the "mul has only one
        input" bug).
      • placed at LCA (root for a 2-level cross edge) → dst is INSIDE
        a collapsed compound, so dst-id doesn't exist in ELK's flat
        node list at root level; ELK errors / drops the edge → empty
        layout.
    Skipping it defers routing to the frontend's `expandDst` recursion,
    which handles BOTH expansion states correctly per scope."""
    global_by_id: Dict[str, dict] = {}
    # spec_id → list of ancestor wrapper ids (innermost-last). Which
    # node of the reader's scope an edge leaves from is read off these.
    ancestors: Dict[str, List[str]] = {}

    def _index(nodes: List[dict], chain: List[str]) -> None:
        for n in nodes:
            global_by_id[n["id"]] = n
            ancestors[n["id"]] = list(chain)
            if n.get("children"):
                _index(n["children"], [*chain, n["id"]])

    _index(nodes_list, [])

    def _resolve_input(inp: str, reader: str) -> str:
        """The producer id a bare parent ref names (port info rides in the
        parallel `input_ports[]` set by `_node_build`). Exact, graph-wide:
        the builder settled every ref to an id before edges are drawn."""
        if inp not in global_by_id:
            raise ValueError(
                f"{reader}: input {inp!r} names no node of the graph")
        return inp

    def _anchor(producer: str, reader: str) -> Optional[str]:
        """The node of the READER's scope this edge leaves from, or None
        when the reader's scope holds nothing that stands for the producer.

        A sibling stands for itself. A producer INSIDE a sibling compound
        is drawn from that compound: a component read by its own name
        (`permute: { in: [x, gate] }` over `moe_ffn`'s router) is settled
        by the IR onto the leaf that actually produces the tensor, which
        lives a scope deeper — and the compound's own `inputs:` says what
        it READS, never who reads it, so nothing else would draw this
        arrow. The region is the anchor, exactly as an exit exchange
        anchors a region rather than one leaf
        (`layout/materializer.consumer_of`); the frontend's `expandSrc`
        routes it on to the inner producer once the compound is opened.

        A producer in an OUTER scope (or under a different parent
        altogether) has no region here: the reader's own wrapper carries
        that edge at the scope where both ends exist."""
        chain, scope = ancestors.get(producer, []), ancestors.get(reader, [])
        if chain == scope:
            return producer
        if len(chain) > len(scope) and chain[:len(scope)] == scope:
            return chain[len(scope)]
        return None

    # `edges_at[parent_id]` = edges to attach to parent's child_edges.
    # `parent_id is None` = root-level edges.
    edges_at: Dict[Optional[str], List[dict]] = {None: []}

    def _emit_edge(src: str, dst: str, idx: int, n_inputs: int,
                   op: str, src_port: str, producer: str) -> None:
        # The edge leaves `src` — the anchor in the reader's scope — but
        # the tensor on it is `producer`'s, so the label, the port and the
        # edge's kind are read off the producer even when a compound
        # stands for it.
        src_node = global_by_id[producer]
        dst_node = global_by_id[dst]
        # When the source produces this edge through a NAMED secondary
        # port (`src_port != ""`), prefer that port's shape over the
        # main `tensor_out`. The shape label on a `hc_pre.post` edge
        # should be `[B,S,Hc]` (post's shape), not `[B,S,H]` (the main
        # `reduced` output). Falls back to main shape when the port
        # didn't get its own shape recorded (custom_ops without
        # secondary port templates, etc.).
        edge_shape = src_node.get("tensor_out")
        if src_port:
            outs = src_node.get("tensor_outs") or {}
            edge_shape = outs.get(src_port, edge_shape)
        # Runtime render invariant: an edge the backend draws but can't
        # label is a graph the frontend can't faithfully express. The
        # engine shapes every producer (unknown ops now raise; secondary
        # ports thread through `_resolve_ref_nr`; synthetic block_stack
        # wrappers inherit the residual-stream shape) — so a `None` here
        # is a genuine drop, not a benign gap. Fail loud at the exact
        # (src, dst, port) instead of shipping a shapeless arrow. Verified
        # zero across all shipped models × {prefill, decode} × {TP, EP};
        # if this ever fires, the producer needs a shape (register the op
        # / add its secondary port template / give the wrapper a tensor_out),
        # not a silent fallback.
        if edge_shape is None:
            port_hint = f".{src_port}" if src_port else ""
            raise ValueError(
                f"Render invariant violated: edge {producer}{port_hint} -> {dst} "
                f"(op {op!r}) has no shape — the producer carries no "
                f"{'port ' + src_port if src_port else 'output'} shape, so "
                f"the frontend would draw a shapeless arrow. The backend "
                f"compute graph must be fully renderable: shape the "
                f"producer rather than emit an unlabelable edge."
            )
        edge = {
            "id": f"{src}->{dst}:{idx}",
            "src": src,
            "dst": dst,
            "tensor": "",
            "shape": edge_shape,
            "port_label": _port_label(dst_node, idx, n_inputs),
        }
        # Edge semantics (7.1): a non-activation producer (a selection /
        # state / cache) marks the edge so the frontend can draw it
        # differently — shape alone can't tell an index list from data.
        # `output_kind` sits on the dict only when it is not "activation"
        # (projected from the producer's declaration in _node_build, or
        # stamped by `kv_marks` on cache-resident data). A fused kernel
        # publishing several tensors kinds them per port (`port_kinds`).
        port_kinds = src_node.get("port_kinds") or {}
        kind = (port_kinds[src_port] if src_port in port_kinds
                else src_node.get("output_kind"))
        if kind:
            edge["kind"] = kind
        if src_port:
            edge["src_port"] = src_port
        if src != producer:
            # An anchored edge keeps the producer: `via` names the member of
            # the compound the arrow leaves from, so opening it steers the
            # arrow to that member instead of fanning from every exit (the
            # frontend's `expandSrc`; the L1 view carries the same field for
            # the same reason). One level deep, which is the member ELK has
            # in the scope it lays out when the compound opens — a producer
            # further in is reached by that member's own exits.
            inside = ancestors[producer][len(ancestors[src]) + 1:]
            edge["via"] = inside[0] if inside else producer
        chain = ancestors.get(dst, [])
        scope = chain[-1] if chain else None
        edges_at.setdefault(scope, []).append(edge)

    def _walk(nodes: List[dict]) -> None:
        for n in nodes:
            inputs = n.get("inputs", []) or []
            input_ports = n.get("input_ports") or [""] * len(inputs)
            for idx, inp in enumerate(inputs):
                # Structurally split parent from port using the parallel
                # `input_ports[]`. When port is "" we use `inp` as-is
                # (bare parent); when port is set, strip the suffix to
                # get the bare parent id to look up.
                src_port = input_ports[idx] if idx < len(input_ports) else ""
                parent_ref = inp[: -(len(src_port) + 1)] if src_port and inp.endswith("." + src_port) else inp
                producer = _resolve_input(parent_ref, n["id"])
                # Drawn from whatever the reader's own scope holds for
                # this producer; an outer-scope ref has nothing there and
                # rides the wrapper's input edge + frontend `expandDst`.
                src = _anchor(producer, n["id"])
                if src is None:
                    continue
                _emit_edge(src, n["id"], idx, len(inputs),
                           n.get("op", ""), src_port, producer)
            if n.get("children"):
                _walk(n["children"])

    _walk(nodes_list)

    # Attach `child_edges` to each compound that owns scope edges.
    for scope, scope_edges in edges_at.items():
        if scope is None:
            continue
        parent = global_by_id.get(scope)
        if parent is not None:
            parent["child_edges"] = scope_edges

    return edges_at[None]
