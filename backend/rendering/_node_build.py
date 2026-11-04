"""UI-tree construction for the block-graph render.

Consumes `spec_tree.iter_tree(resolved)` and decorates it with UI
fields (style / label / detail / formula / badge / module absorption).

The spec walking, id derivation, `<param>` substitution,
rename application, role inheritance, and dim resolution all happen
ONCE in spec_tree — `_node_build` no longer walks the spec itself.
What's left here is "given a SpecIRNode, what does the renderer want
to display": style classes, op-formula tooltips, custom-op flags,
`Top-K/E` MoE badges, and module absorption (drop a moduleless child
from the wrapper's children list and graft in its grand-children
+ inherit matched-tensor metadata).

Output shape unchanged from before (the consumer in builders.py /
_shapes / _memory / frontend SVG renderer all see the same dict
structure).

Design constraints (do NOT change without reading DESIGN.md):
  • `module: null` — a node the checkpoint has no module for. It contributes
    no path segment to the leaves below it, and here it also drops out of
    the wrapper's visible children with its grand-children grafted in, so
    the UI tree has a node exactly where the checkpoint has one. See
    `schema.GraphNode.module`.
  • Alias-adapter inline-flatten happens inside iter_tree — a template
    whose whole body is one delegating node (qwen35_*_block,
    chatglm_block) contributes no namespace, so iter_tree descends into
    the sub-graph and _node_build sees the inlined body directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.arch import roles
from backend.arch.eval.program import BlockTemplate, Program
from backend.arch.layout.axis_use import custom_axis_use
from backend.arch.ops import (
    OpSpec, ShapeKind, build_custom_op_spec, hosts_views, is_out_in_matrix,
)
from backend.arch.schema.custom_ops import CustomOpSpec
from backend.arch.spec_tree import iter_tree
from backend.inference.fusion_plans import FUSION_SCHEMES
from ._shapes import apply_report_fields, dim_glossary


def _fmt(v: Any) -> str:
    """Format a dim value for compact display: comma-thousands for big
    ints, str otherwise."""
    if isinstance(v, int) and v > 999:
        return f"{v:,}"
    return str(v)


# Detail-string keys we suppress for the generic dim-list fallback —
# these are universal axes (B, S, in/out variants, bias flag) that
# don't add information to a per-node tooltip.
_DETAIL_SKIP_DIM_KEYS = frozenset({
    "B", "S", "in", "out", "in_a", "in_b", "bias",
})


def declaration_card(declared: Any, op_name: str = "") -> Optional[dict]:
    """The wire-safe identity/signature of a leaf's op declaration.

    The leaf already carries its declaration (`SpecIRNode.declaration`):
    a registered `OpSpec`, or the spec's `CustomOpSpec`, resolved when the
    tree was built. The graph diff compares declarations after the graph has
    crossed the API boundary, so the builder states the comparable card here,
    once, instead of the diff looking the display name up in a registry.
    """
    if declared is None:
        return None
    custom = isinstance(declared, CustomOpSpec)
    spec = (build_custom_op_spec(op_name, declared) if custom
            else declared)
    identity = (declared.model_dump_json(exclude_none=True) if custom
                else f"op:{op_name}")
    axis = custom_axis_use(declared if custom else spec)
    sigma = (
        ShapeKind(spec.shape_kind).value, spec.output_kind,
        tuple(spec.input_kinds), tuple(spec.input_roles), spec.has_weight(),
        spec.is_view_op, spec.carries_recurrent_state,
        spec.attention_kernel, spec.layout_family == "geometry",
        tuple(sorted((u.category, u.param, u.plan) for u in axis.values())),
    )
    return {
        "identity": identity,
        "sigma": sigma,
        "roles": tuple(spec.input_roles),
        "view": bool(spec.is_view_op),
    }


class NodeBuilder:
    """IR consumer: builds the renderer's `nodes_data` dict tree from
    `spec_tree.iter_tree(resolved)`.

    Public API:
      • `build_graph(template_name)` — top-level entry; returns the
        list of UI node dicts for the block template's IR roots.
    """

    def __init__(self, resolved_spec,
                 style_by_op: Dict[str, str],
                 style_by_role: Dict[str, str],
                 sharded_weight_dims: Optional[Dict[int, Dict[str, int]]] = None,
                 report_index: Optional[Dict[str, Any]] = None,
                 workload_symbolic: bool = False):
        self.resolved_spec = resolved_spec
        # Shape labels with the workload axes left symbolic
        # (`_shapes.apply_report_fields`).
        self.workload_symbolic = workload_symbolic
        self.style_by_op = style_by_op
        self.style_by_role = style_by_role
        # Components AND inline blocks — used to resolve a component
        # invocation's role for style classification (the iter_tree
        # node already carries its own role; this is for the wrapper
        # style-by-role lookup, which uses the COMPONENT's declared
        # role, not the propagated role from above).
        self.components = {
            **(resolved_spec.spec.components or {}),
            **(resolved_spec.spec.blocks or {}),
        }
        self.comp_roles = resolved_spec.spec.component_roles
        self.custom_ops = resolved_spec.spec.custom_ops or {}
        # Model `{dim: int}` glossary (Binding.this) as a str→str map, so
        # `apply_report_fields` can expand a fused feature axis
        # (`Ih*In` → `128·64`) in the numeric shape labels. Derived once
        # here from surviving per-axis symbolic forms + these values; no
        # per-node display field is threaded through the pipeline.
        self.name_to_value = dim_glossary(Program.of(resolved_spec).base_binding)
        # uid → {"in": int, "out": int} from the walk's per-rank shapes.
        # Drives the per-rank `W:in→out` detail on linear leaves. Empty
        # when not provided (no parallelism / Inspector view).
        self.sharded_weight_dims = sharded_weight_dims or {}
        # `{aggregation_id: NodeReport}` — when provided, every node is
        # decorated with matching shape/cost/formula presentation fields.
        # Inspector builds this from the read-only analysis report. The
        # whole-model
        # post-tree path (`serialize_post_tree`) leaves it None and
        # uses its own pre+post+sharded overlay pipeline instead.
        self.report_index = report_index or {}

    # ---- public --------------------------------------------------------

    def build_graph(self, template_name: str) -> List[dict]:
        """Build the UI dict list for one block template's IR roots.

        Pulls `(BlockMeta, root_nodes)` from `iter_tree(resolved).blocks`
        and walks each root SpecIRNode, decorating it with UI fields.
        Module absorption (drop a moduleless child, graft grand-children,
        inherit metadata) applied at every component invocation.

        Parameter-input absorption (fold lone-consumer parameter_vector
        / parameter_tensor into their consumer's `absorbed_params`) is
        applied OUTSIDE this method, by `apply_param_absorption` —
        order matters: dtype + memory pipelines run on the full sibling
        list first so absorbed params still carry weight_dtype / bytes,
        then edge build happens AFTER absorption so spurious edges to
        the folded params are suppressed.
        """
        tree = iter_tree(self.resolved_spec)
        for meta, roots in tree.blocks:
            if meta.template_name == template_name:
                return self.build_roots(BlockTemplate(meta, tuple(roots)).shown)
        return []

    def build_top_level(self) -> List[dict]:
        """Top-level UI nodes (embed_tokens / model.norm / lm_head /
        group sub-layers). Currently the rendering pipeline doesn't
        consume this directly — top-level rendering lives elsewhere —
        but it's exposed for symmetry with `build_graph`."""
        tree = iter_tree(self.resolved_spec)
        return self.build_roots(tree.top_level)

    def build_roots(self, roots) -> List[dict]:
        """Build UI dicts for an explicit list of IR roots.

        `build_graph` / `build_top_level` walk `iter_tree(resolved)` —
        the raw spec tree. This entry instead takes roots a caller
        already holds, so the inference-page "final compute graph" can
        feed the POST-PASS tree (kernel fusion + parallelism comm
        injection applied) through the exact same node-building +
        styling path the Inspector uses. Only the source tree differs;
        the UI dict format is identical (id / op / label / style /
        inputs / children / expandable / …)."""
        return [self._build_node_from_ir(ir) for ir in roots]

    # ---- internals -----------------------------------------------------

    def _build_node_from_ir(self, ir) -> dict:
        """SpecIRNode → renderer UI dict. Recurses into ir.children
        for component invocations; applies module absorption."""
        op_name = ir.op
        is_comp = ir.is_component

        # Style: components use their own declared role, primitives
        # use op→style table.
        if is_comp:
            comp_role = self.comp_roles.get(op_name, "")
            style = next((self.style_by_role[r]
                          for r in roles.lineage(comp_role)
                          if r in self.style_by_role), "op")
        else:
            style = self.style_by_op.get(op_name, "op")

        # Use the IR's resolved input refs — structured (spec_id, port)
        # tuples with wrapper-input aliases walked up to their caller-
        # side source (block-scope `hc_attn_fn` for an inner `fn` ref,
        # etc.). Emit two parallel fields:
        #   inputs       — list of fully-qualified strings, `<spec_id>`
        #                   for main outputs and `<spec_id>.<port>` for
        #                   secondary ports. Kept as the legacy API the
        #                   absorber + frontend `inputs[]` traversal +
        #                   diagrams.py read.
        #   input_ports  — parallel list of port strings ("" for main
        #                   output). Frontend `expandDst` consults this
        #                   to route cross-scope edges to the SPECIFIC
        #                   inner consumer when the producer has
        #                   multiple secondary outputs (e.g. `hc_pre`'s
        #                   `.post` vs `.comb` feeding hc_combine's
        #                   `post_exp` vs `comb_exp`).
        if ir.inputs_obj:
            # src=None entries are unresolved refs — typically a block-input
            # alias (e.g. `pre_norm` block's `inputs: [x]` referenced by
            # `input_layernorm: { in: x }`) that has no in-scope producer.
            # Emit `external_sid` so build_edges can match against the
            # virtual input pseudo-nodes prepended by builders.py for each
            # block.inputs entry. Mirrors `resolved_inputs` @property.
            node_inputs = []
            input_ports = []
            for ref in ir.inputs_obj:
                base = ref.src.aggregation_id if ref.src is not None else ref.external_sid
                if not base:
                    continue
                node_inputs.append(f"{base}.{ref.port}" if ref.port else base)
                input_ports.append(ref.port)
        else:
            node_inputs = list(ir.inputs)
            input_ports = [""] * len(node_inputs)
        n: dict = {
            "id": ir.aggregation_id,  # wire id — `{template}|{display_path}`
            # for block-internal nodes, bare display_path at model root.
            # Single keyspace from SpecIRNode through renderer-dict to
            # frontend ELK input — no boundary translation.
            "uid": ir.uid,
            "op": op_name,
            "label": ir.name,
            "role": ir.role,
            "tensor_name": ir.name,
            "style": style,
            "inputs": node_inputs,
            "input_ports": input_ports,
            "declaration": declaration_card(
                None if is_comp else ir.declaration, op_name),
        }

        # Op formula / source / custom-op flags — from the declaration the
        # leaf carries (`SpecIRNode.declaration`), never a registry lookup.
        if not is_comp:
            op_spec = ir.declaration
            # A call-site `formula:` caption states the local shape
            # semantics (which axis a table broadcasts on, what gets
            # reshaped first); the registry's formula is the op's generic
            # contract, written to cover every use at once.
            local_formula = getattr(ir, "local_formula", None)
            if local_formula or (op_spec and op_spec.formula):
                n["op_formula"] = local_formula or op_spec.formula
            if op_spec and op_spec.description:
                n["op_description"] = op_spec.description
            if isinstance(op_spec, OpSpec):
                # The parts its operands play (q / k / v / probs …): what
                # makes the node an anchor of the structure view, which
                # reads it here.
                if op_spec.input_roles:
                    n["input_roles"] = list(op_spec.input_roles)
                # Edge semantics and absorption read these off the dict
                # node later (the renderer-dict is what _shapes and the
                # absorption pass see), so the declaration's answers are
                # projected here, once.
                if op_spec.output_kind != "activation":
                    n["output_kind"] = op_spec.output_kind
                # A collective draws as one, whatever the op table says:
                # the declaration is the fact, and it overrides the style
                # set from `style_by_op` above.
                if op_spec.is_comm_op:
                    n["style"] = "comm"
                # Fold candidates are nodes that only SOURCE a weight (no
                # inputs, not a buffer) — the same predicate eval.traffic uses
                # to bill such a read to its consumer. Not
                # `is_bare_parameter()`: that is a checkpoint-naming property,
                # and `linear_param` is bare-named yet computes `x @ W`;
                # folding it dropped its FLOPs from the rendered graph.
                if op_spec.is_identity_source():
                    n["identity_source"] = True
                labels = _port_labels(op_spec)
                if labels:
                    n["port_labels"] = labels
            # Whether this kernel runs an adjacent view in its own
            # prologue/epilogue (the compute graph's view absorption reads
            # it): a fused kernel answers for the ops it absorbed.
            scheme = FUSION_SCHEMES.get(op_name)
            if hosts_views(scheme.naive_pattern.ops() if scheme is not None
                           else (op_name,)):
                n["hosts_views"] = True
            if op_spec and op_spec.shape_kind == ShapeKind.SOURCE:
                # Zero-input ops (parameter_vector, rotary buffers, …)
                # are DAG sources; renderer mustn't draw spurious
                # incoming edges from outside the compound.
                n["is_source"] = True
            if isinstance(op_spec, CustomOpSpec):
                # Its formula / description were read above with every
                # declaration's; only the marker is custom-specific.
                n["custom_op"] = True
        else:
            # Compound ops have no hand-written formula (it would drift from
            # the graph) — the derived form is composed from the inner DAG,
            # so it is the graph restated and cannot lie (7.4).
            comp = self.components.get(op_name)
            if comp is not None:
                try:
                    derived = comp.derived_formula()
                except Exception:
                    derived = ""
                if derived:
                    n["op_formula_derived"] = derived

        if is_comp:
            n["subtitle"] = op_name

        # A sibling whose stored tensor it reads (siblings at one `module:`
        # address) is drawn as an input, as a parameter it is handed is.
        for sibling in (ir.parent.children if ir.parent else ()):
            if ir.tied_to_uid == sibling.uid:
                n["inputs"].append(sibling.aggregation_id)
                n["input_ports"].append("")

        # Detail string: linear shows "W:in→out", norm/etc. shows
        # "dim=N", activations show fn name, others fall back to
        # listing non-trivial dim k=v pairs.
        if is_out_in_matrix(ir.declaration):
            # Per-rank (sharded) dims override the spec IR's global dims
            # for linears that participate in TP — `W:1,536→3,072` instead
            # of the unsharded `1,536→24,576`. Falls through to the global
            # dims when no sharded entry exists (Inspector view / TP=1).
            sharded = self.sharded_weight_dims.get(ir.uid)
            global_in = ir.dims.get("in")
            global_out = ir.dims.get("out")
            if sharded and "in" in sharded and "out" in sharded:
                in_v, out_v = sharded["in"], sharded["out"]
            else:
                in_v = global_in
                out_v = global_out
            if in_v and out_v:
                n["detail"] = f"W:{_fmt(in_v)}→{_fmt(out_v)}"
                # When sharded differs from global, surface the unsharded
                # `W:in→out` as a `detail_full` companion so the frontend
                # can render an "originally [...]" hint on hover. Skip
                # when there's no divergence (identity-mode / non-TP).
                if (global_in and global_out
                        and (global_in != in_v or global_out != out_v)):
                    n["detail_full"] = f"W:{_fmt(global_in)}→{_fmt(global_out)}"
        elif ir.dim is not None and ir.dim != "":
            n["detail"] = f"dim={_fmt(ir.dim)}"
        elif ir.dims:
            bits: List[str] = []
            for k, v in ir.dims.items():
                if k in _DETAIL_SKIP_DIM_KEYS:
                    continue
                if isinstance(v, int):
                    bits.append(f"{k}={_fmt(v)}")
            if bits:
                n["detail"] = " ".join(bits)

        # Bias annotation (linear with bias dim non-zero).
        if is_out_in_matrix(ir.declaration) and ir.has_bias:
            n["has_bias"] = True

        # MoE repeat/active badge.
        if ir.repeat is not None:
            n["repeat"] = ir.repeat
            active_val = ir.active if ir.active is not None else ir.repeat
            n["active"] = active_val
            if active_val != ir.repeat:
                n["badge"] = f"Top-{active_val}/{ir.repeat}"
            elif ir.repeat and ir.repeat > 1:
                n["badge"] = f"×{ir.repeat}"

        # Component recursion: build children, then apply module
        # absorption. Children's `inputs` are already fully-qualified
        # spec_ids (resolved at IR build) — wrapper-input aliases walk
        # up to caller-side sources, sibling refs gain the scope prefix.
        # No virtual input pseudo-nodes needed: cross-scope edges are
        # represented natively by `<inner>.inputs` containing an outer
        # node's spec_id, which the frontend's edge layer rewires.
        if is_comp:
            n["expandable"] = True
            sub_children = [self._build_node_from_ir(c) for c in ir.children]

            # A SPEC child the checkpoint has no module for is not a level of
            # the tree. Restricted to `provenance == "spec"` because
            # `module_name is None` answers "is there a module here", and a collective or a fused leaf answers no to that
            # while still being a step the reader needs to see. One field, one
            # question — conflating those two is what `primary` did.
            # The wrapper UI node inherits its matched_tensor
            # metadata so HF-side dtype lookup lands on the linear's
            # weight; the child itself is dropped from the visible
            # children list, with its grand-children grafted into the
            # wrapper so the underlying Module's leaves stay visible.
            #
            # Only a child with children of its own is a LEVEL to collapse —
            # its steps are grafted up, nothing is lost (ChatGLM's adapters
            # wrap the real attention / FFN this way). A LEAF with no module
            # is a step, not a level: the router's `linear` (`module: null`,
            # its weight IS `mlp.gate.weight`) is the gate's matmul, and
            # dropping it hid that matmul — and its FLOPs — from the expanded
            # router while the router's total still counted it. Whether the
            # checkpoint has a module for a node says how weights are named,
            # not what the graph computes.
            absorbed_idx = None
            for i, ir_child in enumerate(ir.children):
                if (ir_child.module_name is None
                        and ir_child.provenance == "spec"
                        and ir_child.children):
                    absorbed_idx = i
                    break
            if absorbed_idx is not None:
                absorbed = sub_children.pop(absorbed_idx)
                if absorbed.get("children"):
                    sub_children.extend(absorbed["children"])
                for k in (
                    "weight_dtype", "_matched_tensor", "weight_dtype_bytes",
                ):
                    if k in absorbed:
                        n[k] = absorbed[k]
                # The popped node no longer exists as a child, but sibling
                # children may still reference it (e.g. a router's
                # `softmax_scores` reads the `linear`). The wrapper ABSORBED
                # it — the wrapper now carries its output/tensor — so
                # redirect those refs to the wrapper. Without this the
                # sibling's input dangles and its edge renders missing
                # (router `linear → softmax` broke on Qwen-MoE /
                # Mixtral-style `router_softmax_topk`, where the gate
                # wrapper absorbs the linear). A leaf maps straight to the
                # wrapper; one with grand-children is represented by the
                # wrapper too, so the same redirect holds.
                _redirect_refs(sub_children, absorbed["id"], n["id"])
            n["children"] = sub_children

        # Report decoration, keyed by the SpecIRNode's aggregation_id. A
        # wrapper and its no-module child have distinct ids (`mlp.gate` vs
        # `mlp.gate.linear`), so each gets its own report;
        # `build_report_index` is first-write-wins only as a guard should two
        # ever share a key (the wrapper, visited first, would keep its
        # component-level shape). Lookup miss → no splat
        # (orphan SpecIRNode without matching NR, or the post-tree path
        # that uses a different shape pipeline).
        report = (
            self.report_index.get(ir.aggregation_id)
            if self.report_index else None
        )
        if report is not None:
            apply_report_fields(n, report, self.name_to_value,
                                self.workload_symbolic)

        return n


# ---------------------------------------------------------------------------
# Parameter-input absorption
# ---------------------------------------------------------------------------

def strip_transport_keys(nodes: List[Dict[str, Any]]) -> None:
    """End the ride of the declaration's answers that LATER passes read off
    the renderer dict: `_shapes.build_edges` reads `port_labels`,
    `output_kind` and `port_kinds`, the absorption passes read
    `identity_source` and `hosts_views`. They travel on
    the node dict through the rewrites the compute graph applies (namespacing,
    fused expansion), which a side table keyed by node id would not survive —
    and they are transport, not wire format. Each graph builder calls this
    once its edges are built, so the API payloads carry none of them.
    """
    for n in nodes:
        n.pop("port_labels", None)
        n.pop("output_kind", None)
        n.pop("port_kinds", None)
        n.pop("identity_source", None)
        n.pop("hosts_views", None)
        if n.get("children"):
            strip_transport_keys(n["children"])


def _port_labels(op_spec: OpSpec):
    """What the declaration says about the roles of `op_spec`'s inputs, in
    the form `_shapes._port_label` decides with at edge time: the per-input
    template's last dim name for a multi-template `shape_in` (a list indexed
    by input position — `attention_score`'s two operands), or the one
    name for a flat template (a str). A flat template's name labels an edge
    only when the node keeps more than one visible input, and that count is
    the edge builder's to take — after absorption has folded parameter
    inputs away — so it is not decided here. None for compound ops (no
    shape_in). B and S are the engine-supplied batch/sequence; the last dim
    is the role."""
    si = op_spec.shape_in
    if not si:
        return None
    if isinstance(si[0], (list, tuple)):
        return [tmpl[-1] if tmpl else "" for tmpl in si]
    return si[-1]


def _redirect_refs(nodes: List[dict], old_id: str, new_id: str) -> None:
    """Rewrite every `inputs[i] == old_id` to `new_id` across `nodes` and
    their descendants, in place. Used after module absorption drops a
    node whose id was folded into its wrapper: siblings that consumed the
    dropped node must now consume the wrapper. `input_ports` ride the
    parallel array and are unchanged (the absorbed node's main output →
    the wrapper's main output)."""
    for n in nodes:
        ins = n.get("inputs")
        if ins:
            n["inputs"] = [new_id if r == old_id else r for r in ins]
        if n.get("children"):
            _redirect_refs(n["children"], old_id, new_id)


def apply_param_absorption(nodes: List[dict]) -> List[dict]:
    """Recursively apply parameter-input absorption over every sibling
    scope in a UI tree, returning the trimmed top-level list.

    Called by `build_block_graph_data` after dtype + memory attach so
    each `absorbed_params` entry carries weight_dtype / weight bytes /
    matched_tensor, but before `build_edges` so spurious edges to the
    folded params don't render.
    """
    for n in nodes:
        if n.get("children"):
            n["children"] = apply_param_absorption(n["children"])
    return _absorb_param_siblings(nodes)


def _absorb_param_siblings(siblings: List[dict]) -> List[dict]:
    """Fold each `parameter_vector` / `parameter_tensor` sibling into its
    lone in-scope consumer, attaching it as `absorbed_params` metadata
    and stripping it from the rendered children list + the consumer's
    `inputs` edge list.

    Why: standalone parameter nodes are storage descriptors with zero
    compute. When their only role is to hand a learnable tensor to one
    consumer op (router bias → router, sink bias → softmax_with_sink,
    HC scale/base/fn → hc_pre), they dominate the top-level graph with
    boxes that carry no dataflow information — the consumer's job is
    already represented. Folding them in keeps the storage leaf
    (needed for HF tensor matching, surfaced via `absorbed_params`)
    while clearing the visual structure.

    Keep standalone (no fold) when:
      • zero consumers in scope (orphan — a legitimate signal that the
        spec models a learnable that no op uses; useful to surface)
      • two or more consumers (shared param — folding into one would
        misleadingly hide the dataflow into the others)
    """
    if not siblings:
        return siblings
    ids = {n["id"] for n in siblings}
    # consumer_count[param_id] = # of sibling nodes referencing it
    # consumer_of[param_id]    = the sole consumer's id (set only when count==1)
    consumer_count: Dict[str, int] = {}
    consumer_of: Dict[str, str] = {}
    for n in siblings:
        for inp in (n.get("inputs") or []):
            if inp in ids:
                consumer_count[inp] = consumer_count.get(inp, 0) + 1
                consumer_of[inp] = n["id"]
    param_to_consumer: Dict[str, str] = {}
    param_payload: Dict[str, dict] = {}
    for n in siblings:
        if not n.get("identity_source"):
            continue
        pid = n["id"]
        if consumer_count.get(pid, 0) != 1:
            continue
        param_to_consumer[pid] = consumer_of[pid]
        # Snapshot the fields that actually matter for the consumer's
        # weight display. The full node dict carries renderer details
        # (style / op / inputs / …) that are irrelevant once absorbed.
        payload: Dict[str, Any] = {
            "id": pid,
            "label": n.get("label") or n.get("tensor_name") or pid,
            "tensor_name": n.get("tensor_name"),
            "op": n.get("op"),
        }
        # Carry forward every field that the consumer's weight-display
        # might want to surface. Order matters here: the absorber runs
        # AFTER attach_dtypes_via_materialize / compute_memory, so
        # weight_dtype / weight_bytes / weight_shape have already been
        # written by the upstream pipeline.
        for k in ("detail", "tensor_out", "weight_dtype",
                  "weight_dtype_bytes", "weight_bytes",
                  "_matched_tensor", "params"):
            if k in n:
                payload[k] = n[k]
        param_payload[pid] = payload
    if not param_to_consumer:
        return siblings
    # Attach payloads to consumers, strip ids from their inputs lists.
    # Also recurse into each consumer's children — when the consumer is
    # a component wrapper, the param actually feeds a specific inner
    # node (e.g. `hc_pre`'s `fn` input is consumed by inner `mixes`,
    # not by the wrapper itself). Dual-attach so the collapsed view
    # shows the summary on the wrapper, and the expanded view shows
    # weights on each real inner consumer.
    by_id = {n["id"]: n for n in siblings}
    # Each absorbed entry records the input position(s) it occupied, counted
    # in the consumer's inputs BEFORE any folding, so a reader of `inputs`
    # (the structure view reads operand roles by position) can still tell
    # which position each remaining input sits at.
    declared = {cid: list(by_id[cid].get("inputs") or [])
                for cid in set(param_to_consumer.values())}
    for pid, cid in param_to_consumer.items():
        consumer = by_id[cid]
        consumer.setdefault("absorbed_params", []).append({
            **param_payload[pid],
            "input_positions": [i for i, inp in enumerate(declared[cid]) if inp == pid],
        })
        consumer["inputs"] = [
            inp for inp in (consumer.get("inputs") or []) if inp != pid
        ]
        if consumer.get("children"):
            _forward_param_into_children(
                consumer["children"], pid, param_payload[pid],
            )
    return [n for n in siblings if n["id"] not in param_to_consumer]


def _forward_param_into_children(
    children: List[dict], param_id: str, payload: dict,
) -> None:
    """Walk a wrapper's children to find inner nodes whose `inputs:`
    reference `param_id` (the outer-scope param's spec_id, threaded
    through wrapper-input alias resolution by iter_tree's
    resolved_inputs). Attach the payload to each match and strip the
    ref. Recurses through nested compounds so deeply-buried consumers
    in multi-level component nesting still get the weight on display.
    """
    for child in children:
        if param_id in (child.get("inputs") or []):
            child.setdefault("absorbed_params", []).append(payload)
            child["inputs"] = [
                inp for inp in child["inputs"] if inp != param_id
            ]
        if child.get("children"):
            _forward_param_into_children(child["children"], param_id, payload)
