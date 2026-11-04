"""Questions answered off the spec IR: blocks, leaves, names, symbols.

`iter_blocks` enumerates the model's block templates, `iter_leaves`
emits one `LeafTensor` per weight-bearing leaf with its checkpoint
name, and the symbol helpers (`resolve_symbols` / `world_of_dim`) say
which dims a node's expressions bind against. Everything here reads the
tree `build` produced — nothing re-walks the authored spec.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.arch.schema.custom_ops import CustomOpSpec


import backend.arch.dims as _dims
from backend.arch.ops_base import ShapeKind

from .build import iter_tree
from .nodes import LeafTensor

_SYMBOL_PAT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def symbols_in(expr: Any) -> frozenset:
    """Extract identifier symbols from a dim expression (e.g.
    `'n_heads * (d_head + d_qk_rope)'` → {'n_heads', 'd_head', 'd_qk_rope'}). Strings, ints, None all
    accepted; non-string-castable values return empty."""
    if expr is None:
        return frozenset()
    return frozenset(_SYMBOL_PAT.findall(str(expr)))
def iter_leaves(resolved) -> List[LeafTensor]:
    """Flat list of weight-bearing LeafTensors — derived as a filter
    over `iter_tree(resolved)`. Memoized on the ResolvedSpec instance.

    Iteration order matches `iter_tree`'s spec-walk order: top-level
    entries (embed_tokens / model.norm / lm_head / group sub-layers)
    first, then block templates in `spec.model` order, with each block's
    leaves emitted depth-first across the IR tree."""
    if resolved._spec_tree_cache is not None:
        return resolved._spec_tree_cache
    tree = iter_tree(resolved)
    custom_ops = resolved.spec.custom_ops or {}
    out: List[LeafTensor] = []
    for ir in tree.top_level:
        collect_leaves(ir, scope_prefix(ir.checkpoint_root), custom_ops, out)
    for _meta, roots in tree.blocks:
        prefix = checkpoint_prefix(_meta)
        for ir in roots:
            collect_leaves(ir, prefix, custom_ops, out)
    resolved._spec_tree_cache = out
    return out


def scope_prefix(root: str) -> str:
    """The name prefix of a scope rooted at `root` (`model` → `model.`)."""
    return f"{root}." if root else ""


def checkpoint_prefix(meta) -> str:
    """The checkpoint address prefix a block stack's tensors live under.

    A property of the STACK, which is why it takes `BlockMeta` and not a leaf:
    every tensor under that stack shares it, and the two consumers that need it
    (leaf enumeration and the evaluator's storage layer) must agree or the same
    weight gets two different on-disk names and neither binds.

    Read off the stack's resolved `checkpoint` address and nothing else:

      * an indexed stack lives at its declared `prefix`, index after it —
        `model.layers.X.`. `index: {after: backbone}` puts the stack's
        tensors at extra TAIL entries of the backbone's index space (DSv3's
        MTP at `model.layers.61.*`), so its prefix is the backbone's path and
        its tensors share the backbone's checkpoint key. Refuses (raises) when
        the backbone layer count didn't resolve: without a bound count the
        tail index the tensors ship under does not exist;
      * a stack with no index (a vision tower wrapped as one component) has
        no address of its own: its tensors hang off its scope's `root`;
      * a component isolated out of a model (`BlockMeta.address`) lives
        where the model's call site put it (`layers.X.attn.`), so a
        name-keyed quant override reaches it the way it does in the model.
    """
    if meta.address is not None:
        return meta.address
    address = meta.checkpoint
    if address is None:
        raise ValueError(
            f"stack {meta.stack_name!r} has no resolved checkpoint address")
    if address.prefix is None:
        return scope_prefix(address.root or "")
    if address.follows_backbone and not meta.backbone_layers:
        raise ValueError(
            f"aux stack {meta.stack_name!r} declares "
            f"`checkpoint.index: {{after: backbone}}`, but the main "
            f"backbone's layer count is unbound — there is no tail "
            f"index for its tensors to ship under."
        )
    return f"{address.prefix}.X."


def collect_leaves(ir, hf_prefix, custom_ops, out):
    """Recursively walk a SpecIRNode, emitting a LeafTensor for every
    weight-bearing leaf. Components recurse into children (their own
    weight is absorbed onto a leaf the checkpoint has the module for,
    which the walker handles via `module_name`)."""
    if ir.is_component:
        for c in ir.children:
            collect_leaves(c, hf_prefix, custom_ops, out)
        return
    if not ir.owns_weight:
        return
    leaf = ir_to_leaf(ir, hf_prefix, custom_ops)
    if leaf is not None:
        out.append(leaf)


def compose_tensor_name(ir, hf_prefix: str, suffix: str) -> str:
    """The on-disk name of the tensor a leaf SpecIRNode stores under the op's
    declared `suffix` (`.weight` / `.bias` / `""`): its module path, then the
    suffix the node stores that tensor under (`GraphNode.tensor_suffix`,
    else the op's own)."""
    require_module(ir)
    return compose_module(ir, hf_prefix) + ir.tensor_suffix.get(suffix, suffix)


def require_module(ir) -> None:
    """A node that stores a tensor names its module: refused when every
    `module:` up its chain is null, which leaves its tensor no name of its
    own. Under an isolated component's call (`provenance == "site"`) the
    scope's path is that call's own module (`BlockMeta.address`), so a
    chain null up to it stores there, as it does in the model where the
    call node names the module."""
    node = ir
    while node is not None:
        if node.provenance == "site":
            return
        node = node.parent
    if not ir.weight_name:
        raise ValueError(
            f"{ir.display_path!r} stores a tensor but no module on its path "
            f"is a checkpoint module (every `module:` up the chain is null); "
            f"its checkpoint name is not declared")


def compose_module(ir, hf_prefix: str) -> str:
    """The checkpoint path of the module a SpecIRNode's tensors live in.

    `hf_prefix` is the scope's (`checkpoint_prefix` of the block stack, or the
    model-level scope root); the rest is the node's module segments
    (`weight_name`, set by `populate_weight_names` from the declared
    `module` names), with an index `X` written after every segment a
    declaration indexes (`index_slots`). A repeat no declaration indexes
    writes no index: the checkpoint stores those copies as one module. A
    node no module stands for (a call whose every `module:` is null) is at
    the scope's path itself.
    """
    segments = _indexed_segments(ir)
    if not segments:
        return hf_prefix[:-1] if hf_prefix.endswith(".") else hf_prefix
    return hf_prefix + ".".join(segments)


def _indexed_segments(ir) -> List[str]:
    """The node's module segments with `X` at each declared index."""
    path = ir.weight_name
    if not path:
        return []
    segments = path.split(".")
    for position in sorted(set(index_slots(ir, segments)), reverse=True):
        segments.insert(position, "X")
    return segments


def index_slots(ir, segments: List[str]) -> List[int]:
    """The position of each index the declarations write into
    `segments` (the node's module path): after a repeated node that declares
    `checkpoint: {index: local}` (or after the stack `prefix` a model-level
    op declares its copies inside), and just before the node's own module in
    an unrolled expert bank (`expert_weight: unrolled`). The node or any
    ancestor may declare either."""
    out: List[int] = []
    unrolled: Optional[str] = None
    node = ir
    while node is not None:
        if node.checkpoint_indexed:
            anchor = (node.checkpoint_index_after or node.weight_name or "")
            anchor = anchor.split(".") if anchor else []
            if not anchor or segments[:len(anchor)] != anchor:
                raise ValueError(
                    f"{node.display_path!r} declares `checkpoint.index` but "
                    f"is not a module on {'.'.join(segments)!r}")
            out.append(len(anchor))
        if node.expert_weight == "unrolled":
            # The encoding's definition: one entry per expert, its index
            # just before the stacked op's own module (`experts.17.w1`,
            # `experts.17.gate_proj`) — `nn.ModuleList` of per-expert modules.
            if unrolled is not None:
                raise ValueError(
                    f"{ir.display_path!r}: `expert_weight: unrolled` is "
                    f"declared by both {unrolled!r} and "
                    f"{node.display_path!r}; one bank declares its encoding "
                    "once")
            unrolled = node.display_path
            if len(segments) < 2:
                raise ValueError(
                    f"{node.display_path!r}: an unrolled expert bank indexes "
                    f"each expert's module, and {'.'.join(segments)!r} has no "
                    "module above its own to index")
            out.append(len(segments) - 1)
        node = node.parent
    return out


def as_matrix(cop, shape) -> Optional[Tuple[int, int]]:
    """`(in_dim, out_dim)` when a custom op's weight is a contraction matrix.

    Downstream asks that question two ways and both are answered here.
    `sharding`, `weight_breakdown` and `parallelism_comm` compare
    `LeafTensor.op` to `"linear"` to decide whether the weight has two shardable
    axes; `materialize._emit_metadata_siblings` reads `in_dim` / `out_dim` to
    resolve a quant scheme's cardinality formula, and REFUSES the tensor when
    they are zero. They are the same fact about the same tensor, so a leaf that
    answers one and not the other is not partially described — it is broken in
    whichever direction it was not filled in.

    Asked of the tensor rather than of the branch that produced it. Which branch
    computed the shape used to decide the op name — a custom op that declared
    its `weight_shape` kept its own name, one that did not was renamed
    `"linear"` — so two unrelated facts shared a switch: what shape the weight
    is, and whether it is a matrix. `mxfp4_fused_linear` is both, and declaring
    the first is what took the second away.

    A source op indexes its weight rather than contracting against an input, so
    a rank-2 lookup table is not a matrix here however square it looks.
    """
    if cop is not None and cop.shape_kind == ShapeKind.SOURCE:
        return None
    dims = tuple(shape)
    if len(dims) != 2:
        return None
    out_v, in_v = int(dims[0]), int(dims[1])
    return (in_v, out_v) if in_v > 0 and out_v > 0 else None


def _leaf(ir, dim_symbols, count_symbols, *, tensor_name, op, shape,
          in_dim=0, out_dim=0, from_custom_op) -> "LeafTensor":
    """The one LeafTensor construction — the identity/multiplicity fields
    every dispatch branch carries identically."""
    return LeafTensor(
        tensor_name=tensor_name,
        op=op,
        shape=shape,
        in_dim=in_dim,
        out_dim=out_dim,
        multiplicity=ir.multiplicity,
        role=ir.role,
        template=ir.template,
        component_path=ir.component_path,
        dim_symbols=dim_symbols,
        count_symbols=count_symbols,
        has_bias=ir.has_bias,
        from_custom_op=from_custom_op,
        stack_name=ir.stack_name,
        ir_node=ir,
    )


def _custom_op_leaf(ir, op, custom_ops: Dict[str, CustomOpSpec], hf_prefix, dim_symbols,
                    count_symbols) -> Optional["LeafTensor"]:
    """Custom-op linears (mxfp4_fused_linear / similar) — not in
    OP_REGISTRY. Geometry comes from `weight_shape`, or the linear
    `(out, in)` default; never from a guessed dim name."""
    cop = custom_ops.get(op)
    # A custom op that declares `weight_shape` emits through the same
    # template path as a registered primitive — that is the only way
    # to state a geometry `out_key` plus a parameter count cannot reach (a
    # norm scale [d_head], a table [n_vocab,d_model], a fused bank whose input extent is
    # named something other than `in`). One that declares none falls through
    # to the (out, in) default below. Which of the two produced the shape
    # says nothing about what KIND of tensor it is, so both hand the naming
    # to `_leaf_op_name`.
    if cop is not None and cop.weight_shape:
        from backend.arch.ops import weight_shape_of
        dims_flat: Dict[str, Any] = {k: to_int(v) for k, v in (ir.dims or {}).items()}
        if ir.dim is not None:
            dims_flat["dim"] = to_int(ir.dim)
        if ir.shape is not None:
            # A declared weight shape with axes ahead of the op's own
            # (`ops.weight_leading_axes`), as the walk reads it.
            dims_flat["_shape"] = [to_int(s) for s in ir.shape]
        # The same resolver the per-rank walk asks, given the same table:
        # the storage and compute sides used to reach a custom op's
        # declared geometry through two different lookups, and only one of
        # them knew spec-scoped ops existed.
        shp = weight_shape_of(op, cop, dims_flat)
        if shp:
            # Whether it is a matrix is a fact about the op's own weight —
            # its last axes; leading ones only say how many there are.
            matrix = as_matrix(cop, shp[len(shp) - len(cop.weight_shape):])
            return _leaf(
                ir, dim_symbols, count_symbols,
                tensor_name=compose_tensor_name(
                    ir, hf_prefix, cop.weight_tensor_suffix),
                op="linear" if matrix else op,
                shape=tuple(shp),
                in_dim=matrix[0] if matrix else 0,
                out_dim=matrix[1] if matrix else 0,
                from_custom_op=True,
            )
        # Declared, but this node's dims do not resolve it — the same
        # "no weight to allocate" as a registered op with an unbound
        # template. Do not fall through to a (out, in) guess.
        return None
    params_expr = str(cop.params or "0") if cop else "0"
    if params_expr.strip() in ("0", "0.0"):
        return None
    dims = ir.dims or {}
    in_v = to_int(dims["in"]) if "in" in dims else 0
    out_v = to_int(dims["out"]) if "out" in dims else 0
    if in_v <= 0 or out_v <= 0:
        where = ir.display_path or ir.name
        have = sorted((ir.dims or {}).keys())
        raise ValueError(
            f"custom op {op!r} at {where!r} stores parameters "
            f"(params={params_expr!r}) but has no weight_shape and no "
            f"in/out dims (have {have}). Declare weight_shape on the "
            f"custom op."
        )
    # Honor the custom op's weight storage convention (default
    # `.weight` matches nn.Linear; empty `""` matches naked
    # nn.Parameter — the same idiom `linear_param` uses).
    suffix = (cop.weight_tensor_suffix
              if cop else ".weight")
    return _leaf(
        ir, dim_symbols, count_symbols,
        tensor_name=compose_tensor_name(ir, hf_prefix, suffix),
        op="linear" if as_matrix(cop, (out_v, in_v)) else op,
        shape=(out_v, in_v),
        in_dim=in_v,
        out_dim=out_v,
        from_custom_op=True,
    )


def ir_to_leaf(ir, hf_prefix, custom_ops) -> Optional[LeafTensor]:
    """Convert one weight-bearing SpecIRNode → LeafTensor.

    Returns None for non-weight ops (add / silu / mul / softmax / view /
    slice / etc.). Dispatches via `OP_REGISTRY[op].weight_shape`:
    each op declares its on-disk template, resolved against the node's
    dims. Adding a new param-storing op = register one template in
    `ops.py`; spec_tree picks it up automatically. Custom-op linears
    take a parallel branch since they're not in OP_REGISTRY.
    """
    op = ir.op
    dim_symbols = ir_dim_symbols(ir)
    count_symbols = ir_count_symbols(ir)

    if ir.from_custom_op:
        return _custom_op_leaf(ir, op, custom_ops, hf_prefix,
                               dim_symbols, count_symbols)

    # OP_REGISTRY-driven dispatch. Build a flat dims dict (merging
    # `ir.dims` with `ir.dim` under key "dim") and ask the op for its
    # weight shape. Every value coerced to int — an unbound dim
    # resolves to nothing, not a zero-width tensor.
    from backend.arch.ops import (OP_REGISTRY, resolve_shape_template,
                                  weight_leading_axes)
    op_spec = OP_REGISTRY.get(op)
    if op_spec is None or not op_spec.has_weight():
        return None   # not a weight-bearing op (silu / add / mul / view / …)
    if op_spec.is_buffer:
        # A precomputed cache occupies bytes but is not a parameter, and this
        # walk feeds the HF-comparable parameter total. Excluded here by the
        # op's own declaration rather than by `_is_quant_meta`, which reaches
        # the same exclusion through the predicate it also uses for
        # quantisation scales and so cannot tell the two apart.
        return None

    dims_for_shape: Dict[str, int] = {
        k: to_int(v) for k, v in (ir.dims or {}).items()
    }
    if ir.dim is not None:
        dims_for_shape["dim"] = to_int(ir.dim)
    if ir.shape is not None:
        # parameter_tensor / similar source ops read the materialized
        # n-D shape via `_shape`, mirroring the engine's _build_node_dims.
        dims_for_shape["_shape"] = [to_int(s) for s in ir.shape]

    shape = resolve_shape_template(op_spec.weight_shape, dims_for_shape)
    if shape is None:
        return None
    shape = (*weight_leading_axes(op_spec, dims_for_shape), *shape)
    shape_tuple: Tuple[int, ...] = tuple(int(s) for s in shape)
    if not shape_tuple or all(s == 0 for s in shape_tuple):
        return None

    # in_dim / out_dim populated for linear-family ops only — used by
    # materialize for sibling-tensor cardinality formulas.
    in_dim = dims_for_shape.get("in", 0) or 0
    out_dim = dims_for_shape.get("out", 0) or 0

    # Tensor name: `OpSpec.weight_tensor_suffix` decides whether the leaf
    # stores under bare spec_id or with `.weight` appended. Bare-spec-id
    # ops (`parameter_vector`, `linear_param`) model nn.Parameter-style
    # storage where the param hangs directly off the parent module; suffixed
    # ops (`linear`, `embedding`, `rmsnorm`, …) wrap an nn.Module that
    # exposes its weight under the `.weight` attribute.
    suffix = op_spec.weight_tensor_suffix if op_spec is not None else ".weight"
    tensor_name = compose_tensor_name(ir, hf_prefix, suffix)

    return _leaf(
        ir, dim_symbols, count_symbols,
        tensor_name=tensor_name,
        op=op,
        shape=shape_tuple,
        in_dim=in_dim,
        out_dim=out_dim,
        from_custom_op=False,
    )



def to_int(v: Any) -> int:
    """A resolved dim value as an int. A value that is nothing, or a
    symbol that never bound, is an error here — it used to become 0, and a
    0-shaped leaf was then dropped from the tree without a word."""
    if v is None:
        raise ValueError("dim value resolved to nothing")
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dim value {v!r} is not an integer in this binding") from exc


def resolve_symbols(ir, symbols) -> frozenset:
    """`symbols`, written in `ir`'s LOCAL vocabulary, in the MODEL's.

    A leaf's own expression is written in its COMPONENT's vocabulary, and a
    component parameter is a local name: `swiglu_ffn`'s `out: d_ffn` is the
    intermediate width *of whatever invoked it*. `moe_ffn` invokes it with
    `dims: { d_ffn: d_ffn_expert }`, so the expert bank's axis IS
    `d_ffn_expert` — reporting the parameter name instead hands consumers the
    dense FFN's identity for an expert tensor. That is not a cosmetic
    difference: the axis's identity decides which parallelism divides it, so an
    expert weight came out sharded by `tp_dense_ffn`, and on a model with no
    dense layers at all — every Qwen3-MoE, Nemotron-3-Ultra, DSv4-Pro — `d_ffn`
    has no value to divide by, and the whole expert bank was counted as
    replicated on every rank.

    So substitute outward through the ancestor `dims_raw` bindings, innermost
    first, the same ancestor walk `ir_count_symbols` does for `repeat_raw`. A
    symbol no enclosing invocation binds is already in the model's own
    vocabulary and passes through untouched.

    The last frame is the one with no node: an `assignments:` entry choosing
    this template for these layers may rebind dims too, and it is not a graph
    node, so the block's roots carry it as `block_dims_raw`. gemma-4's
    full-attention layers are the case — `dims: {n_kv_heads: n_kv_heads_full}`
    — and without that frame every leaf below reported the sliding-window
    axis and answered to its extent.

    Public, and taking the symbol set as an argument, because the weight walk
    is not the only consumer that asks the registry a question about a NAME.
    `eval.transition` asks which world divides a weight's contraction, and it
    asked with the raw local name: on every `moe_ffn`, the expert exit's
    partial sum was placed on `tp_dense_ffn`, so with the two TP worlds at
    different widths it reduced over the wrong group, and with `tp_dense_ffn`
    at 1 — an ordinary EP deployment of a model that has no dense FFN at all —
    `d_ffn` was in no live world, no partial was seeded, and the model
    reported ZERO collectives while its experts were split eight ways. The
    resolution belongs at the boundary between the two vocabularies, once, not
    at each question.
    """
    def _substitute(current: set, binding: Dict[str, str]) -> set:
        """Each symbol replaced by the symbols of what the frame binds to it.

        A binding to a LITERAL — Step-3.7's full-attention layers are invoked
        with `dims: {n_heads: "64"}`, the SWA backbone being 96 — yields no
        symbols, and dropping the axis there says the tensor has no head axis
        at all. It has one; the call site pinned its extent, not its identity.
        The frame KEY is that identity, and it is also what the shard policy
        reads to decide the split, so keeping it is what makes the two agree:
        with the axis dropped, those `q_proj`s reported replicated while the
        per-rank shape divided them eight ways.
        """
        out: set = set()
        for sym in current:
            if sym not in binding:
                out.add(sym)
                continue
            bound = symbols_in(binding[sym])
            out |= bound or {sym}
        return out

    syms = set(symbols)
    node, outermost = getattr(ir, "parent", None), ir
    while node is not None:
        binding = node.dims_raw or {}
        if binding:
            syms = _substitute(syms, binding)
        node, outermost = node.parent, node
    block = getattr(outermost, "block_dims_raw", None) or {}
    if block:
        syms = _substitute(syms, block)
    return frozenset(syms)


def world_of_dim(ir, expr):
    """The world that divides `expr`, an axis expression written in `ir`'s own
    vocabulary — `None` when no symbol in it belongs to a world.

    The join between the two things that have to agree about a split: WHICH
    axis an op divides, which only the spec knows, and WHICH GROUP divides that
    axis, which only the dim registry knows. Both halves are already written
    down; asking them together is what stops a caller inventing a third answer.

    The third answer worth naming, because it shipped: a component's ROLE. It
    is a slot name, not an axis, and `moe_ffn` plugged into a spec's
    `ffn_dense` slot carries `role: ffn_dense` while every expert inside it
    divides `d_ffn_expert`, whose world is `tp_moe_expert`. Deriving the group
    from the role priced Qwen3-30B-A3B's expert all_reduce on `tp_dense_ffn`
    — at `tp_dense_ffn=1, tp_moe_expert=8`, a group of one.

    Raises when one expression's symbols span two worlds: an axis cut by two
    groups at once is a claim about the mesh no caller here can act on, and
    silently taking either one is how a shard goes missing.
    """
    found = {}
    for sym in resolve_symbols(ir, symbols_in(expr)):
        world = _dims.world_of(sym)
        if world is not None:
            found[world.slug] = world
    if len(found) > 1:
        raise ValueError(
            f"axis expression {expr!r} spans more than one world: "
            f"{sorted(found)}")
    return next(iter(found.values()), None)


def ir_dim_symbols(ir) -> frozenset:
    """All identifier symbols across this leaf's dim expressions, in the
    model's vocabulary — consumers intersect with a parent component's
    `parallelism.shard_symbols` for TP-shardability."""
    syms: set = set()
    if ir.dim_raw:
        syms |= symbols_in(ir.dim_raw)
    for k, v in (ir.dims_raw or {}).items():
        if k == "bias":
            continue
        syms |= symbols_in(v)
    return resolve_symbols(ir, syms)


def ir_count_symbols(ir) -> frozenset:
    """Symbols naming how many COPIES of this leaf the model holds — from every
    `repeat:` between it and the model root, its own and its ancestors'.

    Kept apart from `dim_symbols` because a count is not a dim. `dim_symbols`
    answers "which axes does this TENSOR have", and its consumers intersect it
    with a component's `parallelism.shard_symbols` to find a TP-shardable axis;
    a copy count is not an axis of any tensor under it — each copy is a whole
    tensor — so putting it there would offer it to the wrong question.

    Cumulative through ancestors for the same reason `multiplicity` is: a
    `repeat:` sits on the bank, and the leaves that get copied are the linears
    inside the component it repeats.
    """
    syms: set = set()
    node = ir
    while node is not None:
        syms |= symbols_in(getattr(node, "repeat_raw", None))
        node = getattr(node, "parent", None)
    return frozenset(syms)
