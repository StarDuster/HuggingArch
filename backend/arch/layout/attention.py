"""Attention, DSA indexer, and residual-stack layout handlers."""
from __future__ import annotations
import backend.arch.dims as dim_registry
from backend.arch.op_names import (is_join_name, is_view_name,
                                   view_only_components)
from backend.arch.ops import OP_REGISTRY, owns_lookup_table, repeats_input

from enum import Enum
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from backend.arch.spec_tree.nodes import SpecIRNode
    from .propagation import LayoutPropagation

from backend.arch.eval.dim import names_of, require_equal, value_of
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval.transition import InputRequirements, KernelGeometry

from .axis_use import AXIS_USE
from .edge import EdgeState
from .groups import same
from .layout import (
    AxisLayout,
    Block,
    Replicated,
    axis_fully_replicated,
    paired_heads,
)
from .transfer_table import CONTRACT, transfer
from .value import ONLINE_ATTENTION

class AttentionForm(str, Enum):
    """What the attention kernel does when a sequence world is live."""

    HEAD_REPARTITION = "head_repartition"
    CONTEXT_GATHER = "context_gather"
    CACHE_SHARD = "cache_shard"

    def __str__(self) -> str:
        return str(self.value)

    def query_runs(self, rank: int, degree: int
                   ) -> Optional[Tuple[int, int]]:
        """Which of `2·degree` equal chunks of the step's queries rank
        `rank` holds, where this form cuts the queries inside attention;
        None where it does not (a head repartition holds every query for
        its heads, a cache shard cuts keys).

        CONTEXT_GATHER is the balanced split both frameworks run, which is
        the split the engine prices: rank `r` holds chunks `r` and
        `2·degree − 1 − r` — vLLM's DualChunkSwap
        (`vllm/v1/worker/gpu/pcp_manager.py:235-282`, v0.32.0) and
        sglang's zigzag (`python/sglang/srt/layers/cp/zigzag.py:17-29`,
        v0.5.22, one attention call per chunk at `zigzag.py:362-405`). Each
        rank then owns the same share of the causal matrix: the `1/cp` of
        the pairs `physical.pairs_share` prices. The chunks are equal
        rational shares of the queries, the balanced average: vLLM rounds
        each chunk up (`ceil(Q / 2·cp)`, the last ones shorter,
        `pcp_manager.py:262,277-282`) and sglang gives the first `Q mod 2·cp`
        chunks one query more (`zigzag.py:150-157`), so a real chunk differs
        from it by under one query."""
        if self is not AttentionForm.CONTEXT_GATHER:
            return None
        return (rank, 2 * degree - 1 - rank)


#: Score/apply ops that share the main-attention kernel geometry — read off
#: each op's `attention_kernel` declaration, not spelled out here again.
ATTENTION_SCORE_OPS = frozenset(
    name for name, spec in OP_REGISTRY.items() if spec.attention_kernel == "score")
ATTENTION_APPLY_OPS = frozenset(
    name for name, spec in OP_REGISTRY.items() if spec.attention_kernel == "apply")
ATTENTION_KERNEL_OPS = ATTENTION_SCORE_OPS | ATTENTION_APPLY_OPS


def uses_indexer_heads(rule: "LayoutPropagation", node: Any) -> bool:
    """This score contracts indexer-head dims, not residual attention heads."""
    return indexer_heads(node, rule.env.get(node.uid) or {})


def indexer_heads(node: Any, binding: Mapping[str, Any]) -> bool:
    """Whether `node`'s head axes, as `binding` resolves them, are the DSA
    indexer's. The handler asks it of the frame binding; the Ulysses
    legal-degree solver asks it of the program node's own dims, so the
    degrees a picker offers are the ones the handler repartitions. The
    indexer's head axes are the symbols `dims.yaml` puts in the `indexer`
    world (`shardable_by: [indexer]`)."""
    indexer = dim_registry.in_world("indexer")
    for key in ("d_head", "n_heads"):
        if names_of(binding.get(key)) & indexer:
            return True
        raw = (getattr(node, "dims_raw", None) or {}).get(key)
        if raw in indexer:
            return True
    return False


def heads(node: Any, query: AxisLayout, kv: AxisLayout,
           query_extent: int, kv_extent: int) -> AxisLayout:
    return paired_heads(
        query, kv, query_extent, kv_extent, where=node.display_path)

def head_extents(node: Any, in_shapes: Sequence[Any],
                  what: str) -> Tuple[int, int]:
    """The two inputs' GLOBAL head extents, off the arriving shapes."""
    if len(in_shapes) != 2 or any(
            not shape or len(shape) != 4 for shape in in_shapes):
        raise ValueError(
            f"{node.display_path}: {what} expects two rank-4 input "
            "shapes")
    return (int(value_of(in_shapes[0][1])),
            int(value_of(in_shapes[1][1])))

def normalized_attention_scores(rule: LayoutPropagation, node: Any, state: EdgeState) -> EdgeState:
    """Apply the active attention plan's softmax-axis requirement."""
    axis = state.layout[-1]
    if (rule.attention.form == AttentionForm.CACHE_SHARD
            and isinstance(axis, Block)
            and rule.cache_family is not None
            and axis.is_sharded_on(rule.cache_family)):
        # Per-cache-shard statistics are merged at the kernel exit.
        return state
    # The one axis this softmax normalizes: `s_kv` on a dense score matrix,
    # `s_attend` (one query's window) on a windowed one.
    (label, use), = AXIS_USE[node.op].items()
    return rule.whole_axis(
        node, state, len(state.layout) - 1, use, f"{label} (softmax)")

def softmax(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
             in_shapes: Sequence[Any]
             ) -> Tuple[List[EdgeState], EdgeState]:
    index, state = rule.sole(node, states)
    required = list(states)
    need = normalized_attention_scores(rule, node, state)
    required[index] = need
    inputs: Any = required
    if rule.attention.form == AttentionForm.CACHE_SHARD:
        axes = rule.attention.kernel_local_axes(
            node, in_shapes[index], need, "softmax kernel input",
            rule.project_local_extent)
        if rule.attention.dcp_head_buffer_expands():
            axes = rule.attention.expanded_axis(
                axes, 1, rule.cache_family.group_size
            )
        geometry_inputs: List[Optional[Tuple[Any, ...]]] = [
            None for _state in required
        ]
        geometry_inputs[index] = axes
        inputs = InputRequirements(
            tuple(required),
            tuple(required),
            KernelGeometry(tuple(geometry_inputs), axes),
        )
    return inputs, EdgeState(
        need.layout, need.value, need.workload, need.provenance)

def softmax_with_sink(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Windowed softmax plus one per-head side term — a denominator sink
    or a key-0 logit bias. Either is a [n_heads] parameter placed with the
    score heads it acts on, so both follow this one law."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: softmax with sink expects resolved "
            "score and sink inputs")
    scores, sink = states
    if len(scores.layout) != 4 or len(sink.layout) != 1:
        raise ValueError(
            f"{node.display_path}: softmax with sink expects rank-4 "
            f"scores and a rank-1 sink, got {len(scores.layout)} and "
            f"{len(sink.layout)}")
    if sink.value is not None or sink.workload is not None:
        raise ValueError(
            f"{node.display_path}: attention sink must be a complete "
            "per-head parameter")
    need = normalized_attention_scores(rule, node, scores)
    sink_need = EdgeState((need.layout[1],), None)
    required = [need, sink_need]
    if rule.attention.form == AttentionForm.CACHE_SHARD:
        raise NotImplementedError(
            f"{node.display_path}: DCP gathered-head softmax has a "
            "per-head sink but no physical sink replication or side-flow "
            "contract"
        )
    return required, EdgeState(
        need.layout, need.value, need.workload, need.provenance)

def head_shared_operand(rule: LayoutPropagation, node: SpecIRNode,
                        input_index: int, in_shapes: Sequence[Any]) -> bool:
    """Whether an attention operand's logical head axis is stride-zero.

    MLA broadcasts one head-free latent (and its rope term) across query
    heads. The resulting view adopts the query-head layout for local
    kernel geometry, but it is not a set of independently owned KV-head
    shards. DCP legality must therefore inspect the structural source, not
    mistake that view layout for cache ownership.

    Read from declarations, not op names: the operand is head-shared when
    the views it was made through (`is_view_op`, a view-only component
    entered at the output the edge's port names) reach a view that repeats its input
    (`OpSpec.repeats_input`) along this operand's head axis — the walk's
    `width` binding on that view is the head extent, and a repeating view
    the walk bound no `width` for raises rather than reading as unshared.
    A join
    (`ShapeKind.AXIS_CONCAT`) is shared only when every branch is.
    """
    refs = tuple(node.inputs_obj or ())
    if not 0 <= input_index < len(refs) or refs[input_index].src is None:
        return False
    head = names_of(in_shapes[input_index][1])

    def shared(producer: SpecIRNode, port: str) -> bool:
        op = producer.op or ""
        inputs = tuple(producer.inputs_obj or ())
        if op in view_only_components():
            if rule.program is None:
                raise ValueError(
                    f"{producer.display_path}: entering a view-only "
                    "component needs the Program the rule walks")
            return shared(*rule.program.output_leaf(producer, port or "out"))
        if is_join_name(op):
            return bool(inputs) and all(
                ref.src is not None and shared(ref.src, ref.port)
                for ref in inputs)
        if not is_view_name(op):
            return False
        if repeats_input(op):
            binding = rule.env.get(producer.uid)
            if binding is None or "width" not in binding:
                raise ValueError(
                    f"{producer.display_path}: {op} repeats its input but "
                    "the walk bound no `width` for it")
            if names_of(binding["width"]) == head:
                return True
        return bool(inputs) and inputs[0].src is not None and shared(
            inputs[0].src, inputs[0].port)

    ref = refs[input_index]
    return shared(ref.src, ref.port)

def attention_score(rule: LayoutPropagation, node: Any,
                     states: List[Optional[EdgeState]],
                     in_shapes: Sequence[Any]
                     ) -> Tuple[List[EdgeState], EdgeState]:
    """`(B,H,Sq,D) x (B,Hkv,Skv,D) -> (B,H,Sq,Skv)`, under an optional
    selection.

    The third operand is the index list this score reads under. It used to be
    DSA's alone, and it is not: what makes it DSA's is the `Min(s_kv, k)` law,
    while the indices are just the tensor that names which positions — DSv4's
    CSA score is dense over a union whose compressed half an indexer already
    cut, and wires the same operand for the same reason. So the plumbing sits
    here and `dsa_attention_score` keeps only what is sparse about it.
    """
    if len(states) not in (2, 3):
        raise ValueError(
            f"{node.display_path}: attention_score expects Q/K and an "
            "optional selection")
    requirements, scores = _score_pair(
        rule, node, list(states[:2]), in_shapes[:2])
    if len(states) == 2:
        return requirements, scores
    return _under_selection(
        rule, node, states[2], in_shapes, requirements, scores), scores


def _under_selection(rule: LayoutPropagation, node: Any,
                      mask: Optional[EdgeState], in_shapes: Sequence[Any],
                      requirements: Any, scores: EdgeState) -> Any:
    """Fold the index operand into a score's input requirements.

    The indices name absolute positions, so a rank holding a slice of them has
    a list it cannot resolve: they are required whole, on the same batch and
    the same query rows as the score itself.

    `[B, s_query, k]` is one selection for every head (DSA). `[B, Hkv, s_query,
    k]` is one per KV head (MiniMax-M3 MSA: each index head picks its KV
    group's blocks): its head axis is required as the score's KV heads are
    laid out.

    Which axis is which is read off the axes themselves: the batch and query
    axes are the query operand's own (same dim), the last is the selection,
    and an axis left over is a head axis, which must have the key's head
    extent — a selection over query heads is refused, not laid out as KV.
    """
    if mask is None:
        raise ValueError(
            f"{node.display_path}: the selection operand arrived unplaced")
    selection = tuple(in_shapes[2])
    batch_axis = _axis_carrying(node, selection, in_shapes[0][0], "B")
    query_axis = _axis_carrying(node, selection, in_shapes[0][2], "s_query")
    head_axes = [axis for axis in range(len(selection) - 1)
                 if axis not in (batch_axis, query_axis)]
    if len(head_axes) > 1:
        raise ValueError(
            f"{node.display_path}: a selection carries at most one head axis "
            f"besides B and s_query, got {len(head_axes)}")
    for axis in head_axes:
        require_equal(
            selection[axis], in_shapes[1][1],
            lambda at, axis=axis: ValueError(
                f"{node.display_path}: a per-head selection is one per KV "
                f"head; its head axis has {at(value_of(selection[axis]))} "
                f"heads, the key {at(value_of(in_shapes[1][1]))}"))
    if isinstance(requirements, InputRequirements):
        required = list(requirements.edges)
        kernel_required = list(requirements.kernels)
        geometry = requirements.geometry
    else:
        required = list(requirements)
        kernel_required = required
        geometry = None
    if rule.attention.form == AttentionForm.HEAD_REPARTITION:
        mask = rule.attention.gather_resident_axis(
            node, mask, query_axis, "the top-k query rows")
    rule.align(node, required[0].layout[0], mask.layout[batch_axis], "B")
    rule.align(node, required[0].layout[2], mask.layout[query_axis], "s_query")
    if head_axes:
        # Each rank reads the selections of the KV heads it holds: a
        # replicated selection is sliced locally, a differently split one is
        # re-laid out like any operand.
        layout = list(mask.layout)
        layout[head_axes[0]] = required[1].layout[1]
        mask = EdgeState(tuple(layout), mask.value, mask.workload)
    if not axis_fully_replicated(mask.layout[-1]) \
            or mask.value is not None or mask.workload != scores.workload:
        raise ValueError(
            f"{node.display_path}: top-k indices must be "
            "complete, locally whole, and share query ownership")
    required.append(mask)
    if kernel_required is not required:
        kernel_required.append(mask)
    if geometry is not None:
        mask_axes = rule.attention.kernel_local_axes(
            node, in_shapes[2], mask, "top-k mask kernel input",
            rule.project_local_extent)
        geometry = KernelGeometry(
            (*geometry.inputs, mask_axes), geometry.output
        )
    if isinstance(requirements, InputRequirements):
        return InputRequirements(
            tuple(required), tuple(kernel_required), geometry)
    return required


def _axis_carrying(node: Any, shape: Sequence[Any], extent: Any,
                    label: str) -> int:
    """The one non-selection axis of `shape` made of `extent`'s dims."""
    found = [axis for axis, dim in enumerate(shape[:-1])
             if names_of(dim) == names_of(extent)]
    if len(found) != 1:
        raise ValueError(
            f"{node.display_path}: the selection must carry the query's "
            f"{label} axis exactly once, found {len(found)}")
    return found[0]


def _score_pair(rule: LayoutPropagation, node: Any,
                 states: List[Optional[EdgeState]],
                 in_shapes: Sequence[Any]
                 ) -> Tuple[Any, EdgeState]:
    """The Q/K half of a score, with no selection in sight."""
    query, key = pair(node, states, "attention_score")
    rule.align(node, query.layout[0], key.layout[0], "the batch axis")
    rule.align(node, query.layout[3], key.layout[3], "d_head")
    query_heads, kv_heads = head_extents(node, in_shapes,
                                               "attention_score")
    key_head_shared = head_shared_operand(rule, node, 1, in_shapes)
    key_head = rule.replicated() if key_head_shared else key.layout[1]
    head = heads(node, query.layout[1], key_head,
                       query_heads, kv_heads)
    rule.attention.pairing_whole(node, query.layout[2], "s_query")
    pending = rule.contract(
        node, query.layout[3], AXIS_USE["attention_score"]["d_head"],
        "d_head contraction")
    provenance = rule.value_provenance(node, query=0, key=1)
    form = (AttentionForm.CONTEXT_GATHER if uses_indexer_heads(rule, node)
            else rule.attention.form)
    if form == AttentionForm.HEAD_REPARTITION:
        # The entry exchanges: Q and K are required in their interior
        # forms; each edge's difference is one all_to_all.
        required = [
            rule.attention.interior(node, query, query_heads, "the query"),
            rule.attention.interior(
                node, key, kv_heads, "the key",
                gather_if_too_few=True,
            )]
        axes = (query.layout[0], required[0].layout[1],
                rule.replicated(), rule.replicated())
        return required, EdgeState(
            axes, pending, query.workload, provenance)
    if form == AttentionForm.CONTEXT_GATHER:
        required = [query, rule.attention.gather_context(node, key, "the key")]
        axes = (query.layout[0], head, query.layout[2],
                rule.replicated())
        return required, EdgeState(
            axes, pending, query.workload, provenance
        )
    required = [query, key]
    kernel_required = required
    geometry = None
    if form == AttentionForm.CACHE_SHARD:
        head = rule.attention.dcp_gathered_head(
            node, query.layout[1], key_head
        )
        kernel_key_source = key
        if key_head_shared:
            kernel_key_source = EdgeState(
                (key.layout[0], key_head, *key.layout[2:]),
                key.value, key.workload, key.provenance,
            )
        kernel_key = rule.attention.cache_read_state(
            node, kernel_key_source, in_shapes, "s_kv"
        )
        kernel_required = [
            query,
            kernel_key,
        ]
        context = kernel_key.layout[2]
        geometry = rule.attention.dcp_score_geometry(
            node, in_shapes, query, kernel_key, head, key_head_shared,
            rule.project_local_extent)
    else:
        context = rule.attention.kv_side_context(node, key, in_shapes, "s_kv")
    axes = (query.layout[0], head, query.layout[2], context)
    inputs = (
        InputRequirements(
            tuple(required), tuple(kernel_required), geometry
        )
        if kernel_required is not required else required
    )
    return inputs, EdgeState(axes, pending, query.workload, provenance)

def attention_apply(rule: LayoutPropagation, node: Any,
                     states: List[Optional[EdgeState]],
                     in_shapes: Sequence[Any]
                     ) -> Tuple[List[EdgeState], EdgeState]:
    """`(B,H,Sq,Skv) x (B,Hkv,Skv,D) -> (B,H,Sq,D)`."""
    scores, value = pair(node, states, "attention_apply")
    rule.align(node, scores.layout[0], value.layout[0],
                  "the batch axis")
    query_heads, kv_heads = head_extents(node, in_shapes,
                                               "attention_apply")
    value_head_shared = head_shared_operand(rule, node, 1, in_shapes)
    value_head = (
        rule.replicated() if value_head_shared else value.layout[1]
    )
    form = rule.attention.form
    if form == AttentionForm.HEAD_REPARTITION:
        # Scores arrive in the interior form; V takes it on this edge
        # (the third entry exchange), the pairing legality is checked
        # beneath the joint cut at the tensor-parallel level, and the
        # EXIT exchange — head cut back to the sequence cut — happens
        # within this node, reported through `internal`.
        exit_head, exit_rows = rule.attention.exit(node, scores.layout[1])
        heads(node, exit_head, value_head, query_heads,
                    kv_heads)
        rule.attention.pairing_whole(node, scores.layout[2], "s_query")
        required = [scores,
                    rule.attention.interior(
                        node, value, kv_heads, "the value",
                        gather_if_too_few=True,
                    )]
        rule.align(node, scores.layout[3], required[1].layout[2],
                      "s_kv")
        axes = (scores.layout[0], exit_head, exit_rows,
                value.layout[3])
        return required, EdgeState(axes, None, scores.workload)
    if form == AttentionForm.CONTEXT_GATHER:
        head = heads(
            node, scores.layout[1], value_head,
            query_heads, kv_heads,
        )
        rule.attention.pairing_whole(node, scores.layout[2], "s_query")
        required = [
            scores,
            rule.attention.gather_context(node, value, "the value"),
        ]
        rule.align(
            node, scores.layout[3], required[1].layout[2], "s_kv"
        )
        axes = (
            scores.layout[0], head, scores.layout[2], value.layout[3]
        )
        return required, EdgeState(axes, None, scores.workload)
    head = heads(node, scores.layout[1], value_head,
                       query_heads, kv_heads)
    rule.attention.pairing_whole(node, scores.layout[2], "s_query")
    required = [scores, value]
    kernel_required = required
    geometry = None
    s_axis = scores.layout[3]
    if isinstance(s_axis, Replicated):
        # The V side must supply the whole context the scores span.
        rule.attention.kv_side_context(node, value, in_shapes, "s_kv")
        out_value = None
    elif (form == AttentionForm.CACHE_SHARD
          and isinstance(s_axis, Block)
          and same(s_axis.group, rule.cache_family)):
        # Flash decode over the rank's cache shard: the kernel exit is
        # a mergeable online-attention accumulator over the dcp
        # family — the plan-defined output Value.
        if not isinstance(value.layout[2], Replicated):
            raise ValueError(
                f"{node.display_path}: the decode V leaf arrived cut "
                f"({value.layout[2]!r}); the new-token write is "
                "per-rank whole")
        kernel_value_source = value
        if value_head_shared:
            kernel_value_source = EdgeState(
                (value.layout[0], value_head, *value.layout[2:]),
                value.value, value.workload, value.provenance,
            )
        kernel_required = [
            scores,
            rule.attention.cache_read_state(
                node, kernel_value_source, in_shapes, "s_kv"
            ),
        ]
        geometry = rule.attention.dcp_apply_geometry(
            node, in_shapes, scores, kernel_required[1],
            value_head_shared, rule.project_local_extent)
        cell = rule.cell(
            node, "online s_kv contraction",
            transfer(CONTRACT, (s_axis,), reducer=ONLINE_ATTENTION))
        out_value = cell.pending_reduction
        if out_value is None:
            raise ValueError(
                f"{node.display_path}: cache-shard contraction produced "
                "no online-attention Partial")
        score = scores.provenance
        if score is None:
            raise ValueError(
                f"{node.display_path}: attention logits carry no score "
                "provenance")
        out_value = out_value.with_provenance(
            node.uid,
            score=score.producer,
            query=score.operand("query"),
        )
    else:
        use = AXIS_USE["attention_apply"]["s_kv"]
        raise NotImplementedError(
            f"TODO: {node.display_path}: s_kv contraction over "
            f"{s_axis!r} is the {use.plan!r} plan's online form and "
            "no active plan row owns it")
    output_head = (
        rule.attention.declared_query_head(query_heads)
        if form == AttentionForm.CACHE_SHARD else head
    )
    axes = (scores.layout[0], output_head, scores.layout[2],
            value.layout[3])
    inputs = (
        InputRequirements(
            tuple(required), tuple(kernel_required), geometry
        )
        if kernel_required is not required else required
    )
    return inputs, EdgeState(axes, out_value, scores.workload)

def pair(node: Any, states: Sequence[Optional[EdgeState]],
          what: str) -> Tuple[EdgeState, EdgeState]:
    if len(states) != 2 or any(s is None for s in states):
        raise ValueError(
            f"{node.display_path}: {what} expects two resolved input "
            f"states, got {states!r}")
    left, right = states
    if left.workload != right.workload:
        raise ValueError(
            f"{node.display_path}: {what} mixed routed and token-domain "
            "workload evidence")
    for state in (left, right):
        if len(state.layout) != 4:
            raise ValueError(
                f"{node.display_path}: {what} expects rank-4 inputs, "
                f"got rank {len(state.layout)}")
    return left, right


def indexer_logits(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Dense DSA index scores from the query per head `[B, H, s, d]`,
    cached keys `[B, s_kv, d]` and head weights `[B, s, H]`."""
    if len(states) != 3 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: indexer_logits expects query, "
            "cached keys, and weights inputs")
    if len(in_shapes) != 3 or any(shape is None for shape in in_shapes) \
            or [len(shape) for shape in in_shapes] != [4, 3, 3]:
        raise ValueError(
            f"{node.display_path}: indexer_logits expects a rank-4 query "
            "and rank-3 keys and weights")
    query, key, weights = states
    for state in (key, query):
        rule.align(node, weights.layout[0], state.layout[0], "B")
    rule.align(
        node, weights.layout[1], query.layout[2], "s_query")
    if any(state.value is not None for state in (weights, key, query)):
        raise ValueError(
            f"{node.display_path}: indexer_logits inputs must be "
            "complete before its two-axis contraction")
    if any(state.workload != weights.workload for state in (key, query)):
        raise ValueError(
            f"{node.display_path}: indexer_logits mixed routed and "
            "token-domain workload evidence")

    uses = AXIS_USE["indexer_logits"]
    query_need = rule.whole_axis(
        node, query, 1, uses["n_indexer_heads"],
        "n_indexer_heads contraction")
    query_need = rule.whole_axis(
        node, query_need, 3, uses["d_indexer_head"],
        "d_indexer_head contraction")
    weights_need = rule.whole_axis(
        node, weights, 2, uses["n_indexer_heads"],
        "n_indexer_heads contraction")
    key_need = rule.attention.gather_resident_axis(
        node, rule.whole_axis(
            node, key, 2, uses["d_indexer_head"],
            "d_indexer_head contraction"), 1, "the indexer key")
    return [query_need, key_need, weights_need], EdgeState(
        (weights_need.layout[0], weights_need.layout[1],
         key_need.layout[1]),
        None, weights.workload)

def dsa_topk_mask(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Select bounded position indices from a whole dense score row."""
    index, state = rule.sole(node, states)
    if len(state.layout) != 3:
        raise ValueError(
            f"{node.display_path}: dsa_topk_mask expects rank-3 logits")
    need = rule.whole_axis(
        node, state, 2, AXIS_USE["dsa_topk_mask"]["s_kv"],
        "s_kv top-k selection")
    required = list(states)
    required[index] = need
    return required, EdgeState(
        (*need.layout[:-1], rule.replicated()), None, need.workload)

def dsa_attention_score(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Sparse attention score with an optional top-k index operand.

    What is DSA's, once the index operand is `attention_score`'s: the context
    axis comes out whole. A sparse row's `Min(s_kv, k)` positions are chosen
    from anywhere in the history, so no rank owns a contiguous slice of them —
    the reason the same op cannot be handed a cache-sharded read.
    """
    if rule.attention.form == AttentionForm.CACHE_SHARD:
        raise NotImplementedError(
            f"{node.display_path}: DCP sparse top-k attention has no "
            "typed cache-owner workload contract"
        )
    inputs, scores = attention_score(rule, node, states, in_shapes)
    return inputs, EdgeState(
        (*scores.layout[:-1], rule.replicated()), scores.value,
        scores.workload, scores.provenance)


def position_table_add(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Add a learned position table to reference geometry; every rank holds
    the table whole, since any of its positions may be sampled."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: position_table_add expects reference "
            "and table inputs")
    reference, table = states
    table_source = tuple(node.inputs_obj)[1].src
    if table_source is not None \
            and owns_lookup_table(rule.declaration(table_source)) \
            and not all(axis_fully_replicated(axis)
                        for axis in table.layout[:-1]):
        # A lookup table is read by token ids: its rows follow the requests
        # a rank owns. A position table is held whole — any position may be
        # sampled — so a deployment that cuts requests cannot hand it over.
        raise SpecRefusal(
            f"{node.display_path}: position table "
            f"{table_source.display_path!r} is a `{table_source.op}` lookup, "
            "whose rows follow the requests a rank owns, but "
            "position_table_add holds the whole table on every rank",
            declaration="op",
            fix=(f"declare {table_source.display_path!r} as the table "
                 "itself: `op: parameter_tensor` with `shape: [<positions>, "
                 "<width>]`, the learned table position_table_add samples"),
            node=table_source.display_path)
    if len(reference.layout) < 2 or len(table.layout) not in (2, 3):
        raise ValueError(
            f"{node.display_path}: position_table_add expects a rank-2+ "
            "reference and rank-2/rank-3 table")
    if len(in_shapes) != 2 or any(shape is None for shape in in_shapes) \
            or len(in_shapes[0]) != len(reference.layout) \
            or len(in_shapes[1]) != len(table.layout):
        raise ValueError(
            f"{node.display_path}: position_table_add input shapes do "
            "not match its two-port contract")
    if int(value_of(in_shapes[0][-1])) != int(
            value_of(in_shapes[1][-1])):
        raise ValueError(
            f"{node.display_path}: reference and position table "
            "feature extents disagree")
    rule.align(node, reference.layout[-1], table.layout[-1],
                  "feature axis")
    if reference.value is not None or table.value is not None:
        raise ValueError(
            f"{node.display_path}: position_table_add inputs must be "
            "complete")
    if reference.workload is not None or table.workload is not None:
        raise ValueError(
            f"{node.display_path}: position_table_add cannot run inside "
            "a routed zone")
    required_table = EdgeState(tuple(
        rule.replicated() if axis < len(table.layout) - 1 else layout
        for axis, layout in enumerate(table.layout)
    ))
    return [reference, required_table], EdgeState(reference.layout)

def attn_res_combine(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Combine a prefix `[B, S, D]` with a whole history `[B, S, m, D]`,
    scored by a `[D]` gain and a `[1, D]` row; the result is laid out as
    the prefix.

    The softmax runs over the history axis and the score contracts `D`,
    so both are whole where they are read. The scoring weights are
    parameters read whole: a rank does not score against part of `D`.
    """
    if len(states) != 4 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: attn_res_combine expects prefix, "
            "history and two scoring-weight inputs")
    prefix, stack, norm_w, proj_w = states
    if (len(prefix.layout) != 3 or len(stack.layout) != 4
            or len(norm_w.layout) != 1 or len(proj_w.layout) != 2):
        raise ValueError(
            f"{node.display_path}: attn_res_combine expects a rank-3 "
            "prefix, a rank-4 history, a rank-1 gain and a rank-2 row")
    if len(in_shapes) != 4 or any(shape is None for shape in in_shapes) \
            or [len(shape) for shape in in_shapes] != [3, 4, 1, 2]:
        raise ValueError(
            f"{node.display_path}: attn_res_combine input shapes do not "
            "match its four-port contract")
    for prefix_axis, stack_axis, label in (
            (0, 0, "B"), (1, 1, "s_ctx"), (2, 3, "d_model")):
        require_equal(
            in_shapes[0][prefix_axis], in_shapes[1][stack_axis],
            lambda at, label=label: ValueError(
                f"{node.display_path}: {label} extents disagree "
                f"({at(value_of(in_shapes[0][prefix_axis]))} vs "
                f"{at(value_of(in_shapes[1][stack_axis]))})"))
        rule.align(
            node, prefix.layout[prefix_axis], stack.layout[stack_axis],
            label)
    width = int(value_of(in_shapes[0][2]))
    if (int(value_of(in_shapes[2][0])) != width
            or int(value_of(in_shapes[3][1])) != width):
        raise ValueError(
            f"{node.display_path}: the scoring weights are not "
            f"{width} wide")
    if prefix.value is not None or stack.value is not None:
        raise ValueError(
            f"{node.display_path}: attn_res_combine inputs must be "
            "complete")
    if prefix.workload != stack.workload:
        raise ValueError(
            f"{node.display_path}: attn_res_combine mixed routed and "
            "token-domain workload evidence")
    uses = AXIS_USE["attn_res_combine"]
    prefix_need = rule.whole_axis(
        node, prefix, 2, uses["d_model"], "d_model score contraction")
    stack_need = rule.whole_axis(
        node, stack, 2, uses["Sar"], "Sar softmax")
    stack_need = rule.whole_axis(
        node, stack_need, 3, uses["d_model"],
        "d_model score contraction")
    rule.align(
        node, prefix_need.layout[2], stack_need.layout[3], "d_model")
    weights = [EdgeState(tuple(rule.replicated() for _ in weight.layout))
               for weight in (norm_w, proj_w)]
    return [prefix_need, stack_need, *weights], EdgeState(
        prefix_need.layout, None, prefix.workload)

