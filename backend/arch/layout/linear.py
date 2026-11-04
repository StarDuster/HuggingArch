"""Embedding, linear, norm, elementwise, RoPE, and buffer layout handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from backend.arch.spec_tree.nodes import SpecIRNode
    from .propagation import LayoutPropagation

from backend.arch.eval.dim import is_unit, value_of
from backend.arch.eval.storage import is_expert_axis
from backend.arch.eval.transition import InputRequirements
from backend.arch.axis_propagation import signature_for
from backend.arch.ops import OP_REGISTRY, elementwise_broadcast_axes

from .axis_use import AXIS_USE, AxisUse, custom_axis_use
from .edge import EdgeState
from .foldlaw import HOMOMORPHISM_REGISTRY
from .layout import (
    AxisLayout,
    Block,
    Product,
    Replicated,
    UniformFiberError,
    axis_fully_replicated,
    physical_local_extent,
    tensor_layout,
)
from .resolver import layout_for_symbol
from .transfer_table import BROADCAST, transfer
from .value import join_orthogonal_partials


def embedding(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """Token ids `(B, S)` in, `(B, S, d_model)` out.

    The one-hot contraction runs over the WEIGHT's vocab axis; with the
    padded vocab Block each rank holds its slice's rows and the output
    is a Partial(sum) over the family.
    """
    return _table_lookup(rule, node, states, in_shapes, rows_axis=False)


def hash_embedding(rule: LayoutPropagation, node: Any,
                   states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
                   ) -> Tuple[List[EdgeState], EdgeState]:
    """`embedding` with a `rows` axis before the feature: `rows` lookups per
    token into one table. The rows axis indexes lookups, not the table, so
    every rank holds all of it; the table's own cuts are `embedding`'s."""
    return _table_lookup(rule, node, states, in_shapes, rows_axis=True)


def _table_lookup(rule: LayoutPropagation, node: Any,
                  states: List[Optional[EdgeState]], in_shapes: Sequence[Any],
                  *, rows_axis: bool) -> Tuple[List[EdgeState], EdgeState]:
    table = rule.weight(node, "n_vocab")
    feature = rule.weight(node, "d_model")
    pending = rule.contract(node, table, AXIS_USE[node.op]["n_vocab"],
                             "n_vocab contraction")
    # The ids' axes carry through as they arrived — a request cut of the
    # batch, the sequence cut of the rows (`model_input_state`). The table
    # adds the feature axis, and a hashed lookup its rows per token.
    _index, ids = rule.sole(node, states)
    rows = ()
    if rows_axis:
        new_axis = rule.cell(
            node, "broadcast", transfer(BROADCAST, (), scope=rule.universe))
        rows = (new_axis.out_axes[0],)
    return list(states), EdgeState((*ids.layout, *rows, feature), pending)

def linear(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
            in_shapes: Sequence[Any]
            ) -> Tuple[Any, EdgeState]:
    """②-seeded matmul: the weight declaration decides the mode.

    `in` cut = row-parallel — the activation is required to carry the
    same cut and the output owes a Partial(sum). `out` cut = column-
    parallel — whole activation in, sharded feature out. Neither cut =
    a replicated weight, pass-through geometry.
    """
    index, state = rule.sole(node, states)
    edge_state = state
    sampled = list(state.layout)
    for axis, extent in enumerate(in_shapes[index] if in_shapes else ()):
        is_exact = getattr(extent, "is_", None)
        if callable(is_exact) and is_exact("s_sampled"):
            # The serving binding selects the sampled position before the
            # model-level head. It is a domain projection, not a gather of
            # the block's resident sequence buffer.
            sampled[axis] = rule.replicated()
    state = EdgeState(
        tuple(sampled), state.value, state.workload, state.provenance
    )
    w_in = rule.weight(node, "in")
    w_out = rule.weight(node, "out")
    need, produced = linear_map(rule,
        node, state, w_in, w_out, AXIS_USE[node.op]["in"],
        rule.weight_batch_axes(node, 2),
        in_shapes[index] if in_shapes else None)
    required = list(states)
    required[index] = edge_state if state != edge_state else need
    if state != edge_state:
        kernels = list(required)
        kernels[index] = need
        return InputRequirements(tuple(required), tuple(kernels)), produced
    return required, produced

def custom_linear(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Spec-declared projection with a matrix weight: ordinary linear law,
    with the weight's two labels and the in-axis tag read off its own
    declaration via `CustomOpSpec.linear_labels` — `weight_shape: [Out, In]`
    when declared, else the default geometry (out = the `axis_replace`
    target, in = the operand's last label). The in axis contracts by sum."""
    index, state = rule.sole(node, states)
    spec = rule.declaration(node)
    out_label, in_label = spec.linear_labels
    w_in = rule.weight(node, in_label)
    w_out = rule.weight(node, out_label)
    use = custom_axis_use(spec)[in_label]
    need, produced = linear_map(rule,
        node, state, w_in, w_out, use, rule.weight_batch_axes(node, 2),
        in_shapes[index] if in_shapes else None)
    required = list(states)
    required[index] = need
    return required, produced

def linear_external_weight(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Linear map whose `[out, in]` weight is the second data input."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: external-weight linear expects "
            "resolved activation and weight inputs")
    activation, weight = states
    if not activation.layout or len(weight.layout) != 2:
        raise ValueError(
            f"{node.display_path}: external-weight linear expects a "
            "non-scalar activation and rank-2 [out, in] weight")
    if weight.value is not None or weight.workload is not None:
        raise ValueError(
            f"{node.display_path}: external linear weight must be "
            "complete and carry no routed workload")
    if len(in_shapes) != 2 or any(shape is None for shape in in_shapes) \
            or len(in_shapes[1]) != 2:
        raise ValueError(
            f"{node.display_path}: external-weight linear needs its "
            "two input shapes")
    if int(value_of(in_shapes[0][-1])) != int(value_of(in_shapes[1][1])):
        raise ValueError(
            f"{node.display_path}: activation width and external "
            "weight input width disagree")
    need, produced = linear_map(rule, 
        node, activation, weight.layout[1], weight.layout[0],
        AXIS_USE[node.op]["in"])
    return [need, weight], produced

def linear_map(rule: LayoutPropagation, node: Any, state: EdgeState,
                w_in: AxisLayout, w_out: AxisLayout,
                contraction_use: AxisUse,
                w_batch: Sequence[AxisLayout] = (),
                in_shape: Optional[Sequence[Any]] = None,
                ) -> Tuple[EdgeState, EdgeState]:
    """Shared `[out, in]` matmul rule for owned and external weights.

    `w_batch` are the placements of a weight's batch axes, the ones ahead of
    `[out, in]` (an expert bank's `n_expert_copies`). Matrix `i` contracts
    batch slice `i` of the activation, so the activation's batch axes that
    pair with them — the ones just before its row and feature axes — must be
    placed the same way: the rank that holds expert `e`'s rows holds expert
    `e`'s matrix."""
    if not state.layout:
        raise ValueError(f"{node.display_path}: rank-0 linear input")
    if w_batch:
        batch = state.layout[:-2]
        if len(batch) < len(w_batch):
            raise ValueError(
                f"{node.display_path}: the weight has {len(w_batch)} batch "
                f"axes and the activation {len(batch)}")
        for left, right in zip(batch[len(batch) - len(w_batch):], w_batch,
                               strict=True):
            rule.align(node, left, right, "weight batch axis")
    in_cut = not isinstance(w_in, Replicated)
    out_cut = not isinstance(w_out, Replicated)
    if in_cut and out_cut:
        raise ValueError(
            f"{node.display_path}: weight cannot be cut along both axes")
    if out_cut:
        state = gather_rows_for_output(rule, node, state, w_out, in_shape)
    if in_cut:
        need = EdgeState((*state.layout[:-1], w_in), state.value,
                         state.workload)
        pending = rule.contract(node, w_in, contraction_use,
                                 "'in' contraction")
        out_value = pending
        if need.value is not None and pending is not None:
            try:
                out_value = join_orthogonal_partials(
                    need.value, pending).with_provenance(node.uid)
            except ValueError:
                # The arriving debt is legal through the affine map but
                # cannot share one Partial with this contraction. Settle
                # it on the input edge, then publish only the new debt.
                need = EdgeState(need.layout, None, need.workload)
        return need, rule.produced_state(
            node, (*need.layout[:-1], w_out), out_value, need.workload)
    # Column-parallel or replicated: the contraction axis must be whole.
    if not axis_fully_replicated(state.layout[-1]):
        need = EdgeState((*state.layout[:-1], rule.replicated()),
                         state.value, state.workload)
        state = need
    rule.contract(node, state.layout[-1], contraction_use,
                   "'in' contraction")
    return state, rule.produced_state(
        node, (*state.layout[:-1], w_out), state.value, state.workload)

def gather_rows_for_output(rule: LayoutPropagation, node: Any, state: EdgeState, output: AxisLayout,
        in_shape: Optional[Sequence[Any]] = None,
        ) -> EdgeState:
    """Remove a row/output diagonal before column parallelism.

    A token cut preserved across EP can alias the expert-TP family. Keeping
    that cut while adding the output-feature cut would leave only diagonal
    row/feature cells. Sequence parallelism has the same geometry at a TP
    entry. The collaborating ranks first gather rows, then each computes
    its output-feature shard.

    The rows are the token axis of `[B, s, d]`, and of a routed
    `[n_expert_copies, M, d]` the per-expert rows after its expert axis —
    found by the expert axis's identity (`storage.is_expert_axis`).
    """
    try:
        tensor_layout((*state.layout[:-1], output))
        return state
    except UniformFiberError:
        pass
    experts = [index for index, extent in enumerate(in_shape or ())
               if is_expert_axis(extent)]
    if state.workload is not None and len(experts) != 1:
        raise ValueError(
            f"{node.display_path}: a routed tensor's rows follow its one "
            f"expert axis; {list(in_shape or ())!r} has {len(experts)}")
    row_axis = experts[0] + 1 if state.workload is not None else 1
    row = state.layout[row_axis]
    if not isinstance(row, (Block, Product)) or (
            isinstance(row, Block) and row.pad_to is not None):
        raise ValueError(
            f"{node.display_path}: row ownership {row!r} "
            "conflicts with expert tensor parallelism and has no "
            "static gather lowering")
    layout = list(state.layout)
    layout[row_axis] = rule.replicated()
    workload = (
        state.workload.replicate_rows(row.num_shards)
        if state.workload is not None else None
    )
    gathered = EdgeState(
        tuple(layout),
        state.value,
        workload,
        state.provenance,
    )
    tensor_layout((*gathered.layout[:-1], output))
    return gathered

def per_head_linear(rule: LayoutPropagation, node: Any,
                     states: List[Optional[EdgeState]],
                     in_shapes: Sequence[Any]
                     ) -> Tuple[List[EdgeState], EdgeState]:
    index, state = rule.sole(node, states)
    if len(state.layout) != 4:
        raise ValueError(
            f"{node.display_path}: per_head_linear expects rank 4")
    w_head = rule.weight(node, "n_heads")
    w_out = rule.weight(node, "out")
    w_in = rule.weight(node, "in")
    required = list(states)
    need_layout = list(state.layout)
    if state.layout[1] != w_head:
        if axis_fully_replicated(state.layout[1]):
            need_layout[1] = w_head
        elif not axis_fully_replicated(w_head):
            raise ValueError(
                f"{node.display_path}: input heads {state.layout[1]!r} "
                f"do not align with weight heads {w_head!r}")
    if state.layout[-1] != w_in:
        if axis_fully_replicated(w_in):
            if not axis_fully_replicated(state.layout[-1]):
                need_layout[-1] = rule.replicated()
        else:
            need_layout[-1] = w_in
    need = EdgeState(tuple(need_layout), state.value, state.workload)
    required[index] = need
    pending = rule.contract(
        node, w_in, AXIS_USE["per_head_linear"]["in"],
        "'in' contraction")
    out_value = pending or need.value
    if pending is not None and need.value is not None:
        try:
            out_value = join_orthogonal_partials(
                need.value, pending).with_provenance(node.uid)
        except ValueError:
            need = EdgeState(need.layout, None, need.workload)
            required[index] = need
            out_value = pending
    return required, rule.produced_state(
        node, (*need.layout[:-1], w_out), out_value, need.workload)

def sum_reduce(rule: LayoutPropagation, node: SpecIRNode,
                states: List[Optional[EdgeState]],
                in_shapes: Sequence[Any]
                ) -> Tuple[List[EdgeState], EdgeState]:
    """Sum along any declared tensor axis, dropping that axis."""
    index, state = rule.sole(node, states)
    if not in_shapes or not in_shapes[0] \
            or len(in_shapes[0]) != len(state.layout):
        raise ValueError(
            f"{node.display_path}: sum_reduce input shape does not match "
            "the arriving state")
    axis = node.axis
    axis = -1 if axis is None else int(axis)
    if axis < 0:
        axis += len(state.layout)
    if not 0 <= axis < len(state.layout):
        raise ValueError(
            f"{node.display_path}: reduction axis {axis} outside rank "
            f"{len(state.layout)}")
    pending = rule.contract(
        node, state.layout[axis], AXIS_USE["sum_reduce"]["in"],
        f"axis {axis} sum contraction")
    required = list(states)
    required[index] = state
    return required, EdgeState(
        state.layout[:axis] + state.layout[axis + 1:], pending,
        state.workload)

def rmsnorm(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
             in_shapes: Sequence[Any]
             ) -> Tuple[List[EdgeState], EdgeState]:
    return feature_norm(rule, node, states, weight_slot="dim")

def rmsnorm_unscaled(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    return feature_norm(rule, node, states, weight_slot=None)

def l2norm(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    return feature_norm(rule, node, states, weight_slot=None)

def layernorm(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    return feature_norm(rule, node, states, weight_slot="dim")

def feature_norm(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]], *,
        weight_slot: Optional[str],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Normalize the trailing feature axis, with an optional affine.

    RMSNorm, unscaled RMSNorm, L2 norm and LayerNorm share the same distributed
    dependency: each output needs the complete trailing axis.  Their only
    placement difference is whether an affine parameter exists.  A
    sharded affine is legal: every rank first sees the complete statistic,
    then retains the output slice owned by its parameter shard.
    """
    index, state = rule.sole(node, states)
    weight = (rule.weight(node, weight_slot)
              if weight_slot is not None else None)
    required = list(states)
    need = rule.whole_axis(
        node, state, len(state.layout) - 1,
        AXIS_USE[node.op]["dim"],
        f"'dim' ({AXIS_USE[node.op]['dim'].param})")
    required[index] = need
    output_axis = (need.layout[-1] if weight is None
                   or axis_fully_replicated(weight) else weight)
    return required, EdgeState(
        (*need.layout[:-1], output_axis), need.value, need.workload,
        need.provenance)

def unary(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[EdgeState], EdgeState]:
    """Pass arriving layout through. Shape extents may still change via the
    DimMap (`sample` narrows `s_ctx`); this handler does not invent cuts."""
    _index, state = rule.sole(node, states)
    return list(states), state

def elementwise(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
                 in_shapes: Sequence[Any]
                 ) -> Tuple[List[EdgeState], EdgeState]:
    """silu / mul / add: separable with standard trailing-axis broadcast.

    Arriving values are already settled: either everything is complete,
    or a fold law let a pending sum through — every input carries the
    identical Partial (`add`), or one does and the rest are complete
    factors invariant over its group (`mul`) — and that Partial rides
    through. A lower-rank operand is aligned from the trailing
    axis, exactly like the op's shape rule. An expanded size-one input must
    be replicated; an all-unit output may retain only a physically
    representable layout or the semantic axis's declared ownership cut.
    """
    known = [(index, state) for index, state in enumerate(states)
             if state is not None]
    if not known:
        raise ValueError(
            f"{node.display_path}: elementwise op with "
            f"{len(known)}/{len(states)} resolved input states")
    if len(known) != len(states):
        # An enclosing scope's input (`x` at a stack's entry) is unresolved
        # (`src is None`) and simply has no state yet; the sole live operand
        # carries the layout. A `repeat: 0` operand never arrives here: the
        # builder does not build a step whose op declares what an absent
        # operand stands for (`OpSpec.absent_operand_identity`), and any
        # other op has no value for it. A ref that RESOLVED to a producer and
        # still has no state is a dropped edge.
        refs = tuple(node.inputs_obj or ())
        dropped = [
            index for index, state in enumerate(states)
            if state is None and index < len(refs)
            and refs[index].src is not None
        ]
        if dropped:
            raise ValueError(
                f"{node.display_path}: elementwise operand(s) {dropped} "
                "resolved to a producer but no layout state arrived — a "
                "dropped edge, not an optional branch")
        if len(known) != 1:
            raise ValueError(
                f"{node.display_path}: {len(known)} live operands beside "
                "absent or unresolved ones; the broadcast law here is for "
                "the sole surviving operand")
    if len(known) == 1:
        return list(states), known[0][1]

    for input_index, state in known:
        shape = in_shapes[input_index] if input_index < len(in_shapes) else None
        if not shape or len(shape) != len(state.layout):
            raise ValueError(
                f"{node.display_path}: elementwise operand {input_index} "
                "shape/state rank mismatch")
    try:
        axes = elementwise_broadcast_axes(in_shapes)
    except ValueError as error:
        raise ValueError(f"{node.display_path}: {error}") from error

    by_input = dict(known)
    required = list(states)
    output_layout: List[AxisLayout] = []
    broadcast_inputs = set()
    for output_axis, axis in enumerate(axes):
        canonical_input, canonical_axis = axis.canonical
        canonical_layout = by_input[canonical_input].layout[canonical_axis]
        # One operand sharded and the others whole on this axis: the whole
        # ones are sliced to the shard in place — a local slice, no wire
        # traffic (lower_diff prices Replicated -> Block at nothing). Under
        # CP a linear-attention output, gathered for its recurrence, meets
        # its gate branch still in sequence blocks. Any other disagreement
        # stays a violated law.
        members = [axis.canonical, *axis.peers]
        cuts = {by_input[i].layout[a] for i, a in members
                if not axis_fully_replicated(by_input[i].layout[a])}
        if len(cuts) == 1 and not is_unit(axis.extent):
            (canonical_layout,) = cuts
            for input_index, input_axis in members:
                state = required[input_index]
                if axis_fully_replicated(state.layout[input_axis]):
                    fixed = list(state.layout)
                    fixed[input_axis] = canonical_layout
                    required[input_index] = EdgeState(
                        tuple(fixed), state.value, state.workload,
                        state.provenance)
        for input_index, input_axis in axis.peers:
            rule.align(
                node,
                canonical_layout,
                required[input_index].layout[input_axis],
                f"broadcast output axis {output_axis}",
            )
        if is_unit(axis.extent):
            try:
                physical_local_extent(canonical_layout, 1)
            except ValueError as error:
                declared_owners = tuple(
                    rule.logical_axis_owner(
                        in_shapes[input_index][input_axis]
                    )
                    for input_index, input_axis in axis.peers
                )
                if not declared_owners or any(
                        owner is None or owner.layout != canonical_layout
                        for owner in declared_owners):
                    raise ValueError(
                        f"{node.display_path}: size-one broadcast axis "
                        f"{canonical_axis} has an unrepresentable shard"
                    ) from error
        for input_index, input_axis in axis.broadcasts:
            broadcast_inputs.add(input_index)
            if not axis_fully_replicated(
                    by_input[input_index].layout[input_axis]):
                raise ValueError(
                    f"{node.display_path}: size-one broadcast axis "
                    f"{input_axis} is sharded")
        output_layout.append(canonical_layout)

    output_rank = len(axes)
    expanded = {
        input_index
        for input_index, state in known
        if len(state.layout) < output_rank or input_index in broadcast_inputs
    }
    # A law with a coefficient operand (`mul`'s factor) is the one way a
    # pending sum meets a complete operand here: the walk settled it so.
    # Under any other op (`add` folds only matching sums) a complete
    # operand beside a pending one is a settling bug, and stays loud.
    law = HOMOMORPHISM_REGISTRY.get(node.op or "")
    scales = law is not None and bool(law.coefficient_slots)
    pending = [state for _, state in known if state.value is not None]
    if pending and (len(pending) < len(known)) and not scales:
        raise ValueError(
            f"{node.display_path}: a pending sum beside a complete operand "
            f"after settling, under {node.op!r}, whose law folds no factor")
    for input_index in expanded:
        operand = by_input[input_index]
        # Broadcasting copies values, which commutes with the sum: a
        # pending FACTOR-scaled operand may be the expanded one.
        if operand.workload is not None or (
                operand.value is not None and not scales):
            raise ValueError(
                f"{node.display_path}: a broadcast parameter must "
                "be complete and carry no routed workload")
    for other in pending[1:]:
        if other.value != pending[0].value:
            raise ValueError(
                f"{node.display_path}: mixed value states after "
                f"settling ({pending[0].value!r} vs {other.value!r})")

    carriers = [
        state for input_index, state in known if input_index not in expanded
    ]
    if not carriers:
        if pending:
            return required, EdgeState(
                tuple(output_layout), pending[0].value, None,
                pending[0].provenance)
        return required, EdgeState(tuple(output_layout))
    for other in carriers[1:]:
        if other.workload != carriers[0].workload:
            raise ValueError(
                f"{node.display_path}: elementwise op mixed routed and "
                "token-domain workload evidence")
    carrier = pending[0] if pending else carriers[0]
    return required, EdgeState(
        tuple(output_layout), carrier.value, carriers[0].workload,
        carrier.provenance,
    )

def normalize_axis(rule: LayoutPropagation, node: Any,
                    states: List[Optional[EdgeState]],
                    in_shapes: Sequence[Any]
                    ) -> Tuple[List[EdgeState], EdgeState]:
    """A normalization needs its normalized axis whole; every other axis is
    orthogonal to it.

    Which axis that is, is the registry's fact: the op's `couples_on` label,
    located in the signature that describes this node (`softmax_act`'s `dim`
    is the last axis of a router's `[B, S, dim]` and the window axis of DSv4's
    `[B, slots, dim=m, D]`). A call site's `axis:` restates it and must agree;
    a node whose `axis:` points elsewhere is refused, and a coupled axis other
    than the last must be declared. A node no signature describes is refused:
    which axis it normalizes is then nobody's statement.
    """
    index, state = rule.sole(node, states)
    rank = len(state.layout)
    declared = None
    if node.axis is not None:
        declared = int(node.axis) + rank if int(node.axis) < 0 else int(node.axis)
        if not 0 <= declared < rank:
            raise ValueError(
                f"{node.display_path}: normalized axis {node.axis} outside "
                f"rank {rank}")
    coupled = _coupled_positions(node, rank)
    if coupled is None:
        raise ValueError(
            f"{node.display_path}: no signature of {node.op} describes this "
            f"rank-{rank} node, so nothing states which axis it normalizes")
    if len(coupled) != 1:
        raise ValueError(
            f"{node.display_path}: {node.op} couples {len(coupled)} axes "
            "in this signature; a normalization couples one")
    (ax,) = coupled
    if (declared is not None and declared != ax) or (
            declared is None and ax != rank - 1):
        raise ValueError(
            f"{node.display_path}: {node.op} normalizes axis {ax} of "
            f"this signature but the node declares `axis: {node.axis}` — "
            "declare the normalized axis")
    required = list(states)
    if not axis_fully_replicated(state.layout[ax]):
        layout = list(state.layout)
        layout[ax] = rule.replicated()
        need = EdgeState(tuple(layout), state.value, state.workload)
        required[index] = need
        state = need
    return required, state


def _coupled_positions(node: Any, rank: int) -> Optional[Tuple[int, ...]]:
    """Positions of the op's `couples_on` labels in the signature that
    describes this node at `rank`; None when no signature does."""
    spec = OP_REGISTRY.get(node.op)
    if spec is None or not spec.couples_on:
        return None
    sig = signature_for(spec, node, rank)
    if sig is None or not sig.ins:
        return None
    labels = sig.ins[0]
    if len(labels) != rank:
        return None
    return tuple(i for i, lbl in enumerate(labels) if lbl in spec.couples_on)

def rope(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
          in_shapes: Sequence[Any]
          ) -> Tuple[List[EdgeState], EdgeState]:
    _index, state = rule.sole(node, states)
    if not isinstance(state.layout[-1], Replicated):
        # The declared window is the rotation's element pairing; a cut
        # here splits pairs across ranks.
        raise ValueError(
            f"{node.display_path}: sharded rotation pairing axis "
            f"(window {AXIS_USE['rope']['dim'].param!r})")
    return list(states), state

def rope_emb(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
              in_shapes: Sequence[Any]
              ) -> Tuple[List[EdgeState], EdgeState]:
    """RoPE on a head-folded feature axis.

    A plain feature cut would split rotation pairs and is illegal. MLA's
    flattened `n_heads * d_rope` axis is different: its declared Block
    partitions whole heads, so every rank still owns complete pairs.
    """
    _index, state = rule.sole(node, states)
    axis = state.layout[-1]
    if axis_fully_replicated(axis):
        return list(states), state
    binding = rule.env.get(node.uid) or {}
    n_heads = binding.get("n_heads")
    expected = None
    if n_heads is not None:
        expected = layout_for_symbol(
            "n_heads", int(value_of(n_heads)), rule.parallel, rule.phase,
            dict(rule.families))
    names = frozenset(getattr(in_shapes[0][-1], "names", ()) or ()) \
        if in_shapes and in_shapes[0] else frozenset()
    if isinstance(axis, Block) and expected is not None \
            and axis == expected and "n_heads" in names:
        return list(states), state
    raise ValueError(
        f"{node.display_path}: rotation axis {axis!r} is neither whole "
        "nor a declared whole-head Block")

def buffer(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
            in_shapes: Sequence[Any]
            ) -> Tuple[List[EdgeState], EdgeState]:
    """A stored buffer (a precomputed table): its state IS its declared
    weight layout, one axis per stored axis, complete.

    A source reads nothing: whatever states arrive with it, it consumes
    none, requires nothing of them, and produces its own declared layout."""
    placement = rule.weights.get(node.uid)
    if placement is None or not placement.axes:
        raise ValueError(
            f"{node.display_path}: buffer with no weight_axes "
            "declaration — occurrence declarations are the only "
            "placement source (DECLARATION_MISSING)")
    return list(states), EdgeState(placement.axes, None)
