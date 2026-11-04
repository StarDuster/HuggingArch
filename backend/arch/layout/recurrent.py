"""Conv1d, Mamba-2, gated-delta, and headwise-recurrence layout handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .propagation import LayoutPropagation

from backend.arch.eval.dim import require_equal, value_of

from .axis_use import AXIS_USE, RECURRENT, custom_axis_use
from .edge import EdgeState
from .layout import aligned, axis_fully_replicated, paired_heads


def headwise_recurrence(
        rule: LayoutPropagation, node: Any, states: List[Optional[Any]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[Any], Any]:
    """Three-input per-head recurrence with a complete local state."""
    if len(states) != 3 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: headwise recurrence expects three "
            "resolved inputs")
    inputs = list(states)
    if any(len(state.layout) != 4 for state in inputs):
        raise ValueError(
            f"{node.display_path}: headwise recurrence expects rank-4 "
            "inputs")
    if len(in_shapes) != 3 or any(
            shape is None or len(shape) != 4 for shape in in_shapes):
        raise ValueError(
            f"{node.display_path}: headwise recurrence needs three "
            "rank-4 input shapes")
    for axis, label in ((0, "B"), (1, "heads"), (2, "s_ctx")):
        first = inputs[0].layout[axis]
        for other in inputs[1:]:
            rule.align(node, first, other.layout[axis], label)
    for other in inputs[1:]:
        rule.align(
            node, inputs[0].layout[3], other.layout[3], "d_head")
    for axis, label in ((0, "batch"), (1, "head"), (2, "sequence"),
                        (3, "state-width")):
        for shape in in_shapes[1:]:
            require_equal(
                shape[axis], in_shapes[0][axis],
                lambda at, axis=axis, label=label: ValueError(
                    f"{node.display_path}: {label} extents do not agree: "
                    f"{[at(value_of(one[axis])) for one in in_shapes]}"))
    if any(state.workload != inputs[0].workload for state in inputs[1:]):
        raise ValueError(
            f"{node.display_path}: headwise recurrence mixed routed "
            "and token-domain workload evidence")

    # The declaration tags two recurrent axes: the state-width axis (its
    # `param` is `state`) and the sequence axis the state runs along. The
    # handler's positional convention — axis 2 sequence, axis 3 width — is
    # the family's; which labels those are is the op's.
    recurrent = {axis: use for axis, use in custom_axis_use(rule.declaration(node)).items()
                 if use.category == RECURRENT}
    state_axes = [axis for axis, use in recurrent.items() if use.param == "state"]
    if len(recurrent) != 2 or len(state_axes) != 1:
        raise ValueError(
            f"{node.display_path}: a head-wise recurrence declares exactly two "
            f"recurrent axes, one of them the state (`param: state`); "
            f"{node.op!r} declares {sorted(recurrent)}")
    state_axis = state_axes[0]
    seq_axis = next(axis for axis in recurrent if axis != state_axis)
    required = []
    for state in inputs:
        need = rule.whole_axis(
            node, state, 2, recurrent[seq_axis], f"{seq_axis} (recurrent order)")
        need = rule.whole_axis(
            node, need, 3, recurrent[state_axis],
            "d_head (recurrent state)")
        required.append(need)
    query, _key, value = required
    return required, EdgeState(
        (*query.layout[:3], value.layout[3]), None, query.workload)


def gated_delta_rule(
        rule: LayoutPropagation, node: Any, states: List[Optional[Any]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[Any], Any]:
    """Grouped Q/K heads update an independently owned V-head state."""
    if len(states) != 5 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: gated_delta_rule expects Q, K, V, "
            "decay, and beta inputs")
    query, key, value, decay, beta = states
    if any(len(state.layout) != 4 for state in (query, key, value)) \
            or len(decay.layout) not in (1, 3, 4) \
            or len(beta.layout) != 3:
        raise ValueError(
            f"{node.display_path}: gated_delta_rule expects three "
            "rank-4 tensors, a rank-1/rank-3/rank-4 decay, and "
            "rank-3 beta")
    if len(in_shapes) != 5 \
            or any(shape is None for shape in in_shapes) \
            or any(len(shape) != 4 for shape in in_shapes[:3]) \
            or len(in_shapes[3]) not in (1, 3, 4) \
            or len(in_shapes[4]) != 3:
        raise ValueError(
            f"{node.display_path}: gated_delta_rule input shapes do "
            "not match its five-port contract")

    # What the kernel consumes: the recurrence runs along the whole sequence
    # and holds the whole key width in its state, so those axes are gathered
    # (the walk prices it). The laws below hold between the tensors as
    # consumed, not as they arrive — under CP the short conv has already
    # gathered Q/K/V's sequence while the decay and beta arrive in blocks.
    uses = AXIS_USE["gated_delta_rule"]
    required = list(states)
    for index in (0, 1, 2):
        required[index] = rule.whole_axis(
            node, required[index], 2, uses["s_ctx"],
            "s_ctx (recurrent order)")
    for index in (0, 1):
        required[index] = rule.whole_axis(
            node, required[index], 3, uses["d_lin_k_head"],
            "d_lin_k_head (recurrent state)")
    if len(required[3].layout) == 4:
        required[3] = rule.whole_axis(
            node, required[3], 2, uses["s_ctx"],
            "s_ctx (recurrent decay order)")
        required[3] = rule.whole_axis(
            node, required[3], 3, uses["d_lin_k_head"],
            "d_lin_k_head (recurrent decay state)")
    elif len(required[3].layout) == 3:
        required[3] = rule.whole_axis(
            node, required[3], 1, uses["s_ctx"],
            "s_ctx (recurrent decay order)")
    required[4] = rule.whole_axis(
        node, required[4], 1, uses["s_ctx"],
        "s_ctx (recurrent beta order)")
    query, key, value, decay, beta = required

    for axis, label in ((0, "B"), (1, "key heads"),
                        (2, "s_ctx"), (3, "d_lin_k_head")):
        aligned(
            query.layout[axis], key.layout[axis],
            where=node.display_path, what=label)
        require_equal(
            in_shapes[0][axis], in_shapes[1][axis],
            lambda at, label=label: ValueError(
                f"{node.display_path}: Q/K {label} extents disagree"))
    aligned(query.layout[0], value.layout[0],
            where=node.display_path, what="B")
    aligned(query.layout[2], value.layout[2],
            where=node.display_path, what="s_ctx")
    value_heads = int(value_of(in_shapes[2][1]))
    key_heads = int(value_of(in_shapes[0][1]))
    paired_heads(
        value.layout[1], query.layout[1], value_heads, key_heads,
        where=node.display_path)

    if len(decay.layout) == 4:
        aligned(value.layout[0], decay.layout[0],
                where=node.display_path, what="decay batch")
        aligned(value.layout[1], decay.layout[1],
                where=node.display_path, what="decay value heads")
        aligned(value.layout[2], decay.layout[2],
                where=node.display_path, what="decay sequence")
        aligned(query.layout[3], decay.layout[3],
                where=node.display_path, what="decay key width")
        if int(value_of(in_shapes[3][1])) != value_heads:
            raise ValueError(
                f"{node.display_path}: decay head extent does not "
                "match V heads")
        if int(value_of(in_shapes[3][3])) != int(
                value_of(in_shapes[0][3])):
            raise ValueError(
                f"{node.display_path}: decay width does not match "
                "Q/K key width")
    else:
        aligned(value.layout[1], decay.layout[-1],
                where=node.display_path, what="decay value heads")
        if int(value_of(in_shapes[3][-1])) != value_heads:
            raise ValueError(
                f"{node.display_path}: decay head extent does not "
                "match V heads")
        if len(decay.layout) == 3:
            aligned(value.layout[0], decay.layout[0],
                    where=node.display_path, what="decay batch")
            aligned(value.layout[2], decay.layout[1],
                    where=node.display_path, what="decay sequence")
    aligned(value.layout[1], beta.layout[2],
            where=node.display_path, what="beta value heads")
    aligned(value.layout[0], beta.layout[0],
            where=node.display_path, what="beta batch")
    aligned(value.layout[2], beta.layout[1],
            where=node.display_path, what="beta sequence")
    if int(value_of(in_shapes[4][2])) != value_heads:
        raise ValueError(
            f"{node.display_path}: beta head extent does not match "
            "V heads")
    if any(state.value is not None for state in states):
        raise ValueError(
            f"{node.display_path}: gated_delta_rule inputs must be "
            "complete before recurrent execution")
    if any(state.workload != value.workload for state in states):
        raise ValueError(
            f"{node.display_path}: gated_delta_rule mixed routed and "
            "token-domain workload evidence")

    return required, EdgeState(
        value.layout, None, value.workload, value.provenance)


def headwise_gate(
        rule: LayoutPropagation, node: Any, states: List[Optional[Any]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[Any], Any]:
    """Broadcast a `[B, S, H]` scalar gate over `[B, H, S, D]`."""
    del rule
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: headwise gate expects attention and "
            "gate inputs")
    attention, gate = states
    if len(attention.layout) != 4 or len(gate.layout) != 3:
        raise ValueError(
            f"{node.display_path}: headwise gate expects rank-4 "
            "attention and rank-3 gate")
    if len(in_shapes) != 2 or any(shape is None for shape in in_shapes) \
            or len(in_shapes[0]) != 4 or len(in_shapes[1]) != 3:
        raise ValueError(
            f"{node.display_path}: headwise gate input shapes do not "
            "match its two-port contract")
    for attention_axis, gate_axis, label in (
            (0, 0, "B"), (1, 2, "heads"), (2, 1, "s_ctx")):
        require_equal(
            in_shapes[0][attention_axis], in_shapes[1][gate_axis],
            lambda at, label=label: ValueError(
                f"{node.display_path}: {label} extents disagree"))
        aligned(
            attention.layout[attention_axis],
            gate.layout[gate_axis],
            where=node.display_path, what=label)
    if attention.value != gate.value:
        raise ValueError(
            f"{node.display_path}: headwise gate mixed value states")
    if attention.workload != gate.workload:
        raise ValueError(
            f"{node.display_path}: headwise gate mixed routed and "
            "token-domain workload evidence")
    return list(states), EdgeState(
        attention.layout, attention.value, attention.workload,
        attention.provenance)


def conv1d(rule: LayoutPropagation, node: Any, states: List[Optional[Any]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[Any], Any]:
    """Feature-last convolution for full and depthwise channel groups."""
    del in_shapes
    index, state = rule.sole(node, states)
    if len(state.layout) != 3:
        raise ValueError(
            f"{node.display_path}: conv1d expects feature-last rank-3 "
            f"input, got rank {len(state.layout)}")
    binding = rule.env.get(node.uid) or {}
    try:
        c_in = int(value_of(binding["C_in"]))
        c_out = int(value_of(binding["C_out"]))
        groups = int(value_of(binding["groups"]))
    except KeyError as exc:
        raise ValueError(
            f"{node.display_path}: conv1d has no resolved {exc.args[0]} "
            "dimension") from exc
    if groups <= 0 or c_in % groups:
        raise ValueError(
            f"{node.display_path}: C_in={c_in} is not divisible by "
            f"groups={groups}")

    out_channel = rule.weight(node, "C_out")
    in_per_group = rule.weight(node, "C_in // groups")
    kernel = rule.weight(node, "kernel_size")
    if not axis_fully_replicated(kernel):
        raise ValueError(
            f"{node.display_path}: convolution kernel positions must "
            "be whole on every rank")
    required = list(states)
    need = rule.whole_axis(
        node, state, 1, AXIS_USE["conv1d"]["s_ctx"],
        "s_ctx (convolution halo)")

    if groups == c_in == c_out:
        if not axis_fully_replicated(in_per_group):
            raise ValueError(
                f"{node.display_path}: depthwise C_in/groups axis has "
                "extent one and must be replicated")
        input_channel = need.layout[-1]
        if (not axis_fully_replicated(input_channel)
                and not axis_fully_replicated(out_channel)):
            aligned(
                input_channel, out_channel,
                where=node.display_path, what="depthwise channels")
        channel = (input_channel
                   if not axis_fully_replicated(input_channel)
                   else out_channel)
        need = EdgeState(
            (*need.layout[:-1], channel), need.value, need.workload,
            need.provenance)
        required[index] = need
        return required, EdgeState(
            (*need.layout[:-1], channel), need.value, need.workload)

    if groups != 1:
        raise NotImplementedError(
            f"TODO: {node.display_path}: grouped conv1d with "
            f"1 < groups={groups} < C_in={c_in} needs an explicit "
            "group-axis layout")
    wanted = in_per_group
    need = EdgeState(
        (*need.layout[:-1], wanted), need.value, need.workload,
        need.provenance)
    required[index] = need
    pending = rule.contract(
        node, in_per_group, AXIS_USE["conv1d"]["C_in"],
        "C_in contraction")
    return required, EdgeState(
        (*need.layout[:-1], out_channel), pending, need.workload)


def conv2d(rule: LayoutPropagation, node: Any, states: List[Optional[Any]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[Any], Any]:
    """Feature-last strided 2D convolution over a flattened patch grid.

    A full (non-depthwise) conv: the channel contracts `C_in` into `C_out`,
    the kernel positions are whole, and `s_ctx` carries the window halo. The
    stride-driven sequence SHRINK is a size change the DimMap owns; the layout
    keeps the sequence axis whole here, so this is `conv1d`'s full branch with
    a 4-axis (square kernel) weight.
    """
    del in_shapes
    index, state = rule.sole(node, states)
    if len(state.layout) != 3:
        raise ValueError(
            f"{node.display_path}: conv2d expects feature-last rank-3 "
            f"input, got rank {len(state.layout)}")
    placement = rule.weights.get(node.uid)
    if placement is None:
        raise ValueError(
            f"{node.display_path}: conv2d weight-bearing op with no "
            "weight_axes declaration (DECLARATION_MISSING)")
    # The two trailing weight axes are the square kernel; a repeated slot
    # cannot be addressed by name (`require` rejects it), so check them by
    # position. They must be whole on every rank.
    for kax in placement.axes[-2:]:
        if not axis_fully_replicated(kax):
            raise ValueError(
                f"{node.display_path}: convolution kernel positions must "
                "be whole on every rank")
    out_channel = rule.weight(node, "C_out")
    in_channel = rule.weight(node, "C_in")
    required = list(states)
    need = rule.whole_axis(
        node, state, 1, AXIS_USE["conv2d"]["s_ctx"],
        "s_ctx (convolution halo)")
    need = EdgeState(
        (*need.layout[:-1], in_channel), need.value, need.workload,
        need.provenance)
    required[index] = need
    pending = rule.contract(
        node, in_channel, AXIS_USE["conv2d"]["C_in"],
        "C_in contraction")
    return required, EdgeState(
        (*need.layout[:-1], out_channel), pending, need.workload)


def mamba2_scan(rule: LayoutPropagation, node: Any,
                states: List[Optional[Any]],
                in_shapes: Sequence[Any]
                ) -> Tuple[List[Any], Any]:
    """Seven-input Mamba-2 recurrence, partitioned by complete heads."""
    ranks = (4, 3, 1, 4, 4, 1, 1)
    if len(states) != len(ranks) or any(
            state is None for state in states):
        raise ValueError(
            f"{node.display_path}: mamba2_scan expects seven resolved "
            "inputs")
    if len(in_shapes) != len(ranks) or any(
            shape is None or len(shape) != rank
            for shape, rank in zip(in_shapes, ranks, strict=True)):
        raise ValueError(
            f"{node.display_path}: mamba2_scan input ranks must be "
            f"{ranks!r}")
    # The laws below hold between the tensors as the scan consumes them: the
    # recurrence runs along the whole sequence and keeps the whole state
    # width, so those axes are gathered first (the walk prices it). Under CP
    # the short conv has gathered x/B/C's sequence while dt arrives in blocks.
    uses = AXIS_USE["mamba2_scan"]
    required = list(states)
    for input_index in (0, 1, 3, 4):
        required[input_index] = rule.whole_axis(
            node, required[input_index], 1, uses["s_ctx"],
            "s_ctx (recurrent order)")
    for input_index in (3, 4):
        required[input_index] = rule.whole_axis(
            node, required[input_index], 3, uses["d_ssm_state"],
            "d_ssm_state (recurrent state)")
    hidden, dt, a_log, b_in, c_in, d_skip, dt_bias = required

    for other in (dt, b_in, c_in):
        aligned(hidden.layout[0], other.layout[0],
                where=node.display_path, what="B")
        aligned(hidden.layout[1], other.layout[1],
                where=node.display_path, what="s_ctx")
    for other in (dt, a_log, d_skip, dt_bias):
        head_axis = other.layout[2] if len(other.layout) == 3 \
            else other.layout[0]
        aligned(hidden.layout[2], head_axis,
                where=node.display_path, what="n_ssm_heads")
    aligned(b_in.layout[2], c_in.layout[2],
            where=node.display_path, what="n_ssm_groups")
    aligned(b_in.layout[3], c_in.layout[3],
            where=node.display_path, what="d_ssm_state")

    n_heads = int(value_of(in_shapes[0][2]))
    n_groups = int(value_of(in_shapes[3][2]))
    paired_heads(
        hidden.layout[2], b_in.layout[2], n_heads, n_groups,
        where=node.display_path)

    for parameter in (a_log, d_skip, dt_bias):
        if parameter.value is not None or parameter.workload is not None:
            raise ValueError(
                f"{node.display_path}: per-head scan parameters must "
                "be complete and carry no routed workload")
    token_states = (hidden, dt, b_in, c_in)
    if any(state.workload != hidden.workload
           for state in token_states[1:]):
        raise ValueError(
            f"{node.display_path}: mamba2_scan mixed routed and "
            "token-domain workload evidence")

    return required, EdgeState(
        hidden.layout, None, hidden.workload, hidden.provenance)
