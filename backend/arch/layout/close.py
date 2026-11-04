"""Stage and analysis close: interface fixed point, cache write, model output."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from backend.arch.eval.cache import cached_node_uids
from backend.arch.eval.transition import Redistribute, RedistributeSite

from .edge import EdgeState
from .layout import Segmented
from .lower import lower_diff
from .stages import StagePlacement


def close_stage(rule: Any, state: EdgeState, *,
                interface: Optional[EdgeState] = None,
                stage_uid: Optional[int] = None,
                ) -> Optional[Tuple[EdgeState,
                                    Tuple[Tuple[str, str], ...]]]:
    """A repeated block must exit with the same state it entered.

    Top-level stages have no such interface; they only finalize pending
    values such as the vocabulary embedding reduction.
    """
    if not isinstance(state, EdgeState):
        raise ValueError(
            f"foreign state {state!r} at a stage boundary in the "
            "layout rule")
    exit_state = state.finalized()
    if interface is not None and not isinstance(interface, EdgeState):
        raise ValueError(
            f"foreign interface {interface!r} at stage {stage_uid!r}")
    if interface is not None and interface.value is not None:
        raise ValueError(
            f"stage {stage_uid!r} entered with an unsettled interface "
            f"{interface!r}")
    if interface is not None and exit_state != interface:
        raise NotImplementedError(
            f"block interface fixed point violated at stage "
            f"{stage_uid!r}: the region was entered with {interface!r} "
            f"but exits {exit_state!r} — "
            "interface expansion is a later increment")
    if state.value is None:
        return None
    return exit_state, lower_diff(rule, state, exit_state)

def close_analysis(rule: Any, analysis: Any) -> Any:
    """Persistent writes and model outputs after the forward walk.

    Cache storage and the output head have no dataflow consumer, so
    the walk cannot invent an edge. Only a real mismatch is recorded,
    as a Redistribute on the producer.
    """
    records = list(analysis.redistributes)
    if not rule.is_trivial:
        if rule.phase == "prefill":
            records.extend(cache_endpoint_records(rule, analysis))
        records.extend(model_output_records(rule, analysis))
    # A pipeline cut moves values between stages whether or not anything
    # within a stage is distributed: `pp` alone leaves the rule trivial.
    records.extend(stage_crossing_records(rule, analysis))
    analysis.redistributes = tuple(records)
    return analysis

def stage_crossing_records(rule: Any, analysis: Any) -> List[Any]:
    """Every hop across a pipeline cut (`stages.StagePlacement.crossings`):
    each edge whose producer and reader placement puts on different stages
    is one `send_recv` per cut it crosses, sent once however many readers
    read it past that cut, in the state the value is handed on in. The
    readers' own input transitions are the walk's records already, on their
    stages. Each record carries its hop (`Redistribute.crossing`): the cut
    that pays it and the stacks it serves."""
    if rule.pipeline_family is None:
        return []
    return StagePlacement.of(analysis).records(analysis)

def cache_endpoint_records(rule: Any, analysis: Any) -> List[Any]:
    records = []
    for uid in sorted(cached_node_uids(analysis.program)):
        node = analysis.program.by_uid(uid)
        # Identity is the declared child. A wrapper publishes a leaf's
        # axes, so state/shape are read there; the redistribute still
        # names the published node.
        geometry = node
        if node.children:
            geometry, _ = analysis.program.output_leaf(node)
        state = analysis.states.values.get(geometry.uid)
        shape = analysis.shapes.values.get(geometry.uid)
        if state is None or shape is None or shape.out is None:
            raise ValueError(
                f"{node.display_path}: persistent cache endpoint "
                "has no propagated state/shape")
        required, collectives = rule.cache_write_endpoint(
            node, state, shape)
        if state != required and collectives:
            records.append(Redistribute(
                node=uid, input_index=-2, producer=uid,
                arrived=state, required=required,
                collectives=collectives,
                site=RedistributeSite.CACHE_ENDPOINT))
    return records

def model_output_records(rule: Any, analysis: Any) -> List[Any]:
    """The model's outputs leave the walk whole: each output head, and each
    tensor an encoder hands the language model (`Program.handoffs`) — the
    language model reads it as a complete value, so a pending sum is reduced
    where the encoder produces it, in the encode phase."""
    program = analysis.program
    nodes = [node for node in program.walked_nodes()
             if node.boundary_role == "output_head"]
    walked = program.walked_stack_names
    handed = [handoff for handoff in program.handoffs
              if handoff.stack in walked]
    for handoff in handed:
        if not handoff.stream:
            uid, _port = analysis.stage_outputs[handoff.name]
            node = program.by_uid(uid)
            nodes.append(program.output_leaf(node)[0]
                         if node.children else node)
    for handoff in handed:
        if handoff.stream:
            nodes.extend(program.by_uid(uid) for (uid, _port), _state
                         in analysis.appended.get(handoff.name, ()))
    # One record per output node: a node listed twice (every element a
    # repeated stage appends is one template evaluation) is priced once, by
    # its instance count.
    unique = {node.uid: node for node in nodes}
    return [record for node in unique.values()
            for record in _output_record(rule, analysis, node)]


def _output_record(rule: Any, analysis: Any, node: Any) -> List[Any]:
    records = []
    # The state the value is handed on in (`StagePlacement.handed_state`):
    # what a stage close settled, else the node's own output state.
    state = StagePlacement.handed_state(analysis, (node.uid, "out"))
    shape = analysis.shapes.values.get(node.uid)
    if state is None or shape is None or shape.out is None:
        raise ValueError(
            f"{node.display_path}: model output endpoint has no "
            "propagated state/shape")
    if state.workload is not None:
        raise ValueError(
            f"{node.display_path}: model output endpoint must be "
            f"outside a routed zone, got {state!r}")
    if len(state.layout) != len(shape.out):
        raise ValueError(
            f"{node.display_path}: model output state rank "
            f"{len(state.layout)} does not match shape rank "
            f"{len(shape.out)}")
    required_layout = []
    for axis, (layout, extent) in enumerate(
        zip(state.layout, shape.out, strict=True)
    ):
        owner = rule.logical_axis_owner(extent)
        if owner is None or not owner.keep_at_model_output:
            # A concat of replicated pieces (an encoder's output with its
            # begin / end rows) is already whole on every rank.
            whole = rule.replicated()
            required_layout.append(
                layout if isinstance(layout, Segmented) and all(
                    child == whole for _offset, _extent, child
                    in layout.segments)
                else whole)
            continue
        if layout != owner.layout:
            raise ValueError(
                f"{node.display_path}: logical output axis {axis} "
                f"is owned by {owner.layout!r}, not propagated "
                f"layout {layout!r}")
        required_layout.append(layout)
    required = EdgeState(tuple(required_layout), None)
    if state != required:
        records.append(Redistribute(
            node=node.uid, input_index=-3, producer=node.uid,
            arrived=state, required=required,
            collectives=rule.lower_diff(state, required),
            site=RedistributeSite.MODEL_OUTPUT))
    return records

