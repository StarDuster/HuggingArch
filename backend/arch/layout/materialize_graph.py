"""Closed-event-graph checks before grafting collectives onto the spec tree."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from .materialize_comm import FlowRef, PhysicalCollective, Placement, flow_port
from .materialize_flow import _output_axes


def validate_event_graph(analysis: Any, events: Sequence[PhysicalCollective]
) -> None:
    """Validate the closed, ordered event DAG before tree grafting."""
    program = analysis.program
    seen: Dict[str, PhysicalCollective] = {}
    semantic_sources: Dict[str, FlowRef] = {}
    for event in events:
        if not event.event_id or event.event_id in seen:
            raise ValueError(
                f"duplicate or empty physical event id {event.event_id!r}"
            )
        for uid, role in (
            (event.placement_uid, "placement"),
            (event.attribution_uid, "attribution"),
            (event.report_anchor_uid, "report anchor"),
            (event.count_uid, "count"),
        ):
            if program.by_uid(uid) is None:
                raise ValueError(
                    f"event {event.event_id} names unknown {role} uid {uid}"
                )
        if event.source is not None:
            source_node = program.by_uid(event.source.node_uid)
            if source_node is None:
                raise ValueError(
                    f"event {event.event_id} names unknown source uid "
                    f"{event.source.node_uid}"
                )
            source_shape = analysis.shapes.values.get(
                event.source.node_uid
            )
            source_axes = _output_axes(source_shape, event.source.port)
            if source_axes is None:
                raise ValueError(
                    f"event {event.event_id} names unknown source port "
                    f"{source_node.display_path}."
                    f"{event.source.port}"
                )
            semantic_source = event.source
        else:
            dependency = seen.get(str(event.source_event))
            if dependency is None:
                raise ValueError(
                    f"event {event.event_id} depends on missing or later "
                    f"event {event.source_event!r}"
                )
            if dependency.flow_out_shape != event.flow_in_shape:
                raise ValueError(
                    f"event {event.event_id} input {event.flow_in_shape!r} "
                    f"does not continue {dependency.event_id} output "
                    f"{dependency.flow_out_shape!r}"
                )
            semantic_source = semantic_sources[dependency.event_id]
        if event.tree_anchor_event is not None:
            anchor = seen.get(event.tree_anchor_event)
            if anchor is None:
                raise ValueError(
                    f"event {event.event_id} has missing or later tree "
                    f"anchor {event.tree_anchor_event!r}"
                )
            if event.tree_anchor_event != event.source_event:
                raise ValueError(
                    f"event {event.event_id} tree anchor "
                    f"{event.tree_anchor_event!r} is not its flow dependency "
                    f"{event.source_event!r}"
                )
            if (
                anchor.placement_uid != event.placement_uid
                or anchor.placement != event.placement
            ):
                raise ValueError(
                    f"event {event.event_id} cannot graft under "
                    f"{anchor.event_id}: placements differ"
                )
        if event.placement == Placement.AFTER and event.rewrites:
            raise ValueError(
                f"event {event.event_id} is placed after a node but "
                "rewrites consumer inputs"
            )
        if len(set(event.rewrites)) != len(event.rewrites):
            raise ValueError(
                f"event {event.event_id} has duplicate rewrite targets"
            )
        for target in event.rewrites:
            node = program.by_uid(target.node_uid)
            refs = tuple(node.inputs_obj or ())
            if not 0 <= target.input_index < len(refs):
                raise ValueError(
                    f"event {event.event_id} rewrite {target!r} is outside "
                    f"{node.display_path}'s {len(refs)} inputs"
                )
            ref = refs[target.input_index]
            actual = None
            if ref.src is not None:
                actual = FlowRef(
                    ref.src.uid, flow_port(ref.port)
                )
            if actual != semantic_source:
                raise ValueError(
                    f"event {event.event_id} rewrite {target!r} reads "
                    f"{actual!r}, not flow source {semantic_source!r}"
                )
        seen[event.event_id] = event
        semantic_sources[event.event_id] = semantic_source

