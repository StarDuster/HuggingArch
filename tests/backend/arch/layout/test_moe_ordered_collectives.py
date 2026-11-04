"""Ordered MoE collectives read one routed-load record and chain correctly."""
from __future__ import annotations

import json
import math
from fractions import Fraction
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.materializer import collect_events
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import shared_loader

SNAP = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "Qwen__Qwen3-30B-A3B.json"


def _step(**parallel):
    ep_load_factor = parallel.pop("ep_load_factor", 1)
    config = json.loads(SNAP.read_text())["config"]
    resolved = shared_loader().match(config, model_id="Qwen/Qwen3-30B-A3B")
    step = evaluate_step(
        resolved, ParallelConfig(**parallel), phase="prefill", batch=2,
        input_tokens=128, output_tokens=0,
        ep_load_factor=ep_load_factor)
    return step


def _events(**parallel):
    return collect_events(_step(**parallel).analysis)


def test_expert_tp_reduction_then_route_exchange_is_one_ordered_chain():
    events = _events(tp_moe_expert=2, ep_moe=2, dp_attention=4)
    reduction = next(event for event in events if (
        event.op == "all_reduce" and event.world == "tp_moe_expert"))
    exchange = next(event for event in events if (
        event.op == "all_to_all" and event.world == "ep_moe"
        and event.source_event == reduction.event_id))

    assert exchange.flow_in_shape == reduction.flow_out_shape
    assert exchange.flow_out_shape == reduction.flow_out_shape
    assert len(exchange.flow_in_shape) == 2
    assert exchange.flow_in_shape == reduction.flow_out_shape
    assert exchange.payload_bytes == reduction.payload_bytes


def test_routing_skew_reads_one_busiest_rank_record():
    balanced_step = _step(tp_moe_expert=2, ep_moe=2, dp_attention=4)
    skewed_step = _step(tp_moe_expert=2, ep_moe=2, dp_attention=4,
                        ep_load_factor=2.0)
    balanced = collect_events(balanced_step.analysis)
    skewed = collect_events(skewed_step.analysis)

    def payloads(events):
        return {
            event.event_id: event.payload_bytes
            for event in events
            if event.world in ("tp_moe_expert", "ep_moe")
        }

    left, right = payloads(balanced), payloads(skewed)
    assert set(left) == set(right)
    reductions = {
        key for key in left
        if next(event for event in balanced if event.event_id == key).op
        == "all_reduce"
    }
    exchanges = set(left) - reductions
    loads = [
        state.workload for state in skewed_step.analysis.states.values.values()
        if state.workload is not None
    ]
    load = loads[0]
    factor = Fraction(str(load.busiest_rank_load_factor))
    assert factor != 1
    by_id = {event.event_id: event for event in balanced}
    for key in reductions | exchanges:
        rows = int(by_id[key].flow_out_shape[0])
        expected = math.ceil(rows * factor) / rows
        assert right[key] == expected * left[key]


def test_a_product_row_cut_gathers_over_expert_tensor_parallelism():
    events = _events(tp_moe_expert=4, cp=2, dp_attention=2)
    gather = next(event for event in events if (
        event.op == "all_gather" and event.world == "tp_moe_expert"))
    assert gather.group_size == 4
    assert len(gather.flow_in_shape) == 3
    assert len(gather.flow_out_shape) == 3
