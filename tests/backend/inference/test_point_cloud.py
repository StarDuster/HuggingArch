"""One point cloud: coordinates, grid rungs, and absent points as data."""
from __future__ import annotations

import pytest
import ast
import pathlib

from backend.inference.capacity import CapacityKnobs, CapacitySweepPlan
from backend.inference.frontier import (POINT_DIMS, absent_point,
                                        frontier_view, parallel_mode)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.sweep_grid import (CONTEXT_GRID, CURATED_CONTEXTS,
                                          GPU_COUNT_GRID,
                                          SCALING_CONTEXTS,
                                          SCALING_DECODE_LADDER,
                                          SCALING_DEGREES)


def test_only_the_point_cloud_module_makes_points():
    """The vocabulary may live in `frontier`, but point construction — served
    values and absence alike — has one producer interface."""
    root = pathlib.Path(__file__).resolve().parents[3] / "backend"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name != "point_cloud.py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called = node.func
                    name = (called.id if isinstance(called, ast.Name)
                            else called.attr if isinstance(called, ast.Attribute)
                            else "")
                    if name in {"absent_point", "decode_point"}:
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_only_the_producer_appends_cloud_points():
    """Adapters may enumerate records; only `produce_cloud` fills a cloud."""
    root = pathlib.Path(__file__).resolve().parents[3] / "backend"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (path.name == "point_cloud.py"
                    and isinstance(node, ast.Attribute)
                    and node.attr == "append"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "points"):
                continue
            if (isinstance(node, ast.Attribute)
                    and node.attr == "append"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "points"):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_every_surface_context_is_a_measured_context():
    assert 65536 not in CONTEXT_GRID
    assert set(CURATED_CONTEXTS) <= set(CONTEXT_GRID)
    assert list(CURATED_CONTEXTS) == SCALING_CONTEXTS


def test_the_gpu_and_decode_ladders_are_grid_subsets():
    assert [x for x in GPU_COUNT_GRID if x <= 128] == SCALING_DEGREES
    expected = (1, 8, 16, 32, 64, 128, 256, 512, 1024)
    assert expected == SCALING_DECODE_LADDER


def test_an_absent_point_states_every_coordinate_and_a_closed_reason():
    coords = {name: 0 for name in POINT_DIMS}
    point = absent_point(coords, "capacity", "the cache does not fit")
    assert point["refused"] == "capacity"
    assert point["reason"] == "the cache does not fit"
    with pytest.raises(ValueError, match="missing \\['context'\\]"):
        absent_point({k: v for k, v in coords.items() if k != "context"},
                     "capacity", "missing")
    with pytest.raises(ValueError, match="unknown point refusal"):
        absent_point(coords, "somewhere_else", "not a vocabulary word")


def test_a_served_point_also_states_every_coordinate():
    from backend.inference.point_cloud import CloudRecord, produce_cloud

    coords = {name: 0 for name in POINT_DIMS}
    with pytest.raises(ValueError, match="missing=\\['context'\\]"):
        produce_cloud([CloudRecord(
            coordinates={k: v for k, v in coords.items()
                        if k != "context"},
            estimate=_est(0.01))])


def test_the_producer_keeps_each_records_source_token():
    """A Gallery record's coverage index follows its filled point."""
    from backend.inference.point_cloud import CloudRecord, produce_cloud

    coords = {name: 0 for name in POINT_DIMS}
    cloud = produce_cloud([
        CloudRecord(coordinates=coords, estimate=_est(0.01), token="source"),
    ])
    assert cloud.tokens[id(cloud.points[0])] == "source"


def test_a_view_admits_an_absent_point_but_no_front_uses_it():
    coords = {name: 0 for name in POINT_DIMS}
    cloud = [
        {**coords, "gpu_sku": "A", "tok_s_per_user": 10,
         "tok_s_per_gpu": 10},
        absent_point({**coords, "gpu_sku": "A"}, "capacity", "too large"),
    ]
    view = frontier_view(cloud, dims=POINT_DIMS, fix={}, group_by=["gpu_sku"])
    assert len(view["series"][0]["frontier"]) == 1
    assert view["values"]["gpu_sku"] == ["A"]


def test_batch_rung_controls_keep_grid_order():
    base = {name: 0 for name in POINT_DIMS}
    keys = ("b1024", "b128", "b64", "max")
    cloud = [{**base, "batch_key": key} for key in keys]
    view = frontier_view(cloud, dims=POINT_DIMS, fix={}, group_by=["gpu_sku"])
    assert view["values"]["batch_key"] == ["b64", "b128", "b1024", "max"]


def test_parallel_mode_is_the_config_shape_not_a_label():
    assert parallel_mode(None) == "tp1"
    assert parallel_mode(ParallelConfig(tp_attention=1)) == "tp1"
    assert parallel_mode(ParallelConfig(
        dp_attention=8, ep_moe=8)) == "dp_ep"
    assert parallel_mode(ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, ep_moe=8)) == "tp_ep"
    assert parallel_mode(ParallelConfig(tp_attention=3)) == "custom"


def test_a_plan_point_maps_to_the_cloud_coordinates():
    from backend.inference.point_cloud import point_coordinates
    from kernel_bench.plan import PlanPoint

    point = PlanPoint(16384, 8192, 8, 64, 1.5)
    coordinates = point_coordinates(
        mode="tp_tp", num_gpus=2, gpu_sku="H100-SXM",
        context=point.seq_len, batch_key="b64",
        draft_tokens=point.s_verify - 1, acceptance=0.8,
        ep_load_factor=point.ep_load_factor)
    assert coordinates == {
        "mode": "tp_tp", "num_gpus": 2, "gpu_sku": "H100-SXM",
        "context": 16384, "batch_key": "b64", "draft_tokens": 7,
        "acceptance": 0.8, "ep_load_factor": 1.5,
    }


def _est(tpot: float, max_batch: int = 64) -> dict:
    return {
        "max_batch_per_gpu": max_batch,
        "phases": {"decode": {"throughput": {"sol": {
            "time_per_output_token_s": tpot,
            "per_gpu": 1 / tpot,
        }, "concurrent_requests": max_batch}}},
    }


def test_an_estimate_cloud_states_coordinates_and_keeps_absences():
    knobs = CapacityKnobs(
        parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                tp_moe_expert=2),
        s_verify=3, acceptance_rate=0.7, ep_load_factor=1.5)
    plan = CapacitySweepPlan.of(
        seq_lens=[1024, 2048], config={}, gpu_sku="H100-SXM", knobs=knobs,
        batch_grid=[{"key": "b1", "prefill_batch_size": 1,
                     "decode_batch_size": 1},
                    {"key": "b64", "prefill_batch_size": 64,
                     "decode_batch_size": 64}])
    cloud = plan.decode_cloud([
        [_est(0.01), _est(0.02, max_batch=2)],
        [_est(0.03), _est(0.04, max_batch=2)],
    ])
    served = cloud[0]
    assert {k: served[k] for k in POINT_DIMS} == {
        "mode": "tp_tp", "num_gpus": 2, "gpu_sku": "H100-SXM",
        "context": 1024, "batch_key": "b1", "draft_tokens": 2,
        "acceptance": 0.7, "ep_load_factor": 1.5,
    }
    absent = cloud[1]
    assert absent["refused"] == "capacity"
    assert absent["context"] == 1024 and absent["batch_key"] == "b64"
    assert len(cloud) == 4
