"""One canonical SOL walk record serving every public surface."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.sol_pool import SolPool
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import snapshot_config
from tests.sweep_parity import diff, report


MODEL = "meta-llama/Llama-3.2-3B"
PARALLEL = ParallelConfig(
    tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2).to_dict()


@pytest.fixture
def shared(monkeypatch, tmp_path):
    """A Gallery-produced canonical walk and a one-cell Scaling request."""
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import backend.api.gallery_frontier as gallery
    import backend.api.sweep_runtime as scaling

    pool = SolPool(tmp_path / "sol-pool")
    monkeypatch.setattr(gallery, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "_SWEEP_STEP", KeyedCache(
        "sweep-step", MemoryBackend()))

    view = gallery.FrontierView("H100-SXM")
    cards = view.walk_cards()
    law = gallery._layout_law(
        (MODEL, cards, 0.9, gallery._GALLERY_PRECISION.canonical(), PARALLEL))
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    assert gallery.sharedSolPool.write(
        stored.walk_identity(view),
        [(gallery.layoutRecordName(PARALLEL),
          {"parallel": PARALLEL, "walk": law})],
        metadata={
            "cards": cards,
            "layoutsByCard": {sku: [PARALLEL] for sku in cards},
            "facts": gallery.producerWalkFacts(),
            "recordSchema": gallery.walkRecordSchema,
        })

    calls = {"cells": 0}
    realPlan = scaling._PARALLEL.plan
    realSteps = scaling._PARALLEL.steps

    def oneCell(req):
        full = realPlan(req)
        return dataclasses.replace(full, cells=full.cells[:1])

    def counted(cell, prices):
        calls["cells"] += 1
        yield from realSteps(cell, prices)

    monkeypatch.setattr(scaling, "_PARALLEL", dataclasses.replace(
        scaling._PARALLEL, plan=oneCell, steps=counted))
    return gallery, scaling, pool, calls


def _request(scaling, **changes):
    from backend.api._models import InferenceParallelSweepRequest

    body = {
        "model_id": MODEL,
        "gpu_sku": "",
        "context_len": 4096,
        "modes": ["tp_tp"],
        "gpu_skus": ["H100-SXM"],
        "avg_output_tokens": 256,
        "gpu_memory_fraction": 0.9,
        "selection_mode": "envelope",
        "sol_only": True,
    }
    return InferenceParallelSweepRequest(**{**body, **changes})


def _estimateRequest(scaling, **changes):
    from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
    from backend.inference.sweep_grid import (
        ROUTINE_SEQ_LENS,
        SCALING_BATCH_GRID,
    )

    body = {
        "model_id": MODEL,
        "gpu_sku": "H100-SXM",
        "parallel": PARALLEL,
        "seq_lens": list(ROUTINE_SEQ_LENS),
        "batch_grid": [BatchPoint(**point) for point in SCALING_BATCH_GRID],
        "avg_output_tokens": 256,
        "gpu_memory_fraction": 0.9,
        "selection_mode": "envelope",
        "sol_only": True,
    }
    return InferenceEstimateSweepRequest(**{**body, **changes})


def test_scaling_projects_the_gallery_record_without_a_cell(shared, monkeypatch):
    _gallery, scaling, pool, calls = shared
    req = _request(scaling)

    reused = asyncio.run(scaling.compute_parallel_sweep_cached(req))
    assert calls["cells"] == 0
    assert pool.recordIds(
        scaling.sweep_theory_identity("parallel", req).key) is None

    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "sharedSolPool", SolPool(
        pool.root.parent / "empty-pool"))
    direct = asyncio.run(scaling.compute_parallel_sweep_cached(req))
    assert calls["cells"] == 1
    assert scaling.sharedSolPool.recordIds(
        scaling.sweep_theory_identity("parallel", req).key) is None
    differences = diff(reused, direct)
    assert not differences, report(differences, limit=80)


def test_a_point_outside_the_producer_anchor_runs_its_own_cell(shared):
    _gallery, scaling, _pool, _calls = shared
    req = _request(scaling, context_len=6144)
    assert scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(req)) == {}


def test_a_paired_prefill_batch_runs_its_own_cell(shared):
    """Prefill batching is a measurement-campaign axis, not the serving anchor."""
    from backend.api._models import BatchPoint

    _gallery, scaling, _pool, _calls = shared
    req = _estimateRequest(
        scaling,
        batch_grid=[BatchPoint(key="paired", prefill_batch_size=16,
                               decode_batch_size=16)])
    assert scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req)) == {}


def test_context_batch_projects_the_law_without_a_cell(shared, monkeypatch):
    _gallery, scaling, _pool, calls = shared
    calls["estimateCells"] = 0
    realSteps = scaling._ESTIMATE.steps

    def counted(cell, prices):
        calls["estimateCells"] += 1
        yield from realSteps(cell, prices)

    monkeypatch.setattr(scaling, "_ESTIMATE", dataclasses.replace(
        scaling._ESTIMATE, steps=counted))
    req = _estimateRequest(scaling)
    reused = asyncio.run(scaling.compute_estimate_sweep_cached(req))
    assert calls["estimateCells"] == 0

    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "sharedSolPool", SolPool(
        _gallery.sharedSolPool.root.parent / "empty-estimate-pool"))
    direct = asyncio.run(scaling.compute_estimate_sweep_cached(req))
    assert calls["estimateCells"] == 1
    for key, points in reused["results_by_batch"].items():
        assert key in direct["results_by_batch"]
        for got, want in zip(points, direct["results_by_batch"][key],
                             strict=True):
            assert got["phases"]["decode"]["t_sol_s"] == pytest.approx(
                want["phases"]["decode"]["t_sol_s"], rel=1e-12, abs=0)
            assert got["phases"]["prefill"]["t_sol_s"] == pytest.approx(
                want["phases"]["prefill"]["t_sol_s"], rel=1e-12, abs=0), (
                key, got["avg_input_tokens"],
                got["phases"]["prefill"]["batch"],
                want["phases"]["prefill"]["batch"])
            assert got["max_decode_batch_per_gpu"] == (
                want["max_decode_batch_per_gpu"])


def test_compare_rows_read_the_same_context_batch_projection(shared):
    _gallery, scaling, _pool, _calls = shared
    requests = [_estimateRequest(scaling, gpu_sku=gpu)
                for gpu in ("H100-SXM", "H200")]
    projections = [scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req)) for req in requests]
    assert all(set(projection) == {0} for projection in projections)


def test_gallery_and_scaling_name_the_same_producer_anchor(shared):
    from backend.api.estimate_env import resolve_estimate_environment
    from backend.inference.sol_data import producerWalkFacts, walkFacts

    gallery, _scaling, _pool, _calls = shared
    req = gallery._layout_request(
        MODEL, "H100-SXM", 0.9, gallery._GALLERY_PRECISION.canonical(), PARALLEL)
    assert walkFacts(resolve_estimate_environment(req)) == producerWalkFacts()


def test_the_producer_bundle_covers_every_parallelism_cell(shared):
    from backend.inference.sol_data import layoutRecordName
    from backend.inference.parallel_sweep import ALL_MODES, config_for_degree
    from backend.inference.sweep_grid import SCALING_DEGREES

    gallery, scaling, _pool, _calls = shared
    resolved = gallery._resolve_one(MODEL)[1]
    produced = {layoutRecordName(layout)
                for layout in gallery._walk_plan(
                    resolved, gallery.FrontierView("H100-SXM"),
                    scaling=True)["layouts"]}
    req = _request(scaling, gpu_skus=None)
    plan = scaling._parallel_plan(req)
    assert plan.cells
    assert {layoutRecordName(cell.parallel) for cell in plan.cells} <= produced
    assert {layoutRecordName(config_for_degree(mode, degree))
            for mode in ALL_MODES
            for degree in SCALING_DEGREES} <= produced


def test_the_gallery_plan_does_not_pay_for_scaling_tail_rungs(shared):
    from backend.inference.sol_data import layoutRecordName

    gallery, _scaling, _pool, _calls = shared
    resolved = gallery._resolve_one(MODEL)[1]
    view = gallery.FrontierView("H100-SXM")
    frontier = gallery._walk_plan(resolved, view)
    scaling = gallery._walk_plan(resolved, view, scaling=True)
    assert all(gallery.ParallelConfig.from_dict(layout).cluster_size <= 64
               for layout in frontier["layouts"])
    frontier_ids = {layoutRecordName(layout) for layout in frontier["layouts"]}
    scaling_ids = {layoutRecordName(layout) for layout in scaling["layouts"]}
    assert frontier_ids < scaling_ids


def test_a_card_of_another_class_reads_its_own_class_bundle(shared, monkeypatch):
    """A Hopper bundle — even one whose manifest claims B200 among its cards
    — is not B200's to read: B200 (compute capability 10.0) is another walk
    class, whose rows differ, and it names its own class's bundle. With none
    kept for it the cell runs rather than pricing B200 off Hopper rows; a
    series of Hopper cards alone still projects."""
    gallery, scaling, pool, _calls = shared
    claims = SolPool(pool.root.parent / "claims-b200")
    monkeypatch.setattr(gallery, "sharedSolPool", claims)
    monkeypatch.setattr(scaling, "sharedSolPool", claims)
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    hopper = gallery.FrontierView("H100-SXM")
    item = pool.record(stored.walk_identity(hopper).key,
                       gallery.layoutRecordName(PARALLEL))
    cards = [*hopper.walk_cards(), "B200"]
    assert claims.write(
        stored.walk_identity(hopper),
        [(gallery.layoutRecordName(PARALLEL), item)],
        metadata={
            "cards": cards,
            "layoutsByCard": {sku: [PARALLEL] for sku in cards},
            "facts": gallery.producerWalkFacts(),
            "recordSchema": gallery.walkRecordSchema,
        })
    same = _request(scaling, gpu_skus=["H100-SXM", "H200"])
    assert set(scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(same))) == {0}
    mixed = _request(scaling, gpu_skus=["H100-SXM", "B200"])
    assert scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(mixed)) == {}
    blackwell = _estimateRequest(scaling, gpu_sku="B200")
    assert scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(blackwell)) == {}
    # The Gallery reads B200's own class's bundle, not this one.
    assert stored.shared_walk(gallery.FrontierView("B200")) is None
    assert stored.shared_walk(hopper) is not None
