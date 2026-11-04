"""Render physical artifacts on the public weight wire."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.snapshots import shared_loader
from backend.arch.layout.persistent import (
    physical_artifacts,
    resident_weights,
    whole_resident,
)
from backend.arch.layout.stages import StagePlacement
from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from backend.inference.weight_breakdown import per_rank_weight_breakdown


SNAPSHOTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "snapshots"
LANE_COHORT = tuple(json.loads(
    (Path(__file__).resolve().parents[3] / "golden" / "shardv3_lane_snapshots.json").read_text()
))
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


def _resolved(snapshot: str):
    config = json.loads((SNAPSHOTS / snapshot).read_text())["config"]
    resolved = shared_loader().match(
        config, model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return resolved


def _analysis(snapshot: str, parallel: ParallelConfig = TP8,
              precision=None):
    resolved = _resolved(snapshot)
    step = evaluate_step(
        resolved, parallel, phase="prefill", batch=1, input_tokens=128, output_tokens=0,
        precision=precision,
    )
    # The weight tree is a whole-checkpoint question: the step's walk with
    # every encoder walked beside it.
    analysis = step.as_deployment().checkpoint()
    return resolved, analysis, physical_artifacts(analysis)


def _leaf_rows(tree: dict) -> list[dict]:
    out = []

    def walk(row):
        if not row.get("is_component") and not row.get("children"):
            out.append(row)
            return
        for child in row.get("children") or ():
            walk(child)

    for root in (*tree["groups"], *tree["model_level"]):
        walk(root)
    return out


@pytest.mark.parametrize(
    "snapshot",
    (
        "meta-llama__Llama-3.2-3B.json",
        "openai__gpt-oss-20b.json",
        "Qwen__Qwen3-30B-A3B.json",
    ),
)
def test_weight_tree_reads_the_ledger_without_per_rank_shape_replay(snapshot):
    resolved, analysis, artifacts = _analysis(snapshot)
    tree = per_rank_weight_breakdown(
        resolved, TP8, analysis=analysis, artifacts=artifacts,
    )

    expected = resident_weights(artifacts).without_aux(analysis.program)
    assert tree["total_bytes_per_rank"] == expected


def test_endpoint_owned_bias_and_metadata_are_not_scaled_like_the_matrix():
    resolved, analysis, artifacts = _analysis("openai__gpt-oss-20b.json")
    direct = per_rank_weight_breakdown(resolved, TP8)
    reused = per_rank_weight_breakdown(
        resolved, TP8, analysis=analysis, artifacts=artifacts,
    )

    expected = resident_weights(artifacts).without_aux(analysis.program)
    assert direct["total_bytes_per_rank"] == expected
    assert reused["total_bytes_per_rank"] == expected


def test_weight_tree_rejects_an_analysis_from_another_program_or_deployment():
    llama, llama_analysis, artifacts = _analysis(
        "meta-llama__Llama-3.2-3B.json"
    )
    qwen = _resolved("Qwen__Qwen3-30B-A3B.json")
    with pytest.raises(ValueError, match="different resolved program"):
        per_rank_weight_breakdown(
        qwen, TP8, analysis=llama_analysis, artifacts=artifacts,
        )

    with pytest.raises(ValueError, match="different deployment"):
        per_rank_weight_breakdown(
            llama,
            ParallelConfig(),
            analysis=llama_analysis,
            artifacts=artifacts,
        )


def test_declared_corpus_rows_join_the_persistent_ledger_exactly():
    from backend.arch.layout.persistent import physical_weight_leaves

    for snapshot in LANE_COHORT:
        resolved, analysis, artifacts = _analysis(snapshot)
        records = physical_weight_leaves(analysis, artifacts=artifacts)
        for include_aux in (False, True):
            tree = per_rank_weight_breakdown(
                resolved,
                TP8,
                include_aux_weights=include_aux,
                analysis=analysis,
                artifacts=artifacts,
            )
            rows = _leaf_rows(tree)
            assert all(row["uid"] in records for row in rows), snapshot
            assert tree["total_bytes_per_rank"] == sum(
                records[row["uid"]].resident_bytes_per_rank
                for row in rows
            ), snapshot


def test_pipeline_profile_prices_every_stage_from_the_same_artifacts():
    parallel = ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8, pp=16,
    )
    resolved, analysis, artifacts = _analysis(
        "openai__gpt-oss-20b.json", parallel
    )
    placement = StagePlacement.of(analysis, parallel.pp)
    scope = analysis.program.included_stack_names()

    expected = []
    for stage in range(parallel.pp):
        from backend.arch.layout.persistent import (
            artifacts_for_layers,
            resident_weights,
        )

        stage_artifacts = artifacts_for_layers(
            analysis,
            artifacts,
            placement.view(stage).weights(scope),
        )
        resident = resident_weights(stage_artifacts).without_aux(analysis.program)
        expected.append(resident)
        stage_tree = per_rank_weight_breakdown(
            resolved,
            parallel,
            analysis=analysis,
            artifacts=artifacts,
            pp_stage=stage,
        )
        assert stage_tree["total_bytes_per_rank"] == resident
        assert stage_tree["pipeline"]["stage"] == stage
    assert len(expected) == parallel.pp


def test_capacity_derives_storage_and_cache_from_one_analysis(monkeypatch):
    from backend.arch.layout import physical_cache
    from backend.inference import capacity as capacity_mod
    from backend.inference import walk_registry

    # A cold estimate: the walk, the artifacts and the cache views of a
    # model's cells are kept on the cells' entry (`capacity.CellBasis`), so
    # an earlier request for this model, deployment and precision — in this
    # process, any test file — hands them over and this estimate derives
    # nothing. The count is of an estimate that finds no cells.
    walk_registry.WALK_CELLS.clear()
    config = json.loads(
        (SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text()
    )["config"]
    kwargs = dict(
        config=config,
        resolved=_resolved("meta-llama__Llama-3.2-3B.json"),
        gpu_sku="H100-SXM",
        parallel=TP8,
        avg_input_tokens=128,
        avg_output_tokens=32,
        selection_mode="theoretical_only",
    )

    cache_analyses = []
    weight_analyses = []
    real_cache_view = physical_cache.physical_cache_view
    real_weight_breakdown = capacity_mod.per_rank_weight_breakdown

    def recording_cache_view(analysis, **kwargs):
        cache_analyses.append(analysis)
        return real_cache_view(analysis, **kwargs)

    def recording_weight_breakdown(*args, **kwargs):
        weight_analyses.append(kwargs.get("analysis"))
        return real_weight_breakdown(*args, **kwargs)

    monkeypatch.setattr(physical_cache, "physical_cache_view", recording_cache_view)
    monkeypatch.setattr(
        capacity_mod,
        "per_rank_weight_breakdown",
        recording_weight_breakdown,
    )
    estimate_capacity(**kwargs)

    assert len(cache_analyses) == 1
    assert weight_analyses
    assert all(analysis is cache_analyses[0] for analysis in weight_analyses)


def test_capacity_builds_every_pipeline_stage_from_one_analysis(monkeypatch):
    from backend.arch.layout import physical_cache
    from backend.inference import capacity as capacity_mod
    from backend.inference import walk_registry

    # Cold, as above (`capacity.CellBasis`).
    walk_registry.WALK_CELLS.clear()
    config = json.loads(
        (SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text()
    )["config"]
    parallel = ParallelConfig(
        pp=32,
        tp_attention=8,
        tp_dense_ffn=8,
        tp_moe_expert=8,
    )
    kwargs = dict(
        config=config,
        resolved=_resolved("meta-llama__Llama-3.2-3B.json"),
        gpu_sku="H100-SXM",
        parallel=parallel,
        avg_input_tokens=128,
        avg_output_tokens=32,
        selection_mode="theoretical_only",
    )

    calls = []
    profile_calls = []
    real_cache_view = physical_cache.physical_cache_view
    real_stage_profiles = capacity_mod._pipeline_stage_profiles

    def recording_cache_view(analysis, **kwargs):
        view = real_cache_view(analysis, **kwargs)
        calls.append((analysis, view))
        return view

    def recording_stage_profiles(*args, **kwargs):
        profiles = real_stage_profiles(*args, **kwargs)
        profile_calls.append((args[2], args[1], profiles))
        return profiles

    monkeypatch.setattr(physical_cache, "physical_cache_view", recording_cache_view)
    monkeypatch.setattr(
        capacity_mod,
        "_pipeline_stage_profiles",
        recording_stage_profiles,
    )
    estimate_capacity(**kwargs)

    assert len(calls) == 1
    analysis, view = calls[0]
    assert len(profile_calls) == 1
    cache_view, stage_bytes, profiles = profile_calls[0]
    assert cache_view is view
    from backend.arch.layout.persistent import (
        artifacts_for_layers,
        physical_artifacts,
        resident_weights,
    )
    artifacts = physical_artifacts(analysis)
    placement = StagePlacement.of(analysis, parallel.pp)
    scope = analysis.program.included_stack_names()
    assert stage_bytes == [
        resident_weights(
            artifacts_for_layers(
                analysis,
                artifacts,
                placement.view(profile.stage).weights(scope),
            )
        ).without_aux(analysis.program)
        for profile in profiles
    ]
    assert len(profiles) == parallel.pp
    assert any(profile.per_token == 0 for profile in profiles)


def test_pipeline_capacity_budget_is_the_join_not_the_tree():
    """PP>1 used to read `wb['total_bytes_per_rank']`. The join already
    prices a stage (`artifacts_for_layers`); the budget has to be that
    number, or a rank's weights and its module tree can disagree."""
    config = json.loads(
        (SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text()
    )["config"]
    parallel = ParallelConfig(
        pp=4, tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
    )
    resolved, analysis, artifacts = _analysis(
        "meta-llama__Llama-3.2-3B.json", parallel,
    )
    result = estimate_capacity(
         config=config,
        resolved=resolved,
        gpu_sku="H100-SXM",
        parallel=parallel,
        avg_input_tokens=128,
        avg_output_tokens=32,
        selection_mode="theoretical_only",
    )
    from backend.arch.layout.persistent import artifacts_for_layers
    stage = result["memory"]["pp_stage"]
    expected = float(resident_weights(artifacts_for_layers(
        analysis, artifacts,
        StagePlacement.of(analysis, parallel.pp).view(stage).weights(
            analysis.program.included_stack_names()),
    )).without_aux(analysis.program))
    assert result["memory"]["weight_bytes_per_gpu"] == expected


def test_cluster_total_does_not_follow_tp():
    resolved, analysis_1, art_1 = _analysis(
        "meta-llama__Llama-3.2-3B.json", ParallelConfig(),
    )
    resolved, analysis_8, art_8 = _analysis("meta-llama__Llama-3.2-3B.json")
    one = whole_resident(art_1, analysis_1.program, include_aux=False)
    eight = whole_resident(art_8, analysis_8.program, include_aux=False)
    assert one == eight
    config = json.loads(
        (SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text()
    )["config"]
    r1 = estimate_capacity(
         config=config, resolved=resolved, gpu_sku="H100-SXM",
        parallel=ParallelConfig(),
        avg_input_tokens=128, avg_output_tokens=32,
        selection_mode="theoretical_only",
    )
    r8 = estimate_capacity(
         config=config, resolved=resolved, gpu_sku="H100-SXM", parallel=TP8,
        avg_input_tokens=128, avg_output_tokens=32,
        selection_mode="theoretical_only",
    )
    assert r1["weight_bytes_total"] == r8["weight_bytes_total"]
    assert r1["weight_bytes_total"] == one


def test_serving_cast_tree_equals_ledger():
    from backend.arch.eval.precision import PRESETS
    from backend.arch.layout.persistent import physical_artifacts, resident_weights
    from backend.inference.step_eval import evaluate_step

    resolved = _resolved("meta-llama__Llama-3.2-3B.json")
    serving = evaluate_step(
        resolved, TP8, phase="prefill", batch=1, input_tokens=8, output_tokens=0,
        precision=PRESETS["w_fp8_kv_fp8"],
    ).as_deployment().checkpoint()
    artifacts = physical_artifacts(serving)
    tree = per_rank_weight_breakdown(
        resolved, TP8, analysis=serving, artifacts=artifacts,
    )
    expected = resident_weights(artifacts).without_aux(serving.program)
    assert tree["total_bytes_per_rank"] == expected


def test_serving_and_role_casts_keep_storage_derivations_equal():
    from backend.arch.eval.precision import PRESETS, PrecisionProfile
    cases = (
        PRESETS["w_fp8_kv_fp8"],
        PrecisionProfile.canonicalize({"weights": [
            {"select": {"roles": ["attention"]}, "format": {"scheme": "fp8"}}]}),
    )
    for profile in cases:
        resolved, analysis, artifacts = _analysis(
            "meta-llama__Llama-3.2-3B.json", precision=profile)
        incumbent = per_rank_weight_breakdown(resolved, TP8,
                                              precision=profile)
        projected = per_rank_weight_breakdown(
            resolved,
            TP8,
            analysis=analysis,
            artifacts=artifacts,
        )
        assert projected == incumbent


@pytest.mark.parametrize(
    "snapshot",
    (
        "deepseek-ai__DeepSeek-V3.json",
        "MiniMaxAI__MiniMax-M2.7.json",
        "XiaomiMiMo__MiMo-V2-Flash.json",
        "google__gemma-4-12B-it.json",
    ),
)
def test_gallery_debt_models_build_a_weight_tree(snapshot):
    """MTP/vision templates and MiniMax's flat SwiGLU permute used to raise
    before a tree existed; gallery then showed empty charts."""
    resolved, analysis, artifacts = _analysis(snapshot)
    tree = per_rank_weight_breakdown(
        resolved, TP8, analysis=analysis, artifacts=artifacts,
    )
    assert tree["total_bytes_per_rank"] > 0


def test_config_identity_guard_accepts_clone_rejects_cross_model():
    """The walk folds a SHARED CLONE of the spec (object differs), so the guard
    keys on config identity, not object identity. A walk-produced analysis
    (same config) is accepted; an analysis for a different model is rejected."""
    r_a, an_a, art_a = _analysis("meta-llama__Llama-3.2-3B.json")
    # Design: the analysis's resolved is a clone (different object) sharing config.
    assert an_a.program.resolved is not r_a
    assert an_a.program.resolved.config is r_a.config
    tree = per_rank_weight_breakdown(
        r_a, TP8, analysis=an_a, artifacts=art_a)
    assert tree["total_bytes_per_rank"] > 0

    _, an_b, art_b = _analysis("Qwen__Qwen3-30B-A3B.json")
    with pytest.raises(ValueError):
        per_rank_weight_breakdown(
        r_a, TP8, analysis=an_b, artifacts=art_b)
