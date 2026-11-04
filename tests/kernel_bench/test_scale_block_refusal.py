"""The block-scale shard gate (`kernel_bench.plan._scale_block_refusal`)
against the walk's own rows — default-run, where `test_kernel_bench.py` is
not: the facts it reads (`weight_shards`) are the walk's, and a defect that
refuses a loadable cell or plans an unloadable one shows here first."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.mark.parametrize("stem,tp,precision,leaf,refused", [
    # A convolution's taps are no matrix a GEMM loader tiles: under a
    # per-group weight preset GLM-5.3-Flash's KDA conv must not be
    # refused for its 4-tap width (`declaration_is_gemm`).
    ("zai-org__GLM-5.3-Flash", 8, "w_nvfp4_w4a4_kv_fp8",
     "causal_conv1d_packed", False),
    # A routed expert bank's rows, cut per expert: DeepSeek-V3's
    # gate_proj at TP 32 holds 64 of each expert's 2048 rows, half a
    # 128-wide block — refused, whatever the bank's expert count.
    ("deepseek-ai__DeepSeek-V3", 32, "as_shipped",
     "mlp.experts.gate_proj", True),
])
def test_the_walks_rows_are_refused_where_a_loader_would(
        stem, tp, precision, leaf, refused):
    from tests.snapshots import resolve_snapshot
    from backend.arch.precision_profile import PRESETS
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze
    from kernel_bench.plan import _scale_block_refusal
    walk = analyze(resolve_snapshot(stem), parallel=ParallelConfig(
        tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp),
        framework_binding=get_binding("vllm"), batch_size=1,
        avg_input_tokens=256, avg_output_tokens=16, phase="decode",
        breakdown=False, precision=PRESETS[precision])
    rows = [r for r in walk.per_op_timing
            if r["op"] == leaf or r["aggregation_id"].endswith(leaf)]
    assert rows, leaf
    assert {bool(_scale_block_refusal(r)) for r in rows} == {refused}, [
        (r["aggregation_id"], r["weight_shards"]) for r in rows]

@pytest.mark.regression  # two walks (gpt-oss, DeepSeek-V3)
def test_the_walk_publishes_each_packed_partition_and_borrowed_matrix():
    # The facts the refusal reads, from the walk: gpt-oss's packed
    # gate|up bank is one tensor in two partitions, each cut on its own;
    # DeepSeek-V3's absorbed attention multiplies `kv_b_proj`'s matrix,
    # threaded in along an edge, whose heads TP cuts.
    from tests.snapshots import resolve_snapshot
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze

    def rows(stem, tp):
        walk = analyze(resolve_snapshot(stem), parallel=ParallelConfig(
            tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp),
            framework_binding=get_binding("vllm"), batch_size=1,
            avg_input_tokens=256, avg_output_tokens=16, phase="decode",
            breakdown=False)
        return {r["aggregation_id"].split("|")[-1]: r["weight_shards"]
                for r in walk.per_op_timing}

    gpt = rows("openai__gpt-oss-20b", 4)["mlp.experts.gate_up_proj"]
    assert [(s["rows"], s["rows_local"]) for s in gpt] == [(2880, 720)] * 2
    assert len({tuple(s["pack"]) for s in gpt}) == 1
    dsv3 = rows("deepseek-ai__DeepSeek-V3", 8)["self_attn.v_absorbed"]
    assert [(s["rows"], s["rows_local"]) for s in dsv3] == [(32768, 4096)]


@pytest.mark.parametrize("stem,checkpoint_tp,tp,refused,framework", [
    # Default case catches the original false refusal: logical K=192 is
    # inside one N=3392 fused checkpoint chunk, not an independent FP8 grid.
    ("XiaomiMiMo__MiMo-V2.5", 4, 4, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 1, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 2, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 8, True, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 8, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 4, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 2, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 1, False, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 16, True, "sglang"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 4, False, "vllm"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 2, False, "vllm"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 1, False, "vllm"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 8, False, "vllm"),
    ("XiaomiMiMo__MiMo-V2.5", 4, 16, False, "vllm"),
    ("XiaomiMiMo__MiMo-V2.5-Pro", 8, 16, False, "vllm"),
])
def test_fused_checkpoint_chunks_own_the_fp8_grid(
        stem, checkpoint_tp, tp, refused, framework):
    from tests.snapshots import resolve_snapshot
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze
    from kernel_bench.plan import _scale_block_refusal

    walk = analyze(resolve_snapshot(stem), parallel=ParallelConfig(
        tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp),
        framework_binding=get_binding(framework), batch_size=1,
        avg_input_tokens=256, avg_output_tokens=16, phase="decode",
        breakdown=False)
    rows = [r for r in walk.per_op_timing
            if r["aggregation_id"].endswith("self_attn.qkv_proj")]
    assert rows
    for row in rows:
        shards = row["weight_shards"]
        assert len(shards) == 3  # logical Q, K, V placement stays intact
        assert {s["block_scale_layout"]["checkpoint_tp"] for s in shards} == {checkpoint_tp}
        assert {s["block_scale_layout"]["runtime_tp"] for s in shards} == {tp}
        assert {s["block_scale_layout"]["rows_local"] for s in shards} == {
            sum(s["rows_local"] for s in shards)}
        why = _scale_block_refusal(row, framework=framework)
        assert bool(why) == refused, (row["aggregation_id"], why)
        if refused:
            assert f"checkpoint is TP={checkpoint_tp}-interleaved" in why

    if refused:
        return
    # The checkpoint header's grids and the grids read by the kernel are
    # distinct at TP < checkpoint TP: SGLang deinterleaves and requantizes.
    from backend.arch.layout.persistent import physical_artifacts
    all_artifacts = physical_artifacts(walk.analysis)
    primaries = {a.node_uid: a.runtime_shape for a in all_artifacts
                 if a.surface == "stored"}
    artifacts = [a for a in all_artifacts
                 if a.logical_name.endswith("self_attn.qkv_proj.weight_scale_inv")]
    assert artifacts
    if checkpoint_tp % tp:
        assert {a.surface for a in artifacts} == {"resident_metadata"}
        checkpoint_grids = [tuple(r.tensor.shape) for r in walk.analysis.iter_checkpoint()
                            if r.tensor.name.endswith("self_attn.qkv_proj.weight_scale_inv")]
        assert (108 if checkpoint_tp == 4 else 216,
                32 if checkpoint_tp == 4 else 48) in checkpoint_grids
    for artifact in artifacts:
        global_n, k_blocks = artifact.global_shape
        if artifact.surface == "saved_metadata":
            if checkpoint_tp % tp == 0:
                assert artifact.runtime_shape == (global_n // tp, k_blocks)
        else:
            # Both MiMo checkpoints have N=3392 per original chunk in
            # full attention; V2.5 SWA has N=3712 per original chunk.
            n = primaries[artifact.node_uid][-2]
            assert artifact.runtime_shape == ((n + 127) // 128, k_blocks)
            assert global_n == artifact.runtime_shape[0] * tp
            resident_rows = [r for r in walk.analysis.iter_resident()
                             if r.uid == artifact.node_uid and r.is_metadata]
            assert [tuple(r.tensor.shape) for r in resident_rows] == [artifact.global_shape]


def test_interleaved_load_contract_preserves_other_scale_guards():
    from kernel_bench.plan import _scale_block_refusal

    def row(*, width=4096, width_local=4096, runtime_tp=4, layout=True):
        shard = dict(pack=[0, 0], rows=768, rows_local=192,
                     width=width, width_local=width_local,
                     granularity="per-tile:128x128")
        if layout:
            shard["block_scale_layout"] = dict(
                kind="tp_interleaved", checkpoint_tp=4, runtime_tp=runtime_tp,
                world="tp_attention", loaders={"sglang": "whole_chunks"}, rows_local=3392)
        return {"weight_shards": [shard]}

    assert _scale_block_refusal(row(), framework="sglang") is None
    assert "d_out=192" in _scale_block_refusal(row(), framework="vllm")
    assert "d_out=192" in _scale_block_refusal(row(layout=False), framework="sglang")
    assert "d_reduce=192" in _scale_block_refusal(
        row(width=768, width_local=192), framework="sglang")
    assert "TP=8" in _scale_block_refusal(row(runtime_tp=8), framework="sglang")


@pytest.mark.parametrize("framework,tp,refused", [
    ("sglang", 4, False),
    ("sglang", 8, True),
    ("vllm", 8, False),
])
def test_the_planner_uses_the_selected_fused_weight_loader(framework, tp, refused):
    from tests.snapshots import resolve_snapshot
    from kernel_bench.plan import build_jobs_with_unplaced

    jobs, _, unplaced = build_jobs_with_unplaced(
        model_id="XiaomiMiMo/MiMo-V2.5", resolved=resolve_snapshot("XiaomiMiMo__MiMo-V2.5"),
        framework=framework, tp=tp, batch_size=1, seq_len=256,
        phase="decode", avg_output_tokens=16)
    qkv_jobs = [job for job in jobs if job["node_spec_id"].endswith("self_attn.qkv_proj")]
    qkv_unplaced = [record for record in unplaced
                   if record["aggregation_id"].endswith("self_attn.qkv_proj")]
    assert bool(qkv_jobs) != refused
    assert bool(qkv_unplaced) == refused
    if refused:
        assert all("TP=4-interleaved" in record["detail"] for record in qkv_unplaced)


def test_block_scale_layout_is_a_validated_fp8_contract():
    from pydantic import ValidationError
    from backend.arch.schema.quant import BlockScaleLayout, QuantContext

    with pytest.raises(ValidationError, match="greater than 0"):
        BlockScaleLayout(kind="tp_interleaved", checkpoint_tp=0,
                         world="tp_attention", loaders={"sglang": "whole_chunks"})
    with pytest.raises(ValidationError, match="requires method: fp8"):
        QuantContext(per_role={"attention": dict(
            method="none", block_scale_layout=dict(
                kind="tp_interleaved", checkpoint_tp=4, world="tp_attention",
                loaders={"sglang": "whole_chunks"}))})


def test_loader_formats_preserve_theory_and_checkpoint_grids():
    from tests.snapshots import resolve_snapshot
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    from backend.arch.layout.persistent import physical_artifacts, whole_resident

    theory = analyze(resolve_snapshot("XiaomiMiMo__MiMo-V2.5"),
                     parallel=ParallelConfig(), batch_size=1,
                     avg_input_tokens=256, avg_output_tokens=16, phase="decode",
                     breakdown=False).analysis

    def full_grid(analysis, *, checkpoint=False):
        rows = analysis.iter_checkpoint() if checkpoint else analysis.iter_resident()
        return [tuple(r.tensor.shape) for r in rows
                if r.tensor.name.endswith("self_attn.qkv_proj.weight_scale_inv")
                and r.tensor.shape[0] != 116]

    assert set(full_grid(theory)) == {(108, 32)}
    theory_bytes = whole_resident(physical_artifacts(theory))
    for framework in ("sglang", "vllm"):
        loaded = theory.with_load_formats(framework)
        assert set(full_grid(loaded)) == {(106, 32)}
        assert set(full_grid(loaded, checkpoint=True)) == {(108, 32)}
        assert whole_resident(physical_artifacts(loaded)) < theory_bytes
    assert set(full_grid(theory)) == {(108, 32)}
    assert theory.with_load_formats("unregistered-framework") is theory
