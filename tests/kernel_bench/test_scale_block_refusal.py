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
