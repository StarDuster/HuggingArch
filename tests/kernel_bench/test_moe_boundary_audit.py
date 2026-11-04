"""MoE dispatch-boundary facts the 2026-10-25 audit found consumers
disagreeing on, now read off one record (`eval.routed.RoutedLaunch`, on a
row and a job as `routed`). Each test states a fact two consumers must
agree on.

One cell, read through the real plan entry point (`kernel_bench.plan.
build_jobs`): Qwen3-30B-A3B decode, B=8, EP=4 without DP-attention (every
rank holds all 8 tokens; 8 x 8 / 4 = 16 routed rows on the rank, M = 1/2 per
expert over 32 local experts, so the launch runs the 16 experts the rows
reach at one row each). Numbers from the config: d_model 2048,
moe_intermediate 768, 128 experts, top-8, BF16. The jobs built by hand at
the end exercise the job-carried record alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

STEM, MODEL = "Qwen__Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B"
D_MODEL, D_FFN, BPE = 2048, 768, 2


# The tests that build the real cell are per-model and slow (`regression`);
# the hand-built jobs at the end run by default.
@pytest.fixture(scope="module")
def jobs():
    from backend.arch.layout.parallel_config import ParallelConfig
    from kernel_bench.plan import build_jobs
    from tests.snapshots import resolve_snapshot

    parallel = ParallelConfig(tp_attention=4, tp_dense_ffn=4, ep_moe=4)
    out, _dropped = build_jobs(model_id=MODEL, resolved=resolve_snapshot(STEM),
                               batch_size=8, seq_len=1024, phase="decode",
                               avg_output_tokens=256, parallel=parallel)
    return out


def _job(jobs, op, suffix=""):
    return next(j for j in jobs
                if j["op"] == op and str(j["node_spec_id"]).endswith(suffix))


@pytest.mark.regression
def test_combine_flops_follow_the_rows_it_reduces(jobs):
    from backend.arch.eval.routed import RoutedLaunch
    job = _job(jobs, "combine")
    launch = RoutedLaunch.from_wire(job["routed"])
    # The walk bills combine one multiply-add per routed row's element on
    # the rank, the dispatch boundary's `routed_rows` — 16 here, not the
    # step's 8 x 8 token slots — and the launch's numerator is that law at
    # the launched rows.
    assert launch.routed_rows == 16
    assert job["flops_per_rank"] == launch.launch_rows * D_MODEL
    assert job["launch_rows_exact"] == str(launch.routed_rows)


@pytest.mark.regression
def test_geometry_numerator_reads_the_weights_the_launch_reads(jobs):
    """The padded launch touches only the experts that receive rows
    (`RoutedLaunch.touched_experts`), and the geometry numerator reads
    their weights: it used to pad every local expert to ceil(M) rows and
    read all of their weights, while the numerator kept the walk's
    Min(received, E)."""
    from kernel_bench.invocation import Invocation

    job = _job(jobs, "grouped_gemm", "down_proj")
    counts = Invocation.from_job(job).moe_token_counts
    touched, rows = sum(1 for n in counts if n), sum(counts)
    assert (touched, rows) == (16, 16)
    # Exactly: the touched experts' weights, the rows' input read and their
    # output write. Padding every local expert reads 32 experts' weights.
    weights = touched * D_MODEL * D_FFN * BPE
    assert job["access_bytes_per_rank"] == (
        weights + rows * D_FFN * BPE + rows * D_MODEL * BPE)


@pytest.mark.regression
def test_fused_gate_up_job_states_a_buildable_output(jobs):
    """A multi-output routed node (fused gate_up) launches its routed
    outputs at the launch's rows (`RoutedLaunch`, off the row's `routed`
    record): its per-port outputs used to be floored to 0 rows."""
    job = _job(jobs, "grouped_gemm", "gate_proj")
    rows = {tensor[1] for tensor in job["shape_out"]}
    assert rows == {job["dims"]["s_expert_rows"]}


def _routed_job(signature, dims, routed):
    return {"signature": signature, "op": signature, "dims": dims,
            "dtype_env": {"activation": "bf16"}, "phase": "decode",
            "kernel_params": {"moe_token_distribution": "balanced"},
            "routed": routed}


def test_a_routed_job_hands_its_kernel_the_walks_launch():
    """The token counts and the token side a provider builds are the job's
    own RoutedLaunch, which the cell's dims are the projection of."""
    from kernel_bench.invocation import Invocation
    routed = {"dispatch": 7, "source_rows": 8, "routed_rows": 16,
              "local_experts": 32}
    gemm = Invocation.from_job(_routed_job(
        "grouped_gemm", {"n_experts": 16, "s_expert_rows": 1, "n_tokens": 16,
                         "d_reduce": 8, "d_out": 8}, routed))
    assert gemm.moe_token_counts == (1,) * 16
    combine = Invocation.from_job(_routed_job(
        "moe_combine", {"B": 8, "s_query": 1, "n_experts": 16,
                        "s_expert_rows": 1, "n_tokens": 16, "d_model": 8},
        routed))
    assert combine.token_rows == 8


def test_a_cell_that_disagrees_with_its_routed_boundary_is_refused():
    from kernel_bench.invocation import Invocation
    routed = {"dispatch": 7, "source_rows": 8, "routed_rows": 16,
              "local_experts": 32}
    # Every local expert padded, the retired convention: 32 experts of one
    # row where the walk's launch touches 16.
    stale = Invocation.from_job(_routed_job(
        "grouped_gemm", {"n_experts": 32, "s_expert_rows": 1, "n_tokens": 32,
                         "d_reduce": 8, "d_out": 8}, routed))
    with pytest.raises(ValueError):
        _ = stale.moe_token_counts
    combine = Invocation.from_job(_routed_job(
        "moe_combine", {"B": 4, "s_query": 1, "n_experts": 16,
                        "s_expert_rows": 1, "n_tokens": 16, "d_model": 8},
        routed))
    with pytest.raises(ValueError, match="routed boundary holds"):
        _ = combine.token_rows
