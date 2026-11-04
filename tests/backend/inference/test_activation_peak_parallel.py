"""What a rank's activation peak does under each parallelism — the
invariants that replaced `peak_unsharded // min(tp_attention, tp_dense_ffn)`.

That scalar answered every deployment with one division, and each case below
is a way it was wrong:

  * TENSOR parallelism does not divide the peak. The residual stream and every
    layernorm output are replicated on every rank, so the peak falls by less
    than the degree — 1.7x at tp=8 on Llama-3.2-1B, not 8x. The scalar
    under-reported by ~5x, and the activation peak is what OOM headroom is
    computed from.
  * CONTEXT parallelism does divide it, exactly, and the scalar never applied
    it at all — `min(tp_attention, tp_dense_ffn)` is 1 for a cp-only
    deployment, so Kimi-K2.5 at cp=8 was reported 8x too large.
  * EXPERT parallelism divides `n_expert_copies`, and the one live tensor
    that shrinks with it is the routed buffer: a rank gathers the rows ITS
    experts receive, so `ep` ranks each hold `1/ep` of the dispatch. The
    experts themselves are not smaller — a rank runs fewer of them, and the
    live set never held more than one at a time — so nothing else in the
    peak moves. (This bullet used to say `ep` left the peak alone entirely,
    which was true only while the dispatch DECLARED its output as one
    expert's average slice: at `1/n_experts` of the buffer the axis was
    already as small as `ep` could make it. The declaration states the
    buffer now, so the division is visible.)
  * A degree that does not divide its axis degrades. Three-way tensor
    parallelism over 128 heads is a 2-way split with the remainder in data
    parallelism, and that is what the run reports.

The peak is read off a per-rank WALK, so none of these are cases in the code —
they are consequences. The test pins the consequences.
"""
from __future__ import annotations

import json
from pathlib import Path


from tests.snapshots import snapshot_config


SNAPSHOT_DIR = (Path(__file__).resolve().parents[3]
                / "backend" / "arch" / "snapshots")
SEQ = 4096


def _resolve(snapshot: str):
    path = SNAPSHOT_DIR / f"{snapshot}.json"
    snapshot_config(snapshot)
    from tests.snapshots import shared_loader
    return shared_loader().match(json.loads(path.read_text())["config"], model_id=path.stem.replace("__", "/", 1))


def _peak(resolved, **parallel):
    from backend.inference.step_eval import estimate_peak_activation
    from backend.inference.framework_bindings import get_binding
    from backend.inference.parallel_config import ParallelConfig
    return estimate_peak_activation(
        resolved, batch=1, seq_len=SEQ,
        parallel=ParallelConfig(**parallel) if parallel else None,
        framework_binding=get_binding("vllm"),
    )


def test_tensor_parallelism_does_not_divide_the_peak():
    """Between 1x and the degree, strictly — the replicated residual is the
    floor and the sharded projections are the ceiling."""
    resolved = _resolve("meta-llama__Llama-3.2-1B")
    whole = _peak(resolved)["peak_activation_bytes_per_rank"]
    rank = _peak(resolved, tp_attention=8,
                 tp_dense_ffn=8)["peak_activation_bytes_per_rank"]
    ratio = whole / rank
    assert 1.0 < ratio < 8.0, (
        f"tp=8 reduced the activation peak {ratio:.2f}x. Exactly 8x means a "
        f"uniform divisor came back and the replicated residual is being "
        f"sharded; 1.0x means the walk never received the deployment."
    )


def test_context_parallelism_divides_the_peak_exactly():
    """The sequence is genuinely split, so this one IS the degree."""
    resolved = _resolve("meta-llama__Llama-3.2-1B")
    whole = _peak(resolved)["peak_activation_bytes_per_rank"]
    for cp in (2, 4):
        rank = _peak(resolved, cp=cp)["peak_activation_bytes_per_rank"]
        assert whole == rank * cp, (
            f"cp={cp}: expected the peak to divide exactly "
            f"({whole:,} / {cp}), got {rank:,}"
        )


def _routed_paths(resolved):
    """The display path of every tensor laid out `[n_expert_copies, M, ·]`:
    the buffer `permute` writes and every node of the bank it feeds, asked of
    the declarations (`Program.routed_uids`)."""
    from backend.arch.eval.program import Program
    from backend.arch.ops import DISPATCH_OP, is_dispatch_boundary

    program = Program.of(resolved)
    return {node.display_path for node in program.nodes()
            if node.uid in program.routed_uids
            or (is_dispatch_boundary(node.declaration)
                and node.op == DISPATCH_OP)}


def _routed_bytes(peak, paths):
    """The routed tensors' live bytes at the peak, 0 when none is in the
    peak's window. `top_live_at_peak` ids are `<template>|<path>`, a named
    port's `#port=<port>` after it."""
    return sum(int(row["bytes"]) for row in peak["top_live_at_peak"]
               if row["id"].split("|")[-1].split("#port=")[0] in paths)


def test_expert_parallelism_divides_the_routed_tensors_and_nothing_else():
    """`ep` partitions the expert axis. A rank holds fewer experts, each the
    size it always was — and it gathers only the rows ITS experts receive,
    so the live tensors that shrink are the ones led by the expert axis: the
    buffer `permute` writes and the bank's outputs over it.

    So the peak can fall by AT MOST their own reduction, and by exactly that
    while the MoE block is still the window holding the peak.
    """
    resolved = _resolve("Qwen__Qwen3-30B-A3B")
    paths = _routed_paths(resolved)
    base = _peak(resolved)
    whole = base["peak_activation_bytes_per_rank"]
    routed = _routed_bytes(base, paths)
    # The premise: at ep=1 the whole dispatch is on one rank, and the peak
    # is the bank's merged gate|up launch: it reads the permuted rows and
    # writes both halves, so three routed tensors are live. B=1, 4096
    # tokens, k=8 of 128 experts, BF16: 128 experts x 256 rows = 32,768
    # rows; the permuted rows are d_model 2048 wide, each half
    # d_moe_intermediate 768.
    assert routed == 32768 * (2048 + 2 * 768) * 2 == 234881024
    assert routed < whole

    previous = whole
    for ep in (2, 4, 8, 16):
        out = _peak(resolved, ep_moe=ep)
        rank = out["peak_activation_bytes_per_rank"]
        # The axis IS divided, and the run says so.
        assert out["divided_by"].get("n_expert_copies") == ep
        floor = whole - routed + routed // ep
        assert floor <= rank <= previous, (
            f"ep={ep}: the peak is {rank:,}; it may fall to at most "
            f"{floor:,} (the routed tensors' {routed:,} becoming "
            f"{routed // ep:,}) because expert parallelism moves no other "
            f"live tensor, and it may not rise")
        previous = rank
        if ep <= 2:
            # The MoE block still holds the peak, so the bound is tight.
            assert rank == floor, f"ep={ep}"
            assert out["peak_node_id"] == base["peak_node_id"]
            assert _routed_bytes(out, paths) == routed // ep
        else:
            # From ep=4 the rank's share of the routed tensors is small
            # enough that a different window (attention's) becomes the
            # peak, which the run reports by naming a different node. Past
            # that point the peak stops tracking them — not because the
            # division stopped.
            assert rank > floor
            assert out["peak_node_id"] != base["peak_node_id"]


def test_non_divisible_degree_reports_the_resolved_layout():
    """128 heads over 3 ranks stays replicated under the gcd policy."""
    from backend.arch.layout.physical import physical_axis_partitions
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step
    resolved = _resolve("deepseek-ai__DeepSeek-V3")
    assert resolved.dim_values.get("n_heads") == 128, "premise moved"
    analysis = evaluate_step(
        resolved, ParallelConfig(tp_attention=3),
        phase="prefill", batch=1, input_tokens=SEQ, output_tokens=0,
    ).analysis
    divided = physical_axis_partitions(analysis)
    assert "n_heads" not in divided, (
        f"n_heads reported as divided by {divided.get('n_heads')} "
        "at tp_attention=3, but the explicit layout can only partition by "
        "gcd(128, 3)=1"
    )


def test_divided_by_names_only_axes_this_model_has():
    """The policy's key set is registry-wide; the report is not."""
    resolved = _resolve("meta-llama__Llama-3.2-1B")
    divided = _peak(resolved, tp_attention=8, tp_dense_ffn=8)["divided_by"]
    assert divided, "a tp=8 dense model divides something"
    absent = {"n_ssm_heads", "n_vis_heads", "d_vis_ffn", "n_expert_copies"}
    assert not (set(divided) & absent), (
        f"reported axes {sorted(set(divided) & absent)} on a dense Llama, "
        f"which has none of them — the report describes the deployment "
        f"instead of the run"
    )


def test_no_deployment_reports_no_division():
    """Absence over invention: nothing described a split, so nothing is
    claimed to be split, and the peak is the whole model's."""
    resolved = _resolve("meta-llama__Llama-3.2-1B")
    assert _peak(resolved)["divided_by"] == {}
