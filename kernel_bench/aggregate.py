"""Block-level aggregation of measured per-op rows.

The point of the whole per-kernel-impl graph: a fused block's REAL MFU is
Σ(useful GEMM flops) / Σ(time of ALL its ops — the grouped GEMMs *and* the
memory-bound dispatch, activation, norm around them). That block MFU sits well
below any single GEMM's MFU, because permute/combine/act move memory with no
useful matmul FLOPs. This rolls a measured CSV up to per-block MoE numbers so
the dispatch drag is visible.

Grouping is on the corpus identity we have today (node_spec_id =
"template|display_path"), so the block id is the template. A MoE-block row is
one the plan flagged `moe_block_member` — a node the declared dispatch zone
contains, or its permute/combine boundary — no longer an expert/moe name match.
"""
from __future__ import annotations

from backend.inference import benchmarks as _benchmarks

import json
from collections import defaultdict
from typing import Any, Dict, List

from backend.inference.benchmark_identity import declares_routed_token_total
from backend.inference.op_signatures import get_signature, op_signature

def _block_key(node_spec_id: str) -> str:
    # template (before "|") = the transformer block; display_path (after) = the
    # leaf module. Group MoE leaves by their template.
    return (node_spec_id or "").split("|", 1)[0]


def _is_moe_row(r: Dict[str, str]) -> bool:
    # The plan stamps `moe_block_member` on every row from the DECLARED dispatch
    # zone — an expert-bank node (asked of the program, so an expert leaf whose
    # own flag is unset still answers via its bank ancestor) or a permute /
    # combine boundary of it. Read as a "1"/"" string. A schema-7 CSV predates
    # the column and reads as empty, yielding no MoE rows until re-emitted.
    return bool(r.get("moe_block_member"))


def _grouped_flops(r: Dict[str, str]) -> float:
    # Useful matmul flops of an expert GEMM, over the rows its cell says the
    # launch runs. `n_tokens` is the ROUTED ROW TOTAL the cell carries
    # (`declares_routed_token_total`); `s_expert_rows` beside it is the width
    # the per-expert buffer is PADDED to, so `n_experts x s_expert_rows` — what
    # this multiplied before — counts the padding as work wherever the mean
    # rows per expert is not a whole number: 512 rows where DeepSeek-V4-Flash's
    # batch-64 decode routes 384. The MFU it feeds reads a third high.
    #
    # Which rows are expert GEMMs is the signature's declaration: its cell
    # carries the routed token total AND its contract multiplies a weight.
    # The dispatch boundaries carry the total too (a curve of them runs along
    # the routed rows) but move rows and multiply nothing. Such a row's shape
    # must state its whole cell: a shape that does not parse, is not an
    # object or lacks an axis is a broken row, and counting it as zero work
    # raised the block's MFU. Every other row of the block does no useful
    # GEMM work.
    signature = op_signature(r["op"])
    sig = get_signature(signature)
    if (sig is None or not declares_routed_token_total(signature)
            or not any(layout.role == "weight" for layout in sig.inputs)):
        return 0.0
    where = f"{r.get('model')} [{r.get('node_spec_id')}]"
    try:
        d = json.loads(r["shape_json"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: shape_json does not parse: {exc}") from exc
    missing = (sorted(set(sig.cell_dims) - set(d))
               if isinstance(d, dict) else None)
    if missing is None or missing:
        raise ValueError(
            f"{where}: a {signature} row's shape_json {r['shape_json']!r} "
            f"does not state its cell (missing {missing or 'an object'})")
    return 2.0 * d["n_tokens"] * d["d_reduce"] * d["d_out"]


def moe_block_mfu(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """One record per (model, gpu, phase, parallel, block, framework): the
    block's summed op time, useful GEMM flops, and resulting MFU."""
    groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        # A row recording an outcome has no p50 and no peaks, but it DOES
        # have a shape — so left in, its expert GEMM's full FLOPs would be
        # charged against the other ops' time and the block's MFU would rise
        # without bound. The comment below already assumed such a row does
        # not exist ("they OOM'd … so the dispatch-only sum isn't a real
        # efficiency"); recording refusals is what made the assumption false.
        if not _benchmarks.is_measurement(r):
            continue
        if not _is_moe_row(r):
            continue
        key = (r.get("model", ""), r.get("gpu", ""), r.get("phase", ""),
               r.get("parallel", ""), _block_key(r.get("node_spec_id", "")),
               r.get("framework", ""))
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for key, grp in sorted(groups.items()):
        time_us = sum(float(r.get("p50_us") or 0.0) for r in grp)
        gemm_flops = sum(_grouped_flops(r) for r in grp)
        peak_tf = max((float(r.get("peak_tflops_at_test") or 0.0) for r in grp),
                      default=0.0)
        # block_mfu is None (not 0) when the block is INCOMPLETE — no GEMM rows
        # measured (e.g. they OOM'd or failed cross-check), so the dispatch-only
        # sum isn't a real efficiency. 0% would falsely read as "ran at 0 MFU".
        mfu = None
        if time_us > 0 and peak_tf > 0 and gemm_flops > 0:
            mfu = (gemm_flops / (time_us * 1e-6)) / (peak_tf * 1e12)
        out.append({
            "model": key[0], "gpu": key[1], "phase": key[2], "parallel": key[3],
            "block": key[4], "framework": key[5],
            "block_time_us": round(time_us, 2),
            "gemm_tflops_useful": round(gemm_flops / 1e12, 4),
            "block_mfu": None if mfu is None else round(mfu, 4),
            "n_ops": len(grp),
            "ops": sorted({r.get("op", "") for r in grp}),
        })
    return out
