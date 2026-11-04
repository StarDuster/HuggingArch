"""A row's measurement key states the dtype its traffic was billed at.

Two consumers read one operand: traffic bills its bytes, and the key
(`RowIdentity.dtype_env`) tells Kernel Bench what to build. They used to
answer from two places — traffic from the tensor's own declaration (QSA's
indexer logits are fp32, and the in-place scale over them reads fp32,
sglang `qsa/mqa.py:322`), the key from the role's activation dtype — so
Qwen3.8-Flash-Next's `index_reduce` was billed 4 bytes an element and
measured as a bf16 kernel, and its utilisation came out above the card's
peak.

The billed width is read here by a second route: the row's activation-read
bytes over its elements and its count, not the derivation's edge record.
"""
from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.dtypes import dtype_bytes
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_sweep import config_for_degree
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot


def _walk(model: str):
    return analyze(resolve_snapshot(model),
                   parallel=config_for_degree("tp_tp", 4),
                   framework_binding=get_binding("vllm"), batch_size=1,
                   avg_input_tokens=256, avg_output_tokens=0, phase="prefill",
                   breakdown=False)


def _row(walk, aggregation_id: str):
    row, = [r for r in walk.per_op_timing
            if r.get("aggregation_id") == aggregation_id]
    return row


def _billed_read_width(row) -> float:
    """Bytes per element the row's one input was billed at."""
    elements = math.prod(row["shape_in"]) * int(row["count_per_step"])
    return float(row["act_in_bytes_per_rank"]) / elements


@functools.cache
def _qwen():
    # Not a fixture: a failing test's traceback prints its arguments, and
    # rendering a whole walk outlives any timeout.
    return _walk("Qwen/Qwen3.8-Flash-Next")


def test_a_scale_over_fp32_logits_is_keyed_at_the_width_it_was_billed():
    qwen = _qwen()
    row = _row(qwen,
               "qwen_hc_full_block|self_attn.index_reduce__qwen_qsa_attention")
    # Plain values only: an assertion over `row` itself makes pytest render
    # the whole row, laws and all.
    env = RowIdentity(qwen).leaf_dtype_env(row)[0]
    keyed = float(dtype_bytes(json.loads(env)["activation"]))
    billed = _billed_read_width(row)
    assert billed == 4.0
    assert keyed == billed, env


def test_an_ordinary_elementwise_row_keeps_its_bf16_key():
    """Control: a scale over a bf16 activation edge is keyed as it was."""
    qwen = _qwen()
    row = _row(qwen,
               "qwen_hc_full_block|mlp_inject_scale__qwen_hc_full_block")
    billed = _billed_read_width(row)
    env = RowIdentity(qwen).dtype_env(row)
    assert billed == 2.0
    assert env == (
        '{"activation":"bf16","output":"bf16"}')


def test_an_fp8_gemm_keeps_the_dtype_it_runs_at():
    """Control: a block-FP8 GEMM fed a bf16 edge still runs at FP8 — its
    activation slot is the matmul's dtype and scale group, the conversion
    billed apart."""
    walk = _walk("MiniMaxAI/MiniMax-M2.7")
    env = RowIdentity(walk).dtype_env(
        _row(walk, "minimax_m2_decoder_block|self_attn.o_proj"))
    assert env == (
        '{"activation":"fp8_e4m3","activation_granularity":"per-group:128",'
        '"metadata":"fp32","output":"bf16",'
        '"weight":"fp8_e4m3","weight_granularity":"per-tile:128x128"}')


def test_a_top_k_selection_is_keyed_with_its_int32_output_not_declined():
    """`dsa_topk_mask` writes int32 indices (its declared `output_dtype`).
    The key states that, and int32 is in the corpus vocabulary, so the row
    is planned rather than declined: a top-k cell the plan cannot produce
    would shrink coverage silently. It reads the fp32 logits it selects
    from."""
    qwen = _qwen()
    row = _row(qwen,
               "qwen_hc_full_block|self_attn.index_topk__qwen_qsa_attention")
    env, declined = RowIdentity(qwen).leaf_dtype_env(row)
    assert declined == ""
    assert json.loads(env) == {"activation": "fp32", "output": "int32"}
