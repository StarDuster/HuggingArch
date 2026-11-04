"""A fused kernel's per-rank frame is its first origin's, projected where
that origin ran.

A fused node's surfaces join its members' axes, and under a layout the
members need not agree on one name's per-rank extent: vLLM's Gated DeltaNet
prefill prep (`fused_gdn_post_conv_prep`) takes q|k|v after the conv, which
CP gathers for the recurrence, beside the gate inputs, which stay on the
rank's tokens until the recurrence gathers them. Projected over the fused
node, `s_query` had two per-rank values and every CP layout of every GDN
model failed (the Gallery frontier's CP·EP rows for Qwen3.5 / Qwen3.8 /
Qwen3-Coder-Next, 2026-10-29).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_sweep import config_for_degree

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
_CP = 4
_SEQ = 1024


def test_gdn_prep_under_cp_takes_its_first_origins_frame():
    doc = json.loads((SNAP_DIR / "Qwen__Qwen3.5-35B-A3B.json").read_text())
    result = estimate_capacity(
        config=doc["config"], model_id=doc["model_id"], gpu_sku="H100-SXM",
        num_gpus=_CP, avg_input_tokens=_SEQ, avg_output_tokens=256,
        gpu_memory_fraction=0.9, parallel=config_for_degree("cp_ep", _CP),
        framework="vllm", decode_batch_size=1, prefill_batch_size=1)
    rows = [row for row in result["per_op_timing"]
            if row.get("op") == "fused_gdn_post_conv_prep"]
    assert rows, "vLLM's GDN prefill prep did not fuse under CP"
    row = rows[0]
    qkv, dt_raw = row["shape_in"][0], row["shape_in"][1]
    # The members disagree on the rank: q|k|v gathered, the gates cut.
    assert qkv[1] == _SEQ
    assert dt_raw[1] == _SEQ // _CP
    # The frame is the q|k|v split's (the first origin), on its own axes.
    assert row["node_dims"]["s_query"] == _SEQ
