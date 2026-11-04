"""A per-op row carries the function constants its launch takes (`consts`),
read off the walk (`Analysis.consts`) — before fusion on the `clamp` leaves,
after it on the fused clamped SwiGLU that absorbed them. Nothing in
`inference` reads a config or asks an op name for one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import shared_loader


def test_the_clamped_swiglu_rows_carry_the_models_bound():
    model_id = "deepseek-ai/DeepSeek-V4-Flash"
    path = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
            / (model_id.replace("/", "__") + ".json"))
    resolved = shared_loader().match(
        json.loads(path.read_text())["config"], model_id=model_id)
    result = analyze(resolved, parallel=ParallelConfig(),
                     framework_binding=get_binding("vllm"),
                     batch_size=1, avg_input_tokens=128, avg_output_tokens=0, breakdown=True)
    pre = [r for r in result.pre_per_op_timing if r["op"] == "clamp"]
    post = [r for r in result.per_op_timing
            if r["op"] == "silu_and_mul_with_clamp"]
    assert pre and post
    for row in pre + post:
        assert row["consts"] == {"clamp_limit": 10.0}, row["aggregation_id"]
    # An op that takes no constant has none on its row.
    assert not any("consts" in r for r in result.per_op_timing
                   if r["op"] == "linear")
