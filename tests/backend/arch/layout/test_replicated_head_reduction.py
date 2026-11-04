"""A sum finalized onto a row cut it does not scatter over is an all_reduce
and then a free local slice, as `lower_diff` lowers it.

Qwen3-30B-A3B has 32 query heads; at `tp_attention = 64` each head sits in
a replica cell of two ranks, so `o_proj`'s output is a sum pending over the
32-rank cross group (`cross_replica_family`). Sequence parallelism wants the
residual cut over all 64 ranks. The comm row emitted the all_reduce with
the sliced shape as its collective output, which no all_reduce produces,
so the materializer refused a deployment `validate()` and the walk accept.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.materialize_comm import PostTransform
from backend.arch.layout.materializer import collect_events
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import shared_loader

SNAP = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
        / "Qwen__Qwen3-30B-A3B.json")
BATCH, TOKENS = 2, 128


def _post_attention_reduction(config, parallel):
    resolved = shared_loader().match(config, model_id="Qwen/Qwen3-30B-A3B")
    step = evaluate_step(resolved, parallel, phase="prefill", batch=BATCH,
                         input_tokens=TOKENS, output_tokens=0)
    program = step.analysis.program
    (event,) = [e for e in collect_events(step.analysis)
                if e.op == "all_reduce" and program.by_uid(
                    e.placement_uid).display_path == "post_attn_add__block"]
    return event


@pytest.mark.parametrize("tp", (64,))
def test_sequence_parallel_past_the_head_count_slices_after_the_reduce(tp):
    config = json.loads(SNAP.read_text())["config"]
    heads = config["num_attention_heads"]
    assert tp > heads
    sp = _post_attention_reduction(
        config, ParallelConfig(tp_attention=tp, ep_moe=tp,
                               sequence_parallel=True))
    plain = _post_attention_reduction(
        config, ParallelConfig(tp_attention=tp, ep_moe=tp))
    assert sp.world == "tp_attention^cross"
    assert sp.group_size == heads
    assert tuple(sp.collective_out_shape) == tuple(sp.flow_in_shape)
    assert sp.post_transform == PostTransform.LOCAL_SLICE
    assert int(sp.flow_out_shape[1]) * tp == int(sp.flow_in_shape[1])
    # Second route: the sum moves the whole partial, as it does without
    # the row cut after it, and that partial is the config's residual.
    assert sp.payload_bytes == plain.payload_bytes
    assert sp.payload_bytes == BATCH * TOKENS * config["hidden_size"] * 2
