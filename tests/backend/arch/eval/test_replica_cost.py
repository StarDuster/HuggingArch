"""`Analysis.replica_cost`: one data-parallel replica's unsharded cost."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import pytest

from backend.arch.eval.dim import value_of
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import resolve_snapshot

#: A bank of two shared experts (a component whose cost is twice its
#: children's), an MLA MoE and a GQA MoE.
MODELS = ("deepseek-ai/DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V3",
          "Qwen/Qwen3-30B-A3B")


def _analysis(model_id, parallel):
    return evaluate_step(resolve_snapshot(model_id), parallel, phase="prefill",
                         batch=1, input_tokens=512, output_tokens=1).analysis


@pytest.mark.parametrize("model_id", MODELS)
def test_without_data_parallelism_a_replica_is_the_whole(model_id):
    analysis = _analysis(model_id, ParallelConfig(tp_attention=2, tp_dense_ffn=2, ep_moe=2))
    replica = analysis.replica_cost
    assert set(replica.values) == set(analysis.cost.values)
    for uid, cost in analysis.cost.values.items():
        assert value_of(replica.values[uid].flops) == value_of(cost.flops), uid


@pytest.mark.parametrize("model_id", MODELS)
def test_a_replica_does_not_depend_on_how_many_replicas_pool(model_id):
    one = _analysis(model_id, ParallelConfig(tp_attention=2, tp_dense_ffn=2, ep_moe=2))
    pooled = _analysis(model_id, ParallelConfig(dp_attention=2, ep_moe=2))
    roots = {part.node.uid for part in one.program.contributions}
    for uid in roots & set(pooled.cost.values):
        assert (value_of(pooled.replica_cost.values[uid].flops)
                == value_of(one.replica_cost.values[uid].flops)), uid
