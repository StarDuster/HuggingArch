"""DeepSeek-V4.1's candidate-consumer indexer layers score only the candidates.

The paper's Reindex Mode (§2.3.2) and sglang's SM100 path
(`dsv4/v41_indexer/sparse_table.py:303-323`, `consume_decode` →
`sparse_logits` over the candidate schedule) score the `n_topk_blocks ·
s_sparse_block` candidate positions the candidate-source layer published, not
the whole compressed cache. So a consumer layer's indexer score is at most
`Min(⌊s_kv / ratio⌋, n_topk_blocks · s_sparse_block)` wide, and its index-key
read law is bounded the same way — a bounded score over an unbounded read law
is the inconsistency `selected_read_unbounded` names.
"""
from __future__ import annotations

import pytest

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, Deployment
from backend.arch.validate._selection_reads import run_selection_read_check
from tests.snapshots import resolve_snapshot, shared_loader

pytestmark = pytest.mark.regression  # whole-model walks of V4.1

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
CONSUMER = "csa2_indexer_shared_keys"


def _inside(node, component: str) -> bool:
    cursor = node.parent
    while cursor is not None:
        if cursor.op == component:
            return True
        cursor = cursor.parent
    return False


@pytest.fixture(scope="module")
def resolved():
    return resolve_snapshot(MODEL, shared_loader())


@pytest.mark.parametrize("cache_len", (4096, 262144))
def test_a_consumer_layer_scores_at_most_the_candidate_pool(resolved, cache_len):
    dims = resolved.dim_values
    pool = int(dims["n_topk_blocks"]) * int(dims["s_sparse_block"])
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=cache_len)
                .during(DECODE, cache_len=cache_len).analyze())
    scores = [node for node in analysis.program.nodes()
              if node.op == "attention_score" and _inside(node, CONSUMER)]
    assert scores
    for node in scores:
        ratio = int(value_of(analysis.node_dims.values[node.uid]["compress_ratio"]))
        width = int(value_of(analysis.shapes.values[node.uid].out[-1]))
        assert width == min(cache_len // ratio, pool)
        # The keys it reads are the pool's, not the compressed cache's.
        keys = analysis.shapes.values[node.uid].ins[1]
        assert int(value_of(keys[-2])) == min(cache_len // ratio, pool)
    assert run_selection_read_check(analysis) == []
