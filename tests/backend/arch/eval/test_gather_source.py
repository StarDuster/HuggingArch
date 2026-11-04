"""An index gather reads what it picks, not the tensor it picks from.

`OpSpec.gather_source` names the input an op's output is an index gather of
(`permute`'s routed rows, `gather`'s per-token pick), and the traffic law
bills that edge `min(input, output)` elements: one per element written, and
no element twice beyond the tensor itself.

The case that made it visible: expert parallelism without a request cut.
Every rank holds the step's whole token stream and routes `topk/ep` copies
per token to its own experts, so a rank's dispatch gathers fewer rows than
the stream has. Billed as reading the stream, Qwen3-Coder-Next's EP=16
prefill dispatch on H100-SXM read 1.06-1.10x the card's HBM peak (Kernel
Bench `torch_index_permute`, `index_select(bf16[32768,2048], idx[20480])`).

Every number below is written out from the config, not from the engine:
Qwen3-Coder-Next has d_model 2048, 512 experts, 10 per token, BF16.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot

STEM = "Qwen__Qwen3-Coder-Next"
D_MODEL, EXPERTS, TOPK, BPE = 2048, 512, 10, 2
TOKENS = 1024


def _permute_rank_reads(ep: int, *, stem: str = STEM,
                        d_model: int = D_MODEL) -> int:
    from backend.arch.layout.persistent import physical_artifacts
    from backend.arch.layout.physical_cost import physical_access
    from backend.arch.ops import is_dispatch_boundary
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step

    parallel = (ParallelConfig(ep_moe=ep, tp_attention=ep, tp_dense_ffn=ep)
                if ep > 1 else ParallelConfig())
    analysis = evaluate_step(resolve_snapshot(stem), parallel,
                             phase="prefill", batch=1, input_tokens=TOKENS,
                             output_tokens=0).analysis
    access = physical_access(analysis, artifacts=physical_artifacts(analysis),
                             scale_dtype_bytes=4.0)
    rows = [row for uid, row in access.values.items()
            if analysis.program.by_uid(uid).op == "permute"
            and is_dispatch_boundary(analysis.program.by_uid(uid).declaration)]
    assert rows, "no permute in the walk"
    return int(rows[0].act_in_bytes_per_rank)


#: `(ep, rows the rank's permute reads of the token stream)`. EP=16 first:
#: it is the case that reads above HBM peak when the cap is gone. Without
#: EP the routed rows (TOKENS x TOPK) outnumber the tokens and the read is
#: the stream, once — the half that must not move.
@pytest.mark.parametrize("ep, rows", [
    (16, TOKENS * TOPK // 16),     # 640 of the 1024 tokens
    (1, TOKENS),                   # 10240 picks, capped at the stream
])
def test_a_ranks_dispatch_reads_the_rows_its_experts_receive(ep, rows):
    routing = TOKENS * TOPK // ep * BPE    # one selected row number per output row
    assert _permute_rank_reads(ep) == rows * D_MODEL * BPE + routing


def test_a_dispatch_does_not_read_the_routers_candidate_scores():
    """Step-3.7's routing operand is the router's dense score tensor. The
    dispatch follows one selected row number per routed row; billing all 288
    candidate scores made `index_select` read above RTX-4090 HBM peak."""
    ep, d_model, topk = 32, 4096, 8
    rows = TOKENS * topk // ep
    assert _permute_rank_reads(
        ep, stem="stepfun-ai__Step-3.7-Flash", d_model=d_model) == \
        rows * d_model * BPE + rows * BPE


def test_a_topk_gather_reads_the_k_it_picks_of_each_token():
    """`gather` under the same law: the router's `[B, s, E]` scores, of
    which it picks `k` per token."""
    from backend.arch.eval import traffic as T
    from backend.arch.eval.transforms import Deployment
    from tests.backend.arch.eval.test_lookup_contract import (
        ACTIVE, _node, _program)

    analysis = Deployment(_program()).at(batch=2, input_tokens=8).analyze()
    edge = T.derive(analysis).values[_node(analysis, "gather").uid].in_edges[0]
    assert float(edge.elements) == 2 * 8 * ACTIVE
