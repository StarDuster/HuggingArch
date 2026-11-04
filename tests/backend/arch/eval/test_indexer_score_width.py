"""An indexer's scores land in HBM as fp32, whatever the activation dtype.

Every serving kernel that writes indexer scores allocates them float32:
DeepGEMM's MQA logits (vLLM `utils/deep_gemm.py:522-552`, used by
`sparse_attn_indexer.py`; sglang `dsa/dsa_indexer.py:1041`), the Qwen QSA
logits (vLLM `qwen4_exp/nvidia/ops/qsa_indexer.py:397-398`; sglang
`qsa/mqa.py:303`) and MiniMax-M3's block-max scores (vLLM
`minimax_m3/common/ops/index_topk.py:673-677`). The width is declared where the
tensor is made — the registered `indexer_logits` op, the `indexer_block_score`
plan — and an elementwise pass over the scores keeps it: sglang divides QSA's
logits by sqrt(d) in place (`qsa/mqa.py:322`), fp32 in and out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.fusion import Framework
from backend.arch.eval.program import Program
from backend.arch.eval.traffic import derive
from backend.arch.eval.transforms import Deployment
from backend.inference.framework_bindings import get_binding
from backend.arch.eval.fusion import Fusion
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import resolve_snapshot, shared_loader, shared_walk

FP32 = 4.0


def _analysis(program: Program, framework: str = ""):
    deployment = Deployment(program).at(batch=1, input_tokens=256)
    if framework:
        deployment = deployment.fused(
            Framework.of(get_binding(framework), FUSION_SCHEMES))
    return shared_walk(deployment)


def _rows(analysis, op: str):
    traffic = derive(analysis)
    return [(node, traffic.values[node.uid])
            for node in analysis.program.nodes()
            if node.op == op and not node.children]


def _written(row) -> float:
    term, = row[1].out_terms
    return term.width


def _read_from(analysis, reader_op: str, producer_uids) -> set:
    """The widths `reader_op` nodes read the edges from `producer_uids` at."""
    launch_of = {member: fused
                 for fused, members in (analysis.rewrite.origins or {}).items()
                 for member in members}
    out = set()
    for _node, traffic in _rows(analysis, reader_op):
        for edge in traffic.in_edges:
            if launch_of.get(edge.producer, edge.producer) in producer_uids:
                out.add(edge.width)
    return out


def test_dsa_logits_are_written_and_read_as_fp32():
    """DeepSeek-V3.2 writes the registered op itself: no fusion needed."""
    analysis = _analysis(Program.of(resolve_snapshot(
        "deepseek-ai/DeepSeek-V3.2", shared_loader())))
    logits = _rows(analysis, "indexer_logits")
    assert logits
    assert {_written(row) for row in logits} == {FP32}
    assert _read_from(analysis, "dsa_topk_mask",
                      {node.uid for node, _ in logits}) == {FP32}


@pytest.mark.parametrize("framework", ("vllm", "sglang"))
def test_fused_qsa_logits_and_their_in_place_scale_stay_fp32(framework):
    analysis = _analysis(Program.of(resolve_snapshot(
        "Qwen/Qwen3.8-Flash-Next", shared_loader())), framework)
    logits = _rows(analysis, "indexer_logits")
    assert logits
    assert {_written(row) for row in logits} == {FP32}
    scales = [row for row in _rows(analysis, "scale_const")
              if row[0].name == "index_reduce"]
    assert scales
    assert {_written(row) for row in scales} == {FP32}
    assert _read_from(analysis, "dsa_topk_mask",
                      {node.uid for node, _ in scales}) == {FP32}


def test_msa_block_scores_are_written_as_fp32():
    from backend.analyzer import fetch_config
    loader = shared_loader()
    program = Program.of(loader.resolve(
        loader.get("minimax_m3"), fetch_config("MiniMaxAI/MiniMax-M3"),
        model_id="MiniMaxAI/MiniMax-M3"))
    analysis = _analysis(program, "vllm")
    scores = _rows(analysis, "indexer_block_score")
    assert scores
    assert {_written(row) for row in scores} == {FP32}


def test_a_registered_op_states_its_width_once():
    """A plan may state its launch's output dtype only where no registered
    op does: `indexer_logits` states it on the op."""
    restated = FUSION_SCHEMES["indexer_logits"].model_copy(
        update={"output_dtype": "float32"})
    with pytest.raises(ValueError, match="registered op"):
        Fusion.named("indexer_logits", {"indexer_logits": restated})


def test_an_elementwise_op_writes_its_operands_promoted_width():
    """An elementwise op writes the widest of its operands' widths, an
    activation operand counting as the activation width: `add(logits, x)`
    and `add(x, logits)` are one tensor type, as torch's type promotion
    has it, and two activation operands stay at the activation width."""
    from types import SimpleNamespace as N
    from backend.arch.eval.traffic import declared_width

    def node(uid, op, *srcs):
        return N(uid=uid, op=op, children=(), provenance="", output_dtype=None,
                 inputs_obj=tuple(N(src=s) for s in srcs))

    logits, x = node(1, "indexer_logits"), node(2, "linear")
    assert declared_width(node(3, "add", logits, x)) == FP32
    assert declared_width(node(4, "add", x, logits)) == FP32
    assert declared_width(node(5, "add", x, x)) is None


def test_an_elementwise_op_computes_in_its_operands_promoted_dtype():
    """The key's activation slot of an elementwise op is the promoted type
    traffic computes, not whichever operand is first: `add(x, logits)`
    with a bf16 first operand runs, and is keyed, in fp32. Another op keeps
    its first operand; one that declares it reads indices has none."""
    from types import SimpleNamespace as N
    from backend.arch.eval.traffic import operand_dtype

    def node(uid, op, *srcs):
        return N(uid=uid, op=op, children=(), provenance="", output_dtype=None,
                 inputs_obj=tuple(N(src=s, port="") for s in srcs))

    def edges(*dtypes):
        return tuple(N(dtype=d) for d in dtypes)

    logits, x = node(1, "indexer_logits"), node(2, "linear")
    assert operand_dtype(node(3, "add", x, logits),
                         edges("BF16", "F32")) == "F32"
    assert operand_dtype(node(4, "add", x, x), edges("BF16", "BF16")) is None
    assert operand_dtype(node(5, "linear", x), edges("BF16")) == "BF16"
    assert operand_dtype(node(6, "embedding", x), edges("I32")) is None
