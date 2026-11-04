"""A sparse-attention indexer's score matrix never lands in HBM.

Every serving kernel for a lightning indexer scores the query against the
index keys and reduces the scores inside the launch: deep_gemm's MQA logits
sum the heads (DSA, DeepSeek-V4, the GLM kpool drafts — weighted; Qwen QSA —
unweighted), MiniMax-M3's MSA kernels max-pool each head's scores over key
blocks. The spec writes the indexer as plain torch ops, so the fusion plans are
what price it as that kernel. A form no plan matches is billed for writing and
re-reading the full [heads, s_query, s_kv] matrix — at 1M tokens the
difference between a fused 22 s indexer chain and an unfused 250 s one.

Which nodes are an indexer's scores is read off declarations, not names: the
component's `cache.entries` say which tensor is an `index_key`, and the score
op says (`OpSpec.input_roles`) which of its operands is the key.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, List, Set, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.cache import is_view
from backend.arch.eval.fusion import Framework
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import shared_loader, shared_program, shared_walk

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
FRAMEWORKS = ("vllm", "sglang")


def _snapshot_program(stem: str) -> Program:
    return shared_program(stem.replace('__', '/', 1))


def _minimax_m3_program() -> Program:
    """No snapshot: the spec is resolved from the cached config, as
    `test_rotary_positions` does."""
    from backend.analyzer import fetch_config
    loader = shared_loader()
    return Program.of(loader.resolve(
        loader.get("minimax_m3"), fetch_config("MiniMaxAI/MiniMax-M3"),
        model_id="MiniMaxAI/MiniMax-M3"))


def _subtree(node: Any) -> Iterator[int]:
    yield node.uid
    for child in node.children or ():
        yield from _subtree(child)


def _index_key_uids(program: Program) -> Set[int]:
    """Every node of every child a component declares as an `index_key`
    cache entry (a compressor sub-component included)."""
    out: Set[int] = set()
    for node in program.nodes():
        kinds = {e.name: e.kind for e in (node.kv_entries or ())}
        for child in node.children or ():
            if kinds.get(child.name) == "index_key":
                out |= set(_subtree(child))
    return out


def _declares_an_indexer(program: Program) -> bool:
    return any(e.kind == "index_key"
               for node in program.nodes() for e in (node.kv_entries or ()))


def indexer_scores(program: Any) -> Set[int]:
    """The score nodes whose KEY operand is an indexer's cached key: walk
    downstream from the `index_key` tensors and stop at the first op that
    declares operand roles; it is a score of the indexer when the key
    arrives at its `k` position."""
    consumers: dict = {}
    for node in program.walked_nodes():
        for position, ref in enumerate(node.inputs_obj or ()):
            if ref.src is not None:
                consumers.setdefault(ref.src.uid, []).append((node, position))
    found: Set[int] = set()
    seen: Set[int] = set()
    todo = list(_index_key_uids(program))
    while todo:
        uid = todo.pop()
        if uid in seen:
            continue
        seen.add(uid)
        for node, position in consumers.get(uid, ()):
            roles = (tuple(node.declaration.input_roles or ())
                     if node.declaration is not None else ())
            if not roles:
                todo.append(node.uid)
            elif position < len(roles) and roles[position] == "k":
                found.add(node.uid)
    return found


def _framework(name: str) -> Framework:
    return Framework.of(get_binding(name), FUSION_SCHEMES)


def _shape_key(shape: Any) -> Tuple[str, ...]:
    return tuple(str(axis) for axis in shape.out)


def escaped_score_matrices(program: Program, framework: str) -> List[str]:
    """Indexer scores left standing after the framework's rewrite, and live
    nodes that write a tensor of a score's shape (the scores re-materialised
    by an op the kernel was supposed to absorb)."""
    deployment = Deployment(program).at(batch=1, input_tokens=256)
    walked = shared_walk(deployment)
    scores = indexer_scores(walked.program)
    fused = shared_walk(deployment.fused(_framework(framework)))
    problems: List[str] = []
    score_shapes = {}
    for uid in scores:
        node = walked.program.by_uid(uid)
        if uid not in fused.rewrite.dropped:
            problems.append(f"standalone score {node.display_path or node.name}")
        shape = walked.shapes.values.get(uid)
        if shape is not None and shape.out is not None:
            score_shapes[_shape_key(shape)] = node.display_path or node.name
    for node in fused.program.walked_nodes():
        # A view re-indexes and writes nothing (DeepSeek-V4 broadcasts its
        # [B, S, H] head weights to the score's shape for the `mul`).
        if node.children or is_view(node):
            continue
        shape = fused.shapes.values.get(node.uid)
        if shape is None or shape.out is None:
            continue
        if _shape_key(shape) in score_shapes:
            problems.append(
                f"{node.op} {node.display_path or node.name} writes the score "
                f"matrix of {score_shapes[_shape_key(shape)]}")
    return sorted(set(problems))


# -- every indexer in the corpus --------------------------------------------

def _indexer_models() -> List[str]:
    return sorted(p.stem for p in SNAPSHOTS.glob("*.json"))


@pytest.mark.regression  # walks every snapshot
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_no_indexer_score_matrix_reaches_hbm(framework):
    checked = []
    failures = {}
    for stem in _indexer_models():
        program = _snapshot_program(stem)
        if not _declares_an_indexer(program):
            continue
        checked.append(stem)
        problems = escaped_score_matrices(program, framework)
        if problems:
            failures[stem] = problems
    program = _minimax_m3_program()
    checked.append("MiniMaxAI/MiniMax-M3")
    problems = escaped_score_matrices(program, framework)
    if problems:
        failures["MiniMaxAI/MiniMax-M3"] = problems
    assert len(checked) > 1
    assert failures == {}


# -- the two forms the plans cover ------------------------------------------

def _absorbed_ops(fused: Any, walked: Any, fused_op: str) -> List[Tuple[str, ...]]:
    out = []
    for uid, origins in fused.rewrite.origins.items():
        if fused.program.by_uid(uid).op == fused_op:
            out.append(tuple(sorted(walked.program.by_uid(o).op
                                    for o in origins)))
    return out


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_the_unweighted_qsa_indexer_is_one_mqa_logits_launch(framework):
    """Qwen QSA sums ReLU(q·k) over heads with no head weights (sglang
    qsa/mqa.py:168-169, vLLM qwen4_exp/nvidia/ops/qsa_indexer.py:208):
    the `indexer_logits` plan's `mul` by W is optional, so the chain fuses.
    The trailing 1/sqrt(d) stays its own op — sglang's prefill runs it as a
    separate pass (mqa.py:322)."""
    deployment = Deployment(_snapshot_program("Qwen__Qwen3.8-Flash-Next")).at(
        batch=1, input_tokens=128)
    walked = shared_walk(deployment)
    fused = shared_walk(deployment.fused(_framework(framework)))
    absorbed = _absorbed_ops(fused, walked, "indexer_logits")
    assert absorbed
    assert set(absorbed) == {("attention_score", "relu", "sum_reduce")}
    live = {n.op for n in fused.program.walked_nodes()
            if not n.children and (n.name or "").endswith("index_reduce")}
    assert live == {"scale_const"}


def test_the_weighted_dsa_indexer_still_absorbs_its_head_weights():
    deployment = Deployment(_snapshot_program("deepseek-ai__DeepSeek-V4-Flash")).at(
        batch=1, input_tokens=128)
    walked = shared_walk(deployment)
    fused = shared_walk(deployment.fused(_framework("vllm")))
    absorbed = _absorbed_ops(fused, walked, "indexer_logits")
    assert absorbed
    assert set(absorbed) == {("attention_score", "mul", "relu", "sum_reduce")}


# -- MiniMax-M3 MSA: per-head block max, per-head block top-k ---------------

@pytest.fixture(scope="module")
def msa():
    deployment = Deployment(_minimax_m3_program()).at(batch=1, input_tokens=4096)
    return deployment, shared_walk(deployment)


def _msa_node(analysis: Any, name: str) -> Any:
    for node in analysis.program.walked_nodes():
        if not node.children and (node.name or "").split(".")[-1] == name \
                and node.template == "minimax_m3_moe_block":
            return node
    raise AssertionError(f"no MSA node {name!r}")


def test_msa_selects_blocks_per_index_head(msa):
    """HF `MiniMaxM3VLIndexer.forward` and vLLM `minimax_m3_index_score`:
    scores [B, H_idx, S_q, S_k] → amax over each 128-key block →
    [B, H_idx, S_q, S_k/128] → per-head top-16 blocks. No head sum."""
    _deployment, analysis = msa
    dims = analysis.binding
    heads, block, topk = (int(dims["n_indexer_heads"]),
                          int(dims["s_sparse_block"]), int(dims["n_topk_blocks"]))
    pooled = _msa_node(analysis, "index_block_max")
    picked = _msa_node(analysis, "index_topk")
    assert pooled.op == "group_max" and picked.op == "topk"
    assert tuple(int(a) for a in analysis.shapes.values[pooled.uid].out) == (
        1, heads, 4096, 4096 // block)
    assert tuple(int(a) for a in analysis.shapes.values[picked.uid].out) == (
        1, heads, 4096, topk)
    # one max per score element
    assert int(analysis.cost.values[pooled.uid].flops) == heads * 4096 * 4096
    in_indexer = {n.op for n in analysis.program.walked_nodes()
                  if not n.children and n.template == "minimax_m3_moe_block"
                  and (n.name or "").split(".")[-1].startswith("index_")}
    assert "sum_reduce" not in in_indexer and "relu" not in in_indexer


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_msa_scores_and_block_max_are_one_launch(msa, framework):
    deployment, walked = msa
    fused = shared_walk(deployment.fused(_framework(framework)))
    assert set(_absorbed_ops(fused, walked, "indexer_block_score")) == {
        ("attention_score", "group_max")}
    assert escaped_score_matrices(deployment.program, framework) == []


def test_msa_per_head_selection_is_laid_out_with_the_kv_heads():
    """Under attention TP the four KV heads split over the ranks while the
    indexer stays whole (vLLM `ReplicatedLinear`): each rank reads the
    selections of the KV heads it holds. The layout walk accepts the rank-4
    per-head selection (it refused anything but DSA's rank-3 one)."""
    from backend.inference.parallel_config import ParallelConfig
    deployment = Deployment(_minimax_m3_program(),
                            config=ParallelConfig(tp_attention=8)).at(
        batch=1, input_tokens=4096)
    analysis = shared_walk(deployment)
    picked = _msa_node(analysis, "index_topk")
    consumers = [n for n in analysis.program.walked_nodes()
                 if any(ref.src is not None and ref.src.uid == picked.uid
                        for ref in n.inputs_obj or ())]
    assert {n.op for n in consumers} == {"dsa_attention_score"}

