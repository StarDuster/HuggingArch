"""Legal prefill CP degrees are the Ulysses interior divisibility constraint."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace

from backend.analyzer import fetch_config
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.attention import (
    ATTENTION_APPLY_OPS,
    ATTENTION_SCORE_OPS,
)
from backend.arch.layout.attention_plan import (
    degrees_dividing,
    head_repartition_constraints,
    legal_cp_degrees,
)
from backend.arch.layout.layout import physical_local_extent
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.loader import get_default_loader


def _program(model_id: str) -> Program:
    config = fetch_config(model_id)
    resolved = get_default_loader().match(config, model_id=model_id)
    assert resolved is not None
    return Program.of(resolved)


def _walk(model_id: str):
    """The model's one-rank prefill walk the solver reads."""
    return Deployment(_program(model_id)).analyze()


def test_degrees_dividing_includes_one_and_the_extent():
    assert degrees_dividing(28) == (1, 2, 4, 7, 14, 28)
    assert degrees_dividing(1) == (1,)
    assert degrees_dividing(0) == (1,)


def test_ulysses_lists_divisors_pcp_is_unconstrained():
    program = _walk("Qwen/Qwen2.5-7B")
    ulysses = ParallelConfig.from_dict(
        {"tp_attention": 1, "cp_scheme": "ulysses"})
    assert legal_cp_degrees(program, ulysses) == (1, 2, 4, 7, 14, 28)
    assert 8 not in legal_cp_degrees(program, ulysses)
    pcp = ParallelConfig.from_dict({"tp_attention": 1, "cp_scheme": "pcp"})
    assert legal_cp_degrees(program, pcp) is None
    layersplit = ParallelConfig.from_dict(
        {"tp_attention": 1, "cp_scheme": "layersplit"})
    assert legal_cp_degrees(program, layersplit) is None


def test_hybrid_query_heads_intersect_not_top_level_n_heads():
    """Step-3.7-Flash full layers have 64 query heads, SWA 96. Top-level
    n_heads=96 would admit 3 and 96; the 64-head layers reject those."""
    program = _walk("stepfun-ai/Step-3.7-Flash")
    parallel = ParallelConfig.from_dict(
        {"tp_attention": 1, "cp_scheme": "ulysses"})
    legal = legal_cp_degrees(program, parallel)
    assert legal == (1, 2, 4, 8, 16, 32)
    assert 3 not in legal
    assert 96 not in legal
    assert 64 not in legal


def test_per_shard_heads_are_the_tensor_parallel_walks():
    """Each interior edge's per-shard head count is what a TP walk of the
    model holds on one rank at that edge. MiMo-V2-Flash's sliding-window
    layers and its MTP layers bind `n_kv_heads: flash_kv_shards` (8); vLLM
    splits those K/V heads over TP like any other (mimo_v2.py
    `num_kv_heads = total_num_kv_heads // tp_size`), so at TP=2 a rank holds
    4 of them — and 2 of the full layers' 4. No layer keeps all 8: the
    answer does not depend on whether the rebinding is written on a graph
    node or on a stack's assignment row."""
    for model_id in ("XiaomiMiMo/MiMo-V2-Flash", "stepfun-ai/Step-3.7-Flash"):
        program = _program(model_id)
        parallel = ParallelConfig.from_dict(
            {"tp_attention": 2, "cp_scheme": "ulysses"})
        solved = head_repartition_constraints(
            Deployment(program).analyze(), parallel)
        tp = replace(Deployment(program), config=parallel).analyze()
        walked = []
        for node in tp.program.nodes():
            shape = tp.shapes.values.get(node.uid)
            if shape is None or node.uid in tp.program.encoder_uids:
                continue
            edges = (((0, False), (1, True)) if node.op in ATTENTION_SCORE_OPS
                     else ((1, True),) if node.op in ATTENTION_APPLY_OPS
                     else ())
            for index, gather in edges:
                layout = tp.input_states[(node.uid, index)].layout[1]
                walked.append((physical_local_extent(
                    layout, int(value_of(shape.ins[index][1]))), gather))
        assert sorted(set(solved)) == sorted(set(walked)), model_id
    mimo = head_repartition_constraints(
        Deployment(_program("XiaomiMiMo/MiMo-V2-Flash")).analyze(),
        ParallelConfig.from_dict({"tp_attention": 2}))
    assert {(4, True), (2, True)} <= set(mimo)
    assert (8, True) not in mimo
