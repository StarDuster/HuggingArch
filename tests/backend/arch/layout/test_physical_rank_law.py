"""A law the walk evaluates at the step's whole extents is re-evaluated at
the rank's own extents (`physical.rank_definitions`, `rank_law_shape`).

The case that made this a rule: a sliding window's step reads
`Min(s_kv, s_query + s_window − 1)` keys. The global walk evaluates it at the
whole prefill, where it is the whole context, and the K/V operands are whole
on every prefill-CP rank (each gathers the cache) — so the layout alone billed
every rank for every key while its queries reach about `1/cp` of them.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.binding import Definition
from backend.arch.eval.dim import Dim, value_of
from backend.arch.eval.values import Shape
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment, Workload
from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.layout.physical import (
    local_definitions,
    physical_shapes,
    rank_definitions,
    rank_law_shape,
)
from tests.snapshots import resolve_snapshot, shared_loader

GEMMA = "google/gemma-4-12B-it"
BATCH, SEQ = 8, 4096


@lru_cache(maxsize=None)
def _program(model_id: str) -> Program:
    return Program.of(resolve_snapshot(model_id, shared_loader()))


def _analyze(model_id: str, batch: int = BATCH, seq: int = SEQ, **parallel):
    analysis = Deployment(
        _program(model_id), workload=Workload(batch=batch, input_tokens=seq),
        config=ParallelConfig.from_dict(parallel)).analyze()
    return analysis, physical_shapes(analysis)


def _kv_reads(analysis, local, path_part: str):
    """(node, global K extent, per-rank K extent, s_window) of every
    attention score whose path names `path_part`: the score's second
    operand is K, `(B, n_kv_heads, s_kv, d_head)`."""
    out = []
    for node in analysis.program.nodes():
        if path_part not in (node.display_path or "") or not node.op:
            continue
        if not node.display_path.split("|")[-1].split(".")[-1].startswith(
                "qk_score"):
            continue
        g = analysis.shapes.values[node.uid]
        l = local.values[node.uid]
        dims = analysis.node_dims.values[node.uid]
        window = dims.get("s_window")
        out.append((node, value_of(g.ins[1][2]), value_of(l.ins[1][2]),
                    None if window is None else value_of(window)))
    assert out, path_part
    return out


def _mean_zigzag_reach(context: int, queries: int, cp: int,
                       window: int) -> Fraction:
    """The keys each rank's queries reach, counted position by position
    under the frameworks' split and averaged over the ranks: the step's
    `queries` are the last ones of a `context`-key context, cut into
    `2·cp` equal chunks, rank `r` holding chunks `r` and `2·cp − 1 − r`
    (vLLM `pcp_manager.py` DualChunkSwap, sglang `zigzag.py`); query `i`
    attends keys `[i − window + 1, i]`, clipped at 0. A second route: no
    law of the engine is read."""
    assert queries % (2 * cp) == 0
    chunk = queries // (2 * cp)
    first = context - queries
    total = 0
    for rank in range(cp):
        keys = set()
        for index in (rank, 2 * cp - 1 - rank):
            for query in range(first + index * chunk,
                               first + (index + 1) * chunk):
                keys.update(range(max(0, query - window + 1), query + 1))
        total += len(keys)
    return Fraction(total, cp)


def _window_definition(context, queries, window):
    s_kv, s_query, s_window = sp.symbols("s_kv s_query s_window")
    law = sp.Min(s_kv, s_query + s_window - 1)
    return Definition(law, {"s_kv": context, "s_query": queries,
                            "s_window": window},
                      min(context, queries + window - 1))


def _window_shapes(context, queries, window, cp, *, kv_cut=1):
    """A sliding-window score's shapes, globally and on one rank whose
    queries are `1/cp` of the step's and whose K axis is the `kv_cut`
    share of the narrowed read."""
    narrowed = min(context, queries + window - 1)

    def shape(q, k):
        return Shape(
            out=(Dim.symbol("B", 1), Dim.symbol("n_heads", 2),
                 Dim.symbol("s_query", q), Dim.symbol("s_attend", window)),
            ins=((Dim.symbol("B", 1), Dim.symbol("n_heads", 2),
                  Dim.symbol("s_query", q), Dim.symbol("d_head", 8)),
                 (Dim.symbol("B", 1), Dim.symbol("n_kv_heads", 1),
                  Dim.symbol("s_kv", k), Dim.symbol("d_head", 8))))
    return (shape(queries, narrowed),
            shape(Fraction(queries, cp), Fraction(narrowed, kv_cut)))


_NODE = SimpleNamespace(op="swa_attention_score", display_path="score")


def _rule(scheme="pcp", phase="prefill"):
    return SimpleNamespace(parallel=SimpleNamespace(cp_scheme=scheme),
                           phase=phase)


@pytest.mark.parametrize("context,queries,cp,window", [
    # The window crosses from a rank's late chunk into its early one, and
    # the context's start clips the early chunks: the case a per-chunk sum
    # capped at the whole step (`Min(law(r·q), 2·law(q/2))`, 96) got wrong.
    (96, 96, 4, 40),
    (4096, 4096, 16, 1024),
    (1024, 1024, 16, 1024),
    (4096, 4096, 2, 1024),
    (8192, 8192, 8, 4096),
    # A cached prefix: the step's queries are the context's last ones.
    (4096, 3072, 4, 1024),
    (640, 256, 8, 100),
])
def test_rank_law_is_the_direct_count_of_the_ranks_reach(
        context, queries, cp, window):
    """The rank's law is the mean over ranks of the keys each rank's two
    chunks reach, counted key by key."""
    definition = _window_definition(context, queries, window)
    global_shape, local_shape = _window_shapes(context, queries, window, cp)
    at_rank = rank_definitions({"s_kv": definition}, global_shape,
                               local_shape, _rule(), _NODE)["s_kv"]
    assert at_rank.value == _mean_zigzag_reach(context, queries, cp, window)
    assert at_rank.bindings["s_query"] == Fraction(queries, cp)
    assert at_rank.law.xreplace(
        {symbol: sp.nsimplify(at_rank.bindings[symbol.name])
         for symbol in at_rank.law.free_symbols}) == at_rank.value


def test_a_cache_cut_after_the_law_scales_its_definition():
    """Under DCP the decode cache's window is cut after the law
    (`Min(S, w)/dcp`, never `Min(S/dcp, w)`): the K axis holds a share of
    the law's value, and the definition printed beside it says so — it
    comes to the rank's number, not the step's."""
    context, window, dcp = 8192, 1024, 4
    definition = _window_definition(context, 1, window)
    global_shape, local_shape = _window_shapes(context, 1, window, 1,
                                               kv_cut=dcp)
    rule = _rule("pcp", "decode")
    owner = SimpleNamespace(
        rule=rule,
        rank_definitions=lambda defs, whole, local, node, **kw:
            rank_definitions(defs, whole, local, rule, node, **kw))
    projected = local_definitions(
        {"s_kv": definition}, analysis=owner, share=None,
        global_shape=global_shape, local_shape=local_shape,
        node=_NODE)["s_kv"]
    assert projected.value == Fraction(window, dcp)
    assert projected.law.xreplace(
        {symbol: projected.bindings[symbol.name]
         for symbol in projected.law.free_symbols}) == projected.value


def test_windowed_kv_read_is_the_ranks_reach_under_prefill_cp():
    """Gemma-4 sliding-window layers at cp=16: each rank holds 256 of the
    4096 queries in two chunks of 128, and reads the keys those chunks
    reach — 7419/4 averaged over the ranks — not all 4096."""
    cp = 16
    analysis, local = _analyze(GEMMA, cp=cp)
    reads = _kv_reads(analysis, local, "swa_qknorm_attention")
    for node, whole, at_rank, window in reads:
        assert whole == SEQ
        assert at_rank == _mean_zigzag_reach(SEQ, SEQ, cp, window), (
            node.display_path, at_rank)


def test_rank_definition_comes_to_the_ranks_extent():
    """The definition a formula prints beside the rank's `s_kv` is the
    rank's law, bound to the rank's extents, and comes to the same number
    the K/V axis carries."""
    analysis, local = _analyze(GEMMA, cp=16)
    node, _whole, at_rank, _window = _kv_reads(
        analysis, local, "swa_qknorm_attention")[0]
    declared = analysis.node_rebinds.values[node.uid]["s_kv"]
    assert isinstance(declared, Definition)
    projected = local_definitions(
        {"s_kv": declared}, analysis=analysis, share=None,
        global_shape=analysis.shapes.values[node.uid],
        local_shape=local.values[node.uid], node=node)["s_kv"]
    assert projected.value == at_rank
    assert projected.bindings["s_query"] == SEQ // 16
    assert projected.law.xreplace(
        {symbol: projected.bindings[symbol.name]
         for symbol in projected.law.free_symbols}) == at_rank


def test_dense_attention_under_cp_reads_every_key():
    """Control: Gemma-4's global-attention layers read the whole context on
    every CP rank, as before; they record no law to re-evaluate."""
    analysis, local = _analyze(GEMMA, cp=16)
    for node, whole, at_rank, _window in _kv_reads(
            analysis, local, "gqa_qknorm_keqv_attention"):
        assert whole == at_rank == SEQ, node.display_path
        assert "s_kv" not in (analysis.node_rebinds.values.get(node.uid)
                              or {}) or not isinstance(
            analysis.node_rebinds.values[node.uid]["s_kv"], Definition)


def test_windowed_attention_whose_queries_are_whole_is_unchanged():
    """Control: attention TP cuts heads, not queries, so a windowed layer's
    rank reads what the global walk read and its shape is not rebuilt."""
    analysis, local = _analyze(GEMMA, tp_attention=2)
    for node, whole, at_rank, _window in _kv_reads(
            analysis, local, "swa_qknorm_attention"):
        assert at_rank == whole == SEQ, node.display_path
        global_shape = analysis.shapes.values[node.uid]
        assert rank_law_shape(analysis, node, global_shape,
                              local.values[node.uid]) is global_shape


def test_ulysses_reads_whole_queries_on_its_law_face():
    """Control: Ulysses cuts an attention boundary's queries, but the kernel
    that reads K/V scores the whole prompt for its heads. The law reads the
    score-matrix face, not the sequence-cut output beside it, so it is not
    re-evaluated — and the two faces differing is not ambiguity."""
    analysis, local = _analyze(GEMMA, cp=2, cp_scheme="ulysses")
    checked = 0
    for node in analysis.program.nodes():
        if node.op != "swa_attention_apply":
            continue
        global_shape = analysis.shapes.values[node.uid]
        local_shape = local.values[node.uid]
        assert value_of(local_shape.out[2]) == SEQ // 2
        assert value_of(local_shape.ins[0][2]) == SEQ
        assert value_of(local_shape.ins[1][2]) == SEQ
        assert rank_law_shape(
            analysis, node, global_shape, local_shape) is global_shape
        checked += 1
    assert checked


def _moved(model_id: str, **parallel):
    analysis, local = _analyze(model_id, batch=2, **parallel)
    moved = {}
    for node in analysis.program.nodes():
        declared = (analysis.node_rebinds.values.get(node.uid) or {}).get(
            "s_kv")
        if not isinstance(declared, Definition):
            continue
        global_shape = analysis.shapes.values[node.uid]
        moved[node.display_path] = (
            str(declared.law),
            rank_law_shape(analysis, node, global_shape,
                           local.values[node.uid]) is not global_shape)
    return moved


def test_laws_in_no_query_count_are_unchanged_under_cp():
    """Control: a top-k's `Min(s_kv, n_topk_keys)` (DeepSeek-V3.2's DSA)
    reads no count a CP rank holds part of; the rank reads what the step
    reads."""
    moved = _moved("deepseek-ai/DeepSeek-V3.2", cp=2)
    assert moved
    assert not any(changed for _law, changed in moved.values()), moved


def test_pooled_streams_are_not_narrowed_again_under_cp():
    """Control: DeepSeek-V4's compressed streams arrive with their length
    fixed in the workload's names (`⌊s_kv/m⌋`, a window joined with a
    pooled top-k); the attention reading them records no law of its own, so
    no rank applies one a second time. Its sliding-window-only layers do,
    and move."""
    moved = _moved("deepseek-ai/DeepSeek-V4-Flash", cp=2)
    assert all("s_query" in law for law, changed in moved.values()
               if changed), moved
    assert any(changed for _law, changed in moved.values()), moved
    assert not any("csa" in path or "hca" in path for path in moved), moved
