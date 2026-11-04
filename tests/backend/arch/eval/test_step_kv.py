"""`step_kv`: this step's own keys as a KV stream — read, never cached.

DeepSeek-V4.1's DSpark draft block is the case (`DSparkAttention.forward`,
source/model.py:1057-1066): the block's keys are recomputed from `x` every
step, roped, and joined behind the ring buffer of main-model latents before
the one `sparse_attn` call. That kernel is a per-query index gather
(inference/kernel.py:312-365: `kv[by, idxs[i]]`), so the evaluator — not
traffic — decides that its readers are gathered. The union reads the window
and the block; only the window is stored.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import value_of
from backend.arch.eval import cache as cache_mod, traffic as traffic_mod
from backend.arch.ops import get_op
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import resolve_snapshot, shared_loader, shared_walk

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def test_the_marker_reads_its_rows_and_stores_none():
    spec = get_op("step_kv")
    assert spec.kv_stream_marker and not spec.kv_block_anchor
    assert spec.shape_kind == "preserve" and spec.is_view_op
    d = {"s_live": 5, "s_query": 3, "s_kv": 4096}
    assert int(value_of(spec.flops_fn(d))) == 0
    assert int(value_of(spec.kv_positions_per_query_fn(d))) == 5
    assert int(value_of(spec.kv_storage_positions_per_query_fn(d))) == 0
    # Every query of the step reads every live row: no mask among them.
    assert int(value_of(spec.kv_pairs_per_step_fn(d))) == 15


def test_the_draft_block_reads_the_window_and_its_own_keys():
    from backend.arch.eval.transforms import DECODE

    resolved = resolve_snapshot(MODEL)
    analysis = shared_walk(Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=4096).during(DECODE, cache_len=4096))
    program = analysis.program
    nodes = [node for node in program.nodes()
             if node.op == "step_kv"
             and "dsv41_dspark_attention" in node.display_path]
    assert nodes, "the DSpark attention declares its draft keys as a stream"
    for live in nodes:
        # The draft block's roped keys are read, not computed and dropped.
        assert str(live.inputs[0]).startswith("kv_concat")
    scores = [node for node in program.nodes()
              if node.op == "attention_score"
              and "dsv41_dspark_attention" in node.display_path]
    assert scores
    dims = resolved.dim_values
    keys = int(dims["s_window"]) + 1
    for score in scores:
        key_axis = analysis.shapes[score.uid].ins[1][-2]
        assert int(value_of(key_axis)) == keys
    for live in nodes:
        live_axis = analysis.shapes[live.uid].out[-2]
        assert int(value_of(live_axis)) == 1
        assert int(value_of(analysis.kv_lengths[live.uid].read)) == 1


def _walk(prefix=None, *, decode=False):
    workload = {"batch": 1, "input_tokens": 32 if decode else 64}
    if prefix is not None:
        workload["s_prefix_cached"] = prefix
    if decode:
        workload["s_verify"] = 5
    deployment = Deployment(Program.of(resolve_snapshot(
        MODEL, shared_loader()))).at(**workload)
    if decode:
        return shared_walk(deployment.during("decode", cache_len=32))
    return shared_walk(deployment.during("prefill"))


def _dspark(analysis):
    traffic = traffic_mod.derive(analysis)
    return [(node, traffic[node.uid]) for node in analysis.program.nodes()
            if "dsv41_dspark_attention" in node.display_path
            and not node.children]


def _expanded_live_walk(*, decode: bool):
    resolved = resolve_snapshot(MODEL, shared_loader())
    marker = resolved.spec.components[
        "dsv41_dspark_attention"].graph["kv_live"]
    old_live = marker.dims["s_live"]
    marker.dims["s_live"] = "40 * s_dspark_block"
    try:
        deployment = Deployment(Program.of(resolved))
        if decode:
            return shared_walk(deployment.at(
                batch=1, input_tokens=32).during(
                "decode", cache_len=32))
        return shared_walk(deployment.at(
            batch=1, input_tokens=64, s_prefix_cached=32).during(
            "prefill"))
    finally:
        marker.dims["s_live"] = old_live


def test_the_mixed_union_reads_one_draft_block_of_arrived_live_rows():
    analysis = _walk(32)
    coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
    rows = [(node, coverage[node.uid].branches)
            for node in analysis.program.nodes()
            if node.op == "kv_stream_concat"
            and "dsv41_dspark_attention" in node.display_path]
    assert len(rows) == 3
    for _node, branches in rows:
        cache, live = branches
        assert int(cache.rows) == 64 and cache.seeds
        assert int(live.rows) == 5 and not live.seeds

    traffic = traffic_mod.derive(analysis)
    readers = [(node, traffic[node.uid]) for node, _value in _dspark(analysis)
               if node.op in ("attention_score", "attention_apply")]
    assert len(readers) == 6
    cache_by_layer = {}
    live_by_layer = {}
    for _node, value in readers:
        edge = value.in_edges[1]
        live = [branch for branch in edge.branches if not branch.seeds]
        assert len(live) == 1
        assert int(value_of(live[0].rows)) == 5
        block = analysis.program.block_of(_node.uid)
        cache_by_layer[block] = cache_by_layer.get(block, 0) + int(
            value_of(value.kv_reads))
        live_by_layer[block] = live_by_layer.get(block, 0) + int(
            value_of(live[0].elements * edge.width))
    # The cache is 1,552 selected window pairs × 528 B per layer, and the
    # live rows one 32-query × 5-live-row gather of 1,024-B rows. Unfused,
    # score and apply are two kernels and each gathers both itself.
    assert set(cache_by_layer.values()) == {2 * 819456}
    assert set(live_by_layer.values()) == {2 * 163840}


def test_decode_reads_the_window_cache_and_five_live_rows():
    analysis = _walk(decode=True)
    traffic = traffic_mod.derive(analysis)
    readers = [(node, traffic[node.uid]) for node, _value in _dspark(analysis)
               if node.op in ("attention_score", "attention_apply")]
    assert len(readers) == 6
    # Three layers × 170 selected window pairs × 528 B, read by the score
    # and again by the apply.
    assert sum(int(value_of(value.kv_reads))
               for _n, value in readers) == 2 * 269280
    live_by_layer = {}
    for _node, value in readers:
        edge = value.in_edges[1]
        live = [branch for branch in edge.branches if not branch.seeds]
        assert live[0].gathered
        block = analysis.program.block_of(_node.uid)
        live_by_layer[block] = live_by_layer.get(block, 0) + int(
            value_of(live[0].elements * edge.width))
    # One query-by-row live gather per kernel: each of the three layers'
    # score and apply moves 5 queries × 5 rows × 1,024 B, not a five-row
    # stream.
    assert set(live_by_layer.values()) == {2 * 25600}


def test_the_evaluator_decides_dspark_is_gathered():
    analysis = _walk(decode=True)
    readers = {node.uid for node in analysis.program.nodes()
               if "dsv41_dspark_attention" in node.display_path
               and node.op in ("attention_score", "softmax_with_sink",
                               "attention_apply")}
    assert readers
    assert readers <= analysis.gathered
    assert cache_mod.context_span_gaps(analysis) == ()


def test_a_fresh_prefill_gathers_the_window_once_per_query():
    analysis = _walk(0)
    readers = [(node, value) for node, value in _dspark(analysis)
               if node.op in ("attention_score", "attention_apply")]
    live_by_layer = {}
    for _node, value in readers:
        edge = value.in_edges[1]
        live = next(branch for branch in edge.branches if not branch.seeds)
        block = analysis.program.block_of(_node.uid)
        live_by_layer[block] = live_by_layer.get(block, 0) + int(
            value_of(live.elements * edge.width))
    assert sum(int(value_of(value.kv_reads))
               for _node, value in _dspark(analysis)
               if value.kv_reads) == 2 * 3294720
    # 64 queries gather five live rows each, once in the score and once in
    # the apply.
    assert set(live_by_layer.values()) == {2 * 327680}


def test_prefill_never_grows_the_live_union_past_one_draft_block():
    resolved = resolve_snapshot(MODEL, shared_loader())
    input_rows = 64
    expected_union = min(
        input_rows, int(resolved.dim_values["s_window"])
    ) + int(resolved.dim_values["s_dspark_block"])
    for prefix in (0, 32):
        analysis = _walk(prefix)
        unions = [node for node in analysis.program.nodes()
                  if node.op == "kv_stream_concat"
                  and "dsv41_dspark_attention" in node.display_path]
        markers = [node for node in analysis.program.nodes()
                   if node.op == "step_kv"
                   and "dsv41_dspark_attention" in node.display_path]
        assert len(unions) == len(markers) == 3
        for node in unions:
            assert int(value_of(
                analysis.kv_lengths[node.uid].read)) == expected_union
        for node in markers:
            assert int(value_of(analysis.kv_lengths[node.uid].read)) == 5


def test_the_default_decode_block_follows_n_verify():
    analysis = shared_walk(Deployment(Program.of(resolve_snapshot(
        MODEL, shared_loader()))).at(batch=1, input_tokens=32).during(
        "decode", cache_len=32))
    markers = [node for node in analysis.program.nodes()
               if node.op == "step_kv"
               and "dsv41_dspark_attention" in node.display_path]
    assert len(markers) == 3
    for marker in markers:
        assert int(value_of(analysis.shapes[marker.uid].out[-2])) == 1
        assert int(value_of(analysis.kv_lengths[marker.uid].read)) == 1
    assert cache_mod.context_span_gaps(analysis) == ()


def test_an_expanding_live_view_cannot_hide_missing_rows():
    for decode, fresh_rows in ((True, 1), (False, 32)):
        analysis = _expanded_live_walk(decode=decode)
        markers = [node for node in analysis.program.nodes()
                   if node.op == "step_kv"
                   and "dsv41_dspark_attention" in node.display_path]
        assert len(markers) == 3
        for marker in markers:
            assert int(value_of(
                analysis.shapes[marker.uid].out[-2])) == fresh_rows
            assert int(value_of(
                analysis.kv_lengths[marker.uid].read)) == 200

        gaps = cache_mod.context_span_gaps(analysis)
        assert len(gaps) == 15
        assert all("dsv41_dspark_attention" in gap.producer_display
                   for gap in gaps)
        assert all(int(value_of(gap.rows)) == 200
                   and int(value_of(gap.fresh_rows)) == fresh_rows
                   and int(value_of(gap.uncovered)) == 200 - fresh_rows
                   for gap in gaps)
