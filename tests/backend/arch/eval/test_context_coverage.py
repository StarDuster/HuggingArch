"""Branch-preserving coverage for cache, live, and feature context streams."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import Undecided
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.backend.arch.eval.test_eval_symbolic_workload import (
    BATCH, LENGTH, PREFIX, _at, _grid, _read)
from tests.snapshots import resolve_snapshot, shared_loader


def _dspark(prefix: int):
    resolved = resolve_snapshot(
        "deepseek-ai/DeepSeek-V4.1-Flash", shared_loader())
    return (Deployment(Program.of(resolved))
            .at(batch=1, input_tokens=64, s_prefix_cached=prefix)
            .during("prefill").analyze())


def _mla_decode():
    resolved = resolve_snapshot(
        "deepseek-ai/DeepSeek-V2-Lite", shared_loader())
    return (Deployment(Program.of(resolved))
            .at(batch=1, input_tokens=9)
            .during("decode", cache_len=9).analyze())


def test_a_mixed_sequence_union_keeps_cache_and_live_branches():
    analysis = _dspark(32)
    coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
    unions = [node for node in analysis.program.nodes()
              if node.op == "kv_stream_concat"
              and "dsv41_dspark_attention" in node.display_path]
    assert len(unions) == 3
    for union in unions:
        found = coverage[union.uid]
        assert found.disjoint
        assert len(found.branches) == 2
        cache, live = found.branches
        assert cache.seeds and not live.seeds
        assert int(cache.rows) == 64
        assert int(live.rows) == 5
        assert int(found.rows) == 69


def test_a_live_marker_has_first_class_coverage_without_a_seed():
    analysis = _dspark(0)
    coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
    live = next(node for node in analysis.program.nodes()
                if node.op == "step_kv"
                and "dsv41_dspark_attention" in node.display_path)
    found = coverage[live.uid]
    assert found.disjoint and not found.seeds
    assert int(found.rows) == 5


def test_a_stated_seed_length_reaches_the_union_branch():
    resolved = resolve_snapshot(
        "deepseek-ai/DeepSeek-V4-Flash", shared_loader())
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=64, s_prefix_cached=32)
                .during("prefill").analyze())
    coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
    unions = [node for node in analysis.program.nodes()
              if node.op == "kv_stream_concat"
              and node.uid in coverage
              and ("dsv4_csa_attention" in node.display_path
                   or "dsv4_hca_attention" in node.display_path)]
    assert unions
    for node in unions:
        found = coverage[node.uid]
        assert found.rows is not None
        assert all(branch.rows is not None for branch in found.branches)
        # All-cache positional unions keep the whole-edge settlement; only a
        # cache/live union needs a positional branch split.
        assert traffic_mod._split_context(128, found) is None

    # These are the pre-branch-split bytes: the new row split reaches CSA/HCA
    # without changing their all-cache law. The score and the apply each
    # read the cache (two unfused kernels), so its law is billed twice.
    traffic = traffic_mod.derive(analysis)
    readers = [(node, traffic[node.uid])
               for node in analysis.program.nodes()
               if not node.children
               and ("dsv4_csa_attention" in node.display_path
                    or "dsv4_hca_attention" in node.display_path)
               and node.op in ("attention_score", "attention_apply")]
    assert sum(int(value.act_reads) for _n, value in readers) == 7211008
    assert sum(int(value.kv_reads) for _n, value in readers) == 2 * 3152864
    assert sum(int(value.act_writes) for _n, value in readers) == 7208960


def test_a_mixed_union_refuses_an_unstated_branch_share():
    coverage = cache_mod.ContextCoverage((
        cache_mod.ContextBranch(seeds=frozenset((1,)), rows=64,
                                producer=1),
        cache_mod.ContextBranch(seeds=frozenset(), rows=None,
                                producer=2),
    ), disjoint=True)
    with pytest.raises(ValueError, match="unstated"):
        traffic_mod._split_context(128, coverage)


def test_a_sole_cache_branch_keeps_the_whole_edge_law():
    resolved = resolve_snapshot(
        "Qwen/Qwen3.8-Flash-Next", shared_loader())
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=64, s_prefix_cached=32)
                .during("prefill").analyze())
    edges = cache_mod.Edges(analysis)
    coverage = cache_mod.context_sources(edges)
    readers = [node for node in analysis.program.nodes()
               if not node.children
               and "index_score__qwen_qsa_attention" in node.display_path]
    assert readers
    for reader in readers:
        source = edges.source(reader, 1)
        found = coverage[cache_mod.covered_context_source(
            source, coverage, edges).uid]
        assert len(found.branches) == 1
        assert found.branches[0].seeds
        assert traffic_mod._split_context(128, found) is None

    traffic = traffic_mod.derive(analysis)
    assert all(int(traffic[reader.uid].act_reads) == 32768
               and int(traffic[reader.uid].kv_reads) == 4096
               for reader in readers)


def test_a_feature_concat_stays_co_positional():
    analysis = _mla_decode()
    coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
    joins = [node for node in analysis.program.nodes()
             if node.name == "kv_concat" and node.uid in coverage]
    assert joins
    for join in joins:
        found = coverage[join.uid]
        assert not found.disjoint
        assert len(found.branches) == 2
        # Unknown plain-cache rows stay unknown; they are not turned into a
        # sum by the feature join.
        assert found.rows is None
        assert len(found.seeds) == 2


@pytest.mark.regression  # two long-prefill model walks
@pytest.mark.parametrize(("snapshot", "input_tokens", "prefix"), (
    ("deepseek-ai__DeepSeek-V4-Flash", 4096, 2048),
    ("google__gemma-2-9b", 16384, 8192),
))
def test_chunked_prefill_swa_does_not_report_a_prefix_gap(
        snapshot, input_tokens, prefix):
    """A sliding window reads its fresh suffix and the cached rows before it.

    Those are different positions, so their available rows add; taking the
    wider of the two used to turn every long chunked prefill into a false
    structural gap.
    """
    resolved = resolve_snapshot(snapshot, shared_loader())
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=input_tokens,
                    s_prefix_cached=prefix)
                .during("prefill").analyze())
    assert cache_mod.context_span_gaps(analysis) == ()


def _prefill(snapshot, workload):
    """One prefill walk of `snapshot` over `workload`, symbolic or numeric."""
    resolved = resolve_snapshot(snapshot, shared_loader())
    return (Deployment(Program.of(resolved)).at(**workload)
            .during("prefill").analyze())


def _span_gaps(analysis):
    """`context_span_gaps` under the conditions the walk recorded."""
    with analysis.assumed():
        return cache_mod.context_span_gaps(analysis)


@pytest.mark.parametrize(("snapshot", "axes"), (
    # DSpark's mixed cache/live row union.
    ("deepseek-ai__DeepSeek-V4.1-Flash",
     {"B": (1,), "s_ctx": (4096,), "s_prefix": (1024, 2048)}),
    # The sliding-window reader whose rows are covered by fresh PLUS cached.
    # This is the one that caught the bug below, so it is the one that must
    # be here.
    ("google__gemma-2-9b",
     {"B": (1,), "s_ctx": (16384, 32768), "s_prefix": (4096, 8192)}),
))
def test_the_span_gap_reader_answers_every_point_or_refuses(snapshot, axes):
    """`context_span_gaps` is one walk's answer, so it holds at every point.

    The reader decides whether a leaf gets a gap record from the rows left
    over after its fresh and cached rows, and over the workload's symbols
    that leftover is a law. Two answers are allowed, and this pins the
    difference between them:

    * it REFUSES (`dim.Undecided`) — the law cannot decide coverage, so
      there is nothing to read at a point and nothing to disagree with.
      That is today's answer, and it is deliberately NOT pinned as
      permanent: when a spec or the engine can settle `rows - (fresh +
      cached)`, this test keeps running and starts demanding the second
      answer instead of being rewritten.
    * it ANSWERS — then every record it answered with must read, at each
      point, exactly what the walk made at that point answers. A law that
      disagrees with the points it is supposed to answer is the failure
      this catches: deciding the leftover with a Python `==` kept a record
      wherever SymPy merely could not prove coverage, which reported 22
      phantom gaps on DeepSeek-V4.1 and 2 on this Gemma path, every one of
      them reading zero at every point where the numeric walk reported
      none.

    The numeric walk is never allowed to refuse: a walk standing at a point
    has numbers, and `Undecided` there would escape as an error.
    """
    symbolic = _prefill(snapshot, {"batch": BATCH, "input_tokens": LENGTH,
                                   "s_prefix_cached": PREFIX})
    try:
        law = _span_gaps(symbolic)
    except Undecided:
        law = None
    for point in _grid(axes):
        walked = _span_gaps(_prefill(snapshot, {
            "batch": _at(BATCH, point), "input_tokens": _at(LENGTH, point),
            "s_prefix_cached": _at(PREFIX, point)}))
        if law is None:
            continue
        assert _read(law, point) == _read(walked, point), (
            f"the walk's span-gap records read at {point} are not the ones "
            f"the walk made there answers with")
