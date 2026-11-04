"""A prefill context-parallel rank reads the cache its attention reads,
and scores the pairs of the queries it holds.

Two facts decide a CP rank's cache reads, and each once lost them:

  * The cached K/V reach the attention through the collective that hands
    them between ranks (an `all_gather` after the cache write, an
    `all_to_all` at the Ulysses boundary). A collective that MOVES its
    operand leaves it the cache's rows (`OpSpec.reduction`), so the fused
    kernel past it is still the cache's reader. When the pedigree stopped at
    the collective, every CP attention read no cache at all.
  * The queries a kernel scores are its score matrix's (`attention_kernel`).
    Under Ulysses the kernel's boundary is sequence-cut on both sides while
    it scores every query for its heads, so a share read off the boundary
    billed `1/cp` of the gathered pairs.

Ulysses runs the attention on `heads / cp` with the whole sequence — the
geometry TP runs it in — so a latent cache with no head axis (MLA) is read
exactly as TP reads it. PCP keeps queries sequence-cut, so a rank gathers
`1/cp` of the pairs.

The pairs a matrix scores (`s_pairs`) follow the share of its cells a rank
holds (`DimSymbol.score_pairs`): kept whole, they billed a PCP rank — and a
DSA indexer, whose rows are sequence-cut under either scheme — every pair
of the step, `cp` times its work; and a DCP rank, which holds `1/dcp` of
the keys, `dcp` times its decode attention.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.parallel_config import CpScheme, ParallelConfig
from backend.arch.layout import physical_cost
from backend.arch.layout.physical_cost import (
    physical_access,
    physical_access_after_rewrite,
)
from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import algebra, value_of
from backend.inference.capacity import estimate_capacity
from backend.inference.walk_registry import WALK_CELLS as _CELLS
from backend.inference.benchmark_identity import _fanouts
from backend.inference.op_signatures import get_signature, op_signature
from backend.inference.parallel_sweep import config_for_degree

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
_DEGREE = 8


@lru_cache(maxsize=None)
def _capacity(snapshot: str, mode: str, scheme: str, seq: int,
              prefix: int = 0) -> dict:
    """One B200 prefill estimate, shared by the file's row-level laws."""
    doc = json.loads((SNAP_DIR / f"{snapshot}.json").read_text())
    parallel = replace(config_for_degree(mode, _DEGREE),
                       cp_scheme=CpScheme(scheme))
    return estimate_capacity(
        config=doc["config"], model_id=doc["model_id"], gpu_sku="B200",
        num_gpus=_DEGREE, avg_input_tokens=seq, avg_output_tokens=256,
        gpu_memory_fraction=0.9, parallel=parallel,
        selection_mode="measured", decode_batch_size=1,
        prefill_batch_size=1, s_prefix_cached=prefix)


@lru_cache(maxsize=None)
def _attention_rows(snapshot: str, mode: str, scheme: str, seq: int,
                    op: str) -> dict:
    """`{(display_path, count_per_step): row}` of the prefill rows of `op`."""
    result = _capacity(snapshot, mode, scheme, seq)
    return {(row["display_path"], row["count_per_step"]): row
            for row in result["per_op_timing"] if row.get("op") == op}


def _kv(row: dict) -> float:
    return float(row.get("kv_read_bytes_per_rank") or 0)


def _flash_rows(snapshot: str, mode: str, scheme: str, prefix: int) -> dict:
    """`{(display_path, count_per_step): row}` of the S=4096 flash rows."""
    result = _capacity(snapshot, mode, scheme, 4096, prefix)
    return {(row["display_path"], row["count_per_step"]): row
            for row in result["per_op_timing"]
            if row.get("op") == "flash_attention"}


@pytest.mark.regression  # six measured B200 model estimates
@pytest.mark.parametrize("prefix", (0, 2048))
def test_ulysses_reads_v4_cache_unions_as_tp_does(prefix):
    """DeepSeek-V4's window/compressed unions, fresh and chunked: Ulysses
    reads every cache TP reads. A collective between a cache seed and its
    reader once turned the window into live rows and billed it as an
    activation, cutting HCA's cache read about ninefold."""
    snapshot = "deepseek-ai__DeepSeek-V4-Flash"
    cp = _flash_rows(snapshot, "cp_ep", "ulysses", prefix)
    tp = _flash_rows(snapshot, "tp_ep", "ulysses", prefix)
    assert cp and cp.keys() == tp.keys()
    assert any(_kv(row) > 0 for row in cp.values())
    for key, row in cp.items():
        assert _kv(row) == pytest.approx(_kv(tp[key]), rel=1e-9), key


@pytest.mark.regression  # two measured B200 model estimates
def test_pcp_reads_its_share_of_v4_cache_unions():
    """A fresh V4 prefill under PCP gathers `1/cp` of TP's pairs. Only the
    fresh point divides: with a cached prefix, the rank holding the first
    fresh queries still reads a sliding window's whole cached tail."""
    snapshot = "deepseek-ai__DeepSeek-V4-Flash"
    pcp = _flash_rows(snapshot, "cp_ep", "pcp", 0)
    tp = _flash_rows(snapshot, "tp_ep", "ulysses", 0)
    assert pcp and pcp.keys() == tp.keys()
    assert any(_kv(row) > 0 for row in pcp.values())
    for key, row in pcp.items():
        assert _kv(row) == pytest.approx(_kv(tp[key]) / _DEGREE,
                                         rel=1e-9), key


@pytest.mark.regression  # one measured B200 model estimate
@pytest.mark.parametrize("scheme", ("ulysses", "pcp"))
def test_a_cache_union_keeps_its_seeds_past_the_cp_collectives(
        monkeypatch, scheme):
    """On the program CP actually runs — collectives spliced between each
    cache seed and its window — every branch of a V4 cache union still
    names its seed. A branch that lost it is priced as this step's
    activation instead of as the cache."""
    taken = []

    def take_after_rewrite(analysis, **kwargs):
        taken.append((analysis, kwargs["program"], kwargs["rewrite"]))
        return physical_access_after_rewrite(analysis, **kwargs)

    monkeypatch.setattr(physical_cost, "physical_access_after_rewrite",
                        take_after_rewrite)
    _CELLS.clear()
    doc = json.loads(
        (SNAP_DIR / "deepseek-ai__DeepSeek-V4-Flash.json").read_text())
    estimate_capacity(
        config=doc["config"], model_id=doc["model_id"], gpu_sku="B200",
        num_gpus=_DEGREE, avg_input_tokens=4096, avg_output_tokens=256,
        gpu_memory_fraction=0.9,
        parallel=replace(config_for_degree("cp_ep", _DEGREE),
                         cp_scheme=CpScheme(scheme)),
        selection_mode="measured", decode_batch_size=1,
        prefill_batch_size=1, s_prefix_cached=2048)
    _CELLS.clear()
    assert taken
    analysis, program, rewrite = taken[0]
    unions = [coverage for coverage in cache_mod.context_sources(
        cache_mod.Edges(analysis, program, rewrite)).values()
        if coverage.disjoint and len(coverage.branches) > 1]
    assert unions
    for coverage in unions:
        assert all(branch.seeds for branch in coverage.branches), coverage


def test_ulysses_reads_an_mla_sparse_cache_as_tp_does():
    """DeepSeek-V3.2's DSA: the gathered latent under Ulysses is TP's."""
    args = ("deepseek-ai__DeepSeek-V3.2", 8192)
    cp = _attention_rows(args[0], "cp_ep", "ulysses", args[1], "flash_dsa")
    tp = _attention_rows(args[0], "tp_ep", "ulysses", args[1], "flash_dsa")
    assert cp and cp.keys() == tp.keys()
    for key, row in cp.items():
        assert _kv(row) > 0, key
        assert _kv(row) == pytest.approx(_kv(tp[key]), rel=1e-9), key
        assert row["flops_per_rank"] == pytest.approx(
            tp[key]["flops_per_rank"], rel=1e-9), key


def test_pcp_reads_its_share_of_the_gathered_pairs():
    """PCP scores `1/cp` of the queries, so it gathers `1/cp` of TP's pairs."""
    args = ("deepseek-ai__DeepSeek-V3.2", 8192)
    pcp = _attention_rows(args[0], "cp_ep", "pcp", args[1], "flash_dsa")
    tp = _attention_rows(args[0], "tp_ep", "ulysses", args[1], "flash_dsa")
    assert pcp and pcp.keys() == tp.keys()
    for key, row in pcp.items():
        assert _kv(row) == pytest.approx(_kv(tp[key]) / _DEGREE, rel=1e-9), key


def test_two_unfused_gathering_readers_price_under_pcp():
    """MLA+DSA under prefill context parallelism, at cp=2 with EP=2.

    Unfused, the sparse score and the sparse apply each gather the selected
    latent rows. When one law was split between them in proportion to their
    spans, the two spans were in different frames per rank (the apply's
    fan-out compared its gathered K/V with the producer's sequence-cut rows,
    halving its span), so the apply took a third of an integer law and the
    estimate refused the non-integer per-rank bytes. Each kernel now bills
    its own read, so the per-rank bytes are whole and the estimate returns
    (its weights do not fit four H100s, which it reports, not raises)."""
    doc = json.loads((SNAP_DIR / "tencent__Hy4-preview.json").read_text())
    result = estimate_capacity(
        config=doc["config"], model_id=doc["model_id"], gpu_sku="H100-SXM",
        parallel=ParallelConfig(cp=2, ep_moe=2, dp_attention=1),
        avg_input_tokens=1024, avg_output_tokens=1024,
        prefill_batch_size=1, decode_batch_size=1)
    assert set(result["phases"]) == {"prefill", "decode"}


def test_a_fresh_pcp_prefill_reads_no_cache_before_or_after_fusion():
    """A fresh prompt's DSA indexer scores the keys this step wrote: no
    cache read. Per rank under PCP the indexer takes every rank's keys
    while the keys landed are this rank's slice, gathered by a collective
    the pre-fusion program does not yet hold; comparing those two frames
    billed the pre-fusion indexer a whole cache read the post-fusion one
    (past the grafted all-gather) never had. Whether a reader reads the
    cache is asked in the unsplit frame, so both programs agree."""
    from backend.inference.framework_bindings import (
        DEFAULT_FRAMEWORK, get_binding)
    from backend.inference.pipeline import analyze
    from tests.snapshots import resolve_snapshot

    result = analyze(
        resolve_snapshot("deepseek-ai__DeepSeek-V3.2"),
        parallel=ParallelConfig(cp=2, ep_moe=2, dp_attention=1),
        framework_binding=get_binding(DEFAULT_FRAMEWORK), phase="prefill",
        batch_size=1, avg_input_tokens=1024, avg_output_tokens=0,
        peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3e12,
        interconnect_bw_bytes_per_sec=4e11, gpus_per_node=8, breakdown=True)
    for rows in (result.pre_per_op_timing, result.per_op_timing):
        indexer = [row for row in rows if row.get("op") == "indexer_logits"]
        assert indexer
        assert all(_kv(row) == 0 for row in indexer)


def test_ulysses_decompresses_only_its_resident_mla_rows():
    """The headed MLA decompression follows the K/V view's resident rows.

    PCP intentionally decompresses the whole gathered context on every rank;
    Ulysses computes the same local shape TP would before its entry A2A. The
    weight and cache residency laws do not move with this invocation cut.
    """
    args = ("deepseek-ai__DeepSeek-V3", 8192)
    results = {
        label: _capacity(args[0], mode, scheme, args[1])
        for label, mode, scheme in (("tp", "tp_ep", "ulysses"),
                                    ("ulysses", "cp_ep", "ulysses"),
                                    ("pcp", "cp_ep", "pcp"))
    }

    def projection(result, name):
        return {
            (row["display_path"], row["count_per_step"]):
                row["flops_per_rank"]
            for row in result["per_op_timing"]
            if row.get("op") == "linear"
            and row.get("display_path", "").endswith(f".{name}")
        }

    tp = projection(results["tp"], "kv_b_proj")
    ulysses = projection(results["ulysses"], "kv_b_proj")
    pcp = projection(results["pcp"], "kv_b_proj")
    assert tp and tp.keys() == ulysses.keys() == pcp.keys()
    for key, expected in tp.items():
        assert ulysses[key] == pytest.approx(expected, rel=1e-9), key
        assert pcp[key] == pytest.approx(expected * _DEGREE, rel=1e-9), key

    uly, whole = results["ulysses"], results["pcp"]
    assert uly["weight_bytes_total"] == whole["weight_bytes_total"]
    assert (uly["memory"]["weight_bytes_per_gpu"]
            == whole["memory"]["weight_bytes_per_gpu"])
    assert (uly["weight_breakdown"]["total_bytes_per_rank"]
            == whole["weight_breakdown"]["total_bytes_per_rank"])
    assert uly["kv_cache"] == whole["kv_cache"]
    assert uly["max_tokens_per_gpu"] == whole["max_tokens_per_gpu"]
    assert uly["max_batch_per_gpu"] == whole["max_batch_per_gpu"]


def test_ulysses_keeps_gqa_cache_storage_on_the_persistent_face():
    """TP's head cut survives the Ulysses invocation-only sequence cut.

    GQA K/V producers do not lie between a whole-cache view and attention, so
    they cannot enter the head-repartition region.  Their persistent cache
    width therefore remains the attention-TP shard under both context schemes
    (four of Qwen3-8B's eight KV heads here), rather than collapsing back to
    the global eight-head cache.
    """
    doc = json.loads(
        (SNAP_DIR / "Qwen__Qwen3-8B.json").read_text())
    parallel = dict(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=1,
                    ep_moe=1, cp=4, pp=1, dp_attention=1, dp_replicas=1)
    results = {
        scheme: estimate_capacity(
            config=doc["config"], model_id=doc["model_id"],
            gpu_sku="B200", num_gpus=8, avg_input_tokens=8192,
            avg_output_tokens=256, gpu_memory_fraction=0.9,
            parallel=ParallelConfig(cp_scheme=scheme, **parallel),
            selection_mode="measured", decode_batch_size=1,
            prefill_batch_size=1)
        for scheme in ("ulysses", "pcp")
    }
    ulysses, pcp = results["ulysses"], results["pcp"]
    assert (ulysses["kv_cache"]["per_token_bytes"]
            == pcp["kv_cache"]["per_token_bytes"] == 73728)
    assert (ulysses["max_tokens_per_gpu"]
            == pcp["max_tokens_per_gpu"] == 2_248_197)
    assert (ulysses["max_batch_per_gpu"]
            == pcp["max_batch_per_gpu"])


def test_ulysses_reads_only_its_resident_prefix_rows(monkeypatch):
    """A cut whole-cache view bills only local rows, and only cached rows.

    Fresh prefill rows are activations: when the reader spans no more than
    the producer wrote this step, the existing fresh/span rule must keep the
    read out of `kv_read`.  With a prefix, both whole-cache views (latent and
    RoPE key) read their resident share, making Ulysses exactly TP/cp.
    """
    doc = json.loads(
        (SNAP_DIR / "deepseek-ai__DeepSeek-V3.json").read_text())
    results = {}
    captured = []

    def record_access(prefix: int, label: str):
        def take(analysis, **kwargs):
            layer = physical_access(analysis, **kwargs)
            if not any(row[:3] == (prefix, label, "pre")
                       for row in captured):
                captured.append((prefix, label, "pre", analysis, layer))
            return layer

        def take_after_rewrite(analysis, **kwargs):
            layer = physical_access_after_rewrite(analysis, **kwargs)
            # The access of the graph before the fusion rewrite and after
            # it are both derivations over a rewritten graph once the casts
            # are grafted (`serving.build_serving_layers`): the pre-fusion
            # one passes the identity rewrite.
            rewrite = kwargs.get("rewrite")
            stage = ("pre" if rewrite is None or rewrite.is_identity()
                     else "post")
            if not any(row[:3] == (prefix, label, stage)
                       for row in captured):
                captured.append((prefix, label, stage, analysis, layer))
            return layer

        monkeypatch.setattr(physical_cost, "physical_access", take)
        monkeypatch.setattr(
            physical_cost, "physical_access_after_rewrite",
            take_after_rewrite)

    for prefix in (0, 16384):
        for label, parallel in (
            ("tp", ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                  tp_moe_expert=1, ep_moe=8,
                                  cp_scheme="ulysses")),
            ("ulysses", ParallelConfig(cp=8, ep_moe=8, dp_attention=1,
                                      cp_scheme="ulysses")),
        ):
            record_access(prefix, label)
            results[(prefix, label)] = estimate_capacity(
                config=doc["config"], model_id=doc["model_id"],
                gpu_sku="B200", num_gpus=8, avg_input_tokens=32768,
                avg_output_tokens=256, s_prefix_cached=prefix,
                parallel=parallel, selection_mode="theoretical_only",
                decode_batch_size=1, prefill_batch_size=1)
            if prefix == 0:
                # Model cells deliberately ignore workload counts: one graph
                # serves every prefix reading.  Clearing that cell cache
                # between the two readings keeps both pre/post layers real.
                _CELLS.clear()

    def cache_bytes(result: dict) -> float:
        # Per-op timing rows each carry their own count scaling.  The gate
        # compares those per-op bytes; multiplying by `count_per_step` again
        # changes the dimension from per-op-row bytes to invocation bytes.
        return sum(float(row["kv_read_bytes_per_rank"] or 0)
                   for row in result["per_op_timing"])

    tp_fresh = cache_bytes(results[(0, "tp")])
    ulysses_fresh = cache_bytes(results[(0, "ulysses")])
    assert tp_fresh == ulysses_fresh == 0

    tp_prefix = cache_bytes(results[(16384, "tp")])
    ulysses_prefix = cache_bytes(results[(16384, "ulysses")])
    assert tp_prefix > 0
    assert ulysses_prefix == pytest.approx(tp_prefix / _DEGREE, rel=1e-9)

    # The latent decompression reads those same resident rows once, and it
    # reads them through the cast that quantises the latent for the GEMM
    # (`materialize_casts`): the cast's operand IS the cache read, never the
    # cache read plus a gathered copy, and the GEMM downstream reads only
    # the fresh narrow copy the cast wrote.
    def decompression(result: dict, field: str,
                      suffix="kv_b_proj__cast") -> float:
        rows = [row for row in result["per_op_timing"]
                if row.get("display_path", "").endswith(suffix)]
        assert rows
        return sum(float(row[field] or 0) for row in rows)

    tp_in = decompression(results[(16384, "tp")], "act_in_bytes_per_rank")
    ulysses_in = decompression(results[(16384, "ulysses")],
                               "act_in_bytes_per_rank")
    assert ulysses_in == pytest.approx(
        decompression(results[(16384, "ulysses")], "kv_read_bytes_per_rank"),
        rel=1e-9)
    assert ulysses_in == pytest.approx(tp_in / _DEGREE, rel=1e-9)
    for result in (results[(16384, "tp")], results[(16384, "ulysses")]):
        assert decompression(result, "kv_read_bytes_per_rank",
                             suffix="kv_b_proj") == 0

    def read_bytes(analysis, value, at_prefix: int, *,
                   at_seq: int = 32768):
        law = algebra(value_of(value or 0))
        values = {
            "B": 1,
            "s_input": at_seq,
            "s_prefix": at_prefix,
        }
        substitutions = {}
        for symbol in law.free_symbols:
            assert str(symbol) in values, str(symbol)
            substitutions[symbol] = values[str(symbol)]
        return float(law.subs(substitutions))

    def kv_bytes(want_prefix: int, stage: str, label: str, *,
                 kv_b_only: bool = False) -> float:
        total = 0
        for (got_prefix, got_label, got_stage, analysis, layer) in captured:
            if (got_prefix != want_prefix or got_label != label
                    or got_stage != stage):
                continue
            for uid, access in layer.values.items():
                # The GEMM and the cast quantising its operand: the cache
                # read bills the cast, the node that makes it
                # (`materialize_casts`).
                if kv_b_only and not layer.program.by_uid(
                        uid).display_path.endswith(
                            ("kv_b_proj", "kv_b_proj__cast")):
                    continue
                total += read_bytes(
                    analysis, access.kv_read_bytes_per_rank, got_prefix)
        return total

    fresh = [row for row in captured if row[0] == 0 and row[1] == "ulysses"]
    assert fresh
    for _prefix, _label, stage, analysis, layer in fresh:
        for uid, access in layer.values.items():
            assert read_bytes(
                analysis, access.kv_read_bytes_per_rank, 0) == 0, (
                stage, layer.program.by_uid(uid).display_path)

    pre = kv_bytes(16384, "pre", "ulysses", kv_b_only=True)
    post = kv_bytes(16384, "post", "ulysses", kv_b_only=True)
    reference = kv_bytes(16384, "pre", "tp", kv_b_only=True)
    assert pre > 0
    assert pre == post == pytest.approx(reference / _DEGREE, rel=1e-9)


@pytest.mark.regression  # minutes-long: two DeepSeek-V4 walks
def test_ulysses_reads_a_compressed_cache_past_its_collectives():
    """DeepSeek-V4's compressed streams reach the attention through
    `all_gather` / `all_to_all`; a CP rank reads them as TP does. A fresh
    prefill's window reads this step's rows, not the cache, under every
    layout — so the comparison is row by row, and some row must read."""
    args = ("deepseek-ai__DeepSeek-V4-Flash", 8192)
    cp = _attention_rows(args[0], "cp_ep", "ulysses", args[1], "flash_attention")
    tp = _attention_rows(args[0], "tp_ep", "ulysses", args[1], "flash_attention")
    assert cp and cp.keys() == tp.keys()
    assert any(_kv(row) > 0 for row in cp.values())
    for key, row in cp.items():
        assert _kv(row) == pytest.approx(_kv(tp[key]), rel=1e-9), key


@pytest.mark.regression  # four measured model walks
def test_a_ulysses_reader_clips_head_fanout_to_its_own_heads():
    """A broadcast-head K/V is read by the heads the consumer holds.

    A head broadcast's own frame can show the full query-head repetition
    (and, before Ulysses' interior A2A, a sequence-cut producer).  The reader
    is nevertheless in the Ulysses frame: it holds `h_C` heads and at least
    one stored head serves each of them.  The traffic-derived fan-out must
    divide the operand's head extent, and the resulting cache read must be
    the stored-head read TP makes.  The elementwise/key-axis broadcasts used
    by an indexer are also here through their attention kernels: they either
    have no head fan-out or remain divisible by the stored head count.
    """
    cases = (
        ("deepseek-ai__DeepSeek-V4-Pro", "flash_attention"),
        ("deepseek-ai__DeepSeek-V3.2", "flash_dsa"),
        ("deepseek-ai__DeepSeek-V4-Flash", "flash_attention"),
        ("Qwen__Qwen3-8B", "flash_attention"),
    )
    checked = 0
    for snapshot, op in cases:
        uly = _attention_rows(snapshot, "cp_ep", "ulysses", 8192, op)
        tp = _attention_rows(snapshot, "tp_ep", "ulysses", 8192, op)
        assert uly and uly.keys() == tp.keys(), snapshot
        for key, row in uly.items():
            signature = get_signature(op_signature(row["op"]))
            for index, fanout in sorted(_fanouts(row).items()):
                operand = row["shape_in"][index]
                axis = (None if signature is None
                        else signature.head_axis(index, len(operand)))
                if fanout <= 1 or axis is None:
                    continue
                heads = int(row["shape_in"][index][axis])
                assert fanout <= heads, (key, index, fanout, heads)
                assert heads % fanout == 0, (key, index, fanout, heads)
                checked += 1
            assert _kv(row) == pytest.approx(
                _kv(tp[key]), rel=1e-9), (key, "Ulysses KV left TP's frame")
    assert checked, "no broadcast-head operand was checked"


def test_a_rank_scores_the_pairs_of_the_queries_it_holds():
    """Ulysses cuts heads, PCP cuts query rows: either way a rank does
    `1/cp` of one rank's attention work (`invariants._inv_cp_scheme`), and
    a DSA indexer — sequence-cut under both — `1/cp` of its scoring."""
    def flops(snapshot, mode, scheme, op):
        rows = _attention_rows(snapshot, mode, scheme, 8192, op)
        assert rows, (snapshot, mode, scheme, op)
        return sum(row["flops_per_rank"] for row in rows.values())

    for snapshot, op in (("Qwen__Qwen3-8B", "flash_attention"),
                         ("deepseek-ai__DeepSeek-V3.2", "flash_dsa"),
                         ("deepseek-ai__DeepSeek-V3.2", "indexer_logits")):
        whole = flops(snapshot, "dp_ep", "ulysses", op)
        for scheme in ("ulysses", "pcp"):
            assert flops(snapshot, "cp_ep", scheme, op) == pytest.approx(
                whole / _DEGREE, rel=1e-9), (snapshot, op, scheme)


def test_a_dcp_rank_scores_the_pairs_of_the_keys_it_holds():
    """Decode context parallelism cuts the other side of the matrix: a rank
    holds `1/dcp` of the keys and gathers the heads, so its attention work
    is the dcp=1 rank's — plain and speculative decode alike. Llama-3.2-3B
    under TP4 × PCP2 keeps the dcp group inside one KV-head shard."""
    doc = json.loads((SNAP_DIR / "meta-llama__Llama-3.2-3B.json").read_text())

    def decode_flops(dcp, s_verify):
        result = estimate_capacity(
            config=doc["config"], model_id=doc["model_id"], gpu_sku="B200",
            num_gpus=_DEGREE, avg_input_tokens=8192,
            # The cache length divides by dcp: 8192 + 250/2 + s_verify - 1.
            avg_output_tokens=250 if s_verify == 4 else 256,
            gpu_memory_fraction=0.9, selection_mode="theoretical_only",
            parallel=ParallelConfig(tp_attention=4, tp_dense_ffn=4, cp=2,
                                    dcp=dcp, cp_scheme="pcp"),
            decode_batch_size=1, prefill_batch_size=1, s_verify=s_verify)
        rows = [row for row in result["per_op_timing_decode"]
                if row.get("op") == "flash_attention"]
        assert rows
        return sum(row["flops_per_rank"] for row in rows)

    for s_verify in (1, 4):
        assert decode_flops(2, s_verify) == pytest.approx(
            decode_flops(1, s_verify), rel=1e-9), s_verify
