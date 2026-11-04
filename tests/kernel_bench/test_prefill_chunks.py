"""Pure chunked-prefill policy tests.

Everything in this module is arithmetic over ``PlanPoint`` values: no model
snapshot is resolved, no CUDA device is requested, and no lake artifact is
opened.  That boundary is part of the contract being tested.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from backend.inference.phases import phase_binding
from kernel_bench.plan import PlanPoint
from kernel_bench.prefill_chunks import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    ChunkLaunchTerm,
    PrefillChunkPoint,
    chunk_term_count,
    decompose_prefill,
    merge_chunk_launch_terms,
    normalize_prefill_chunk_size,
    prefill_chunk_map,
)


def _chunks(seq_len, prefix=0, batch=1, chunk_size=DEFAULT_PREFILL_CHUNK_SIZE):
    return decompose_prefill(
        PlanPoint(seq_len, prefix, 1, batch),
        prefill_chunk_size=chunk_size,
    )


def test_campaign_map_and_runtime_map_are_one_decomposition():
    endpoint = PlanPoint(1234, 234, 1, 5)
    chunks = decompose_prefill(endpoint, prefill_chunk_size=100)
    from backend.inference.prefill_geometry import (
        PrefillEndpointGeometry, prefill_chunk_map as runtime_map)

    campaign = prefill_chunk_map(
        endpoint, prefill_chunk_size=100, deployment="tp=1")
    runtime = runtime_map(
        PrefillEndpointGeometry(1234, 234, 1, 5), prefill_chunk_size=100)
    assert campaign == {
        **runtime, "deployment": "tp=1", "endpoint_index": 0}
    assert campaign["chunks"] == [{
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "width": chunk.width,
        "launch_batch": chunk.launch_batch,
    } for chunk in chunks]
    assert [chunk.point for chunk in chunks] == [
        PlanPoint(chunk.end, chunk.start, 1, chunk.launch_batch)
        for chunk in chunks]


def test_a_runtime_schedule_serializes_its_materialized_intervals():
    from backend.inference.prefill_geometry import (
        PrefillChunkSchedule, PrefillEndpointGeometry)

    schedule = PrefillChunkSchedule.materialize(
        PrefillEndpointGeometry(1234, 234, 1, 1),
        prefill_chunk_size=100)
    rebatched = schedule.with_endpoint_batch(5)

    assert [(c.start, c.end) for c in rebatched.chunks] == [
        (c.start, c.end) for c in schedule.chunks]
    assert rebatched.chunk_map() == prefill_chunk_map(
        PlanPoint(1234, 234, 1, 5), prefill_chunk_size=100)


def test_one_million_tokens_are_128_full_chunks_with_no_tail():
    chunks = _chunks(1_048_576, prefix=0, batch=3, chunk_size=8192)

    assert len(chunks) == 128
    assert [(c.start, c.end) for c in chunks] == [
        (index * 8192, (index + 1) * 8192) for index in range(128)
    ]
    assert all(c.width == 8192 for c in chunks)
    assert all(c.launch_batch == 1 for c in chunks)
    assert all(c.endpoint.batch == 3 for c in chunks)


def test_a_prefix_with_remainder_is_three_full_chunks_and_one_tail():
    chunk, remainder = 100, 7
    prefix = 23
    seq = prefix + 3 * chunk + remainder
    chunks = _chunks(seq, prefix=prefix, batch=5, chunk_size=chunk)

    assert len(chunks) == 4
    assert [(c.index, c.start, c.end, c.width) for c in chunks] == [
        (0, 23, 123, 100),
        (1, 123, 223, 100),
        (2, 223, 323, 100),
        (3, 323, 330, 7),
    ]
    assert all(c.point.seq_len - c.point.s_prefix_cached <= chunk
               for c in chunks)
    # First-stage scheduler assumption: endpoint B_total is fold provenance,
    # not a second request packed into an already-full chunk launch.
    assert all(c.launch_batch == 1 for c in chunks)
    assert {c.endpoint for c in chunks} == {
        PlanPoint(seq, prefix, 1, 5)
    }


def test_a_query_within_one_chunk_is_exactly_todays_point():
    for prefix, query, batch in ((0, 8192, 1), (1_040_384, 8192, 4),
                                 (1_048_192, 4096, 7)):
        endpoint = PlanPoint(prefix + query, prefix, 1, batch)
        chunks = decompose_prefill(endpoint, prefill_chunk_size=8192)

        assert len(chunks) == 1
        assert chunks[0].point == endpoint
        assert chunks[0].launch_batch == batch


def test_explicit_unchunked_values_keep_the_endpoint_batch_and_geometry():
    endpoint = PlanPoint(1_048_576, 0, 1, 3)
    for explicit in (0, None):
        chunks = decompose_prefill(endpoint,
                                   prefill_chunk_size=explicit)

        assert len(chunks) == 1
        assert chunks[0].point == endpoint
        assert chunks[0].launch_batch == 3


def test_each_chunk_reads_the_incumbent_phase_binding():
    chunk, remainder = 8192, 11
    prefix = 100_000
    seq = prefix + 3 * chunk + remainder
    chunks = _chunks(seq, prefix=prefix, chunk_size=chunk)

    for chunk_point in chunks:
        got = phase_binding(
            phase="prefill",
            avg_input_tokens=chunk_point.point.seq_len,
            avg_output_tokens=0,
            s_prefix_cached=chunk_point.point.s_prefix_cached,
        )
        width = chunk_point.width
        assert got["s_query"] == width
        assert got["s_kv"] == chunk_point.end
        assert got["s_pairs"] == Fraction(
            width * chunk_point.start + width * (width + 1) // 2
        )


def test_chunk_causal_pairs_sum_to_the_endpoint_pairs():
    prefix, chunk, remainder = 12345, 700, 29
    seq = prefix + 2 * chunk + remainder
    endpoint_binding = phase_binding(
        phase="prefill", avg_input_tokens=seq, avg_output_tokens=0,
        s_prefix_cached=prefix)
    summed = sum(
        phase_binding(
            phase="prefill",
            avg_input_tokens=chunk_point.point.seq_len,
            avg_output_tokens=0,
            s_prefix_cached=chunk_point.point.s_prefix_cached,
        )["s_pairs"]
        for chunk_point in _chunks(seq, prefix=prefix, chunk_size=chunk)
    )

    assert summed == endpoint_binding["s_pairs"]


def test_a_short_cached_prefix_query_is_batched_by_new_tokens():
    from kernel_bench.sweep import _feasible_batches, PREFILL_LAUNCH_TOKEN_CAP

    class Cap:
        max_batch = 64
        max_prefill_batch = 64
        beyond_max_context = False

    query = 4096
    seq = 1_048_576
    prefix = seq - query
    got = _feasible_batches(
        Cap(), "prefill", seq, ("max",), prefill_query_tokens=query)

    # The old total-context reading would floor this endpoint to B=1; the
    # scheduler admits B by the 4096 fresh query tokens.
    assert max(got) == PREFILL_LAUNCH_TOKEN_CAP // query
    endpoint = PlanPoint(seq, prefix, 1, max(got))
    chunks = decompose_prefill(endpoint, prefill_chunk_size=8192)
    assert len(chunks) == 1
    assert chunks[0].point == endpoint
    assert chunks[0].point.batch == 8


def test_chunk_policy_refuses_ambiguous_widths_and_accepts_explicit_unchunked():
    assert DEFAULT_PREFILL_CHUNK_SIZE == 8192
    assert normalize_prefill_chunk_size(0) is None
    assert normalize_prefill_chunk_size(None) is None
    assert normalize_prefill_chunk_size(4096) == 4096
    with pytest.raises(ValueError):
        normalize_prefill_chunk_size(-1)
    with pytest.raises(TypeError):
        normalize_prefill_chunk_size("8192")
    with pytest.raises(TypeError):
        normalize_prefill_chunk_size(True)


def test_invalid_endpoints_are_refused_before_any_model_work():
    for endpoint in (PlanPoint(0, 0, 1, 1), PlanPoint(100, 100, 1, 1),
                     PlanPoint(100, -1, 1, 1), PlanPoint(100, 20, 1, 0)):
        with pytest.raises((TypeError, ValueError)):
            decompose_prefill(endpoint, prefill_chunk_size=8192)


def test_the_wrapper_is_a_plain_launch_provenance_record():
    endpoint = PlanPoint(2048, 1024, 1, 2)
    chunk = PrefillChunkPoint(endpoint, 0, 1024, 2048, 2)

    assert chunk.width == 1024
    assert chunk.point == PlanPoint(2048, 1024, 1, 2)


def _term(key, launches, **over):
    over.setdefault("formula_weight", 1)
    over.setdefault("inputs", {"flops": 100, "anchor": None})
    over.setdefault("context", {"gpu": "H100"})
    over.setdefault("cadence", 1)
    return ChunkLaunchTerm(
        key=("sig", key), launches=launches, **over)


def test_safe_duplicate_chunk_keys_add_launches_only():
    terms = [
        _term("k", 2),
        _term("k", 3, formula_weight=1, inputs={"flops": 100, "anchor": None},
              context={"gpu": "H100"}, cadence=1),
        _term("other", 7),
    ]

    merged = merge_chunk_launch_terms(terms)

    assert merged == [
        ChunkLaunchTerm(
            key=("sig", "k"), launches=5, formula_weight=1,
            inputs={"flops": 100, "anchor": None},
            context={"gpu": "H100"}, cadence=1),
        _term("other", 7),
    ]
    assert chunk_term_count(terms) == {
        "terms": 3, "distinct_keys": 2, "zero_cost_terms": 0}


def test_duplicate_chunk_keys_with_different_fold_laws_stay_separate():
    baseline = _term("k", 2)
    variants = (
        _term("k", 3, formula_weight=3),
        _term("k", 3, zero_cost=True),
        _term("k", 3, inputs={"flops": 101, "anchor": None}),
        _term("k", 3, context={"gpu": "B300"}),
        _term("k", 3, cadence=Fraction(1, 3)),
    )

    for variant in variants:
        assert merge_chunk_launch_terms([baseline, variant]) == [
            baseline, variant]


def test_a_one_token_tail_is_its_own_prefill_geometry():
    chunks = _chunks(2 * 8192 + 1, chunk_size=8192)

    assert len(chunks) == 3
    assert (chunks[-1].width, chunks[-1].start, chunks[-1].end) == (
        1, 2 * 8192, 2 * 8192 + 1)
    binding = phase_binding(
        phase="prefill", avg_input_tokens=chunks[-1].end,
        avg_output_tokens=0,
        s_prefix_cached=chunks[-1].start)
    assert binding["s_query"] == 1
    assert binding["s_kv"] == 2 * 8192 + 1
