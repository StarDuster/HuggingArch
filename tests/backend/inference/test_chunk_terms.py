"""The backend-owned invariant for folding duplicate chunk launch keys."""
from __future__ import annotations

from fractions import Fraction

import pytest

from backend.inference.chunk_terms import (ChunkLaunchTerm, chunk_term_count,
                                           merge_chunk_launch_terms)


def _term(key, launches, **over):
    over.setdefault("formula_weight", 1)
    over.setdefault("inputs", {"flops": 100, "anchor": None})
    over.setdefault("context", {"gpu": "H100"})
    over.setdefault("cadence", 1)
    return ChunkLaunchTerm(key=("sig", key), launches=launches, **over)


def test_safe_duplicate_chunk_keys_add_launches_only():
    terms = [
        _term("k", 2),
        _term("k", 3, formula_weight=1,
              inputs={"flops": 100, "anchor": None},
              context={"gpu": "H100"}, cadence=1),
        _term("other", 7),
    ]

    assert merge_chunk_launch_terms(terms) == [
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


def test_negative_launch_counts_are_refused():
    with pytest.raises(ValueError, match="negative"):
        merge_chunk_launch_terms([_term("k", -1)])
