"""A score's selection operand is read by its axes, not by its rank.

The batch and query axes are the query operand's own dims; an axis left over
is a per-head selection and must be one per KV head (MiniMax-M3 MSA). A
rank-4 selection over the query heads is refused rather than laid out as if
it were per KV head. The accepted shapes (DSA's `[B, s_query, k]`, MSA's
`[B, Hkv, s_query, k]` under attention TP) are walked end to end by
`tests/backend/arch/eval/test_indexer_fusion.py`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy as sp

from backend.arch.eval.dim import Dim
from backend.arch.layout.attention import _under_selection
from backend.arch.layout.edge import EdgeState


def _shape(*axes):
    return tuple(Dim(sp.Symbol(name), value) for name, value in axes)


QUERY = _shape(("B", 1), ("n_heads", 64), ("s_query", 128), ("d_head", 128))
KEY = _shape(("B", 1), ("n_kv_heads", 4), ("s_kv", 128), ("d_head", 128))
NODE = SimpleNamespace(display_path="attn.qk_score")


def _fold(selection):
    mask = EdgeState((None,) * len(selection), None, None)
    return _under_selection(None, NODE, mask, (QUERY, KEY, selection), [],
                            None)


def test_a_selection_over_query_heads_is_refused():
    selection = _shape(("B", 1), ("n_heads", 64), ("s_query", 128),
                       ("k", 16))
    with pytest.raises(ValueError, match="one per KV head"):
        _fold(selection)


def test_a_selection_must_carry_the_query_rows():
    selection = _shape(("B", 1), ("n_kv_heads", 4), ("k", 16))
    with pytest.raises(ValueError, match="s_query axis exactly once"):
        _fold(selection)
