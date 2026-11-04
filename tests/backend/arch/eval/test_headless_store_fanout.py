"""A head broadcast says WHETHER the tensor under it stores heads at all.

`traffic._Fanout` already carried how many times a view repeats a head axis,
and a reader that must rebuild the stored shape asked only that number. The
number stops answering at one head per rank: a view holding a single head
repeats its store once, which is also what a per-head store reads unviewed.
Absorbed MLA's one latent was keyed there as the per-head K/V of the
non-absorbed prefill geometry — a decode kernel nobody ships — so every
such cell (128-way and wider attention tensor parallelism on each MLA model)
was refused instead of measured. The fact is now kept beside the ratio.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import traffic
from backend.arch.eval.dim import Dim


@dataclass
class _Shape:
    """A node's output axes, as the derivation reads them (`.out`)."""
    out: List[Any]


def _axes(*named: Any) -> _Shape:
    return _Shape([Dim(sp.Symbol(name), value) for name, value in named])


def test_a_broadcast_over_one_head_still_states_a_headless_store():
    # Absorbed MLA at 128-way attention TP: one query head per rank, so the
    # latent's view [B, 1, s_kv, d] repeats its store exactly once.
    view = _axes(("B", 1), ("n_heads", 1), ("s_kv", 1152), ("d_head", 576))
    store = _axes(("B", 1), ("s_kv", 1152), ("d_head", 576))
    factor = traffic._own_fanout(view, store)
    assert factor.headless_store
    assert factor.total == 1 and factor.head == 1


def test_the_same_broadcast_over_many_heads_states_it_too():
    view = _axes(("B", 1), ("n_heads", 128), ("s_kv", 1152), ("d_head", 576))
    store = _axes(("B", 1), ("s_kv", 1152), ("d_head", 576))
    factor = traffic._own_fanout(view, store)
    assert factor.headless_store
    assert factor.head == 128


def test_a_grouped_kv_store_is_not_headless():
    # GQA: 8 stored KV heads shown to 32 query heads. The store HAS heads,
    # and the stored count is the extent over the fan-out.
    view = _axes(("B", 1), ("n_heads", 32), ("s_kv", 64), ("d_head", 128))
    store = _axes(("B", 1), ("n_kv_heads", 8), ("s_kv", 64), ("d_head", 128))
    factor = traffic._own_fanout(view, store)
    assert not factor.headless_store
    assert factor.head == 4


def test_one_stored_head_shown_to_one_query_head_is_not_headless():
    # The case the ratio cannot tell from the first test: an MQA store with
    # its own head axis, viewed at one head. Nothing may be squeezed here.
    view = _axes(("B", 1), ("n_heads", 1), ("s_kv", 64), ("d_head", 128))
    store = _axes(("B", 1), ("n_kv_heads", 1), ("s_kv", 64), ("d_head", 128))
    assert not traffic._own_fanout(view, store).headless_store


def test_a_packed_head_axis_split_apart_is_not_a_headless_store():
    # `head_split` and friends are view-only components too, and the tensor
    # under them has no head AXIS either — its heads are packed into one.
    # Reading the rank alone, every GQA fused rope lost its projection.
    view = _axes(("B", 8), ("n_kv_heads", 8), ("s_query", 1), ("d_head", 128))
    store = _Shape([Dim(sp.Symbol("B"), 8), Dim(sp.Symbol("s_query"), 1),
                    Dim(sp.Symbol("n_kv_heads") * sp.Symbol("d_head"), 1024)])
    assert not traffic._own_fanout(view, store).headless_store


def test_a_broadcast_on_another_axis_says_nothing_about_heads():
    # DeepSeek-V4's indexer head weights [B, S, H] expanded to the score's
    # [B, H, S, T]: the repetition is over keys, and the store's head axis
    # is intact.
    view = _axes(("B", 1), ("n_heads", 64), ("s_query", 8), ("s_kv", 2048))
    store = _axes(("B", 1), ("s_query", 8), ("n_heads", 64))
    factor = traffic._own_fanout(view, store)
    assert not factor.headless_store


def test_the_product_of_a_chain_keeps_the_store_it_found():
    headless = traffic._Fanout.of(
        total=1, head=1, head_names=frozenset({"n_heads"}),
        headless_store=True)
    assert (headless * traffic._Fanout()).headless_store
    assert (traffic._Fanout() * headless).headless_store
    assert not (traffic._Fanout() * traffic._Fanout()).headless_store
