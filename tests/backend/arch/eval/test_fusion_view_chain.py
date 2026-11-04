"""A quantised write commutes above the views it reads through
(`backend.arch.eval.fusion.ViewChains`).

The serving graph keeps an activation cast where the framework quantises —
on the tensor the GEMM is handed, after any view (`materialize_cast`). A
view moves no bytes, so where it hands every run of elements sharing one
scale to one such run, quantising after it is quantising before it, and a
plan that names the cast as a member of the producer's launch
(`fused_inv_rope`) binds the producer across the views.

Synthetic graphs, so each case holds one clause of the law:

    attn_out ─┬ slice ─ rope_emb_inverse ─┐
              └ slice ────────────────────┴ concat ─ <views> ─ cast ─ matmul
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.eval.fusion import ViewChains, _dag_matches
from backend.arch.eval.traffic import TensorReaders
from backend.arch.eval.values import Shape
from backend.inference.fusion_plans import FUSION_SCHEMES

FP8_128 = ACTIVATION_FORMATS["fp8_token_group128_ue8m0"]
HEADS, D_HEAD, ROPE = 4, 512, 64


class _Graph:
    """Leaves, their walked shapes and the program surface `ViewChains`
    reads (`nodes`)."""

    def __init__(self) -> None:
        self.leaves: List[Any] = []
        self.shapes = {}
        #: uid -> `(template, parent)`, as `fusion._scopes` states it.
        self.scopes = {}

    def add(self, op: str, srcs: Sequence[Any], ins: Sequence[Tuple],
            out: Tuple, **attrs: Any) -> Any:
        node = N(uid=len(self.leaves) + 1, op=op, children=(), parent=None,
                 secondary_outputs={},
                 display_path=f"{op}{len(self.leaves) + 1}",
                 out_shape=None, axis=attrs.pop("axis", None),
                 axes=attrs.pop("axes", None),
                 output_format=attrs.pop("output_format", None),
                 inputs_obj=tuple(N(src=s, port="", external_sid=None)
                                  for s in srcs))
        self.leaves.append(node)
        self.scopes[node.uid] = attrs.pop("scope", ("t", 0))
        self.shapes[node.uid] = Shape(ins=tuple(tuple(i) for i in ins),
                                      out=tuple(out))
        return node

    def nodes(self):
        return iter(self.leaves)

    def match(self, shapes: Any = None):
        pattern = FUSION_SCHEMES["fused_inv_rope"].naive_pattern
        views = ViewChains(self, self.shapes if shapes is None else shapes,
                           TensorReaders(self, pinned_uids=frozenset()))
        roles = N(of_node=lambda uid: frozenset())
        return _dag_matches(pattern, self.leaves, self.scopes, roles, None,
                            views)


def _producer(g: _Graph):
    """`attn_out` through the un-rotation to `concat`, `[1, H, 1, D]`."""
    o = (1, HEADS, 1, D_HEAD)
    attn = g.add("attention_apply", (), (), o)
    pe = g.add("slice", (attn,), (o,), (1, HEADS, 1, ROPE))
    nope = g.add("slice", (attn,), (o,), (1, HEADS, 1, D_HEAD - ROPE))
    rope = g.add("rope_emb_inverse", (pe,), ((1, HEADS, 1, ROPE),),
                 (1, HEADS, 1, ROPE))
    return g.add("concat", (nope, rope),
                 ((1, HEADS, 1, D_HEAD - ROPE), (1, HEADS, 1, ROPE)), o)


def _head_merge(g: _Graph, src: Any, scope: Any = ("t", 0)):
    """`[1, H, 1, D]` → transpose → `[1, 1, H, D]` → view_merge →
    `[1, 1, H·D]`: the innermost axis kept, then merged whole."""
    t = g.add("transpose", (src,), ((1, HEADS, 1, D_HEAD),),
              (1, 1, HEADS, D_HEAD), axes=(0, 2, 1, 3), scope=scope)
    return g.add("view_merge", (t,), ((1, 1, HEADS, D_HEAD),),
                 (1, 1, HEADS * D_HEAD), scope=scope)


def _cast_and_matmul(g: _Graph, src: Any, dims: Tuple, fmt: Any = FP8_128):
    cast = g.add("quantise_activation", (src,), (dims,), dims,
                 output_format=fmt)
    g.add("linear", (cast,), (dims,), (*dims[:-1], 7168))
    return cast


def _bound_ops(matches) -> List[set]:
    return [{node.op for node in bound.values()} for bound in matches]


def test_a_cast_behind_block_keeping_views_joins_the_launch():
    """`concat → transpose → view_merge → cast`: the transpose keeps the
    innermost axis, the merge ends in it and 128 divides 512 and 2048, so
    the plan binds `concat` across the views and takes the cast in."""
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    # The larger match is met first, and the rewrite keeps the first.
    matches = g.match()
    assert _bound_ops(matches)[0] == {
        "slice", "rope_emb_inverse", "concat", "quantise_activation"}
    root, port, chain = ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0)
    assert root == concat.uid and port == "out"
    assert chain == {merged.uid, merged.inputs_obj[0].src.uid}
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).through(
        tuple(matches[0].values())) == chain


def test_a_cast_behind_a_narrowing_slice_stays_out():
    """DeepSeek-V3's `q_nope_slice`: 128 of a 192-wide row. The slice hands
    the cast fewer elements than it was handed — and a 128-group over 192
    would straddle rows — so it does not commute."""
    g = _Graph()
    wide = (1, HEADS, 1, 192)
    q = g.add("linear", (), (), wide)
    cut = g.add("slice", (q,), (wide,), (1, HEADS, 1, 128))
    cast = _cast_and_matmul(g, cut, (1, HEADS, 1, 128))
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None


def test_a_merge_whose_rows_a_group_straddles_stays_out():
    """Equal elements, but `[H, 192] → [H·192]` with 128-groups: a group of
    the merged row spans two heads' rows, so the scales are not the same
    scales."""
    g = _Graph()
    d = 192
    src = g.add("attention_apply", (), (), (1, 1, HEADS, d))
    merged = g.add("view_merge", (src,), ((1, 1, HEADS, d),),
                   (1, 1, HEADS * d))
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * d))
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None


def test_a_per_token_cast_does_not_commute_with_a_merge():
    """One scale per row: merging heads into the row changes the rows."""
    per_token = next(fmt for fmt in ACTIVATION_FORMATS.values()
                     if fmt.scale_granularity == "per-token")
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD), per_token)
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None


def test_a_view_another_reader_reads_keeps_the_cast_out():
    """The transpose's output is also read wide by someone else: quantising
    above it would hand that reader the FP8 tensor, so the cast stays out
    of the launch, as closure keeps any read member out."""
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    transpose = merged.inputs_obj[0].src
    g.add("rmsnorm", (transpose,), ((1, 1, HEADS, D_HEAD),),
          (1, 1, HEADS, D_HEAD))
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None
    assert _bound_ops(g.match()) == [
        {"slice", "rope_emb_inverse", "concat"}]


def test_a_producer_read_wide_elsewhere_keeps_the_cast_out():
    """`concat` itself has a second, wide reader."""
    g = _Graph()
    concat = _producer(g)
    g.add("rmsnorm", (concat,), ((1, HEADS, 1, D_HEAD),),
          (1, HEADS, 1, D_HEAD))
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None


def test_views_in_another_scope_keep_the_cast_out():
    """A launch is one component instance (`fusion._scopes`): views of
    another instance — another layer, another component — are not taken
    in, so neither is the cast behind them."""
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat, scope=("t", 99))
    _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    assert all("quantise_activation" not in ops
               for ops in _bound_ops(g.match()))


def test_the_scale_block_is_judged_on_the_ranks_tensor():
    """The cast groups the tensor ONE rank holds. Head dim 512 cut eight
    ways along the innermost axis leaves 64 per rank: a 128-group divides
    the model's 512 and 2048, not the rank's 64 and 256, so the views do
    not commute there although they do in the whole model."""
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    assert ViewChains(g, g.shapes, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is not None
    cut = D_HEAD // 8
    local = dict(g.shapes)
    transpose = merged.inputs_obj[0].src
    local[transpose.uid] = Shape(ins=((1, HEADS, 1, cut),),
                                 out=(1, 1, HEADS, cut))
    local[merged.uid] = Shape(ins=((1, 1, HEADS, cut),),
                              out=(1, 1, HEADS * cut))
    assert ViewChains(g, local, TensorReaders(g, pinned_uids=frozenset())).root(cast, 0) is None
    assert all("quantise_activation" not in ops
               for ops in _bound_ops(g.match(local)))


def test_a_distributed_walk_with_no_rank_shapes_refuses_to_guess():
    """A walk whose rule distributes tensors is no per-rank answer of its
    own; asked without the caller's per-rank layer, the question raises
    rather than reading the model's shapes as the rank's."""
    import pytest

    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    readers = TensorReaders(g, pinned_uids=frozenset())

    def chains(walk):
        return replace(ViewChains.of(g, walk), tensor_readers=readers)

    walk = N(rule=N(is_trivial=False), shapes=N(values=g.shapes))
    with pytest.raises(ValueError, match="per-rank"):
        chains(walk).root(cast, 0)
    trivial = N(rule=N(is_trivial=True), shapes=N(values=g.shapes))
    assert chains(trivial).root(cast, 0) is not None


def test_a_view_read_past_the_graph_edges_is_not_read_through():
    """A view whose output something reads past the edges (cached, a stream,
    a block's output: `TensorReaders.pinned`) keeps its tensor: fusion does
    not read a cast through it, as the graft does not move one above it."""
    g = _Graph()
    concat = _producer(g)
    merged = _head_merge(g, concat)
    cast = _cast_and_matmul(g, merged, (1, 1, HEADS * D_HEAD))
    assert ViewChains(g, g.shapes, TensorReaders(
        g, pinned_uids=frozenset())).root(cast, 0) is not None
    assert ViewChains(g, g.shapes, TensorReaders(
        g, pinned_uids=frozenset({merged.uid}))).root(cast, 0) is None
