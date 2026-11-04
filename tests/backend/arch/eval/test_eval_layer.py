"""`Layer[T]` / `Rewrite` — the annotation layer and the graph-transform map.

Falsification targets, in the order they would actually bite:
  - absence must be distinguishable from a `None` value, or an un-inferred shape
    and an un-covered node become the same thing;
  - `zip` must intersect, because sideband layers (measured corpus, quant
    formats) legitimately cover fewer nodes than propagated ones;
  - `rebase` must refuse to guess a fused node's value, because the plausible
    guess (take the first origin) under-counts by exactly the fan-in.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.layer import (
    Layer, MissingValue, Rewrite,
)


@dataclass
class _N:
    """Stand-in node: a layer only ever needs `uid` off a node."""
    uid: int
    name: str = "n"
    children: tuple = ()


class _P:
    """Stand-in program: `leaves` and `map_with_node` only need `by_uid`."""

    def __init__(self, *nodes: _N) -> None:
        self._by_uid = {n.uid: n for n in nodes}

    def by_uid(self, uid: int) -> _N:
        return self._by_uid[uid]


class TestReading(unittest.TestCase):
    def test_accepts_node_or_bare_uid(self):
        lay = Layer.build(None, [(_N(1), "a"), (2, "b")])
        self.assertEqual(lay[_N(1)], "a")
        self.assertEqual(lay[1], "a")
        self.assertEqual(lay[2], "b")

    def test_absent_raises_but_none_value_returns(self):
        lay = Layer.build(None, [(1, None)])
        self.assertIsNone(lay[1], "None is a value, not an absence")
        with self.assertRaises(MissingValue):
            lay[2]
        self.assertNotIn(2, lay)

    def test_get_defaults(self):
        lay = Layer.build(None, [(1, "a")])
        self.assertEqual(lay.get(1), "a")
        self.assertIsNone(lay.get(2))
        self.assertEqual(lay.get(2, "z"), "z")

    def test_non_node_is_a_type_error(self):
        lay = Layer.build(None, [(1, "a")])
        with self.assertRaises(TypeError):
            lay["not-a-node"]


class TestDeriving(unittest.TestCase):
    def test_map_is_lazy_in_the_sense_that_matters(self):
        """`display` is `map`-derived rather than stored. Cheap to assert:
        map produces a new layer with the same keys."""
        lay = Layer.build(None, [(1, 2), (2, 3)])
        doubled = lay.map(lambda v: v * 2)
        self.assertEqual([doubled[1], doubled[2]], [4, 6])
        self.assertEqual([lay[1], lay[2]], [2, 3], "map must not mutate")

    def test_zip_intersects(self):
        """A sideband layer covers fewer nodes; requiring equal domains would
        make the normal case an error."""
        theory = Layer.build(None, [(1, 10), (2, 20), (3, 30)])
        measured = Layer.build(None, [(2, 21)])
        both = theory.zip(measured)
        self.assertEqual(len(both), 1)
        self.assertEqual(both[2], (20, 21))
        self.assertNotIn(1, both)

    def test_zip_with_combines(self):
        shapes = Layer.build(None, [(1, (4, 8))])
        formats = Layer.build(None, [(1, 2)])          # bytes per element
        nbytes = shapes.zip_with(formats,
                                lambda sh, w: sh[0] * sh[1] * w)
        self.assertEqual(nbytes[1], 64)

    def test_filter(self):
        lay = Layer.build(None, [(1, 0), (2, 5)])
        self.assertEqual(len(lay.filter(bool)), 1)


class TestAggregating(unittest.TestCase):
    """`leaves` + `group_by` — attribution as a join, not a second traversal."""

    def test_leaves_drops_the_parents(self):
        """A component and its children both carry a role, so an attribution
        that summed both would count the subtree twice."""
        leaf, other = _N(2), _N(3)
        parent = _N(1, children=(leaf, other))
        lay = Layer(_P(parent, leaf, other), {1: 30, 2: 10, 3: 20})
        self.assertEqual(sorted(lay.leaves().values), [2, 3])

    def test_leaves_needs_a_program_rather_than_assuming(self):
        with self.assertRaises(ValueError):
            Layer(None, {1: 10}).leaves()

    def test_group_by_folds_into_the_key_layers_buckets(self):
        cost = Layer.build(None, [(1, 10), (2, 20), (3, 5)])
        roles = Layer.build(None, [(1, "attention"), (2, "attention"),
                                   (3, "ffn")])
        self.assertEqual(cost.group_by(roles, sum),
                         {"attention": 30, "ffn": 5})

    def test_a_node_with_no_key_is_absent_rather_than_neither(self):
        """The engine's classifier falls through a chain of tests into a
        catch-all bucket; a node that is not in the key layer simply does not
        appear, which is the same statement without the fall-through."""
        cost = Layer.build(None, [(1, 10), (2, 20)])
        roles = Layer.build(None, [(1, "attention")])
        self.assertEqual(cost.group_by(roles, sum), {"attention": 10})

    def test_none_is_a_bucket_and_not_an_absence(self):
        """Same distinction `MissingValue` draws on the value side: a key layer
        may legitimately hold `None` for a node with no role, and that is an
        answer about the node rather than a gap in the layer."""
        cost = Layer.build(None, [(1, 10), (2, 20)])
        keys = Layer.build(None, [(1, None), (2, None)])
        self.assertEqual(cost.group_by(keys, sum), {None: 30})


class TestRewrite(unittest.TestCase):
    def test_identity_rewrite_leaves_a_layer_alone(self):
        lay = Layer.build(None, [(1, 10), (2, 20)])
        out = lay.rebase(Rewrite(), None, combine=sum)
        self.assertEqual((out[1], out[2]), (10, 20))
        self.assertTrue(Rewrite().is_identity())

    def test_fused_node_takes_the_combination_of_its_origins(self):
        """Three projections fuse into one QKV node: cost combines by sum."""
        cost = Layer.build(None, [(1, 100), (2, 100), (3, 100), (9, 7)])
        rw = Rewrite(origins={10: (1, 2, 3)}, dropped=frozenset({1, 2, 3}))
        out = cost.rebase(rw, None, combine=sum)
        self.assertEqual(out[10], 300, "fused cost must be the sum, not one of")
        self.assertEqual(out[9], 7, "untouched node keeps its value")
        for gone in (1, 2, 3):
            self.assertNotIn(gone, out)

    def test_combine_is_mandatory(self):
        """No default: the plausible default (first origin) under-counts a fused
        node by its fan-in, which is the failure mode that reads as fine."""
        cost = Layer.build(None, [(1, 100)])
        with self.assertRaises(TypeError):
            cost.rebase(Rewrite(origins={10: (1,)}), None)   # type: ignore[call-arg]

    def test_dropped_without_replacement_disappears(self):
        lay = Layer.build(None, [(1, "a"), (2, "b")])
        out = lay.rebase(Rewrite(dropped=frozenset({1})), None, combine=sum)
        self.assertNotIn(1, out)
        self.assertEqual(out[2], "b")

    def test_origins_absent_from_the_layer_produce_no_value(self):
        """A sideband layer that never covered the fused inputs must not
        materialise a value for the fused node out of nothing."""
        sparse = Layer.build(None, [(9, "x")])
        rw = Rewrite(origins={10: (1, 2)}, dropped=frozenset({1, 2}))
        out = sparse.rebase(rw, None, combine=lambda vs: vs[0])
        self.assertNotIn(10, out)

    def test_composition_flattens_through_the_intermediate_graph(self):
        """Two transforms in sequence: a→b fuses 1,2→10; b→c fuses 10,3→20.
        A layer keyed on the original graph must reach 20 from {1,2,3}."""
        first = Rewrite(origins={10: (1, 2)}, dropped=frozenset({1, 2}))
        second = Rewrite(origins={20: (10, 3)}, dropped=frozenset({10, 3}))
        both = first.then(second)
        self.assertEqual(set(both.sources(20)), {1, 2, 3})
        cost = Layer.build(None, [(1, 1), (2, 2), (3, 3)])
        out = cost.rebase(both, None, combine=sum)
        self.assertEqual(out[20], 6)


if __name__ == "__main__":
    unittest.main()
