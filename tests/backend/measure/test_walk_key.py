"""A walk is named by the facts it is a function of (`scope.of_walk`).

A walk — the graph walked once for one deployment, every row carrying its
law in the workload's counts and the card rather than a number — is the
expensive half of every number this site publishes, and it is shared
across the cards of one walk class. So what it is keyed BY is the whole
enforcement: a component missing from the key lets one card's rows answer
another card's request, and a component that does not belong splits one
walk into entries holding identical rows.

These tests state both directions, component by component.
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.measure.scope import NONE, of_walk


def _key(**over):
    args = dict(version="rv1", model="m/x", layout={"tp_attention": 1},
                card_class="device=cuda-cc=9.0-peaks=bf16", precision="shipped",
                pp_stage=None, anchor={"s_input": 4096})
    args.update(over)
    return of_walk(**args).key


class TestEveryComponentIsInTheKey(unittest.TestCase):
    """Each of the six moves the key, because each changes which rows the
    walk has or what they are laws of."""

    def test_each_component_moves_the_key(self):
        base = _key()
        for name, other in (
                # the engine + this model's footprint: a plan, binding or
                # sheet edit lands on a fresh key with no invalidation pass
                ("version", "rv2"),
                ("model", "m/y"),
                # the per-rank projection is a rule ON the walk
                ("layout", {"tp_attention": 8}),
                # the structural half of a card: a dispatch it answers
                # differently fuses the same graph differently
                ("card_class", "device=cuda-cc=10.0-peaks=bf16+fp4"),
                ("precision", "fp8-blockwise"),
                # two cards of one class with different HBM can run out on
                # different stages, and then they are different walks
                ("pp_stage", 1),
                # the point the walk was taken at, and what removes rows
                # before anything is priced
                ("anchor", {"s_input": 8192}),
        ):
            with self.subTest(component=name):
                self.assertNotEqual(base, _key(**{name: other}), name)

    def test_the_same_facts_are_the_same_entry(self):
        """Two spellings of one walk are one entry — including the two
        consumers that used to key the same facts under different `kind`
        strings and so could never share 33 GB of identical work."""
        self.assertEqual(_key(), _key())
        self.assertEqual(_key(layout={"tp_attention": 1}),
                         _key(layout={"tp_attention": 1}))


class TestAWalkIsTheoryUnderNoScope(unittest.TestCase):
    def test_reads_no_measurements_and_may_be_shared(self):
        """A walk is theory: the measured track is folded over its points
        afterwards. So one entry serves every selection mode and every
        Kernel Bench state, and it may enter the tier other machines read.
        """
        identity = of_walk("rv1", model="m/x", layout=None, card_class="k",
                           precision="shipped", pp_stage=None, anchor=None)
        self.assertEqual(identity.kind, "walk")
        self.assertFalse(identity.reads)
        self.assertIs(identity.scope, NONE)
        self.assertTrue(identity.remote_ok)
        self.assertEqual(identity.measurements, "none")


class TestPointTimeAxesAreNotWalkComponents(unittest.TestCase):
    def test_routing_skew_is_substituted_not_keyed(self):
        """`ep_load_factor` changes a routed tensor's busiest-rank reading at
        a point; it changes neither the row set nor the walk's identity.
        """
        self.assertNotIn("ep_load_factor",
                         inspect.signature(of_walk).parameters)
        identity = of_walk("rv1", model="m/x", layout=None, card_class="k",
                           precision="shipped", pp_stage=None, anchor=None)
        self.assertNotIn("ep_load_factor", identity.content)


if __name__ == "__main__":
    unittest.main()
