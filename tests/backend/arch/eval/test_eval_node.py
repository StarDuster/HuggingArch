"""The structural layer: what a node must provide, and what a stage is.

These pin the contracts rather than the numbers. The numbers are the oracle's
job, and it passes unchanged either way — which is the point of a refactor that
only names things. What can still break here is the naming itself: a protocol
that drifts from the type it describes, or a `Stage` subclass that answers for a
kind of stage it is not.

Two of the four assertions were the reason to write the contracts down at all:

  - `Evaluable` is strictly narrower than `Identified`, and `FusedNode`
    satisfies exactly one of them. That asymmetry is what `Rewrite` encodes — a
    fusion hands out a correspondence between IDENTITIES, so the node it creates
    needs nothing else — and it was previously enforced by nothing.
  - a template with a declared count of ZERO contributes zero layers. `or 1`
    reads it as absent, which is the GLM-5.1 over-count, and now the rule lives
    on the template rather than in each of its three readers.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest

from tests.snapshots import (
    any_snapshot,
    shared_loader,
    shared_program,
    shared_walk)
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.launch import FusedNode
from backend.arch.eval.layer import Layer
from backend.arch.eval.node import (
    Evaluable, Identified,
)
from backend.arch.eval.program import (
    BlockStage, BlockTemplate, OpStage, Program, Stage,
)
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.values import Cost
from backend.arch.spec_tree import BlockMeta, SpecIRNode

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: A model with nested components and many layers, so the two rules a total
#: obeys are both non-trivially in force. Any snapshot would do for the shape of
#: the assertion; this one makes the margins large.
DEEP = "deepseek-ai__DeepSeek-V2-Lite"


def _resolved_snapshots():
    for path in sorted(glob.glob(str(SNAPSHOT_DIR / "*.json"))):
        try:
            config = json.loads(Path(path).read_text())["config"]
            yield os.path.basename(path), shared_loader().match(
                config, model_id=Path(path).stem.replace("__", "/", 1))
        except Exception:
            continue


def _first_analysis():
    """One analysis of `DEEP`, for the assertions that need a model rather than
    the corpus."""
    program = shared_program(DEEP.replace('__', '/', 1))
    return shared_walk(Deployment(program).at(batch=1, input_tokens=128))


class TestTheTwoContracts(unittest.TestCase):
    """`Evaluable` ⊂ `Identified`, and the two node kinds sit either side."""

    def test_a_spec_node_satisfies_both(self):
        node = SpecIRNode(name="q_proj", op="linear", is_component=False)
        self.assertIsInstance(node, Identified)
        self.assertIsInstance(node, Evaluable)

    def test_a_fused_node_has_an_identity_and_not_a_body(self):
        """It carries a boundary, but no dims or counts to evaluate again."""
        fused = FusedNode(uid=7, op="fused_add_rmsnorm", origins=(1, 2))
        self.assertIsInstance(fused, Identified)
        self.assertNotIsInstance(fused, Evaluable)

    def test_the_walk_only_ever_sees_evaluable_nodes(self):
        """Whatever `iter_tree` emits, the evaluator can evaluate. A node that
        satisfied only the narrower contract would fail at the first `dims_raw`,
        four frames into a recursion, rather than here."""
        for name, resolved in _resolved_snapshots():
            with self.subTest(name):
                program = Program.of(resolved)
                for node in program.nodes():
                    self.assertIsInstance(node, Evaluable)


class TestBlockTemplate(unittest.TestCase):

    def test_a_declared_zero_is_zero_layers(self):
        template = BlockTemplate(
            meta=BlockMeta(template_name="b", count=0, stack_name="layers"),
            roots=())
        self.assertEqual(template.count, 0)

    def test_an_absent_count_is_one_layer(self):
        """A singleton wrapper — no `repeat:`, no `assignments:` — declares no
        count and exists once."""
        template = BlockTemplate(
            meta=BlockMeta(template_name="b", count=None,  # type: ignore[arg-type]
                           stack_name=None),
            roots=())
        self.assertEqual(template.count, 1)

    def test_every_shipped_template_is_reachable_by_name(self):
        """Every invocation carries its own formals on BlockMeta. Empty
        inputs is a broken tree, not a silent bind-nothing."""
        for name, resolved in _resolved_snapshots():
            with self.subTest(name):
                program = Program.of(resolved)
                for template in program.blocks:
                    self.assertTrue(template.name)
                    self.assertTrue(template.meta.inputs)
                    self.assertEqual(tuple(template.input_names()),
                                     tuple(template.meta.inputs))

    def test_declared_output_wins_over_a_later_side_root(self):
        program = shared_program('stepfun-ai/Step-3.7-Flash')
        template = next(
            item for item in program.blocks
            if item.name == "step3p7_mtp_block")
        # The MTP layer is one call of the library Step 3.5 layer, which
        # returns its inner block's hidden state; the shared head's logits
        # are a later side root of that call, not its output.
        (layer,) = template.output_roots()
        self.assertEqual(layer.output_names, ("transformer",))
        self.assertEqual(
            layer.children[-1].name, "transformer.shared_head.output")


class TestStagesAreDispatched(unittest.TestCase):

    def test_every_stage_is_one_of_the_two_kinds(self):
        for name, resolved in _resolved_snapshots():
            with self.subTest(name):
                program = Program.of(resolved)
                for stack in program.stacks:
                    for stage in stack.stages:
                        self.assertIsInstance(stage, (OpStage, BlockStage))
                        self.assertIsInstance(stage, Stage)

    def test_a_stages_roots_are_the_nodes_it_evaluates(self):
        """An op stage evaluates one node; a block stage evaluates its template
        body. Both answer `roots`, which is what removed the branch."""
        for name, resolved in _resolved_snapshots():
            with self.subTest(name):
                program = Program.of(resolved)
                for stack in program.stacks:
                    for stage in stack.stages:
                        if isinstance(stage, OpStage):
                            self.assertEqual(tuple(stage.roots), (stage.node,))
                        else:
                            self.assertEqual(tuple(stage.roots),
                                             stage.template.roots)

class TestContributions(unittest.TestCase):
    """The two rules a model total obeys, in the one place they now live.

    Both are invisible when broken. Summing every node instead of the roots
    inflates a model by roughly the depth of its tree; dropping the block
    multiplier shrinks it by `n_layers`. Neither produces a number that looks
    wrong, so these assert that both rules are load-bearing rather than that
    the total happens to come out right.
    """

    def _corpus(self):
        for name, resolved in _resolved_snapshots():
            yield name, Program.of(resolved)

    def test_a_contribution_is_always_a_root(self):
        for name, program in self._corpus():
            with self.subTest(name):
                roots = {id(n) for n in program.ir.top_level}
                roots |= {id(n) for t in program.blocks for n in t.roots}
                for part in program.contributions:
                    self.assertIn(id(part.node), roots)

    def test_a_block_root_counts_once_per_layer_using_its_template(self):
        for name, program in self._corpus():
            with self.subTest(name):
                by_uid = {n.uid: t.count
                          for t in program.blocks for n in t.roots}
                for part in program.contributions:
                    self.assertEqual(part.count, by_uid.get(part.node.uid, 1))

    @staticmethod
    def _unscaled_roots(analysis) -> Cost:
        return Cost.combine([analysis.cost.values[p.node.uid]
                             for p in analysis.program.contributions
                             if p.node.uid in analysis.cost.values])

    def test_a_component_already_holds_its_childrens_cost(self):
        """Why roots suffice, stated as the invariant it rests on rather than
        as a difference between two sums. A component with no `repeat:` of its
        own IS the combination of its children, so a traversal that visited
        both would add the subtree to itself."""
        analysis = _first_analysis()
        costs = analysis.cost.values
        checked = 0
        for node in analysis.program.nodes():
            if not node.children or node.uid not in costs:
                continue
            if node.repeat is not None:      # its own multiplier, applied above
                continue
            kids = Cost.combine([costs[c.uid] for c in node.children
                                 if c.uid in costs])
            self.assertEqual(kids.params, costs[node.uid].params, node.name)
            checked += 1
        self.assertGreater(checked, 0, "no nested components to check")

    def test_summing_every_node_double_counts(self):
        """The same fact as a number: against the SAME baseline — neither side
        scaled — visiting every node comes out above visiting the roots, by the
        nesting. Comparing it to the scaled total instead compares two errors
        at once and the block multiplier wins."""
        analysis = _first_analysis()
        every = Cost.combine(list(analysis.cost.values.values()))
        self.assertGreater(every.params, self._unscaled_roots(analysis).params)

    def test_dropping_the_layer_multiplier_is_not_the_total(self):
        analysis = _first_analysis()
        self.assertLess(self._unscaled_roots(analysis).params,
                        analysis.total().params)

    def test_the_stacks_partition_the_total(self):
        """`by_stack` and `total` fold the same roots, so their sum is an
        identity rather than two accountings that have to agree."""
        for name, resolved in _resolved_snapshots():
            with self.subTest(name):
                analysis = shared_walk(Deployment(Program.of(resolved)).at(
                    batch=1, input_tokens=128))
                summed = Cost.combine(list(analysis.by_stack().values()))
                self.assertEqual(summed.params, analysis.total().params)
                self.assertEqual(summed.flops, analysis.total().flops)


class TestLayersKnowTheirGraph(unittest.TestCase):
    """`Layer.graph` was typed to a class that does not exist and was passed
    `None` at all seventeen construction sites, so `map_with_node` — its only
    consumer — raised on every possible call and was never called."""

    def _analysis(self):
        for _name, resolved in _resolved_snapshots():
            return shared_walk(Deployment(Program.of(resolved)).at(
                batch=1, input_tokens=128))
        any_snapshot()  # the corpus is not checked out at all
        raise AssertionError("snapshots present but none resolved")

    def test_every_layer_an_analysis_hands_out_carries_its_program(self):
        analysis = self._analysis()
        for field in ("cost", "shapes", "roles", "storage", "formats",
                      "node_dims", "instances", "active_share"):
            with self.subTest(field):
                self.assertIs(getattr(analysis, field).program,
                              analysis.program)

    def test_a_derivation_can_ask_the_layer_for_the_node(self):
        analysis = self._analysis()
        ops = analysis.cost.map_with_node(lambda node, cost: node.op)
        self.assertTrue(ops.values)
        for uid, op in ops.values.items():
            self.assertEqual(op, analysis.program.by_uid(uid).op)

    def test_a_layer_with_no_program_still_refuses_rather_than_guessing(self):
        bare: Layer[int] = Layer(None, {1: 10})
        with self.assertRaises(ValueError):
            bare.map_with_node(lambda node, value: value)


if __name__ == "__main__":
    unittest.main()
