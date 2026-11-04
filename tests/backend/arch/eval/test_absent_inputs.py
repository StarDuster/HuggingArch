"""A step that reads a branch the model does not build.

`moe_out: add [combine, shared_experts]` is authored once and instantiated
by models with and without a shared expert; where `n_shared_experts` is 0 the
builder skips the branch. `add` declares what an absent term stands for
(`OpSpec.absent_operand_identity`: nothing to add), so the sum is not built
and its readers read `combine` — the forward's `if self.shared_experts:`. An
op that declares no such value cannot be handed a missing operand: the spec
is refused where it is built, rather than any later pass giving the operand a
shape or letting the live one through.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from tests.snapshots import shared_program
from backend.arch.layout.linear import elementwise
from backend.arch.loader import ResolvedSpec
from backend.arch.ops import COMBINE_OP
from backend.arch.schema import (ArchSpec, BlockGraph, GraphNode,
                                 ModelInputSpec, ModelLayer)

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
_W = [{"replicated": True}, {"replicated": True}]


def _program(stem: str) -> Program:
    return shared_program(stem.replace('__', '/', 1))


def _moe_outputs(program: Program):
    """The output leaf of each call that combines routed experts."""
    return [program.output_leaf(node)[0] for node in program.nodes()
            if node.is_component
            and any(child.op == COMBINE_OP for child in node.children)]


class TestTheSumOfAnAbsentTerm(unittest.TestCase):
    def test_without_a_shared_expert_the_moe_output_is_combine(self):
        leaves = _moe_outputs(_program("Qwen__Qwen3-30B-A3B"))
        self.assertTrue(leaves)
        self.assertEqual({leaf.op for leaf in leaves}, {"combine"})

    def test_with_a_shared_expert_the_moe_output_sums_two_live_terms(self):
        leaves = _moe_outputs(_program("deepseek-ai__DeepSeek-V3"))
        self.assertTrue(leaves)
        for leaf in leaves:
            self.assertEqual(leaf.op, "add")
            self.assertEqual(len(leaf.inputs_obj), 2)
            self.assertTrue(all(ref.src is not None for ref in leaf.inputs_obj))


def _branch_model(op: str) -> ResolvedSpec:
    dims = ({"in_a": "d_model", "in_b": "d_model", "out": "2 * d_model"}
            if op == "concat" else {"dim": "d_model"})
    comp = BlockGraph(inputs=["x"], graph={
        "a": GraphNode(op="linear", inputs=["x"],
                       dims={"in": "d_model", "out": "d_model"}, weight_axes=_W),
        "b": GraphNode(op="linear", inputs=["x"],
                       dims={"in": "d_model", "out": "d_model"},
                       repeat="n_b", weight_axes=_W),
        "out": GraphNode(op=op, inputs=["a", "b"], dims=dims),
    }, outputs=["out"])
    block = BlockGraph(inputs=["x"],
                       graph={"c": GraphNode(op="comp", inputs=["x"])},
                       outputs=["c"])
    spec = ArchSpec(
        name="demo", dims={"d_model": "config.hidden_size"},
        model_inputs={"input_ids": ModelInputSpec(kind="tokens")},
        model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"]),
        ],
        blocks={"block": block}, components={"comp": comp})
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 1, "n_b": 0}, {}, {})


class TestAnOpWithNoValueForAnAbsentOperand(unittest.TestCase):
    def test_a_product_of_an_absent_factor_reads_through(self):
        names = [n.name for n in Program.of(_branch_model("mul")).nodes()]
        self.assertNotIn("out", names)

    def test_a_concat_of_an_absent_branch_is_refused_where_it_is_built(self):
        with self.assertRaisesRegex(ValueError, "declares no value for an "
                                                "absent operand"):
            Program.of(_branch_model("concat"))


class TestElementwiseOperandsWithNoState(unittest.TestCase):
    """The handler decides before touching the rule, so `rule=None` is enough."""

    def _node(self, op, second):
        """`second` is what the second ref is: "scope" (an enclosing input,
        unresolved) or "resolved"."""
        ref = {"scope": SimpleNamespace(src=None),
               "resolved": SimpleNamespace(src=object())}[second]
        return SimpleNamespace(
            op=op, display_path=f"t.{op}",
            inputs_obj=(SimpleNamespace(src=object()), ref))

    def test_a_scope_input_with_no_state_yet_leaves_the_live_operand(self):
        live = object()
        required, produced = elementwise(
            None, self._node("add", "scope"), [live, None], [None, None])
        self.assertIs(produced, live)
        self.assertEqual(required, [live, None])

    def test_a_resolved_producer_with_no_state_is_a_dropped_edge(self):
        with self.assertRaisesRegex(ValueError, "dropped edge"):
            elementwise(None, self._node("add", "resolved"),
                        [object(), None], [None, None])


if __name__ == "__main__":
    unittest.main()
