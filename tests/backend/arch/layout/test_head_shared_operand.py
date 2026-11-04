"""`head_shared_operand` reads the edge's port and refuses a missing fact.

An MLA key reaches attention through a view-only component; the operand is
head-shared exactly when the child behind the port the edge reads repeats
its input along the query-head axis. A repeating view with no `width` bound
by the walk has no answer, and must raise rather than read as unshared.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import sympy as sp

from backend.arch.layout.attention import head_shared_operand
from backend.arch.spec_tree.nodes import InputRef, SpecIRNode

H, S, D, B = sp.symbols("n_heads s_ctx d_head B")


def _node(name, op, uid, inputs=(), **kw):
    return SpecIRNode(name=name, op=op, is_component=bool(kw.get("children")),
                      uid=uid, inputs_obj=[InputRef(src=src, port=port)
                                           for src, port in inputs], **kw)


def _score_over(producer, port=""):
    query = _node("q", "transpose", 1)
    return _node("score", "attention_score", 99,
                 inputs=[(query, ""), (producer, port)])


SHAPES = ((B, H, S, D), (B, H, S, D))


class HeadSharedOperand(unittest.TestCase):
    def test_broadcast_along_query_heads_is_shared(self):
        expand = _node("expand", "broadcast", 10)
        rule = SimpleNamespace(env={10: {"width": H}})
        self.assertTrue(head_shared_operand(
            rule, _score_over(expand), 1, SHAPES))

    def test_broadcast_without_bound_width_raises(self):
        expand = _node("expand", "broadcast", 10)
        for env in ({}, {10: {}}):
            with self.subTest(env=env), self.assertRaisesRegex(
                    ValueError, "bound no `width`"):
                head_shared_operand(SimpleNamespace(env=env),
                                    _score_over(expand), 1, SHAPES)

    def test_view_only_component_enters_where_the_program_says(self):
        # Which child a component's port names is the Program's answer
        # (`Program.output_leaf`), asked with the port this edge reads.
        expand = _node("expand", "broadcast", 10)
        other = _node("other", "transpose", 12)
        wrapper = _node("kpe", "head_broadcast", 20,
                        children=[expand, other])
        asked = []

        def output_leaf(node, port="out"):
            asked.append((node.name, port))
            return (expand, "out") if port == "out" else (other, "out")

        rule = SimpleNamespace(env={10: {"width": H}},
                               program=SimpleNamespace(output_leaf=output_leaf))
        self.assertTrue(head_shared_operand(
            rule, _score_over(wrapper), 1, SHAPES))
        self.assertFalse(head_shared_operand(
            rule, _score_over(wrapper, "plain"), 1, SHAPES))
        self.assertEqual(asked, [("kpe", "out"), ("kpe", "plain")])

    def test_a_rule_without_its_program_refuses_to_guess(self):
        wrapper = _node("kpe", "head_broadcast", 20,
                        children=[_node("expand", "broadcast", 10)])
        rule = SimpleNamespace(env={10: {"width": H}}, program=None)
        with self.assertRaisesRegex(ValueError, "needs the Program"):
            head_shared_operand(rule, _score_over(wrapper), 1, SHAPES)

    def test_the_walk_hands_the_rule_its_program(self):
        from backend.arch.eval.program import Program
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        from tests.snapshots import resolve_snapshot

        analysis = evaluate_step(
            resolve_snapshot("deepseek-ai__DeepSeek-V3"),
            ParallelConfig(tp_attention=8, dcp=2), phase="decode",
            batch=1, input_tokens=128, output_tokens=0).analysis
        program = analysis.rule.program
        self.assertIsInstance(program, Program)
        # The program the walk ran: every node it bound is one of its nodes.
        self.assertTrue(all(program.by_uid(uid) is not None
                            for uid in analysis.rule.env))

if __name__ == "__main__":
    unittest.main()
