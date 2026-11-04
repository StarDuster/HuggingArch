"""An analysis refusal names its graph node in a structured attribute, and a
refusal the spec resolves names the declaration and the fix."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.refusal import (
    SpecRefusal, node_context, refusal_node, refusal_stage, spec_refusal,
    stage_context,
)


class TestNodeContext(unittest.TestCase):
    def test_the_loop_names_the_node_and_keeps_the_type(self):
        class Indivisible(ValueError):
            pass
        with self.assertRaises(Indivisible) as caught, \
                node_context("layers.self_attn.q_proj"):
            raise Indivisible("extent 5 not divisible by 2")
        self.assertEqual(refusal_node(caught.exception), "layers.self_attn.q_proj")

    def test_the_innermost_loop_wins(self):
        with self.assertRaises(ValueError) as caught, node_context("block"), \
                node_context("block.attn.o_proj"):
            raise ValueError("x")
        self.assertEqual(refusal_node(caught.exception), "block.attn.o_proj")

    def test_a_cause_chain_is_followed(self):
        try:
            try:
                with node_context("lm_head"):
                    raise ValueError("gather")
            except ValueError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            self.assertEqual(refusal_node(outer), "lm_head")

    def test_a_wrapped_refusal_keeps_the_innermost_node(self):
        with self.assertRaises(ValueError) as caught, node_context("block"):
            try:
                with node_context("block.attn.o_proj"):
                    raise ValueError("inner")
            except ValueError as inner:
                raise ValueError("wrapped") from inner
        self.assertEqual(refusal_node(caught.exception), "block.attn.o_proj")

    def test_no_loop_no_node(self):
        self.assertIsNone(refusal_node(ValueError("model-level")))


class TestStageContext(unittest.TestCase):
    def test_the_innermost_stage_wins_on_any_exception(self):
        @stage_context("capacity")
        def ceilings():
            raise KeyError("kv")
        with self.assertRaises(KeyError) as caught, stage_context("phases"):
            ceilings()
        self.assertEqual(refusal_stage(caught.exception), "capacity")

    def test_a_refusal_re_raised_later_is_named_where_it_was_raised(self):
        with self.assertRaises(RuntimeError) as caught, stage_context("phases"):
            try:
                with stage_context("preparation"):
                    raise ValueError("x")
            except ValueError as inner:
                raise RuntimeError("wrapped") from inner
        self.assertEqual(refusal_stage(caught.exception), "phases")

    def test_no_stage_no_name(self):
        self.assertIsNone(refusal_stage(ValueError("x")))


class TestSpecRefusal(unittest.TestCase):
    def test_it_is_a_value_error_with_declaration_fix_and_node(self):
        exc = SpecRefusal("dup", declaration="template", fix="own template",
                          node="layer.input_layernorm")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual((exc.declaration, exc.fix), ("template", "own template"))
        self.assertEqual(refusal_node(exc), "layer.input_layernorm")
        self.assertIs(spec_refusal(exc), exc)
        self.assertIsNone(spec_refusal(ValueError("engine")))


if __name__ == "__main__":
    unittest.main()
