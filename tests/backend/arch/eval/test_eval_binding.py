"""`Binding` — the dim environment as a chained scope.

Extents stay global. Per-rank geometry is the layout's, so this file does
not install a divisor on lookup.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.binding import Binding
from backend.arch.eval.dim import expr_of


class TestScoping(unittest.TestCase):
    def test_child_shadows_without_touching_the_parent(self):
        base = Binding({"n_heads": 64, "d_model": 8192})
        inner = base.child(n_heads=96)
        self.assertEqual(inner["n_heads"], 96)
        self.assertEqual(inner["d_model"], 8192, "inherits what it did not bind")
        self.assertEqual(base["n_heads"], 64, "parent frame is untouched")

    def test_nesting_depth_resolves_innermost_first(self):
        b = Binding({"x": 1}).child(x=2).child(y=3).child(x=4)
        self.assertEqual((b["x"], b["y"]), (4, 3))

    def test_missing_symbol_raises(self):
        b = Binding({"x": 1})
        with self.assertRaises(KeyError):
            b["nope"]
        self.assertEqual(b.get("nope", 7), 7)
        self.assertNotIn("nope", b)

    def test_flat_and_symbols_see_the_whole_chain(self):
        b = Binding({"a": 1}).child(b=2).child(a=3)
        self.assertEqual(b.symbols(), frozenset({"a", "b"}))
        self.assertEqual(b.flat(), {"a": 3, "b": 2})

    def test_rebased_collapses_the_chain(self):
        b = Binding({"a": 1}).child(b=2)
        flat = b.rebased()
        self.assertIsNone(flat.parent)
        self.assertEqual(flat.flat(), {"a": 1, "b": 2})

    def test_raw_is_the_bound_value(self):
        b = Binding({"n_heads": 64})
        self.assertEqual(b.raw("n_heads"), b["n_heads"])

    def test_s_ctx_reads_s_query(self):
        """Op templates write s_ctx; the phase binds s_query. One number."""
        b = Binding({"s_query": 6, "B": 2})
        self.assertEqual(b["s_ctx"], 6)
        self.assertIn("s_ctx", b)
        self.assertEqual(b.flat()["s_ctx"], 6)
        self.assertIs(b["s_ctx"], b["s_query"])

    def test_an_explicit_s_ctx_bind_wins(self):
        b = Binding({"s_query": 6}).child(s_ctx=1)
        self.assertEqual(b["s_ctx"], 1)
        self.assertEqual(b["s_query"], 6)


class TestAWrittenShape(unittest.TestCase):

    def test_a_written_shape_is_read_under_the_walks_binding(self):
        """`shape:` is a dim expression like `dims:`, so a
        `parameter_tensor`'s `n_copies` is the model's `n_copies` by
        identity, and the weights it stores are a law in it."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.loader import ResolvedSpec
        from backend.arch.schema import (
            ArchSpec, BlockGraph, GraphNode, ModelLayer)

        heads = {"n_heads": "n_copies", "d_head": "d_head"}
        block = BlockGraph(
            inputs=["x"],
            graph={
                "xs": GraphNode(op="view_split", inputs=["x"], axis=-1,
                                dims={"in": "d_model", **heads}),
                "w": GraphNode(op="parameter_tensor",
                               shape=["n_copies", "d_head"]),
                "y": GraphNode(op="mul", inputs=["xs", "w"], dim="d_head"),
                "out": GraphNode(op="view_merge", inputs=["y"], dims=heads),
            },
            outputs=["out"])
        spec = ArchSpec(
            model_inputs={"input_ids": {"kind": "tokens"}},
            name="demo",
            dims={"d_model": "config.hidden_size",
                  "n_copies": "config.n_copies", "d_head": "config.head_dim"},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="n_layers",
                           inputs=["embed"]),
            ],
            blocks={"block": block})
        dims = {"d_model": 16, "n_copies": 4, "d_head": 4, "n_layers": 1,
                "n_vocab": 32}
        config = {"hidden_size": 16, "n_copies": 4, "head_dim": 4,
                  "torch_dtype": "bfloat16"}
        d = Deployment(Program.of(ResolvedSpec(spec, dims, config, {}))).at(
            input_tokens=8)
        params = d.analyze().total().params
        self.assertIn(sp.Symbol("n_copies"),
                      sp.sympify(expr_of(params)).free_symbols)


if __name__ == "__main__":
    unittest.main()
