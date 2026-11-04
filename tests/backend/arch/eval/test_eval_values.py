"""`Cost` / `Shape` keep independently derived concerns in separate layers.

Falsification targets:
  - `scaled` must keep STORAGE, ACTIVE-PARAMETER, and EXECUTION counts separate.
    A dispatched bank executes E narrowed expert workloads while only K expert
    parameter copies are active;
  - fusing costs must sum and fusing shapes must span, not pick one;
  - an unknown shape must contribute zero elements, not one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.values import Cost, Shape
from backend.arch.eval.dim import Dim


class TestCost(unittest.TestCase):
    def test_add_is_componentwise(self):
        a = Cost(flops=10, params=2, active_params=1, quant_metadata_params=3)
        b = Cost(flops=1, params=20, active_params=5, quant_metadata_params=7)
        self.assertEqual(a + b, Cost(11, 22, 6, 10))

    def test_combine_of_nothing_is_zero(self):
        self.assertTrue(Cost.combine([]).is_zero)

    def test_storage_and_active_copies_scale_differently(self):
        """256 stored copies, 8 active per token: params ×256, FLOPs ×8."""
        one = Cost(flops=100, params=50, active_params=50)
        bank = one.scaled(storage=256, active=8)
        self.assertEqual(bank.params, 50 * 256)
        self.assertEqual(bank.flops, 100 * 8)
        self.assertEqual(bank.active_params, 50 * 8)

    def test_dense_scaling_is_the_degenerate_case(self):
        one = Cost(flops=100, params=50, active_params=50)
        self.assertEqual(one.scaled(1, 1), one)

    def test_quant_metadata_follows_storage_not_active(self):
        """Scale tensors exist per stored copy — an inactive expert's scales are
        still on disk and still in HBM."""
        one = Cost(params=50, quant_metadata_params=2)
        bank = one.scaled(storage=256, active=8)
        self.assertEqual(bank.quant_metadata_params, 2 * 256)

    def test_symbolic_values_flow_through(self):
        E, K = sp.Symbol("E"), sp.Symbol("K")
        bank = Cost(flops=100, params=50).scaled(storage=E, active=K)
        self.assertEqual(bank.params, 50 * E)
        self.assertEqual(bank.flops, 100 * K)

    def test_combine_preserves_dim_value_after_symbolic_zero(self):
        extent = Dim.symbol("d_model", 3072)
        combined = Cost.combine([Cost(flops=sp.Integer(0)),
                                 Cost(flops=2 * extent)])
        self.assertIsInstance(combined.flops, Dim)
        self.assertEqual(int(combined.flops), 6144)
        # Plain symbols, no assumptions: `Dim.is_` decides axis identity with
        # `expr == sp.Symbol(name)`, so an assumption set would be a second
        # spelling of the same axis that compares unequal to the first.
        self.assertEqual(combined.flops.expr, 2 * sp.Symbol("d_model"))


class TestShape(unittest.TestCase):
    def test_inputs_stay_separable(self):
        """A two-input contraction must keep its inputs distinguishable; the
        engine's flat-list-or-list-of-lists convention makes every consumer
        re-derive which case it is holding."""
        sh = Shape(out=(2, 4, 8), ins=((2, 4, 16), (16, 8)))
        self.assertEqual(len(sh.ins), 2)
        self.assertEqual(sh.ins[1], (16, 8))
        self.assertEqual(sh.rank, 3)

    def test_elements_is_the_product(self):
        self.assertEqual(Shape(out=(2, 4, 8)).elements(), 64)

    def test_unknown_shape_contributes_nothing(self):
        """0, not 1: a missing shape must not silently count as one element and
        add a dtype's worth of bytes."""
        self.assertEqual(Shape().elements(), 0)
        self.assertEqual(Shape(out=()).elements(), 0)
        self.assertIsNone(Shape().rank)

    def test_symbolic_extents_multiply(self):
        B, S = sp.Symbol("B"), sp.Symbol("s_ctx")
        self.assertEqual(Shape(out=(B, S, 8)).elements(), 8 * B * S)

    def test_fusion_spans_first_input_to_last_output(self):
        """A fused chain reads what the first link read and produces what the
        last produced; interior tensors do not survive it."""
        a = Shape(out=(1, 4, 16), ins=((1, 4, 8),))
        b = Shape(out=(1, 4, 32), ins=((1, 4, 16),))
        fused = Shape.combine([a, b])
        self.assertEqual(fused.ins, ((1, 4, 8),))
        self.assertEqual(fused.out, (1, 4, 32))

    def test_secondary_ports_survive_from_the_last_link(self):
        a = Shape(out=(1, 8))
        b = Shape(out=(1, 4)).with_port("comb", (1, 2))
        fused = Shape.combine([a, b])
        self.assertEqual(fused.ports, {"comb": (1, 2)})

    def test_with_port_does_not_mutate(self):
        base = Shape(out=(1, 8))
        base.with_port("p", (1, 1))
        self.assertEqual(base.ports, {})

    def test_combine_of_nothing_is_empty(self):
        self.assertIsNone(Shape.combine([]).out)


class TestDimNumericProtocol(unittest.TestCase):
    """A `Dim` stands in for the number it resolves to, everywhere a number
    is asked for — including in the string formats a diagnostic renders."""

    def test_a_format_spec_renders_the_value(self):
        d = Dim.symbol("d_model", 4096)
        self.assertEqual(f"{d:,.0f}", "4,096")
        self.assertEqual(f"{d:d}", format(4096, "d"))

    def test_a_bare_interpolation_keeps_the_symbol(self):
        d = Dim.symbol("d_model", 4096)
        self.assertEqual(f"{d}", "4096⟨d_model⟩")

    def test_a_symbolic_value_has_no_number_to_render(self):
        with self.assertRaises(TypeError):
            f"{Dim(sp.Symbol('n'), sp.Symbol('n')):,.0f}"


if __name__ == "__main__":
    unittest.main()
