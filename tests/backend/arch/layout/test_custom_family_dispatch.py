"""A spec-declared op is laid out by what it declares, not by its name.

Four custom ops used to appear by name in three layout tables — the
handler dispatch, the axis-use tags, the law-slot table — so a new custom
op needed a Python edit in a package that must not know it exists. The
declaration now decides: a `[Out, In]` weight is the linear family, a
recurrent state is the recurrent family, and a geometry op says how it
couples in its own `axis_use`. The law-slot table is gone too: a
homomorphism law names its operands itself.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError

from backend.arch.layout import linear as _linear, recurrent as _recurrent
from backend.arch.layout.foldlaw import HOMOMORPHISM_REGISTRY
from backend.arch.layout.propagation import custom_handler, layout_gap, layout_handler
from backend.arch.layout.axis_use import registry_violations
from backend.arch.ops_base import AxisUseDecl, OpSpec
from backend.arch.ops import build_custom_op_spec
from backend.arch.schema.custom_ops import CustomOpSpec

LINEAR = dict(flops="2 * B * s_ctx * In * Out", params="In * Out",
              weight_shape=["Out", "In"], signature={"x": ["B", "s_ctx", "In"]},
              shape_kind="axis_replace", out_key="Out", axis_locked=-1,
              couples_on=["In"], description="a packed expert projection no primitive stores")
RECURRENT = dict(flops="0", params="0",
                 signature={"q": ["B", "s_ctx", "H", "d_head"]},
                 shape_kind="preserve", couples_on=["s_ctx", "d_head"],
                 carries_recurrent_state=True,
                 description="a gated recurrence no primitive expresses",
                 axis_use={"s_ctx": {"category": "recurrent", "param": "d_head"},
                           "d_head": {"category": "recurrent", "param": "state"}})
GEOMETRY = dict(flops="0", params="0", signature={"x": ["B", "s_ctx", "d"]},
                shape_kind="axis_replace", out_key="Out", axis_locked=-1,
                couples_on=["s_ctx"], description="a patch fold no primitive expresses",
                axis_use={"s_ctx": {"category": "window", "param": "2x2 fold"}})


def _spec(name, raw):
    return CustomOpSpec(**raw)


class TestFamilyComesFromTheDeclaration(unittest.TestCase):
    def test_the_three_families(self):
        self.assertIs(custom_handler(_spec("p", LINEAR)), _linear.custom_linear)
        self.assertIs(custom_handler(_spec("r", RECURRENT)), _recurrent.headwise_recurrence)
        self.assertIsNone(custom_handler(_spec("g", GEOMETRY)))

    def test_linear_tags_are_derived_not_declared(self):
        spec = _spec("p", LINEAR)
        self.assertEqual({axis: use.model_dump() for axis, use in spec.layout_axis_use().items()},
                         {"In": {"category": "contract", "param": "sum", "plan": "", "note": ""}})
        # The registered-op form built from it declares no layout family of
        # its own; the walk is handed the CustomOpSpec, not this.
        self.assertIsNone(build_custom_op_spec("p", spec).layout_family)
        with self.assertRaisesRegex(ValidationError, "derived"):
            CustomOpSpec(**{**LINEAR, "axis_use": {"In": {"category": "contract", "param": "sum"}}})

    def test_every_coupled_axis_says_how(self):
        with self.assertRaisesRegex(ValidationError, "exactly the tagged axes"):
            CustomOpSpec(**{**GEOMETRY, "couples_on": ["s_ctx", "d"]})
        with self.assertRaisesRegex(ValidationError, "not one of"):
            CustomOpSpec(**{**GEOMETRY, "axis_use": {"s_ctx": {"category": "sideways"}}})


def _proposed(**over):
    """A registered op an op proposal adds: no hand-written handler."""
    base = dict(name="proposed_proj", flops_fn=lambda d: 0, params_fn=lambda d: 0,
                shape_in=("B", "s_ctx", "In"), shape_out=("B", "s_ctx", "Out"),
                shape_kind="axis_replace", out_key="Out",
                weight_shape=("Out", "In"), couples_on=("In",),
                layout_family="linear")
    base.update(over)
    return OpSpec(**base)


class TestAProposedOpDeclaresItsFamily(unittest.TestCase):
    """An op a proposal registers has no `_HANDLERS` entry; the layout family
    it declares is what places it, exactly as for a spec-declared op."""

    def test_the_declared_family_dispatches(self):
        self.assertIs(layout_handler("proposed_proj", _proposed()), _linear.custom_linear)
        recurrent = _proposed(
            name="proposed_scan", weight_shape=(), out_key=None,
            shape_kind="preserve", couples_on=("s_ctx", "d_head"),
            carries_recurrent_state=True, layout_family="recurrent",
            axis_use={"s_ctx": AxisUseDecl(category="recurrent", param="d_head"),
                      "d_head": AxisUseDecl(category="recurrent", param="state")})
        self.assertIs(layout_handler("proposed_scan", recurrent),
                      _recurrent.headwise_recurrence)
        self.assertFalse([v for v in registry_violations({"proposed_scan": recurrent})
                          if v.startswith("proposed_scan")])

    def test_no_family_no_law_and_the_gap_names_the_fix(self):
        bare = _proposed(layout_family=None, weight_shape=(), couples_on=())
        self.assertIsNone(layout_handler("proposed_proj", bare))
        self.assertIn('layout_family=\"linear\"', layout_gap("proposed_proj", bare))

    def test_an_incomplete_declaration_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no layout law"):
            _proposed(layout_family="geometry")
        with self.assertRaisesRegex(ValueError, r"weight_shape=\(Out, In\)"):
            _proposed(weight_shape=("In", "Out"))
        with self.assertRaisesRegex(ValueError, "two axes"):
            _proposed(layout_family="recurrent", weight_shape=(), out_key=None)
        with self.assertRaisesRegex(ValueError, "declare `layout_family`"):
            _proposed(layout_family=None, axis_use={
                "In": AxisUseDecl(category="contract", param="sum")})


class TestTheLawNamesItsOperands(unittest.TestCase):
    """What the hand-kept `_LAW_SLOTS` table used to restate."""

    def test_named_laws(self):
        self.assertEqual(HOMOMORPHISM_REGISTRY["linear"].slots(1), ("in",))
        self.assertEqual(HOMOMORPHISM_REGISTRY["linear_external_weight"].slots(2), ("in", "weight"))
        self.assertEqual(HOMOMORPHISM_REGISTRY["combine"].slots(2), ("expert_out", "routing"))
        self.assertEqual(HOMOMORPHISM_REGISTRY["add"].slots(2), ("a", "b"))

    def test_an_arity_the_law_does_not_describe_gets_no_slots(self):
        self.assertIsNone(HOMOMORPHISM_REGISTRY["linear"].slots(2))

    def test_a_wildcard_law_is_positional(self):
        self.assertEqual(len(HOMOMORPHISM_REGISTRY["concat"].slots(3)), 3)


if __name__ == "__main__":
    unittest.main()
