"""`LayerSelection` — one spelling for "a set of layers of one stack".

Falsification targets:
  - a block assignment's authored YAML must be unchanged by the selection
    moving to a shared base, because every model spec in the corpus writes
    `range:` / `layers:` / `stride:` flat on the assignment;
  - one resolver must answer for both users, or a layer a template runs at
    and a layer an override speaks about become two different readings of
    the same words;
  - a layer bound must stay a dim expression, because the layers an override
    carves out are the last ones of a stack whose depth is a config value
    (`n_layers - 3`), not three literals that happen to be right for one
    checkpoint;
  - a spec that declares no layer scope must resolve exactly as before.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.schema.model import BlockAssignment, LayerSelection
from backend.arch.schema.quant import OverrideEntry, OverrideSelector


class TestOneSpelling(unittest.TestCase):
    def test_a_block_assignment_is_a_layer_selection(self):
        """The shared base, not a copy of its fields."""
        self.assertTrue(issubclass(BlockAssignment, LayerSelection))

    def test_an_assignment_keeps_its_authored_fields(self):
        assignment = BlockAssignment(range=["0", "n_dense_layers - 1"],
                                     ffn="dense_ffn")
        self.assertEqual(assignment.range, ["0", "n_dense_layers - 1"])
        self.assertEqual(assignment.ffn, "dense_ffn")
        # `rest:` stays on the assignment: "whatever the other assignments
        # did not claim" is about a container's assignments, and means
        # nothing to a row that names tensors.
        self.assertFalse(hasattr(LayerSelection(), "rest"))

    def test_every_selector_a_block_assignment_writes_is_on_the_base(self):
        """The corpus writes these; none may be left behind on the subclass."""
        for field in ("range", "layers", "stride", "offset", "config_list",
                      "config_list_optional", "equals", "config_list_base",
                      "within"):
            self.assertIn(field, LayerSelection.model_fields, field)

    def test_a_selection_naming_nothing_says_so(self):
        self.assertFalse(LayerSelection().selects_layers)
        self.assertTrue(LayerSelection(range=["88", "90"]).selects_layers)
        self.assertTrue(LayerSelection(stride="2").selects_layers)
        self.assertTrue(
            LayerSelection(config_list="layer_types",
                           equals="full").selects_layers)


class TestOverrideLayerScope(unittest.TestCase):
    """The scope Step-5 needs: three layers of 91 whose routed experts the
    checkpoint leaves unquantized, with nothing structural to tell them
    apart."""

    def test_a_mapping_is_the_full_selection(self):
        selector = OverrideSelector(
            layers={"range": ["n_layers - 3", "n_layers - 1"]},
            tensors_re=r"moe\.experts\.w(13|2)_weight$")
        self.assertEqual(selector.layers.range,
                         ["n_layers - 3", "n_layers - 1"])

    def test_a_bare_list_is_the_explicit_form(self):
        """Spelled as a `BlockAssignment` spells it, so an author who knows
        one knows the other."""
        selector = OverrideSelector(layers=[88, 89, 90], tensors_re="w13")
        self.assertEqual(selector.layers.layers, ["88", "89", "90"])

    def test_a_config_list_selects_layers_by_schedule(self):
        selector = OverrideSelector(
            layers={"config_list": "layer_types", "equals": "full"},
            tensors_re="o_proj")
        self.assertEqual((selector.layers.config_list,
                          selector.layers.equals),
                         ("layer_types", "full"))

    def test_a_layer_scope_naming_no_layers_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            OverrideSelector(layers={"equals": "full"}, tensors_re="w13")
        self.assertIn("names no layers", str(caught.exception))

    def test_an_old_row_with_no_layer_scope_is_unchanged(self):
        """Every row in the corpus today. `layers` is absent, and absent
        from the dump, so a spec's fingerprint does not move."""
        entry = OverrideEntry(tensors_re=r"mlp\.gate\.expert_bias$",
                              dtype="F32")
        self.assertIsNone(entry.select)
        self.assertNotIn("layers", entry.model_dump())
        scoped = OverrideEntry(
            select=OverrideSelector(stack="mtp", tensors=["a.weight"]),
            dtype="BF16")
        self.assertIsNone(scoped.select.layers)
        self.assertNotIn("layers", scoped.model_dump()["select"])

    def test_the_layer_scope_is_part_of_a_row_s_identity(self):
        """Two rows differing only by the layers they speak about are two
        rules; the duplicate-selector check must not call them one."""
        from backend.arch.schema.quant import QuantContext

        context = QuantContext(overrides=[
            OverrideEntry(select=OverrideSelector(layers=[88, 89, 90],
                                                  tensors_re="w13"),
                          dtype="BF16"),
            OverrideEntry(select=OverrideSelector(layers=[3],
                                                  tensors_re="w13"),
                          dtype="F32"),
        ])
        self.assertEqual(len(context.overrides), 2)

    def test_a_layer_scope_outside_select_points_at_select(self):
        with self.assertRaises(ValueError) as caught:
            OverrideEntry(tensors_re="w13", dtype="BF16", layers=[88])
        self.assertIn("move it under", str(caught.exception))

    def test_the_selector_text_names_the_layers(self):
        entry = OverrideEntry(
            select=OverrideSelector(layers={"range": ["88", "90"]},
                                    tensors_re="w13"),
            dtype="BF16")
        self.assertIn("range: [88, 90]", entry.selector_text())


if __name__ == "__main__":
    unittest.main()
