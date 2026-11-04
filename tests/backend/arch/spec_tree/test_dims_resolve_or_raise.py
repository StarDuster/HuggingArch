"""A dim expression the binding cannot resolve is an error with the
expression in it, not a zero.

`spec_tree` used to fold every resolution failure into a number: a
misspelt `repeat:` became one layer, a misspelt `dim:` became a 0-shaped
leaf that `ir_to_leaf` then dropped. `eval/dim.py` raised all along; the
tree builder caught it. The corpus is the positive half of this gate:
every snapshot still builds, so nothing in it was living on a silent
zero.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader
from backend.arch.eval.program import Program

SPEC = """\
match: {{ model_type: [{model_type}], model_id: [{model_type}] }}
dims:
  d_model: config.hidden_size
  n_layers: config.num_hidden_layers
blocks:
  fixed:
    graph:
      n: {{ op: rmsnorm, in: x, dim: {norm_dim} }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }} }}
  - {{ name: layers, repeat: "{repeat}", template: fixed, inputs: [embed],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers] }}
"""


def _build(model_type: str, **fields) -> Program:
    text = SPEC.format(model_type=model_type, **fields)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{model_type}.yaml").write_text(text)
        loader = SpecLoader(base_dir=str(base))
        return Program.of(loader.match(
            {"model_type": model_type, "hidden_size": 8, "num_hidden_layers": 2},
            model_id=model_type))


class TestAnUnresolvedDimIsAnError(unittest.TestCase):
    def test_the_well_formed_spec_builds(self):
        program = _build("test_dims_ok", norm_dim="d_model", repeat="n_layers")
        self.assertEqual([n.op for n in program.nodes() if n.op == "rmsnorm"],
                         ["rmsnorm"])

    def test_a_misspelt_node_dim_raises_with_the_node_and_the_expression(self):
        with self.assertRaisesRegex(Exception, r"\bn\b.*d_modle"):
            _build("test_dims_typo", norm_dim="d_modle", repeat="n_layers")

    def test_a_registered_dim_nothing_binds_is_named_at_its_node(self):
        """`d_vis_head` is a real symbol this model never binds — a caller
        still binding a component's renamed dim by its old name. It is named
        with its component and node when the tree is built, not left free
        until a byte count fails to become an integer."""
        with self.assertRaisesRegex(
                Exception, r"fixed/n: unresolved dim d_vis_head"):
            _build("test_dims_unbound", norm_dim="d_vis_head",
                   repeat="n_layers").nodes()

    def test_a_misspelt_repeat_raises_instead_of_one_layer(self):
        with self.assertRaisesRegex(Exception, r"layers.*n_layerz"):
            _build("test_repeat_typo", norm_dim="d_model", repeat="n_layerz")


if __name__ == "__main__":
    unittest.main()
