"""A model-root layer's `inputs:` are its edges, named whole.

Root and group layers in `model:` are named by checkpoint path, so their
names carry literal dots (`mtp.pre_fc_norm_embedding`). A ref to one is the
whole layer name with the main-output port, never `<node>.<port>`; and a
primitive that declares `inputs:` reads those layers and nothing else.
Qwen3.5-MoE's MTP head is the corpus case: `mtp.concat` joins two norms.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.spec_tree import iter_tree
from backend.arch.spec_tree.build import SpecTreeWalker

SNAPSHOT = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "Qwen__Qwen3.5-35B-A3B.json"


def _resolved():
    config = json.loads(SNAPSHOT.read_text())["config"]
    return shared_loader().match(config, model_id="Qwen/Qwen3.5-35B-A3B")


def _top_level_by_name(tree):
    found = {}

    def walk(nodes):
        for node in nodes:
            found.setdefault(node.display_path, node)
            walk(node.children or ())

    walk(tree.top_level)
    return found


class TestRootRefResolution(unittest.TestCase):
    def test_root_ref_is_the_whole_layer_name(self):
        walker = SpecTreeWalker(_resolved())
        self.assertEqual(walker._scope_stack, [])
        for ref in ("mtp.pre_fc_norm_embedding", "model.language_model.norm",
                    "layers"):
            with self.subTest(ref=ref):
                self.assertEqual(walker._resolve_input_ref(ref), (ref, "", False))


class TestRootPrimitiveInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = _top_level_by_name(iter_tree(_resolved()))

    def test_concat_reads_both_norms(self):
        concat = self.nodes["mtp.concat"]
        refs = tuple(concat.inputs_obj)
        self.assertEqual(len(refs), 2)
        self.assertTrue(all(ref.src is not None for ref in refs))
        self.assertEqual(
            [ref.src.display_path for ref in refs],
            ["mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"])
        self.assertEqual([ref.port for ref in refs], ["", ""])

    def test_single_declared_input_resolves_across_stacks(self):
        refs = tuple(self.nodes["mtp.pre_fc_norm_hidden"].inputs_obj)
        self.assertEqual(len(refs), 1)
        self.assertIsNotNone(refs[0].src)
        self.assertEqual(refs[0].src.display_path, "model.language_model.norm")
        self.assertEqual(refs[0].port, "")

    def test_residual_edge_is_the_declared_layer(self):
        # Every model-level edge is named: `mtp.fc` names `mtp.concat`.
        refs = tuple(self.nodes["mtp.fc"].inputs_obj)
        self.assertEqual(len(refs), 1)
        self.assertIsNotNone(refs[0].src)
        self.assertEqual(refs[0].src.display_path, "mtp.concat")
        self.assertEqual(refs[0].port, "")


if __name__ == "__main__":
    unittest.main()
