"""Smoke test for the resolved-structure renderer (arch/structure.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import ResolvedSpec
from backend.arch.schema import (
    ArchSpec,
    BlockGraph,
    CacheSpec,
    GraphNode,
    ModelLayer,
)
from backend.arch.structure import render_structure


def _resolved():
    attn = BlockGraph(
        inputs=["x"],
        graph={"wq": GraphNode(op="linear", inputs=["x"], dims={"in": "d_model", "out": "d_model"})},
        outputs=["wq"],
        role="attention",
        cache=CacheSpec(family="GQA", entries=[{"node": "wq", "kind": "kv"}]),
    )
    block = BlockGraph(
        inputs=["x"],
        graph={"a": GraphNode(op="my_attn", inputs=["x"])},
        outputs=["a"],
    )
    spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
        name="demo",
        dims={"d_model": "config.hidden_size"},
        model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"]),
        ],
        blocks={"block": block},
        components={"my_attn": attn},
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 2}, {}, {})


class TestRenderStructure(unittest.TestCase):
    def test_renders_component_nodes_and_dataflow(self):
        text = render_structure(_resolved())
        # component reached via params/op, its node op + dataflow + role.
        self.assertIn("[my_attn]", text)
        self.assertIn("role=attention", text)
        self.assertIn("kv_label=GQA", text)
        self.assertIn("wq: linear", text)
        self.assertIn("<- x", text)


if __name__ == "__main__":
    unittest.main()
