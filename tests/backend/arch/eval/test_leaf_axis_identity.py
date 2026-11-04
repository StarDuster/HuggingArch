"""A leaf's axis is the one its CALL SITE named, and an axis with no value is loud.

A leaf's dim expression is written in its component's vocabulary, and a
component parameter is a local name. `swiglu_ffn` says `out: d_ffn`; `moe_ffn`
invokes it as `dims: { d_ffn: d_ffn_expert }`, so the expert bank's axis IS
`d_ffn_expert`. Reporting the parameter name instead hands every consumer the
dense FFN's identity for an expert tensor, and identity is what decides which
parallelism divides the axis.

The two tests here are the two halves of one defect:

  * on a model that has both kinds of layer, the dense and expert widths must
    come back as DIFFERENT symbols — the expert bank divided by eight only
    because `d_ffn` happened to be a divisible int, i.e. it borrowed the dense
    FFN's width to get the right answer for the wrong reason;
  * on a model with no dense layer at all, `d_ffn` has no value, and the
    storage walk read that absence as "replicated" — 454 GB of Qwen3-MoE expert
    weight on every rank, silently, which is what made the whole family look
    infeasible on 8×H100. An axis the leaf CARRIES and cannot value is not a
    default, it is a contradiction, so it raises.

`TestStorageGates` states what is left of the copy divisor once the storage
path stopped recovering a scalar divisor from the axis names: every per-rank
division of a tensor is its shape's, and only pipeline parallelism drops whole
copies.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.spec_tree import (
    iter_leaves,
)
from backend.inference.sharding import dim_to_tp_field
from backend.inference.weight_breakdown import (
    _copies_dropped,
)

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolve(name: str):
    return shared_loader().match(
        json.loads((SNAPSHOTS / f"{name}.json").read_text())["config"],
        model_id=name.replace("__", "/", 1))


def _axes(resolved, suffix: str) -> frozenset:
    """Dim symbols of the first leaf whose HF name ends in `suffix`."""
    for lt in iter_leaves(resolved):
        if lt.tensor_name.endswith(suffix):
            return frozenset(lt.dim_symbols)
    raise AssertionError(f"no leaf named *{suffix}")


class TestCallSiteAxisIdentity(unittest.TestCase):

    def test_dense_and_expert_widths_are_different_symbols(self):
        """DSv3 has both, from the SAME `swiglu_ffn` graph.

        One invocation leaves `d_ffn` bound to the model's own `d_ffn`; the
        other rebinds it to `d_ffn_expert`. If the walk reported the parameter
        name, these two would come back identical and every expert tensor in
        the corpus would shard under `tp_dense_ffn`.
        """
        r = _resolve("deepseek-ai__DeepSeek-V3")
        dense = _axes(r, "layers.X.mlp.gate_proj.weight")
        expert = _axes(r, "layers.X.mlp.experts.X.gate_proj.weight")

        self.assertIn("d_ffn", dense)
        self.assertNotIn("d_ffn_expert", dense)
        self.assertIn("d_ffn_expert", expert)
        self.assertNotIn("d_ffn", expert)

        self.assertEqual(dim_to_tp_field()["d_ffn"], "tp_dense_ffn")
        self.assertEqual(dim_to_tp_field()["d_ffn_expert"], "tp_moe_expert")

    def test_a_dense_free_model_names_no_dense_axis(self):
        """Qwen3-235B is pure MoE — `d_ffn` is a symbol it never binds.

        This is the case the old walk could not survive: it named an axis the
        model does not have, so the value lookup came back `None`.
        """
        r = _resolve("Qwen__Qwen3-235B-A22B")
        self.assertIn("d_ffn_expert",
                      _axes(r, "layers.X.mlp.experts.X.gate_proj.weight"))
        self.assertNotIn("d_ffn", {s for lt in iter_leaves(r)
                                   for s in lt.dim_symbols})
        self.assertIsNone(r.dim_values.get("d_ffn"))


class TestStorageGates(unittest.TestCase):
    """Per-rank storage is two questions. How thin each copy is comes from
    the forward-propagated SHAPE — the expert axis of a bank among its axes,
    which expert parallelism divides there. How many whole copies a rank
    drops is pipeline parallelism's alone: it partitions the layer index
    space, which no tensor carries."""

    def test_only_pipeline_parallelism_drops_whole_copies(self):
        self.assertEqual(_copies_dropped(), [])
        self.assertEqual(_copies_dropped(1.0), [])
        self.assertEqual(_copies_dropped(7.0), ["pp"])


if __name__ == "__main__":
    unittest.main()
