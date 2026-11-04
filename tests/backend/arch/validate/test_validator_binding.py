"""`bind_leaves_to_hf` — object shape + join invariants.

The binding API bridges spec leaves and checkpoint header tensors: a
leaf's stored tensors bind the header tensors their names select. These
tests pin the properties that make that join meaningful: every binding owns
the leaves it reports params for, and no header tensor is claimed twice.

Offline-only: each test pulls from `backend/arch/snapshots/` and
skips when the snapshot for a model is missing.
"""
from __future__ import annotations

import os

from tests.snapshots import resolve_snapshot
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))




def _analysis(resolved):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    return Deployment(Program.of(resolved)).at(batch=1, input_tokens=8).analyze()

def _resolve(model_id: str):
    """Match the model's snapshot config through the spec loader, the
    same path `validate()` and `estimate_capacity` use. Missing snapshot →
    skip; present but unresolvable → failure (tests.snapshots decides).

    The config comes from the snapshot file, not from `fetch_metadata` —
    snapshot *metadata* is the safetensors side (patterns, dtypes, byte
    totals) and carries no `config` key.
    """
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from backend.arch.loader import get_default_loader
        return resolve_snapshot(model_id, get_default_loader())
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def _hf_data(model_id: str):
    """Build the `hf_meta` dict that `bind_leaves_to_hf` consumes,
    matching the shape `_get_safetensors_data` produces."""
    from backend.arch.validate._matching import _get_safetensors_data
    return _get_safetensors_data(model_id)


class TestHFBindingObjectShape(unittest.TestCase):
    """Smoke test: result dataclasses carry the documented fields."""

    def test_bind_returns_expected_object_shape(self):
        from backend.arch.validate import (
            BindResult,
            HFBinding,
            bind_leaves_to_hf,
        )
        model_id = "meta-llama/Llama-3.2-1B"
        resolved = _resolve(model_id)
        prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            hf_meta = _hf_data(model_id)
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev

        result = bind_leaves_to_hf(hf_meta["header"], _analysis(resolved))
        self.assertIsInstance(result, BindResult)
        self.assertGreater(len(result.bindings), 0)
        b = result.bindings[0]
        self.assertIsInstance(b, HFBinding)
        # Object-reference invariant: every leaf in `leaves` is a real
        # LeafTensor (not a string), and `ir_node` reference is present.
        from backend.arch.spec_tree import LeafTensor
        for lt in b.leaves:
            self.assertIsInstance(lt, LeafTensor)
            self.assertIsNotNone(lt.ir_node)
        # Field types match documented schema.
        self.assertIsInstance(b.spec_node_label, str)
        self.assertIsInstance(b.hf_name, str)
        self.assertIsInstance(b.rows, tuple)
        self.assertIsInstance(b.spec_params, int)
        self.assertIsInstance(b.hf_params, int)
        self.assertIsInstance(b.diff, int)
        self.assertIsInstance(b.shape_ok, bool)


class TestHFBindingJoinInvariants(unittest.TestCase):
    """Properties the equality join must have on real checkpoints.

    The join replaced a greedy suffix-walk that would pair a spec leaf
    to any HF module its canonical path merely *ended with*. That walk
    could report params for a module while the binding held none of the
    leaves those params came from, and it could let two spec groups
    claim overlapping HF patterns. Both are unrepresentable now; these
    tests keep them that way.
    """

    _MODELS = (
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V4-Pro",
    )

    def _analysis_of(self, model_id):
        return _analysis(_resolve(model_id))

    def _bind(self, model_id):
        resolved = _resolve(model_id)
        if resolved is None:
            return None
        prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            hf_meta = _hf_data(model_id)
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev
        from backend.arch.validate import bind_leaves_to_hf
        return (bind_leaves_to_hf(hf_meta["header"], _analysis(resolved)),
                hf_meta["header"])

    def test_every_binding_owns_the_leaves_it_counts(self):
        for model_id in self._MODELS:
            with self.subTest(model_id=model_id):
                bind, _header = self._bind(model_id)
                self.assertGreater(len(bind.bindings), 0)
                for b in bind.bindings:
                    self.assertGreater(
                        len(b.leaves), 0,
                        f"{model_id} / {b.hf_name}: binding reports "
                        f"{b.spec_params:,} spec params but holds no leaves",
                    )
                    uids = {lt.ir_node.uid for lt in b.leaves}
                    self.assertTrue(
                        all(row.uid in uids and row.name == b.hf_name
                            for row in b.rows),
                        f"{model_id} / {b.hf_name}: a counted row is not a "
                        f"tensor of that name stored by the binding's leaves",
                    )
                    self.assertEqual(
                        b.spec_params,
                        sum(int(row.elements) * int(row.count)
                            for row in b.rows),
                        f"{model_id} / {b.hf_name}: spec_params is not "
                        f"the sum over the rows the binding holds",
                    )

    def test_no_header_tensor_is_claimed_twice(self):
        for model_id in self._MODELS:
            with self.subTest(model_id=model_id):
                bind, header = self._bind(model_id)
                names = [b.hf_name for b in bind.bindings]
                self.assertEqual(len(names), len(set(names)), model_id)
                claimed = {}
                for b in bind.bindings:
                    for group, n in b.selection.parts:
                        claimed[id(group)] = claimed.get(id(group), 0) + n
                for group in header:
                    self.assertLessEqual(
                        claimed.get(id(group), 0), group.count,
                        f"{model_id}: {group.display()} is claimed by more "
                        f"bindings than it holds tensors")


class TestSpecVsTensorsCheck(unittest.TestCase):
    def test_absent_or_empty_meta_is_silent(self):
        from backend.arch.validate import spec_vs_tensors_check
        self.assertEqual(spec_vs_tensors_check(object(), {}), [])
        self.assertEqual(spec_vs_tensors_check(object(), {"header": None}), [])


if __name__ == "__main__":
    unittest.main()
