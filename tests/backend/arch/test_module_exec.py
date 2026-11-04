"""Unit tests for module-execution validation (validate/_module_exec.py).

The check logic is exercised by mocking the eval compatibility entry to
return controlled results (clean / non-linear / non-finite / raising);
a snapshot smoke test confirms a real, correct spec produces 0 warnings.
"""
from __future__ import annotations

import json
import sys
import unittest

from tests.snapshots import snapshot_config
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import ResolvedSpec
from backend.arch.schema import (
    ArchSpec,
    BlockGraph,
    GraphNode,
    ModelLayer,
)
from backend.arch.validate._module_exec import run_module_exec_checks


def _br(flops, params):
    return SimpleNamespace(total_flops=flops, total_params=params)


def _one_template_spec():
    block = BlockGraph(
        inputs=["x"],
        graph={"n": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
        outputs=["n"],
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
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 2}, {}, {})


def _run(resolved, per_call, *, new_components=None):
    """Run the checks with block analysis replaced by `per_call(template,
    batch_size, seq_len)`. KV sanity is stubbed out to isolate the engine
    checks."""
    def _analyze_blocks(resolved, *, batch=1, input_tokens=1):
        return {
            template: per_call(template, batch, input_tokens)
            for template in {layer.template for layer in resolved.spec.model}
        }

    with mock.patch(
            "backend.arch.validate._module_exec.project_blocks",
            _analyze_blocks,
    ), \
            mock.patch("backend.arch.validate._module_exec._kv_sanity", return_value=[]):
        return run_module_exec_checks(resolved, new_components=new_components)


def _kinds(ws):
    return {w.kind for w in ws}


class TestModuleExec(unittest.TestCase):
    def setUp(self):
        self.resolved = _one_template_spec()

    def test_linear_finite_is_clean(self):
        # params batch-independent, flops exactly ×2.
        ws = _run(self.resolved, lambda t, b, s: _br(100 * b, 50))
        self.assertEqual(ws, [])

    def test_exception_reported_not_raised(self):
        def boom(t, b, s):
            raise ValueError("shape inference blew up")
        ws = _run(self.resolved, boom)
        self.assertIn("module_exec_error", _kinds(ws))

    def test_nonfinite_flagged(self):
        ws = _run(self.resolved, lambda t, b, s: _br(float("inf"), 50))
        self.assertIn("module_nonfinite", _kinds(ws))

    def test_batch_nonlinear_flops_flagged(self):
        # quadratic-in-batch flops violates linearity.
        ws = _run(self.resolved, lambda t, b, s: _br(100 * b * b, 50))
        self.assertIn("module_batch_nonlinear", _kinds(ws))

    def test_params_batch_variant_flagged(self):
        ws = _run(self.resolved, lambda t, b, s: _br(100 * b, 50 * b))
        self.assertIn("module_batch_noninvariant", _kinds(ws))

    def test_new_module_tagged(self):
        ws = _run(self.resolved, lambda t, b, s: _br(float("inf"), 50),
                  new_components={"block"})
        nf = [w for w in ws if w.kind == "module_nonfinite"]
        self.assertTrue(nf and nf[0].extra["new"] is True)


class TestModuleExecRealSpec(unittest.TestCase):
    SNAP = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
            / "deepseek-ai__DeepSeek-V4-Pro.json")

    def test_dsv4_computes_clean(self):
        snapshot_config(self.SNAP.name)
        from tests.snapshots import shared_loader
        cfg = json.loads(self.SNAP.read_text())["config"]
        resolved = shared_loader().match(
            cfg, model_id=self.SNAP.stem.replace("__", "/", 1))
        # Real engine + real KV: a correct spec runs at basic configs cleanly.
        self.assertEqual(run_module_exec_checks(resolved), [])

    def test_a_template_no_layer_runs_is_not_executed(self):
        """GLM-5.1 shares its spec with GLM-5.2, whose `shared` indexer
        layers GLM-5.1's config selects none of: the model instantiates no
        such block, so there is no module of it to execute."""
        from tests.snapshots import resolve_snapshot
        self.assertEqual(
            run_module_exec_checks(resolve_snapshot("zai-org/GLM-5.1")), [])


if __name__ == "__main__":
    unittest.main()
