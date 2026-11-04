"""A check that cannot evaluate one template says so on that template and
keeps checking the others.

The validate modules used to `except: continue` around each template; a
block that never evaluated left the gate green. Raising instead would have
been the other silence: `_guarded` turns a raise into ONE `check_crashed`
and every sibling template's findings are gone with it. The unit here is
the template (or the stream / the model): one finding for the failure,
the rest of the loop unaffected, and nothing collected then dropped.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import shared_loader
from backend.arch.validate import _axioms, _structural
from backend.arch.validate._orchestrator import _guarded
from backend.arch.validate._cross_layer_channels import _StreamShapeCheck

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolved(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    return shared_loader().match(config, model_id=stem.replace("__", "/", 1))


class TestUnknownTemplateIsOneFinding(unittest.TestCase):
    def test_structural_diff_names_the_missing_template_and_checks_the_rest(self):
        resolved = _resolved("meta-llama__Llama-3.2-1B")
        real = _structural._collect_templates(resolved)
        self.assertTrue(real)
        seen = []
        def check(_source, block):
            seen.append(block)
            return []
        out = _structural._structural_diff_warnings(
            check, "def forward(): pass", resolved, [real[0], "nope", *real[1:]])
        self.assertEqual([(f.kind, f.template) for f in out],
                         [("unknown_template", "nope")])
        self.assertEqual(len(seen), len(real), "the siblings were still checked")


class TestAnUnwalkableModelLeavesStreamShapesUnjudged(unittest.TestCase):
    def test_the_walk_failure_is_one_finding_not_a_pass(self):
        """Declared stream shapes are compared against the model's walk. A
        model that will not walk is one finding carrying the walk's own
        refusal — not a `check_crashed`, and not a silent pass."""
        from backend.arch.eval.program import Program
        from backend.arch.validate import _cross_layer_channels
        resolved = _resolved("zai-org__GLM-5.1")
        self.assertIn("dsa_topk", resolved.spec.stream_names().inner_names())
        with mock.patch.object(_cross_layer_channels, "Deployment",
                               side_effect=RuntimeError("no eval today")):
            out = _StreamShapeCheck(resolved, Program.of(resolved)).run()
        self.assertEqual([(f.kind, f.template) for f in out],
                         [("stream_shape_unjudged", "<model>")])
        self.assertIn("no eval today", out[0].detail)


class TestAnUnanalysableModelIsUnjudged(unittest.TestCase):
    def test_the_analysis_failure_is_a_finding_not_a_pass(self):
        """Claims are judged on the model's analysis. A model that will not
        analyse grew no cache to judge: that is one finding carrying the
        analysis's own refusal, and the checks that do not need it still run."""
        resolved = _resolved("google__gemma-2-9b")
        with mock.patch.object(_axioms, "Deployment",
                               side_effect=RuntimeError("analysis down")):
            out = _guarded("axiom checks", _axioms.run_axiom_checks,
                           "google/gemma-2-9b", resolved)
        unjudged = [f for f in out if f.kind == "axiom_unjudged"]
        self.assertEqual([f.template for f in unjudged], ["<model>"])
        self.assertIn("analysis down", unjudged[0].detail)
        self.assertNotIn("check_crashed", [f.kind for f in out])


class TestAStackBlockDeclaringACacheIsUnjudged(unittest.TestCase):
    def test_a_cache_no_call_grew_is_a_finding_without_claims(self):
        """A cache belongs to a component call. A block a stack runs directly
        that declares one grew a cache nothing can judge — its window contract
        included — so it is reported whether or not the block has claims."""
        from backend.arch.loader import ResolvedSpec
        from backend.arch.schema import (ArchSpec, BlockGraph, CacheSpec,
                                         GraphNode, ModelLayer)
        block = BlockGraph(
            inputs=["x"],
            graph={"wq": GraphNode(op="linear", inputs=["x"],
                                   dims={"in": "d_model", "out": "d_model"})},
            outputs=["wq"],
            cache=CacheSpec(family="SWA", entries=[{"node": "wq", "kind": "kv"}]))
        spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
            name="demo", dims={"d_model": "config.hidden_size"},
            model=[
                ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                           boundary_role="input_embedding",
                           dims={"n_vocab": "d_model", "d_model": "d_model"}),
                ModelLayer(name="layers", template="block", repeat="n_layers",
                           inputs=["embed"]),
            ],
            blocks={"block": block})
        resolved = ResolvedSpec(spec, {"d_model": 16, "n_layers": 1},
                                {"hidden_size": 16, "torch_dtype": "bfloat16"}, {})
        out = _axioms.run_axiom_checks("demo", resolved)
        self.assertEqual([(f.kind, f.template) for f in out],
                         [("axiom_unjudged", "block")])


if __name__ == "__main__":
    unittest.main()
