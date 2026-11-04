"""Validate refuses a spec the multi-rank analysis would refuse, with the
node and the fix, and runs an injected gate only once every other gate
passed."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import snapshot_config

from backend.arch.validate._analysable import ENGINE_KIND, SPEC_KIND
from backend.arch.validate._finding import Finding
from backend.arch.validate._orchestrator import GateOutcome

MODEL_ID = "meta-llama/Llama-3.2-1B"
NORM = ("{ name: norm, op: rmsnorm, dim: d_model, weight_axes: "
        "[{ replicated: true }] , inputs: [layers]}")
#: A weightless, stateless custom op: the geometry family, which no layout
#: law places.
GEOMETRY_OP = """
custom_ops:
  mask_channel:
    description: "A one-channel per-token flag; weightless."
    flops: "0"
    params: "0"
    signature: { x: [B, s_ctx, d_model] }
    couples_on: []
    shape_kind: axis_replace
    out_key: Out
"""


def _validate(spec_text=None, gates=()):
    from backend.arch.validate import validate
    snapshot_config(MODEL_ID)
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}):
        spec_name = None
        if spec_text is not None:
            models = Path(tmp) / "models"
            models.mkdir()
            (models / "llama_glmsol_case.yaml").write_text(spec_text)
            spec_name = "llama_glmsol_case"
        return validate(MODEL_ID, spec_name=spec_name, overlay_dir=tmp,
                        downstream_gates=gates)


def _llama(edit):
    text = (PROJECT_ROOT / "backend" / "arch" / "models" / "llama.yaml").read_text()
    assert NORM in text
    return edit(text)


def _with_geometry_op(text):
    text = text.replace("\nmodel:\n", GEOMETRY_OP + "model:\n", 1)
    return text.replace(NORM, NORM + "\n  - { name: mask, op: mask_channel, "
                        "dims: { Out: \"1\" }, inputs: [norm] }")


class TestDispatchScan(unittest.TestCase):
    def test_a_geometry_custom_op_is_refused_with_node_and_fix(self):
        result = _validate(_llama(_with_geometry_op))
        self.assertFalse(result["valid"])
        (refusal,) = result["analysis_refusals"]
        self.assertEqual(refusal["kind"], SPEC_KIND)
        self.assertEqual(refusal["node"], "mask")
        self.assertEqual(refusal["declaration"], "op")
        self.assertIn("registered ops", refusal["fix"])
        self.assertIn("analysis refuses node `mask`", " ".join(result["errors"]))


class TestDownstreamGates(unittest.TestCase):
    def test_a_passing_gate_reports_passed(self):
        result = _validate(gates=(lambda resolved, config, model_result: GateOutcome([], {"deployments": []}),))
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["analysis"], "passed")
        self.assertEqual(result["analysis_replay"], {"deployments": []})

    def test_no_gate_is_not_requested(self):
        self.assertEqual(_validate()["analysis"], "not_requested")

    def test_a_failed_earlier_gate_short_circuits(self):
        called = []
        result = _validate(_llama(_with_geometry_op),
                           gates=(lambda *args: called.append(args) or GateOutcome([]),))
        self.assertEqual(result["analysis"], "not_run")
        self.assertFalse(result["valid"])
        self.assertEqual(called, [])

    def test_only_engine_refusals_block_by_engine(self):
        engine = Finding(kind=ENGINE_KIND, detail="engine refuses", node="lm_head")
        result = _validate(gates=(lambda *args: GateOutcome([engine]),))
        self.assertFalse(result["valid"])
        self.assertEqual(result["analysis"], "refused")
        self.assertTrue(result["blocked_by_engine"])
        self.assertEqual(result["analysis_refusals"][-1]["node"], "lm_head")

    def test_a_spec_refusal_does_not_block_by_engine(self):
        spec = Finding(kind=SPEC_KIND, detail="fix the dims", node="x")
        result = _validate(gates=(lambda *args: GateOutcome([spec]),))
        self.assertFalse(result["blocked_by_engine"])


class TestEngineGapEntries(unittest.TestCase):
    """The stored shape of an engine refusal is read off validate's own
    finding, never off its text."""

    def _refusals(self):
        from backend.arch.eval.refusal import EngineGap
        from backend.arch.validate._analysable import Refusal, grouped_findings

        def raised(deployment):
            try:
                raise EngineGap("no closure for the gather", capability="gather extent",
                                node="layers.0.lm_head")
            except EngineGap as exc:
                return Refusal.of(exc, deployment=deployment, stage="decode")
        return [f.as_dict() for f in grouped_findings([raised("tp8"), raised("ep8")])]

    def test_one_entry_per_cause_with_its_deployments(self):
        from backend.arch.validate._analysable import engine_gap_entries
        refusals = self._refusals()
        (entry,) = engine_gap_entries({"analysis_refusals": refusals})
        self.assertEqual(entry["capability"], "gather extent")
        self.assertEqual(entry["node"], "layers.0.lm_head")
        self.assertEqual(entry["deployments"], ["ep8", "tp8"])
        self.assertEqual(entry["stages"], ["decode"])
        self.assertEqual(entry["exception"], "EngineGap")
        self.assertEqual(entry["detail"], refusals[0]["detail"])
        self.assertNotIn("kind", entry)

    def test_only_unattributed(self):
        from backend.arch.validate._analysable import OPEN_KIND, only_unattributed
        findings = [{"kind": ENGINE_KIND, "detail": "e"}, {"kind": OPEN_KIND, "detail": "o"},
                    {"kind": SPEC_KIND, "detail": "s"}]
        self.assertTrue(only_unattributed({"errors": ["o", "e"], "analysis_refusals": findings}))
        self.assertFalse(only_unattributed({"errors": ["o", "s"], "analysis_refusals": findings}))
        self.assertFalse(only_unattributed({"errors": ["e"], "analysis_refusals": findings}))
        self.assertFalse(only_unattributed({"errors": ["o", "x"], "analysis_refusals": findings}))


class TestCliStdout(unittest.TestCase):
    def test_a_blocked_result_is_one_json_object_on_stdout(self):
        """The writer and the verdict gate parse stdout whole; a blocked run
        says so in that JSON, and the human line goes to stderr."""
        import io
        import json
        from contextlib import redirect_stderr, redirect_stdout
        from backend.arch.validate import __main__ as cli
        blocked = {"valid": False, "errors": ["engine refuses"], "blocked_by_engine": True,
                   "analysis_refusals": [{"kind": ENGINE_KIND, "detail": "engine refuses"}]}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "validate", lambda *a, **k: blocked), \
                mock.patch.object(sys, "argv", ["validate", "acme/Foo"]), \
                redirect_stdout(out), redirect_stderr(err), \
                self.assertRaises(SystemExit):
            cli.main()
        self.assertEqual(json.loads(out.getvalue()), blocked)
        self.assertIn("blocked by the engine", err.getvalue())


if __name__ == "__main__":
    unittest.main()
