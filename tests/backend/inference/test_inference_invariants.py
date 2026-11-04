"""Tests for the inference invariant battery (backend/inference/invariants.py).

Two halves:
  • the battery is GREEN on the real snapshot models (V3 / V3.2 / V4-Pro);
  • the invariants actually FIRE on crafted violations (not vacuous) — this
    is what makes a green battery meaningful.
"""
from __future__ import annotations

import json
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.invariants import (
    _inv_batch_linearity,
    _inv_cp_scheme,
    _inv_phase_closure,
    run_inference_invariants,
)

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
_CASES = [
    "deepseek-ai__DeepSeek-V3",
    "deepseek-ai__DeepSeek-V3.2",
    "deepseek-ai__DeepSeek-V4-Pro",
]


class TestBatteryGreenOnSnapshots(unittest.TestCase):
    """Every exact invariant holds for the reconciled DeepSeek family."""

    @pytest.mark.regression  # minutes-long
    def test_all_invariants_hold(self):
        from tests.snapshots import shared_loader

        ran = 0
        for snap in _CASES:
            p = SNAP_DIR / f"{snap}.json"
            if not p.exists():
                continue
            with self.subTest(model=snap):
                config = json.loads(p.read_text())["config"]
                resolved = shared_loader().match(config, model_id=snap.replace("__", "/", 1))
                results = run_inference_invariants(config, resolved)
                self.assertTrue(results, f"{snap}: battery produced no results")
                failed = [r for r in results if not r.passed]
                self.assertEqual(
                    failed, [],
                    f"{snap}: " + "; ".join(f"{r.name}: {r.detail}" for r in failed))
                ran += 1
        self.assertGreater(ran, 0, "no snapshot cases ran")


class TestInvariantsAreNotVacuous(unittest.TestCase):
    """The checks must FAIL on crafted violations — a battery that can't go
    red proves nothing."""

    #: One attention row as the two schemes shape it at cp=8: the query is
    #: cut by heads under ulysses and by tokens under pcp (same size), the
    #: KV is whole under pcp, and the head-less per-token index is whole
    #: under ulysses.
    @staticmethod
    def _cp_rows(q, kv, index, out, flops=1000,
                 kv_producer=False):
        row = {
            "aggregation_id": "x", "op": "flash_dsa",
            "flops_per_rank": flops,
            "shape_in": [q, kv, index], "shape_out": out}
        result = {"per_op_timing": [row]}
        if kv_producer:
            result["head_repartition_kv_producers"] = ["x"]
        return result

    def _uly(self, **kw):
        return self._cp_rows([1, 2, 8192, 128], [1, 2, 8192, 128],
                             [1, 8192, 2048], [1, 16, 1024, 128], **kw)

    def _pcp(self, **kw):
        return self._cp_rows([1, 16, 1024, 128], [1, 16, 8192, 128],
                             [1, 1024, 2048], [1, 16, 1024, 128], **kw)

    def test_cp_scheme_holds_on_the_schemes_own_ratios(self):
        """Query x1, KV x cp, head-less per-token x 1/cp, output equal."""
        self.assertTrue(_inv_cp_scheme(self._uly(), self._pcp(), 8).passed)

    def test_cp_scheme_producer_ratio_comes_from_layout_provenance(self):
        """A marked producer owes cp even when the two rows happen to match.

        The earlier gate inferred the expected ratio from those same output
        axes, so this equality licensed itself.  Provenance from the Ulysses
        layout walk now demands the ratio, and the walked output check still
        sees that it did not happen.
        """
        r = _inv_cp_scheme(self._uly(kv_producer=True), self._pcp(), 8)
        self.assertFalse(r.passed)
        self.assertIn("expected ratio 8", r.detail)

    def test_cp_scheme_fires_on_a_leaked_sharding_factor(self):
        """A KV operand that is neither whole nor cut by cp — the head-shard
        applied where the scheme does not shard."""
        leaked = self._cp_rows([1, 16, 1024, 128], [1, 5, 8192, 128],
                               [1, 1024, 2048], [1, 16, 1024, 128])
        r = _inv_cp_scheme(self._uly(), leaked, 8)
        self.assertFalse(r.passed)
        self.assertIn("flash_dsa", r.detail)

    def test_cp_scheme_fires_when_the_work_differs(self):
        """Per-rank FLOPs is the invariant; the schemes cut one matrix."""
        r = _inv_cp_scheme(self._uly(), self._pcp(flops=2000), 8)
        self.assertFalse(r.passed)
        self.assertIn("FLOPs", r.detail)

    def test_cp_scheme_fires_when_the_output_differs(self):
        """Both schemes hand the next op the same rank-local tensor."""
        wider = self._cp_rows([1, 16, 1024, 128], [1, 16, 8192, 128],
                              [1, 1024, 2048], [1, 16, 8192, 128])
        r = _inv_cp_scheme(self._uly(), wider, 8)
        self.assertFalse(r.passed)
        self.assertIn("output", r.detail)

    def test_phase_closure_fires_on_mismatch(self):
        res = {"phases": {"prefill": {"t_sol_s": 1.0}},
               "per_op_timing": [{"t_op_s": 0.5}]}  # Σ parts ≠ displayed total
        results = _inv_phase_closure(res)
        self.assertTrue(results and not results[0].passed)

    def test_phase_closure_holds_when_sum_matches(self):
        res = {"phases": {"prefill": {"t_sol_s": 1.0}},
               "per_op_timing": [{"t_op_s": 0.6}, {"t_op_s": 0.4}]}
        results = _inv_phase_closure(res)
        self.assertTrue(results and results[0].passed)

    def test_phase_closure_fires_on_a_millisecond_step(self):
        """The check used to run through `_approx`, whose scale floors at 1.0.
        For a count that is the smallest meaningful unit; for seconds it is a
        fifty-fold overstatement of a decode step, and the comparison degraded
        to "agree within 20 ms". A phase publishing 0.0 against 8.4 ms of
        per-op time — which is what an infeasible DeepSeek-V3 decode did —
        passed. The two sides are one addition over one list, so nothing here
        needs slack."""
        res = {"phases": {"decode": {"t_sol_s": 0.0}},
               "per_op_timing_decode": [{"t_op_s": 0.008359}]}
        results = _inv_phase_closure(res)
        self.assertTrue(results and not results[0].passed, results)
        self.assertIn("0.008359", results[0].detail)

    def test_phase_closure_holds_for_a_millisecond_step_that_matches(self):
        """The tolerance is relative, so the same small numbers still close
        when they agree — the check did not simply get stricter everywhere."""
        res = {"phases": {"decode": {"t_sol_s": 0.008359}},
               "per_op_timing_decode": [{"t_op_s": 0.004}, {"t_op_s": 0.004359}]}
        results = _inv_phase_closure(res)
        self.assertTrue(results and results[0].passed, results)

    def test_batch_linearity_runs_the_real_walk(self):
        """The gate asks eval's walk directly — no legacy projection, no
        mock: a mocked producer verifies the mock. One real model proves the
        invariant is not vacuous."""
        import json
        from pathlib import Path

        from tests.snapshots import shared_loader

        snapshot = (Path(__file__).resolve().parents[3] / "backend"
                    / "arch" / "snapshots"
                    / "meta-llama__Llama-3.2-1B.json")
        resolved = shared_loader().match(
            json.loads(snapshot.read_text())["config"],
            model_id="meta-llama/Llama-3.2-1B")
        result = _inv_batch_linearity(resolved)
        self.assertTrue(result.passed, result.detail)


if __name__ == "__main__":
    unittest.main()
