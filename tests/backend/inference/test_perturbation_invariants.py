"""Perturbation invariants — output-vs-output exact checks that hold
regardless of the (un-knowable) correct constants. See
docs/validation.md §11 (class C). Generic: no per-model
knowledge, no hardcoded expected values.

CP-scheme invariance
--------------------
Ulysses and pcp are two communication schemes for the SAME
context-parallel work. Each rank ends with the same per-rank compute
and activation FOOTPRINT — ulysses shards heads / keeps full seq, pcp
shards seq / keeps full heads, but `N/cp·S·S == N·S/cp·S`. So a leaf's
per-rank ACTIVATION access (access minus the cp-invariant weight bytes)
must NOT depend on `cp_scheme`.

A divergence means one scheme's sharding silently dropped or
double-applied a factor. This is exactly the DSA bug
(`indexer_logits` / `dsa_topk_mask`): they carry a context axis (Lq)
but no head axis, so the ulysses head-shard no-op'd and left per-rank
access at full S² while pcp correctly gave S²/cp. With the fix both
schemes shard the query-seq axis and agree. Reverting the fix makes
this test fire on those two ops.
"""
from __future__ import annotations

import json
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from backend.inference.invariants import _inv_cp_scheme

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

# Models whose attention is CP-shardable, spanning DSA (V3.2), hybrid
# CSA/HCA (V4-Pro) and dense MLA (V3). Same TP1/CP8/EP8 deployment the
# recon used. cp>1 is what makes ulysses≠pcp meaningful.
_CASES = [
    ("DeepSeek-V3", "deepseek-ai__DeepSeek-V3"),
    ("DeepSeek-V3.2", "deepseek-ai__DeepSeek-V3.2"),
    ("DeepSeek-V4-Pro", "deepseek-ai__DeepSeek-V4-Pro"),
]
_PARALLEL = dict(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1,
                 ep_moe=8, cp=8, pp=1, dp_attention=1, dp_replicas=1)
_SEQ = 65536


def _config(snapshot: str):
    p = SNAP_DIR / f"{snapshot}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["config"]


def _resolved(snapshot: str):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    config = _config(snapshot)
    if config is None:
        return None
    return shared_loader().match(config, model_id=snapshot.replace("__", "/", 1))


def _estimate(config, scheme: str, resolved=None) -> dict:
    """One prefill estimate under `scheme`. Batch is fixed at 1 so the
    comparison is not polluted by the max-batch swing decode introduces."""
    par = ParallelConfig(cp_scheme=scheme, **_PARALLEL)
    return estimate_capacity(
         config=config, resolved=resolved, gpu_sku="B200",
        num_gpus=8 * _PARALLEL["cp"],
        avg_input_tokens=_SEQ, avg_output_tokens=256,
        parallel=par, selection_mode="measured",
    )


class TestCpSchemeInvariantActivation(unittest.TestCase):
    """The two context-parallel schemes owe each other identical per-rank
    work, and operands that follow the axis each scheme cuts.

    The law itself is `invariants._inv_cp_scheme`; this exercises it on real
    models under a mesh the battery does not use, rather than restating it.
    An earlier version asserted that per-rank ACTIVATION was equal too, which
    is false by construction: ulysses cuts heads and keeps every token, pcp
    cuts tokens and keeps every head, so the KV each rank must attend is
    whole under pcp and `1/cp` under ulysses.
    """

    @pytest.mark.regression  # minutes-long
    def test_the_schemes_agree_where_they_must(self):
        ran = 0
        for label, snapshot in _CASES:
            config = _config(snapshot)
            if config is None:
                continue
            with self.subTest(model=label):
                resolved = _resolved(snapshot)
                uly = _estimate(config, "ulysses", resolved)
                pcp = _estimate(config, "pcp", resolved)
                # Don't pass vacuously: an empty per-op table means the
                # calibration overlay errored out (the law then has nothing
                # to compare). This guard would have caught the
                # cross-component fusion-match failure on DSv4's rope path.
                self.assertTrue(
                    (uly.get("per_op_timing") and pcp.get("per_op_timing")),
                    f"{label}: per-op timing empty — calibration overlay "
                    "produced no rows (it likely errored)")
                result = _inv_cp_scheme(uly, pcp, _PARALLEL["cp"])
                self.assertTrue(result.passed, f"{label}: {result.detail}")
                ran += 1
        self.assertGreater(ran, 0, "no cases ran — snapshots missing?")


class TestCalibrationOverlayComputes(unittest.TestCase):
    """The measured-mode calibration overlay must actually run — `error`
    empty and `per_op_timing` non-empty. Locks the rewrite fix: a fusion
    match referencing a spec id absent from the scope index (DSv4's rope
    scheme reaching the indexer's q_rope) must be rejected, not KeyError into
    an empty result."""

    @pytest.mark.regression  # minutes-long
    def test_dsv4_family_overlay_runs(self):
        par = ParallelConfig(cp_scheme="ulysses", **_PARALLEL)
        ran = 0
        for label, snapshot in _CASES:
            config = _config(snapshot)
            if config is None:
                continue
            with self.subTest(model=label):
                r = estimate_capacity(
                     config=config, resolved=_resolved(snapshot), gpu_sku="B200",
                    num_gpus=8 * _PARALLEL["cp"],
                    avg_input_tokens=4096, avg_output_tokens=256,
                    parallel=par, selection_mode="measured",
                )
                self.assertFalse(
                    r.get("error"),
                    f"{label}: calibration overlay error: {r.get('error')!r}")
                self.assertTrue(
                    r.get("per_op_timing"),
                    f"{label}: per_op_timing empty despite no error")
                ran += 1
        self.assertGreater(ran, 0, "no cases ran — snapshots missing?")


if __name__ == "__main__":
    unittest.main()
