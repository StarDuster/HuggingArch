"""Golden: the audit-curves battery + reconciliation stay green on a class-diverse
cohort. This is the reusable acceptance gate the spec-gen pipeline can call —
a generated spec is "cost-correct" only if Tier-0 holds and every chart reconciles.

Asserts, per representative model (one per attention class):
  - every Tier-0 (universal monotonicity) axiom passes,
  - in fact ALL axioms pass (these are reconciled-correct models),
  - the first-principles reconciliation has no ⚠ (every chart's number is
    reproducible from the roofline / capacity identities).

Heavier than the other goldens (it sweeps seq×batch), but bounded: 4 models ×
3 seqs × 3 batches. Run from the project root on a local disk (importing
from a networked filesystem can hang).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_SKILL = ROOT / ".claude/skills/audit-curves"
sys.path.insert(0, str(_SKILL))
sys.path.insert(0, str(ROOT))

from sweep import build_curves
from axioms import derive_expected_shape, run_axioms
from reconcile import reconcile_model
from tests.snapshots import snapshot_config


# One model per attention class — the cohort that exercises every Tier-1
# derivation branch (dense / compressed-sparse / linear-recurrent / windowed).
# The smallest first: the default run takes a parametrised test's first case
# (tests/conftest.py), and `-m regression` the other three.
_COHORT = {
    "google__gemma-2-9b": "GQA + SWA",
    "deepseek-ai__DeepSeek-V3": "dense MLA",
    "deepseek-ai__DeepSeek-V4-Pro": "CSA + hyper-connection",
    "Qwen__Qwen3-Coder-Next": "GatedDeltaNet (linear)",
}
_SEQS = [2048, 32768, 1048576]   # short, mid, and 1M (the long-context regime)
_BATCHES = [1, 64, "max"]


class TestCurveAxiomsGolden:
    @pytest.mark.parametrize("stem, cls", list(_COHORT.items()),
                             ids=list(_COHORT))
    def test_battery_green_and_reconciles(self, stem, cls):
        cfg = snapshot_config(stem)
        model_id = stem.replace("__", "/", 1)
        label = stem.split("__")[-1]
        curves = build_curves(label, cfg, model_id, _SEQS, _BATCHES)
        expected = derive_expected_shape(model_id, cfg)
        results = run_axioms(curves, expected)
        assert results, f"{label}: battery produced no results"

        tier0 = [r for r in results if r.tier == "0" and not r.passed]
        assert tier0 == [], (
            f"{label} ({cls}) Tier-0 violated: "
            + "; ".join(f"{r.axiom_id}: {r.measured}" for r in tier0))

        failed = [r for r in results if not r.passed]
        assert failed == [], (
            f"{label} ({cls}) axioms violated: "
            + "; ".join(f"{r.axiom_id}: {r.detail or r.measured}" for r in failed))

        warns = [ln for ln in reconcile_model(curves, expected) if "⚠" in ln]
        assert warns == [], (
            f"{label} ({cls}) reconciliation did not self-reconcile:\n  "
            + "\n  ".join(warns))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
