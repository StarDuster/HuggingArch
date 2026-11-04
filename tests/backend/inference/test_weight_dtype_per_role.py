"""Per-role weight rules of a precision profile, read through the weight
ledger (`per_rank_weight_breakdown(precision=...)`).

A weight rule selects matrices by role and runs them at the deployment
format: a quant scheme stores its data at the scheme's width AND its
scales; a plain dtype stores `params × width` with no scale. Roles no
rule selects keep the bytes the checkpoint saves. An empty profile is
the as-shipped ledger byte for byte.
"""
from __future__ import annotations

import sys
import unittest

import pytest

from tests.snapshots import resolve_snapshot
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.precision import (
    PrecisionError,
    PrecisionProfile,
)
from backend.arch.spec_tree import iter_leaves
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import (
    per_rank_weight_breakdown,
)


def _resolved(model_name):
    return resolve_snapshot(model_name)


def _weights(*rules):
    return PrecisionProfile.canonicalize({"weights": [
        {"select": {"roles": [role]}, "format": fmt} for role, fmt in rules
    ]})


_STEP37_TP8 = dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
                   ep_moe=1, cp=1, dp_replicas=1)


class TestEmptyProfileIsAsShipped(unittest.TestCase):

    def test_empty_profile_byte_identical(self):
        """An empty profile produces exactly the as-shipped ledger."""
        empty = PrecisionProfile.canonicalize({})
        for name in ("meta-llama__Llama-3.2-3B",
                     "XiaomiMiMo__MiMo-V2.5",
                     "stepfun-ai__Step-3.7-Flash"):
            resolved = _resolved(name)
            baseline = per_rank_weight_breakdown(resolved, ParallelConfig())
            with_empty = per_rank_weight_breakdown(
                resolved, ParallelConfig(), precision=empty)
            self.assertEqual(
                baseline["deployment_total_bytes"],
                with_empty["deployment_total_bytes"],
                msg=f"{name}: empty profile drifted the total",
            )
            self.assertEqual(
                baseline["total_bytes_per_rank"],
                with_empty["total_bytes_per_rank"],
                msg=f"{name}: empty profile drifted per-rank",
            )

    def test_unknown_role_is_a_repairable_error(self):
        """A rule naming a role roles.yaml does not register is refused
        with an error that names the rule, not silently ignored."""
        with self.assertRaises(PrecisionError) as ctx:
            _weights(("nonexistent_role_zzz", {"scheme": "fp8"}))
        self.assertTrue(ctx.exception.errors)
        self.assertIn("weights[0]", ctx.exception.errors[0]["path"])


class TestWeightRuleReachesTheLedger(unittest.TestCase):

    def test_step37_ffn_moe_fp8_hits_doc_target(self):
        """Step-3.7-Flash deployed as BF16 attention + FP8 MoE (TP=8): the
        deployment doc's target is W/rank ≈ 27 GB. The rule must free a
        MoE-dominated margin and land in a plausible window around it
        (attention stays BF16, so it sits a bit above the doc's quote)."""
        resolved = _resolved("stepfun-ai__Step-3.7-Flash")
        parallel = ParallelConfig(**_STEP37_TP8)
        baseline = per_rank_weight_breakdown(resolved, parallel)
        cast = per_rank_weight_breakdown(
            resolved, parallel,
            precision=_weights(("ffn_moe", {"scheme": "fp8"})))

        delta_gb = (baseline["total_bytes_per_rank"]
                    - cast["total_bytes_per_rank"]) / 1e9
        self.assertGreater(
            delta_gb, 10.0,
            msg=f"expected MoE FP8 to free >10 GB/rank; got {delta_gb:.1f}",
        )
        cast_gb = cast["total_bytes_per_rank"] / 1e9
        self.assertGreater(cast_gb, 25.0, msg=f"cast={cast_gb:.1f} GB too low")
        self.assertLess(cast_gb, 40.0, msg=f"cast={cast_gb:.1f} GB too high")

    def test_step37_ffn_moe_nvfp4_saves_more_than_fp8_net_of_scales(self):
        """From a BF16 checkpoint, FP8 saves ~1 B/element (block-128 F32
        scales cost 4 B per 16384 elements) and NVFP4 saves 1.5 B/element
        minus its group-16 FP8 scales (1/16 B/element). Both schemes'
        scales are counted, so the NVFP4 saving is ~1.44× FP8's, strictly
        under the 1.5× a scale-less count would give."""
        resolved = _resolved("stepfun-ai__Step-3.7-Flash")
        parallel = ParallelConfig(**_STEP37_TP8)
        baseline = per_rank_weight_breakdown(resolved, parallel)
        fp8 = per_rank_weight_breakdown(
            resolved, parallel,
            precision=_weights(("ffn_moe", {"scheme": "fp8"})))
        fp4 = per_rank_weight_breakdown(
            resolved, parallel,
            precision=_weights(("ffn_moe", {"scheme": "nvfp4",
                                            "activation": "w4a4"})))

        d_fp8 = baseline["total_bytes_per_rank"] - fp8["total_bytes_per_rank"]
        d_fp4 = baseline["total_bytes_per_rank"] - fp4["total_bytes_per_rank"]
        ratio = d_fp4 / d_fp8
        self.assertGreater(ratio, 1.42, msg=f"nvfp4/fp8 saving ratio {ratio}")
        self.assertLess(ratio, 1.46, msg=f"nvfp4/fp8 saving ratio {ratio}")

    def test_plain_dtype_rule_moves_only_its_role_by_params(self):
        """`{attention: F32, ffn_dense: BF16}` on a BF16 checkpoint:
        attention matrices grow by exactly `params × 2 B` (no scale for a
        plain dtype); the ffn_dense rule restates the shipped dtype and
        moves nothing."""
        resolved = _resolved("meta-llama__Llama-3.2-3B")
        attn_params = 0
        for lt in iter_leaves(resolved):
            if lt.role != "attention" or len(lt.shape) < 2:
                continue
            numel = 1
            for d in lt.shape:
                numel *= int(d)
            attn_params += numel * int(lt.multiplicity)

        baseline = per_rank_weight_breakdown(resolved, ParallelConfig())
        override = per_rank_weight_breakdown(
            resolved, ParallelConfig(),
            precision=_weights(("attention", {"dtype": "F32"}),
                               ("ffn_dense", {"dtype": "BF16"})))
        actual_delta = (override["deployment_total_bytes"]
                        - baseline["deployment_total_bytes"])
        self.assertAlmostEqual(
            actual_delta, 2 * attn_params,
            delta=max(1.0, attn_params * 1e-3),
            msg=f"delta={actual_delta:.0f} B, expected attention "
                f"matrix params×2B={2 * attn_params:.0f} B",
        )


@pytest.mark.parametrize("snapshot", [
    # FIRST because it is the only model where the file's width and the
    # rank's differ: gemma-2 is saved F32 and loaded BF16
    # (`dtypes.served_dtype`), so a ledger that went back to indexing the
    # FILE's tensor list reads 36,966,823,936 for its leaf fold against
    # capacity's 18,483,411,968 and this reds (checked by restoring the
    # defect). On every other model the two lists carry the same widths and
    # the mistake is invisible — which is how it lived in
    # `weight_breakdown` across the whole corpus unnoticed.
    "google__gemma-2-9b",
    # The aux half of the same claim: DSv3's MTP stack is stored and not
    # loaded by default, so the ledger equals capacity's `without_aux`
    # and not its total.
    "deepseek-ai__DeepSeek-V3",
])
def test_the_ledger_totals_what_capacity_says_the_rank_holds(snapshot):
    """One number, two readers — the page and the budget.

    `per_rank_weight_breakdown` publishes two totals reached two ways:
    `total_bytes_per_rank` folds the layout's per-rank artifacts, and
    `deployment_total_bytes` folds the per-leaf bytes it reads off the
    analysis row index. Capacity asks the walk's rule for the rank's
    resident weights. On one rank all three are the same quantity, and
    nothing used to say so — which is exactly why the leaf fold could read
    the FILE's widths for a whole release without a number looking wrong.

    The silence is directional, and that is the argument for this test: a
    reader that takes the file's width where it wants the rank's is caught
    by no gate at all, while the mirror mistake — the file's list carrying a
    loaded width — is caught loudly by the checkpoint byte gate. Only this
    side needs a ratchet, and it has to be in the DEFAULT set, because the
    surfaces that would otherwise notice (`sol_surface`) are all
    `regression`.
    """
    from backend.arch.eval import capacity as capacity_mod
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    resolved = _resolved(snapshot)
    ledger = per_rank_weight_breakdown(resolved, ParallelConfig())
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=1).analyze()
    held = float(capacity_mod.weights(analysis).without_aux(analysis.program))
    # The artifact fold, and then the leaf fold — the one that reads the
    # row index and so the one that can take the wrong projection.
    assert float(ledger["total_bytes_per_rank"]) == held
    assert float(ledger["deployment_total_bytes"]) == held


if __name__ == "__main__":
    unittest.main()
