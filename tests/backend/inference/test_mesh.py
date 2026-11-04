"""Region-aware device mesh — the single source for the deployment's
rank layout, consumed by both ParallelConfig.validate (group sizes) and
the comm graft sites (base_stride). These lock the layout convention
and that both consumers stay derived from it."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import mesh
from backend.inference.parallel_config import ParallelConfig
from backend.arch import dims
from backend.inference.sharding import dim_to_tp_field


def _ffn_weight_degree(pc: ParallelConfig) -> int:
    """The degree that actually divides a dense FFN weight, read the way the
    shape walk reads it: the axis `d_ffn` names its world, the world names its
    ParallelConfig field. Asked here rather than asserted as a number so the
    claim below — cp spans the stage but does NOT divide the weight — is a
    statement about the registry and not about a table that agreed with it."""
    return int(getattr(pc, dim_to_tp_field()["d_ffn"]))


class TestMeshGroupSize(unittest.TestCase):
    def test_group_size_matches_per_region_product(self):
        pc = ParallelConfig(tp_attention=2, cp=2, dp_attention=2,
                            tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2)
        self.assertEqual(mesh.group_size(pc, "attention"), 2 * 2 * 2)
        # dense spans the stage via cp too (token-shard, weight replicated):
        # tp_dense_ffn × cp × dp_attention.
        self.assertEqual(mesh.group_size(pc, "dense"), 8 * 2 * 2)
        self.assertEqual(mesh.group_size(pc, "moe"), 4 * 2)

    def test_per_stage_size_is_max_region_group(self):
        # per_stage_size must read the SAME group products mesh exposes.
        pc = ParallelConfig(tp_attention=8, cp=1, dp_attention=1)
        self.assertEqual(
            pc.per_stage_size,
            max(mesh.group_size(pc, r) for r in ("attention", "dense", "moe")),
        )


class TestMeshBaseStride(unittest.TestCase):
    """base_stride = product of axes nested INSIDE the collective's axis.
    TP is innermost (stride 1); CP is spaced by tp_attention, EP by
    tp_moe_expert."""

    def test_tp_is_innermost_stride_one(self):
        pc = ParallelConfig(tp_attention=8, tp_moe_expert=4)
        self.assertEqual(mesh.base_stride(pc, "attention", "tp_attention"), 1)
        self.assertEqual(mesh.base_stride(pc, "moe", "tp_moe_expert"), 1)

    def test_cp_spaced_by_tp_attention(self):
        pc = ParallelConfig(tp_attention=8, cp=4, dp_attention=1)
        self.assertEqual(mesh.base_stride(pc, "attention", "cp"), 8)

    def test_ep_spaced_by_tp_moe_expert(self):
        pc = ParallelConfig(tp_moe_expert=4, ep_moe=2)
        self.assertEqual(mesh.base_stride(pc, "moe", "ep_moe"), 4)

    def test_dp_attention_outer_of_cp(self):
        # dp_attention is outermost in the attention region → spaced by
        # tp_attention × cp.
        pc = ParallelConfig(tp_attention=2, cp=2, dp_attention=2)
        self.assertEqual(mesh.base_stride(pc, "attention", "dp_attention"), 4)

    def test_unknown_axis_for_region_raises(self):
        pc = ParallelConfig()
        with self.assertRaises(ValueError):
            mesh.base_stride(pc, "moe", "cp")  # cp is not a MoE-region axis


class TestMeshLayoutCompleteness(unittest.TestCase):
    def test_pp_and_dp_replicas_reserved_as_global_outer(self):
        # PP / DP-replicas have no grafted collective yet, but the mesh
        # lists them as the outermost axes so a future PP p2p comm gets
        # the right (cross-node) stride for free.
        self.assertEqual(mesh.GLOBAL_OUTER_AXES, ("pp", "dp_replicas"))

    def test_every_region_axis_is_a_real_config_field(self):
        pc = ParallelConfig()
        for region, axes in mesh.REGION_AXES.items():
            for axis in axes:
                self.assertTrue(hasattr(pc, axis), f"{region}:{axis}")


class TestDenseRegionSpansStageViaCp(unittest.TestCase):
    """cp is a TOKEN-distribution axis: it shards the sequence and REPLICATES
    the FFN weight, so — exactly like attention — it spans the dense region's
    ranks but does NOT enter the FFN weight divisor. The realistic CP
    deployment therefore keeps `tp_dense_ffn == tp_attention`: both regions
    fill the same stage via the shared cp/dp factors, and only the inner TP
    axis splits weight. `tp_dense_ffn = tp_attn × cp` is OVER-sharding the FFN
    (it idles attention ranks), not the way to combine cp with the FFN.

    These pin the corrected semantics so the dense region can't drift back to
    omitting cp (which forced tp_dense to absorb cp → FFN weight ÷cp too much).
    """

    def test_cp_is_a_dense_region_axis(self):
        # cp spans the dense region's ranks (token-shard) just like attention.
        self.assertIn("cp", mesh.REGION_AXES["dense"])
        self.assertIn("cp", mesh.REGION_AXES["attention"])

    def test_attn8_ffn4_is_illegal_and_reports_idle(self):
        # tp_attn=8, tp_dense=4 → dense uses only 4 of 8 stage ranks → 4 idle.
        pc = ParallelConfig(tp_attention=8, tp_dense_ffn=4)
        self.assertEqual(mesh.group_size(pc, "attention"), 8)
        self.assertEqual(mesh.group_size(pc, "dense"), 4)
        errs = pc.validate()
        self.assertTrue(errs, "attn8×ffn4 must be rejected")
        self.assertTrue(any("dense FFN" in e and "per-stage" in e for e in errs),
                        errs)

    def test_realistic_cp_keeps_tp_dense_equal_tp_attn(self):
        # cp>1 with tp_dense == tp_attn is the real CP config and must be valid;
        # dense fills the stage via cp (token-shard), not by inflating tp_dense.
        ok = ParallelConfig(tp_attention=4, cp=2, tp_dense_ffn=4)
        self.assertEqual(ok.validate(), [], ok.validate())
        self.assertEqual(mesh.group_size(ok, "attention"),
                         mesh.group_size(ok, "dense"))
        # FFN weight divides by tp_dense_ffn ONLY — cp replicates it.
        self.assertEqual(_ffn_weight_degree(ok), 4)
        self.assertNotIn("d_ffn", dims.divided_by("cp"))

    def test_tp_dense_times_cp_is_oversharding(self):
        # tp_dense = tp_attn × cp asks dense for tp_attn×cp×cp ranks while the
        # stage only has tp_attn×cp → attention idles → rejected. (Was the
        # ONLY config validate used to accept; now correctly rejected.)
        bad = ParallelConfig(tp_attention=4, cp=2, tp_dense_ffn=8)
        self.assertTrue(bad.validate(), "tp_dense=tp_attn×cp must be rejected")

    def test_dp_x_cp_dense_valid_and_weight_unscaled_by_cp(self):
        # dp_attention>1 × cp>1 with tp_dense==tp_attn: was wrongly REJECTED
        # (forced tp_dense=tp_attn×cp → FFN weight ÷cp too much). Now valid and
        # the FFN weight divisor is tp_dense_ffn alone.
        pc = ParallelConfig(tp_attention=4, cp=2, dp_attention=2, tp_dense_ffn=4)
        self.assertEqual(pc.validate(), [], pc.validate())
        self.assertEqual(_ffn_weight_degree(pc), 4)

    def test_pure_cp_replicates_dense_weight(self):
        # tp_dense=1 under cp>1: dense spans the stage via cp (group == cp) but
        # the weight divisor is 1 — FFN weight replicated across cp ranks
        # (data-parallel FFN). cp must NOT divide the weight.
        pc = ParallelConfig(cp=8, tp_dense_ffn=1)
        self.assertEqual(pc.validate(), [])
        self.assertEqual(mesh.group_size(pc, "dense"), 8)    # spans stage via cp
        self.assertEqual(mesh.group_size(pc, "attention"), 8)
        self.assertEqual(_ffn_weight_degree(pc), 1)  # weight NOT sharded


if __name__ == "__main__":
    unittest.main()
