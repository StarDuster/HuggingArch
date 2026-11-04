"""Direct regression net for the three mesh functions layout groups
replaced — `alias_map`, `contains`, `subgroup_stride` (and its companion
`subgroup_degrees`). Until now they were covered only through
`test_parallelism_comm` and golden runs; comparing against the current
layout groups needs their CURRENT answers pinned directly, else a
divergence cannot say which side moved.

Every expected value below was captured from the implementation at the
commit that introduces this file. These tests assert what the functions
DO, deliberately including the documented soft spots (the dp×cp×ep
three-class split, the stride-interleave containment miss) — those are
the behaviors ProcessGroupFamily relations must reproduce or knowingly
change, and a knowing change starts by failing here."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import mesh
from backend.inference.parallel_config import ParallelConfig


def _degrees(**kw) -> dict:
    return mesh.axis_degrees(ParallelConfig(**kw))


class TestAliasMap(unittest.TestCase):
    def test_three_tp_worlds_at_equal_degree_are_one_group(self):
        # Stride 1 in each region, same size: one alias class. This is the
        # fact that lets sequence parallelism meet attention and FFN without
        # a spurious re-shard per block.
        alias = mesh.alias_map(_degrees(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8))
        self.assertEqual(alias["tp_dense_ffn"], alias["tp_attention"])
        self.assertEqual(alias["tp_moe_expert"], alias["tp_attention"])

    def test_tp_worlds_at_different_degrees_do_not_alias(self):
        alias = mesh.alias_map(_degrees(tp_attention=8, tp_dense_ffn=4))
        self.assertNotEqual(alias["tp_attention"], alias["tp_dense_ffn"])

    def test_moe_seam_aliases_request_cut(self):
        # dp_attention and ep_moe over the same eight ranks: same span,
        # one class — the seam is an exchange, not a pair of gathers.
        alias = mesh.alias_map(_degrees(dp_attention=8, ep_moe=8))
        self.assertEqual(alias["ep_moe"], alias["dp_attention"])

    def test_moe_seam_aliases_position_cut(self):
        # The REGION_AXES["moe"] no-cp comment, as behavior.
        alias = mesh.alias_map(_degrees(cp=8, ep_moe=8))
        self.assertEqual(alias["ep_moe"], alias["cp"])

    def test_contained_cut_is_not_an_alias(self):
        # cp spans (2, 4) inside ep_moe's (1, 8): containment, not aliasing.
        degrees = _degrees(tp_attention=2, cp=4, ep_moe=8)
        alias = mesh.alias_map(degrees)
        self.assertNotEqual(alias["cp"], alias["ep_moe"])
        self.assertTrue(mesh.contains(degrees, "ep_moe", "cp"))

    def test_same_size_different_stride_does_not_alias(self):
        # Both spans have size 2, strides 4 vs 1: stride AND size must
        # match — the docstring's own condition, pinned directly.
        alias = mesh.alias_map(_degrees(tp_attention=4, cp=2, ep_moe=2))
        self.assertNotEqual(alias["cp"], alias["ep_moe"])

    def test_singleton_classes_map_to_themselves(self):
        # The docstring says un-aliased worlds are "absent"; the
        # implementation returns them as their own representative. Pinned
        # as-is: consumers rely on lookups never KeyError-ing.
        alias = mesh.alias_map(_degrees(tp_attention=8, tp_dense_ffn=4))
        self.assertEqual(alias["tp_dense_ffn"], "tp_dense_ffn")

    def test_dp_cp_ep_falls_in_three_classes(self):
        # The documented DSv3-dpcpep soft spot: the two-axis token cut and
        # the expert cut land in three alias classes. v3's owner-map
        # comparison is the fix; until it ships, THIS is the behavior.
        alias = mesh.alias_map(_degrees(
            tp_attention=1, cp=2, dp_attention=2, ep_moe=4))
        self.assertEqual(
            len({alias["cp"], alias["dp_attention"], alias["ep_moe"]}), 3)


class TestContains(unittest.TestCase):
    def test_ep_contains_both_token_cuts_in_dpcpep(self):
        degrees = _degrees(tp_attention=1, cp=2, dp_attention=2, ep_moe=4)
        self.assertTrue(mesh.contains(degrees, "ep_moe", "cp"))
        self.assertTrue(mesh.contains(degrees, "ep_moe", "dp_attention"))
        self.assertFalse(mesh.contains(degrees, "dp_attention", "ep_moe"))

    def test_aliased_worlds_contain_each_other(self):
        degrees = _degrees(dp_attention=8, ep_moe=8)
        self.assertTrue(mesh.contains(degrees, "ep_moe", "dp_attention"))
        self.assertTrue(mesh.contains(degrees, "dp_attention", "ep_moe"))

    def test_stride_interleave_is_a_documented_miss(self):
        # tp_moe_expert=4 puts ep_moe at stride 4 while dp_attention sits at
        # stride 1: the exchange is real but containment answers no — the
        # docstring calls this out. Pinned so the v3 owner-inclusion rule's
        # deliberate divergence here is visible as a failing assertion, not
        # a silent drift.
        degrees = _degrees(tp_moe_expert=4, ep_moe=2, dp_attention=2)
        self.assertFalse(mesh.contains(degrees, "ep_moe", "dp_attention"))

    def test_axis_at_degree_one_is_never_contained(self):
        degrees = _degrees(dp_attention=8, ep_moe=8)
        self.assertFalse(mesh.contains(degrees, "ep_moe", "cp"))


class TestSubgroupStrideAndDegrees(unittest.TestCase):
    def test_dcp_inside_tp_reuses_tp_ranks_at_stride_one(self):
        # No pcp: dcp partitions tp_attention's ranks directly.
        pc = ParallelConfig(tp_attention=8, dcp=8)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 1)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 4, 8))

    def test_dcp_fills_cp_before_tp(self):
        # dcp == pcp: the group is one PCP column, spaced by the tp width
        # below it — vLLM's transpose-then-reshape, as a stride.
        pc = ParallelConfig(tp_attention=4, cp=2, dcp=2)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 4)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 8))

    def test_dcp_swallowing_cp_spills_into_tp(self):
        # dcp == tp*pcp: the group swallows the PCP column whole and runs
        # contiguous through TP.
        pc = ParallelConfig(tp_attention=4, cp=2, dcp=8)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 1)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
