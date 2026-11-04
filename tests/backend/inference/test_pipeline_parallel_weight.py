"""Pipeline parallelism is a STAGE SELECTION, not a divisor.

`pp` is the one deployment axis that partitions no tensor: it slices the
model's layer index space, contiguously, into whole layers, and a rank holds
one slice. Dividing every leaf by `pp` was wrong twice, and both errors
under-reported what a rank holds:

  * it divides EXACTLY, so a 28-layer model at pp=8 reported 3.5 layers per
    stage when the heaviest stage holds 4 (12.5% low);
  * it divides the EMBEDDINGS, which live whole on one stage.

A stage holds what placement puts on it (`layout.stages.StageView.weights`):
its layers, the model-level steps placed there — the embedding with layer 0,
the norm and the head with the last layer — and the tables it reads through
`tied_weight`. An untied embedding and head are therefore on different
stages, and a tied one is resident where it is read.
"""
from __future__ import annotations

import math
import unittest
from functools import lru_cache

from backend.arch.layout.parallel_config import stage_layer_ranges
from backend.inference.capacity import CapacityKnobs, _capacity_preparation
from backend.inference.parallel_config import ParallelConfig
from backend.inference.weight_breakdown import per_rank_weight_breakdown
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

GiB = 1 << 30


@lru_cache(maxsize=None)
def _stages(model_id: str, pp: int, tp: int, aux: bool = False):
    """Every stage's resident weight per rank, GiB, as capacity weighs it."""
    parallel = ParallelConfig(pp=pp, tp_attention=tp, tp_dense_ffn=tp,
                              tp_moe_expert=tp)
    prep = _capacity_preparation(
        config=snapshot_config(model_id), gpu_sku="H200",
        knobs=CapacityKnobs(
            resolved=resolve_snapshot(model_id, shared_loader()),
            selection_mode="theoretical_only", parallel=parallel,
            include_aux_weights=aux))
    return prep


def _gib(prep):
    return [round(s.weight_bytes / GiB, 3) for s in prep.pp_stages]


class TestStagePartitionFollowsVLLM(unittest.TestCase):
    """`stage_layer_ranges` is the engine's canonical partition, and it
    coincides with vLLM's `get_pp_indices`: an even split, then the
    remainder handed out from the second-to-last stage backwards, because
    the first and last stages also hold the embedding and the head."""

    def test_stages_are_contiguous_and_cover_every_layer(self):
        for n_layers in (1, 3, 24, 28, 61, 78, 128):
            for pp in (1, 2, 3, 4, 5, 7, 8, 16):
                if pp > n_layers:
                    continue
                stages = stage_layer_ranges(n_layers, pp)
                with self.subTest(n_layers=n_layers, pp=pp):
                    self.assertEqual(len(stages), pp)
                    flat = [i for s in stages for i in s]
                    self.assertEqual(flat, list(range(n_layers)),
                                     "stages must tile 0..L-1 in order")

    def test_remainder_goes_to_the_inner_stages_first(self):
        self.assertEqual([len(s) for s in stage_layer_ranges(28, 8)],
                         [3, 3, 3, 4, 4, 4, 4, 3])
        self.assertEqual([len(s) for s in stage_layer_ranges(78, 4)],
                         [19, 20, 20, 19])
        self.assertEqual([len(s) for s in stage_layer_ranges(7, 4)],
                         [2, 2, 2, 1])

    def test_heaviest_stage_is_the_ceiling(self):
        for n_layers in (24, 28, 61, 78):
            for pp in (2, 3, 4, 5, 7, 8):
                with self.subTest(n_layers=n_layers, pp=pp):
                    self.assertEqual(
                        max(len(s) for s in stage_layer_ranges(n_layers, pp)),
                        math.ceil(n_layers / pp))


class TestPipelineIsInertAtOneStage(unittest.TestCase):
    """pp=1 reaches the same builder it always did."""

    def test_pp_one_matches_the_default_deployment(self):
        resolved = resolve_snapshot("meta-llama/Llama-3.2-3B",
                                    shared_loader())
        explicit = per_rank_weight_breakdown(resolved, ParallelConfig(pp=1))
        default = per_rank_weight_breakdown(resolved, ParallelConfig())
        self.assertEqual(explicit["total_bytes_per_rank"],
                         default["total_bytes_per_rank"])
        self.assertEqual(explicit["parallel_warnings"],
                         default["parallel_warnings"])
        self.assertIsNone(explicit["pipeline"])

    def test_a_pipelined_tree_names_its_stage(self):
        resolved = resolve_snapshot("meta-llama/Llama-3.2-3B",
                                    shared_loader())
        with self.assertRaises(ValueError):
            per_rank_weight_breakdown(resolved, ParallelConfig(pp=2))


class TestTheEndpointsAreOnTheirStages(unittest.TestCase):
    """The worked weights of the design (PP_LAYOUT_PROPOSAL §3 (a), (g),
    (h)), H200, BF16, GiB per rank."""

    def test_an_untied_embedding_and_head_sit_on_two_stages(self):
        """Qwen3-8B pp2 x tp4: the embedding is on stage 0 only and the head
        on stage 1 only. Counting both on both stages read 2.197."""
        self.assertEqual(_gib(_stages("Qwen/Qwen3-8B", 2, 4)),
                         [1.907, 1.907])

    def test_a_tied_table_is_resident_where_it_is_read(self):
        """Llama-3.2-3B pp2 x tp4: the head reads the embedding table on the
        last stage, so the table is resident on both."""
        self.assertEqual(_gib(_stages("meta-llama/Llama-3.2-3B", 2, 4)),
                         [0.840, 0.840])

    def test_an_uneven_split_with_a_head_stack(self):
        """DeepSeek-V3 pp4 x tp2 (15 | 15 | 16 | 15 layers). The MTP head is
        weighed only when `include_aux_weights` asks, and then only on stage
        3, where it runs; the weight-heaviest stage moves from 2 to 3."""
        default = _stages("deepseek-ai/DeepSeek-V3", 4, 2)
        self.assertEqual(_gib(default), [66.133, 80.542, 85.911, 81.405])
        aux = _stages("deepseek-ai/DeepSeek-V3", 4, 2, aux=True)
        self.assertEqual(_gib(aux), [66.133, 80.542, 85.911, 88.692])
        # The cache is the backbone's either way.
        self.assertEqual([s.per_token for s in default.pp_stages],
                         [s.per_token for s in aux.pp_stages])


class TestTheTreeDescribesTheBoundStage(unittest.TestCase):
    """The weight tree is built for the stage capacity binds, from the same
    projection, so the tree and the number beside it describe one rank."""

    def test_the_tree_total_is_the_bound_stages_weight(self):
        prep = _stages("deepseek-ai/DeepSeek-V3", 4, 2)
        self.assertEqual(prep.wb["pipeline"]["stage"], prep.pp_bound.stage)
        self.assertAlmostEqual(prep.wb["total_bytes_per_rank"],
                               prep.pp_bound.weight_bytes,
                               delta=prep.pp_bound.weight_bytes * 1e-9)

    def test_a_template_off_the_stage_reports_zero_and_says_why(self):
        """V3's 3 dense layers are 0-2: a stage that holds none reports 0 B
        for that template."""
        resolved = resolve_snapshot("deepseek-ai/DeepSeek-V3",
                                    shared_loader())
        wb = per_rank_weight_breakdown(
            resolved, ParallelConfig(pp=8, ep_moe=8), pp_stage=3)
        self.assertEqual(wb["groups"][0]["bytes_per_rank"], 0.0)
        self.assertGreater(wb["groups"][1]["bytes_per_rank"], 0.0)
        self.assertTrue(any("BINDING" in w and "stage 3" in w
                            for w in wb["parallel_warnings"]))


class TestPipelineNeverUnderReports(unittest.TestCase):
    """The heaviest stage holds at least an even share, and adding stages
    never asks the heaviest rank to hold more."""

    def test_at_least_an_even_share_and_monotone_in_pp(self):
        for model_id in ("meta-llama/Llama-3.2-3B", "zai-org/GLM-5.2"):
            whole = _stages(model_id, 2, 1).pp_stages
            total = sum(s.weight_bytes for s in whole)
            previous = max(s.weight_bytes for s in whole)
            for pp in (4, 8):
                heaviest = max(s.weight_bytes
                               for s in _stages(model_id, pp, 1).pp_stages)
                with self.subTest(model_id=model_id, pp=pp):
                    self.assertGreaterEqual(heaviest, total / pp - 1.0)
                    self.assertLessEqual(heaviest, previous + 1.0)
                previous = heaviest


if __name__ == "__main__":
    unittest.main()
