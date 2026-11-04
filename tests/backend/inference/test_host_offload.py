"""Weights kept in host memory leave the per-GPU weights.

A spec declares the weights a serving framework can keep in host memory
(`host_offload`, `Program.host_offloadable`): the n-gram tables vLLM's Engram
`cpu_offload` puts in pinned host memory, on by default
(`VLLM_PLE_CPU_OFFLOAD=1`). With `host_offload` the GPU's weights are the
device artifacts alone: smaller by exactly the tables, the cache room larger
by the same, and the model's whole weight unchanged. A model that declares
none does not move — DeepSeek-V4's hash-routing table is a lookup too, and
stays on the GPU.
"""
from __future__ import annotations

import unittest
from functools import lru_cache

from backend.arch.eval.program import Program
from backend.inference.capacity import CapacityKnobs, _capacity_preparation
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import (
    resolve_snapshot, shared_loader, snapshot_config,
)

PLE = "Qwen__Qwen3.8-Flash-Next"
DENSE = "Qwen__Qwen3-8B"
V41 = "deepseek-ai__DeepSeek-V4.1-Flash"
DSV4 = "deepseek-ai__DeepSeek-V4-Flash"


@lru_cache(maxsize=None)
def _prep(model_id: str, tp: int, on_host: bool, pp: int = 1):
    parallel = ParallelConfig(pp=pp, tp_attention=tp, tp_dense_ffn=tp,
                              tp_moe_expert=tp)
    return _capacity_preparation(
        config=snapshot_config(model_id), gpu_sku="H100-SXM",
        knobs=CapacityKnobs(
            resolved=resolve_snapshot(model_id, shared_loader()),
            selection_mode="theoretical_only", parallel=parallel,
            host_offload=on_host))


class TestLookupTablesOnHost(unittest.TestCase):

    def test_the_declared_tables_are_offloadable_and_nothing_else(self):
        expect = {
            PLE: {"ple.ple_embedding.ngram_embedding"},
            V41: {"engram.embed"},
            DSV4: set(),
        }
        for model_id, paths in expect.items():
            p = Program.of(resolve_snapshot(model_id, shared_loader()))
            got = {n.display_path for n in p.nodes()
                   if n.uid in p.host_offloadable}
            self.assertEqual(got, paths, model_id)

    def test_host_tables_leave_the_gpu_weights_exactly(self):
        for tp in (1, 8):
            on, off = _prep(PLE, tp, True), _prep(PLE, tp, False)
            table = off.host_offload_bytes[0]
            self.assertEqual(table, on.host_offload_bytes[0])
            # 320,001,536 x 160 BF16, split over TP by its rows.
            self.assertEqual(table, 102_400_491_520 // tp)
            self.assertEqual(off.weight_bytes_per_rank - on.weight_bytes_per_rank,
                             table)
            self.assertEqual(off.weight_bytes_total, on.weight_bytes_total)
            if off.fits:
                self.assertAlmostEqual(
                    on.usable_for_tokens - off.usable_for_tokens, table,
                    delta=1)

    def test_pipeline_stages_drop_only_their_own_tables(self):
        on, off = _prep(PLE, 1, True, pp=2), _prep(PLE, 1, False, pp=2)
        self.assertEqual(sum(on.host_offload_bytes), 102_400_491_520)
        for s_on, s_off, table in zip(on.pp_stages, off.pp_stages,
                                      on.host_offload_bytes, strict=True):
            self.assertEqual(s_off.weight_bytes - s_on.weight_bytes, table)

    def test_a_model_without_tables_does_not_move(self):
        on, off = _prep(DENSE, 1, True), _prep(DENSE, 1, False)
        self.assertEqual(on.host_offload_bytes, (0.0,))
        self.assertEqual(on.weight_bytes_per_rank, off.weight_bytes_per_rank)


class TestHostOffloadDeclaration(unittest.TestCase):
    """The declaration names a stored weight; on a node that stores none
    there is nothing to keep in host memory, and the walk refuses it."""

    def test_a_node_that_stores_no_weight_is_refused(self):
        from backend.arch.eval.refusal import SpecRefusal
        from backend.arch.eval.transforms import Deployment
        from tests.backend.arch.eval.test_hash_embedding_and_stored_split import (
            REPLICATED, _program, _spec,
        )
        stored = _program(_spec(REPLICATED).replace(
            "stored_split: {", 'host_offload: "x", stored_split: {'))
        Deployment(stored).at(batch=1, input_tokens=4).analyze()
        self.assertEqual(len(stored.host_offloadable), 1)
        bare = _program(_spec(REPLICATED, after=(
            '  - { name: act, op: silu, inputs: [ngram], host_offload: "x" }\n')))
        with self.assertRaisesRegex(SpecRefusal, "stores no weight"):
            Deployment(bare).at(batch=1, input_tokens=4).analyze()


if __name__ == "__main__":
    unittest.main()
