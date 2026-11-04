"""Evaluator storage ledger laws.

Per-rank weights come from `local_tensors()` on the public layout walk.
What is pinned is the vocabulary — see `test_the_vocabulary_is_sharded`.
"""
from __future__ import annotations

import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import capacity as cap
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_loader, snapshot_header
from backend.arch.ops_base import OP_REGISTRY
from backend.inference.parallel_config import ParallelConfig

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"

@lru_cache(maxsize=None)
def _resolved(name: str):
    config = json.loads((SNAPSHOTS / f"{name}.json").read_text())["config"]
    return shared_loader().match(config, model_id=name.replace("__", "/", 1))


@lru_cache(maxsize=None)
def _analysis(name: str, tp: int = 1):
    config = None
    if tp > 1:
        config = ParallelConfig(
            tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp)
    return Deployment(
        Program.of(_resolved(name)),
        config=config,
    ).at(batch=1, input_tokens=128).analyze()


def _names():
    return sorted(p.stem for p in SNAPSHOTS.glob("*.json"))


class TestWeights(unittest.TestCase):
    def test_buffers_are_resident_and_are_not_parameters(self):
        """`rotary_emb.inv_freq` and its kind are stored, loaded and on the
        device, and no parameter count includes them. The incumbent reports
        parameters as the per-rank weight bytes and is short by exactly these:
        8,192 on Llama-2-7B, 6,832 on Kimi-K2."""
        for name, expected in (("meta-llama__Llama-2-7b-hf", 8192),
                               ("moonshotai__Kimi-K2-Base", 6832),
                               ("THUDM__glm-4-9b", 64)):
            with self.subTest(model=name):
                weights = cap.weights(_analysis(name))
                self.assertEqual(float(weights.buffers), float(expected))
                self.assertEqual(
                    weights.resident,
                    weights.parameters + weights.metadata + weights.buffers,
                )

    def test_an_fp32_checkpoint_is_saved_at_four_and_served_at_two(self):
        """The file and the rank are two answers. Gemma-2's header is F32,
        but a default deployment loads wider-than-16-bit floats at BF16
        (`ResolvedPrecision.model_dtype`): the checkpoint byte gate still
        sees four bytes per element, while the resident weight ledger sees
        the two bytes the server holds."""
        for name in ("google__gemma-2-9b", "google__gemma-2-27b"):
            with self.subTest(model=name):
                header = snapshot_header(name)
                self.assertEqual({g.dtype for g in header}, {"F32"})
                elements = sum(g.elements * g.count for g in header)
                analysis = _analysis(name)
                self.assertEqual(analysis.checkpoint_param_bytes(),
                                 elements * 4.0)
                weights = cap.weights(_analysis(name))
                mine = float(weights.parameters + weights.metadata)
                self.assertEqual(mine, elements * 2.0)

    def test_parameters_and_metadata_are_reported_apart(self):
        """A checkpoint's parameter total excludes quantisation scales and its
        file size includes them, so a single number cannot be checked against
        both."""
        weights = cap.weights(_analysis("deepseek-ai__DeepSeek-V3"))
        self.assertGreater(weights.metadata, 0, "V3 is FP8 with block scales")
        self.assertEqual(weights.resident,
                         weights.parameters + weights.metadata)

    def test_the_per_node_ledger_adds_up_to_the_whole(self):
        """`by_uid` splits the same three kinds by the node that stores them,
        so its sum is the ledger, kind by kind — on one device and per rank.
        A scale sits with the weight it scales, and a buffer with the node
        that declares it."""
        for name, tp in (("meta-llama__Llama-2-7b-hf", 1),
                         ("deepseek-ai__DeepSeek-V3", 1),
                         ("deepseek-ai__DeepSeek-V3", 4)):
            with self.subTest(model=name, tp=tp):
                analysis = _analysis(name, tp)
                weights = cap.weights(analysis)
                nodes = weights.by_uid.values()
                self.assertEqual(
                    (sum(w.parameters for w in nodes),
                     sum(w.metadata for w in nodes),
                     sum(w.buffers for w in nodes)),
                    (weights.parameters, weights.metadata, weights.buffers))
                self.assertEqual(sum(w.resident for w in nodes),
                                 weights.resident)
                for uid, w in weights.by_uid.items():
                    # Nothing is scaled without a weight, and a node's own
                    # buffers are what its op declares.
                    self.assertFalse(w.metadata and not w.parameters, uid)
                    op = analysis.program.by_uid(uid).op
                    self.assertEqual(bool(w.buffers),
                                     op in OP_REGISTRY and OP_REGISTRY[op].is_buffer,
                                     uid)

    def test_aux_filter_rejects_an_unowned_stack_bucket(self):
        analysis = _analysis("meta-llama__Llama-3.2-1B")
        ledger = cap.Weights(by_stack={"not-a-program-stack": 1})
        with self.assertRaisesRegex(ValueError, "undeclared stack"):
            ledger.without_aux(analysis.program)


class TestPerRank(unittest.TestCase):
    def test_tensor_parallelism_shrinks_what_it_shards(self):
        whole = float(cap.weights(_analysis("meta-llama__Llama-3.2-1B")).resident)
        rank = float(cap.weights(
            _analysis("meta-llama__Llama-3.2-1B", tp=4)).resident)
        self.assertLess(rank, whole)
        # More than a quarter, because the norms have no axis to split and are
        # whole on every rank.
        self.assertGreater(rank, whole / 4)

    def test_logical_tensors_stay_global_under_tp(self):
        analysis = _analysis("meta-llama__Llama-3.2-1B", 4)
        logical = {
            row.tensor.name: row.tensor.elements for row in analysis.tensors()
        }
        local = {
            row.tensor.name: row.tensor.elements
            for row in analysis.local_tensors()
        }
        self.assertEqual(logical["model.embed_tokens.weight"], 128256 * 2048)
        self.assertEqual(local["model.embed_tokens.weight"], 128256 * 2048 // 4)

    def test_the_vocabulary_is_sharded(self):
        """Both the embedding and the LM head split along it.

        vLLM's `VocabParallelEmbedding` divides `num_embeddings_padded` by the
        tensor-parallel size, and `ParallelLMHead` is a SUBCLASS of it — the head
        is the same split with a logits gather on the end. 139 of vLLM's model
        files use the first and 134 the second; sglang mirrors both. Leaving the
        table whole reported an embedding replicated on every rank that no
        framework replicates, which on Llama-3.2-1B is 525 MB of a 2.4 GB model.
        """
        # Two models because one cannot show both: Llama-3.2-1B ties its head to
        # the embedding, so it stores no `lm_head.weight` at all.
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                tied = {
                    row.tensor.name: row.tensor.elements
                    for row
                    in _analysis("meta-llama__Llama-3.2-1B", tp).local_tensors()
                }
                self.assertEqual(tied["model.embed_tokens.weight"],
                                 128256 * 2048 // tp)
                self.assertNotIn("lm_head.weight", tied)

                untied = {
                    row.tensor.name: row.tensor.elements
                    for row
                    in _analysis("meta-llama__Llama-2-7b-hf", tp).local_tensors()
                }
                self.assertEqual(untied["lm_head.weight"],
                                 untied["model.embed_tokens.weight"])
                self.assertEqual(untied["model.embed_tokens.weight"],
                                 32000 * 4096 // tp)


if __name__ == "__main__":
    unittest.main()
