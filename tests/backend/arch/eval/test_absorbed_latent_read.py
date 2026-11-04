"""Absorbed MLA / DSA read the head-shared latent once per position, not once per head.

The domain fact: FlashMLA and its sparse variant run absorbed attention as MQA
over one KV head — vLLM passes `num_heads_k=1` and a `(..., 1, 576)` cache
(`v1/attention/backends/mla/flashmla.py`, `flashmla_sparse.py`), and a CTA
loads each latent row once for its whole group of query heads. So one decode
step reads `positions x (d_kv_lora + d_qk_rope)` elements per request, where
`positions` is the context (dense MLA) or the top-k (DSA), whatever `n_heads`
is. The spec spells the sharing as a `head_broadcast`, whose fan-out traffic
divides back out.

Falsification targets:
  - the fused view of an isolated absorbed component (the Playground's module
    bench) must read no more than its naive view, and its attention kernel
    exactly one latent row per position plus the query. Before the fix a
    `fused_qk_rope` that absorbed `k_pe_rope` left the broadcast's producer
    without a shape row, the fan-out fell back to 1, and the kernel's K edge
    was priced `n_heads` times over (305 MB against 7 MB for V3.2);
  - a fusion changes the bytes of the nodes it rewrote and of no other node:
    the non-absorbed prefill concatenates per-head K with the broadcast rope
    half, and that concat read the rope half once per head only in the fused
    graph;
  - in the full model (the serving walk) the attention kernel's read is exactly
    the latent once per position plus the query (plus the int32 top-k indices
    for DSA).
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.fusion import Framework
from backend.arch.eval.launch import FusedNode
from backend.arch.eval.program import Program
from backend.arch.eval.traffic import derive
from backend.arch.eval.transforms import (
    DECODE, Deployment, isolate_resolved,
)
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import backbone_call, resolve_snapshot, shared_loader

#: (model, the absorbed-at-decode attention component it binds)
CASES = (
    ("deepseek-ai/DeepSeek-V3", "mla_attention"),
    ("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention"),
    ("zai-org/GLM-5.1", "mla_dsa_attention"),
)
BATCH = 2
CACHE_LEN = 4096
#: BF16 activations (the model's own dtype); int32 top-k indices. The cached
#: latent row is read at the cache layer's own width (`_row_bytes`): V3.2's
#: is FP8 with its group scales beside a BF16 RoPE key.
WIDTH = 2
INDEX_WIDTH = 4
ATTENTION_KERNELS = ("flash_attention", "flash_dsa")

_LOADER = shared_loader()
_RESOLVED = {}


def _resolved(model_id: str):
    if model_id not in _RESOLVED:
        _RESOLVED[model_id] = resolve_snapshot(model_id, _LOADER)
    return _RESOLVED[model_id]


def _isolated(resolved, component):
    return isolate_resolved(resolved, component, None,
                            backbone_call(resolved, component))


def _framework() -> Framework:
    return Framework.of(get_binding("vllm"), FUSION_SCHEMES)


def _decode(resolved) -> Deployment:
    return (Deployment(Program.of(resolved))
            .at(batch=BATCH, input_tokens=CACHE_LEN)
            .during(DECODE, cache_len=CACHE_LEN))


def _row_bytes(analysis) -> int:
    """Bytes of one cached position of the absorbed attention: the latent
    and the RoPE key, each at its own stored format with the dynamic scales
    it stores beside the data (`CachedTensor.position_bytes`) — the row a
    kernel reads (V3.2: 512 FP8 + 4 F32 group scales + 64 BF16 = 656 B,
    vLLM `flashmla_sparse.py`'s fp8_ds_mla layout)."""
    rows = {sum(float(t.position_bytes)
                for t in entry.tensors
                if t.kind in ("latent", "rope_key"))
            for entry in cache_mod.derive(analysis).values.values()}
    rows.discard(0.0)
    (row,) = rows
    return int(row)


def _geometry(resolved):
    """`(n_heads, latent row width, positions a step attends, bytes of the
    top-k indices a sparse step also reads)` per request."""
    dims = resolved.dim_values
    row = int(dims["d_kv_lora"]) + int(dims["d_qk_rope"])
    topk = dims.get("n_topk_keys")
    if topk is None:
        return int(dims["n_heads"]), row, CACHE_LEN, 0
    positions = min(CACHE_LEN, int(topk))
    return int(dims["n_heads"]), row, positions, positions * INDEX_WIDTH


def _kernel_reads(analysis):
    """Bytes each attention kernel reads, activations plus cache."""
    leaves = derive(analysis).leaves().values
    return [int(t.act_reads) + int(t.kv_reads)
            for uid, t in leaves.items()
            if analysis.program.by_uid(uid).op in ATTENTION_KERNELS]


class TestIsolatedFusedLatentReadOnce(unittest.TestCase):
    """The Playground's view: one component, naive and fused, at decode."""

    def test_fused_reads_no_more_than_naive(self) -> None:
        for model_id, component in CASES:
            with self.subTest(model=model_id):
                deployment = _decode(
                    _isolated(_resolved(model_id), component))
                naive = derive(deployment.analyze()).leaves().values
                fused = derive(deployment.fused(_framework()).analyze()
                               ).leaves().values
                self.assertLessEqual(
                    sum(int(t.act_reads) for t in fused.values()),
                    sum(int(t.act_reads) for t in naive.values()))

    def test_kernel_reads_the_latent_once_per_position(self) -> None:
        for model_id, component in CASES:
            with self.subTest(model=model_id):
                resolved = _resolved(model_id)
                n_heads, row, positions, indices = _geometry(resolved)
                deployment = _decode(_isolated(resolved, component))
                analysis = deployment.fused(_framework()).analyze()
                (read,) = _kernel_reads(analysis)
                # Exactly the serving read (`TestServingLatentReadExact`):
                # isolation calls the component as a model block does, so its
                # cache is declared and the value half (`d_kv_lora`) is read
                # inside the one latent read, not again as an edge of its own.
                self.assertEqual(read, BATCH * (positions
                                                * _row_bytes(analysis)
                                                + n_heads * row * WIDTH
                                                + indices))


class TestFusionMovesOnlyWhatItRewrites(unittest.TestCase):
    """Every node a fusion left alone bills the same bytes in both graphs,
    except the read of an operand a fused launch now writes already
    quantised (the conversion it absorbed, `traffic._settle_conversions`)."""

    def test_untouched_nodes_keep_their_traffic(self) -> None:
        for model_id, component in CASES:
            resolved = _resolved(model_id)
            for scope, target in (("model", resolved),
                                  ("isolated",
                                   _isolated(resolved, component))):
                for phase in ("prefill", DECODE):
                    with self.subTest(model=model_id, scope=scope,
                                      phase=phase):
                        deployment = (Deployment(Program.of(target))
                                      .at(batch=BATCH, input_tokens=1024))
                        if phase == DECODE:
                            deployment = deployment.during(
                                DECODE, cache_len=1024)
                        naive_an = deployment.analyze()
                        fused_an = deployment.fused(_framework()).analyze()
                        naive = derive(naive_an).leaves().values
                        fused = derive(fused_an).leaves().values
                        for node in fused_an.program.nodes():
                            if (isinstance(node, FusedNode) or node.children
                                    or node.uid not in naive
                                    or node.uid not in fused):
                                continue
                            before, after = naive[node.uid], fused[node.uid]
                            if (int(before.convert_bytes)
                                    and not int(after.convert_bytes)):
                                # A launch that writes its output already
                                # quantised (the cast a member) absorbed this
                                # node's input conversion: it reads the narrow
                                # tensor and pays no conversion. Only that
                                # read moves, and only downward.
                                self.assertLess(int(after.act_reads),
                                                int(before.act_reads),
                                                node.name)
                                before = replace(before,
                                                 act_reads=after.act_reads)
                            self.assertEqual(
                                (int(before.act_reads),
                                 int(before.act_writes),
                                 int(before.kv_reads)),
                                (int(after.act_reads),
                                 int(after.act_writes),
                                 int(after.kv_reads)),
                                node.name)


class TestServingLatentReadExact(unittest.TestCase):
    """The full model, as serving walks it: the latent once, exactly."""

    def test_each_kernel_reads_latent_query_and_indices(self) -> None:
        for model_id, _component in CASES:
            with self.subTest(model=model_id):
                resolved = _resolved(model_id)
                n_heads, row, positions, indices = _geometry(resolved)
                analysis = _decode(resolved).fused(_framework()).analyze()
                reads = _kernel_reads(analysis)
                self.assertTrue(reads)
                expected = BATCH * (positions * _row_bytes(analysis)
                                    + n_heads * row * WIDTH + indices)
                for read in reads:
                    self.assertEqual(read, expected)


class TestEachKernelPaysItsOwnCacheRead(unittest.TestCase):
    """Unfused, absorbed MLA's score and apply are two kernels, and each reads
    the latent from HBM: two reads, each on its own row, nothing split
    between them. The framework's attention kernel scores and applies in one
    launch and reads it once. The difference is the read fusion saves."""

    def test_two_unfused_readers_and_one_fused_reader(self) -> None:
        from backend.arch.eval.fusion import rebase_analysis
        from backend.arch.eval.traffic import _kv_reads

        analysis = _decode(_resolved("deepseek-ai/DeepSeek-V3")).analyze()
        laws = {}
        for uid, law in _kv_reads(analysis).items():
            laws.setdefault(analysis.program.by_uid(uid).name, set()).add(
                int(law))
        (latent,) = laws["kv_a_layernorm"]
        (positional,) = laws["k_pe_rope"]

        def by_op(walk):
            reads = {}
            for uid, t in derive(walk).leaves().values.items():
                if int(t.kv_reads):
                    reads.setdefault(walk.program.by_uid(uid).op,
                                     set()).add(int(t.kv_reads))
            return reads

        unfused = by_op(analysis)
        self.assertEqual(unfused["attention_apply"], {latent})
        self.assertEqual(unfused["attention_score"], {latent + positional})

        fused, rewrite = _framework().rewrite(analysis.program, analysis)
        self.assertEqual(by_op(rebase_analysis(analysis, fused, rewrite)),
                         {"flash_attention": {latent + positional}})


if __name__ == "__main__":
    unittest.main()
