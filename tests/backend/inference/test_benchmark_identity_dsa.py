"""A sparse-MLA cell attends the rows the walk bills, read off the cache operand.

The walk sizes the attention kernel's cache operand (the score's K, input 1)
at the keys one query attends — `Min(s_kv, k)`, `k` the score op's own top-k
budget — and bills the launch for those rows. The cell used to read its
`s_kv` off the SELECTION operand instead, and that width is the indexer's
selection, not the keys attended: GLM-5.3-Flash's k-pool indexer selects
`n_topk_keys / compress_ratio` pools and expands each back to
`compress_ratio` keys plus the open tail (vLLM `expand_pools_and_append_tail`,
`topk + pool_size - 1` indices a query). Keyed at 288 pools over a cache of
1152 keys, every provider built and gathered 288 rows while the row billed
1152 — RTX-4090 rows at MBU 1.0-1.1, H100 rows at 4x the GLM-5.x level.

The same shape search took Hy4's stored latent `[B, s_kv, 576]` for the
selection whenever `s_kv == s_query` (prefill), keying `s_kv = 576`, its
head width. Operands are now selected by the signature's declared engine
layouts (position and rank), the same declaration the provider's canonical
inputs are built from (`kernel_bench.refs._carried_axes`).
"""
from __future__ import annotations


import json
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.benchmark_identity import _dsa_identity
from backend.inference.op_signatures import get_signature

SPARSE = get_signature("mla_sparse_attention")
SINK = get_signature("mla_sparse_sink_attention")
#: DSv3.2 / GLM-5.x latent: 512 + 64.
SPEC = {"n_topk_keys": 2048, "d_kv_lora": 512, "d_qk_rope": 64}
#: GLM-5.3-Flash: NoPE latent (no rope lane), k-pool of 4.
KPOOL_SPEC = {"n_topk_keys": 2048, "d_kv_lora": 512}

#: DSv3.2 decode over a 4096-key cache, as the row keys it: Q, the latent at
#: the attended rows as STORED (no head axis — the walk marks its broadcast
#: headless and `stored_operands` drops it), the top-k list, the value (the
#: latent's leading 512 columns, stored the same way).
V32_DECODE = ([[2, 128, 1, 576], [2, 2048, 576], [2, 1, 2048],
               [2, 2048, 512]], [2, 128, 1, 512])
#: GLM-5.3-Flash decode over a 1152-key cache: 288 POOLS selected, every
#: one of the 1152 keys attended (the stored latent's rows).
KPOOL_DECODE = ([[2, 4, 1, 512], [2, 1152, 512], [2, 1, 288]],
                [2, 4, 1, 512])
#: GLM-5.3-Flash prefill of 4096 tokens, as the walk publishes it (the
#: latent in its stored form): 512 pools selected, 2051 keys attended.
KPOOL_PREFILL = ([[2, 4, 4096, 512], [2, 2051, 512], [2, 4096, 512]],
                 [2, 4, 4096, 512])
#: Hy4 prefill of 1024 tokens: the stored latent with no head axis, the
#: top-k list, the sink, the value.
HY4_PREFILL = ([[1, 1, 1024, 576], [1, 1024, 576], [1, 1024, 1024], [1],
                [1, 1, 1024, 512]], [1, 1, 1024, 512])


def _cell(sig, operands, spec):
    ident = _dsa_identity(sig, *operands, spec)
    return None if ident is None else json.loads(ident)


class TestAttendedKeysAreTheCacheOperandRows(unittest.TestCase):

    def test_kpool_cell_attends_the_keys_not_the_pools(self):
        dims = _cell(SPARSE, KPOOL_DECODE, KPOOL_SPEC)
        self.assertIsNotNone(dims)
        self.assertEqual((dims["s_kv"], dims["n_topk_keys"]), (1152, 1152))

    def test_kpool_prefill_attends_the_budget_plus_the_open_tail(self):
        # Past the budget a query attends n_topk_keys + compress_ratio - 1 =
        # 2051 keys (vLLM `expand_pools_and_append_tail`), more than the
        # spec's 2048: the cache rows are the count, not the spec clamped to
        # them (`min(n_topk_keys, rows)` keyed 2048 and dropped the tail).
        dims = _cell(SPARSE, KPOOL_PREFILL, KPOOL_SPEC)
        self.assertIsNotNone(dims)
        self.assertEqual((dims["s_query"], dims["s_kv"], dims["n_topk_keys"]),
                         (4096, 2051, 2051))

    def test_plain_topk_cell_is_unchanged(self):
        dims = _cell(SPARSE, V32_DECODE, SPEC)
        self.assertEqual((dims["B"], dims["n_heads"], dims["s_query"],
                          dims["d_head"], dims["d_v_head"]),
                         (2, 128, 1, 576, 512))
        self.assertEqual((dims["s_kv"], dims["n_topk_keys"]), (2048, 2048))

    def test_stored_latent_is_the_cache_even_when_it_is_query_long(self):
        dims = _cell(SINK, HY4_PREFILL, SPEC)
        self.assertIsNotNone(dims)
        self.assertEqual((dims["s_kv"], dims["n_topk_keys"]), (1024, 1024))



class TestTheGatedSpanAttendsWhatItSelects(unittest.TestCase):
    """Qwen QSA's gated prefill over fewer cached keys than the indexer's
    budget: 1024 keys under n_topk_keys = 2048. The cell keyed the budget,
    its indices `[B, s_query, 2048]` outgrew the 1024 rows any build can
    select, and the H100 run skipped every such job ("built cell disagrees
    with the row")."""

    def test_the_selection_is_the_keys_cached(self):
        gated = [[16, 3, 1024, 256], [16, 1, 1024, 256], [16, 1024, 256],
                 [16, 1, 1024, 256], [16, 3, 1024, 256]]
        dims = _cell(get_signature("gqa_sparse_gated_attention"),
                     (gated, [16, 1024, 768]), {"n_topk_keys": 2048})
        self.assertEqual(dims, {"B": 16, "n_heads": 3, "n_kv_heads": 1,
                                "s_query": 1024, "d_head": 256,
                                "s_kv": 1024, "n_topk_keys": 1024})


class TestTopKIsDeclared(unittest.TestCase):
    """A sparse kernel without a declared top-k is not identifiable as one:
    substituting the candidate length matched a sparse measurement against
    dense-attention work."""

    def test_missing_or_zero_topk_is_not_this_kernel(self):
        latent = {"d_kv_lora": 512, "d_qk_rope": 64}
        for spec in (latent, {**latent, "n_topk_keys": 0}):
            self.assertIsNone(_cell(SPARSE, V32_DECODE, spec))
        self.assertIsNone(_dsa_identity(SPARSE, *V32_DECODE, None))


class TestTheLatentAndThePerHeadGeometries(unittest.TestCase):
    """The latent cell reads one shared row a key; a span carrying K and V
    per KV head is the per-head geometry (`n_kv_heads`, K at input 1, V at
    input 3) — top-k GQA. GLM-5 prefill's per-head K/V was once timed as a
    one-latent gather and billed for 2·n_heads copies (MBU 1.9); Qwen QSA
    and Step-5, with no latent at all, were refused outright and left their
    attention unmeasured."""

    def test_per_head_span_keys_the_gqa_geometry(self):
        # Qwen QSA decode at one KV head a rank: Q, K, pools, a separate V.
        qsa = [[2, 3, 1, 256], [2, 1, 1152, 256], [2, 1, 288],
               [2, 1, 1152, 256]]
        dims = _cell(SPARSE, (qsa, [2, 3, 1, 256]), {"n_topk_keys": 2048})
        self.assertEqual(dims, {"B": 2, "n_heads": 3, "n_kv_heads": 1,
                                "s_query": 1, "d_head": 256, "d_v_head": 256,
                                "s_kv": 1152, "n_topk_keys": 1152})

    def test_the_stored_latent_keys_no_kv_heads(self):
        self.assertNotIn("n_kv_heads", _cell(SPARSE, V32_DECODE, SPEC))

    def test_a_per_head_store_at_the_latent_width_is_per_head(self):
        # How the cache is STORED decides, not what widths the spec declares:
        # a K kept per KV head whose width happens to be the latent's (512
        # with no rope lane) is top-k GQA. At one query head a rank its K
        # and the broadcast latent have the same shape; choosing by
        # `d_kv_lora` keyed it as the latent — the class of the stale Qwen
        # QSA cells that reached FlashMLA ("Invalid d_qk: 256").
        q = [2, 1, 1, 512]
        dims = _cell(SPARSE, ([q, [2, 1, 1152, 512], [2, 1, 288],
                               [2, 1, 1152, 512]], q), KPOOL_SPEC)
        self.assertEqual((dims["n_kv_heads"], dims["d_head"],
                          dims["d_v_head"]), (1, 512, 512))

    def test_per_head_width_under_a_latent_is_per_head(self):
        q, out = [2, 64, 4096, 192], [2, 64, 4096, 128]
        dims = _cell(SPARSE, ([q, [2, 64, 2048, 192], [2, 4096, 2048],
                               [2, 64, 2048, 128]], out), SPEC)
        self.assertEqual((dims["n_kv_heads"], dims["d_head"],
                          dims["d_v_head"]), (64, 192, 128))

    def test_a_per_head_span_without_its_value_is_no_cell(self):
        # The per-head geometry lays V out at input 3; an operand list that
        # stops short of it would otherwise bind on Q and K alone.
        q = [2, 3, 1, 256]
        self.assertIsNone(_dsa_identity(
            SPARSE, [q, [2, 1, 1152, 256], [2, 1, 288]], q,
            {"n_topk_keys": 2048}))

    def test_ungrouped_heads_are_no_cell(self):
        q = [2, 3, 1, 256]
        self.assertIsNone(_dsa_identity(
            SPARSE, [q, [2, 2, 1152, 256], [2, 1, 288], [2, 2, 1152, 256]],
            q, {"n_topk_keys": 2048}))


class TestTheWalkSaysHowTheCacheIsStored(unittest.TestCase):
    """End to end: the walk's rows keyed by `RowIdentity`. DSv3.2's latent
    reaches the kernel head-broadcast, and the walk's headless-store fact
    (`stored_operands`) is what keys it as the latent; Hy4's graph hands its
    sink kernel the stored latent; Qwen3.8 QSA's and Step-5's K are stored
    per KV head and key the per-head geometry."""

    def _sparse_cells(self, model_id, op="flash_dsa"):
        from backend.inference.benchmark_identity import RowIdentity
        from backend.inference.framework_bindings import (
            DEFAULT_FRAMEWORK, get_binding)
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        from tests.snapshots import resolve_snapshot, shared_loader
        result = analyze(resolve_snapshot(model_id, shared_loader()),
                         parallel=ParallelConfig(), avg_input_tokens=64,
                         avg_output_tokens=0, phase="decode",
                         framework_binding=get_binding(DEFAULT_FRAMEWORK),
                         breakdown=True)
        identity = RowIdentity(result, kv_block_size=64)
        cells = []
        for row in result.per_op_timing:
            got, shape_json = identity.of(row)
            if got == op and shape_json:
                cells.append(json.loads(shape_json))
        self.assertTrue(cells, model_id)
        return cells

    def test_the_broadcast_latent_keys_the_latent_cell(self):
        for dims in self._sparse_cells("deepseek-ai/DeepSeek-V3.2"):
            self.assertNotIn("n_kv_heads", dims)
            self.assertEqual((dims["d_head"], dims["d_v_head"]), (576, 512))

    # Per model walks: with the regression set (the per-head choice itself is
    # pinned by the operand-level tests above).
    @pytest.mark.regression
    def test_the_per_head_store_keys_the_per_head_cell(self):
        for dims in self._sparse_cells("Qwen/Qwen3.8-Flash-Next"):
            self.assertEqual(dims["n_heads"] % dims["n_kv_heads"], 0, dims)

    @pytest.mark.regression
    def test_step5s_per_kv_head_cache_keys_the_per_head_cell(self):
        # Step-5's region-sparse attention: K and V per KV head (4 of them
        # under 64 query heads), V at the head width.
        for dims in self._sparse_cells("stepfun-ai/Step-5-Preview"):
            self.assertEqual((dims["n_heads"], dims["n_kv_heads"],
                              dims["d_head"], dims["d_v_head"]),
                             (64, 4, 192, 192), dims)

    def test_hy4s_stored_latent_keys_the_latent_cell(self):
        for dims in self._sparse_cells("tencent/Hy4-preview",
                                       op="flash_dsa_sink"):
            self.assertNotIn("n_kv_heads", dims)
            self.assertEqual((dims["d_head"], dims["d_v_head"]), (576, 512))


class TestOneRuleForTheLatent(unittest.TestCase):
    """The latent is the cache stored with no head axis, on every sparse
    signature and in the stored operands every row is keyed from."""

    def test_the_sink_signature_takes_no_cache_stored_per_head(self):
        # Hy4's sink signature has no per-head geometry: a rank-4 cache at
        # input 1 is no cell, not the latent read by shape.
        q, out = [1, 1, 1024, 576], [1, 1, 1024, 512]
        self.assertIsNone(_dsa_identity(
            SINK, [q, [1, 1, 1024, 576], [1, 1024, 1024], [1],
                   [1, 1, 1024, 512]], out, SPEC))

    def test_a_one_head_store_keeps_its_head_axis(self):
        # A K stored per KV head and broadcast to four query heads, one KV
        # head left: divided, not squeezed to the latent's rank 3. Only the
        # walk's headless answer drops the axis.
        from fractions import Fraction
        from backend.inference.benchmark_identity import stored_operands
        walked = [[2, 4, 1, 512], [2, 4, 1152, 512], [2, 1, 288],
                  [2, 4, 1152, 512]]
        (form,) = stored_operands(SPARSE, walked,
                                  {1: Fraction(4), 3: Fraction(4)})
        self.assertEqual(form[1], [2, 1, 1152, 512])
        dims = _cell(SPARSE, (form, [2, 4, 1, 512]), KPOOL_SPEC)
        self.assertEqual(dims["n_kv_heads"], 1)
        (latent,) = stored_operands(SPARSE, walked, {1: Fraction(4)},
                                    frozenset({1, 3}))
        self.assertEqual(latent[1], [2, 1152, 512])


if __name__ == "__main__":
    unittest.main()
