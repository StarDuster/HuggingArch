"""A selected attention scores what each query selects, summed over the step.

The domain facts, from the kernels that run these attentions:

  * top-k sparse attention (DeepSeek-V3.2 DSA, GLM-5): the query whose causal
    context is `c` keys attends `min(c, k)` of them — the indexer's top-k over
    the keys it may see, padded with -1 when `c < k` (vLLM
    `sparse_attn_indexer.py`, FlashMLA sparse prefill skips -1 slots);
  * DeepSeek-V4's attention over a window joined with a compressed stream:
    `min(c, W)` window keys plus the `⌊c/m⌋` compressed blocks that have closed
    by its position — all of them on an HCA layer, the indexer's top
    `index_topk` on a CSA layer (vLLM `deepseek_v4/common/ops/cache_utils.py`:
    `topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)`,
    `swa_len = min(pos + 1, WINDOW_SIZE)`, `combined_len = topk_len + swa_len`).

A step's queries have contexts `s_past + 1 … s_past + s_query`, so a prefill
step is not `s_query` copies of its last query: the dense causal factor
`(S+1)/(2S)` halves a full lower triangle, and a selection capped at `k` is not
one. Decode (one query) is unchanged.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import (
    DECODE, Deployment, Phase, Workload, isolate_resolved,
)
from backend.arch.layout.physical import physical_kernel_shapes
from backend.arch.layout.physical_cost import physical_flops
from backend.arch.ops_base import _selected_pairs
from tests.snapshots import backbone_call, resolve_snapshot, shared_loader

_LOADER = shared_loader()
_RESOLVED = {}
S = 4096


def _resolved(model_id: str):
    if model_id not in _RESOLVED:
        _RESOLVED[model_id] = resolve_snapshot(model_id, _LOADER)
    return _RESOLVED[model_id]


def _brute(past: int, queries: int, cap=None, ratio=None) -> int:
    """Σ over the step's query contexts of what one query sees."""
    total = 0
    for context in range(past + 1, past + queries + 1):
        seen = context // (ratio or 1)
        total += seen if cap is None else min(seen, cap)
    return total


def _flops(analysis, op: str, name: Optional[str] = None):
    out = [analysis.cost.values[n.uid].flops
           for n in analysis.program.nodes()
           if n.op == op and (name is None or n.name == name)]
    assert len(out) == 1, (op, name, out)
    return int(out[0])


def _analysis(model_id, component, phase="prefill", prefix=0, cp=1, **dims):
    resolved = _resolved(model_id)
    deployment = Deployment(Program.of(isolate_resolved(
        resolved, component, dims, backbone_call(resolved, component)))).on(cp=cp).at(
            batch=1, input_tokens=S, s_prefix_cached=prefix)
    if phase == DECODE:
        deployment = deployment.during(DECODE, cache_len=S)
    return deployment.analyze()


class TestPhaseOffset(unittest.TestCase):
    """`s_past` is where the step's first query sits, in every workload."""

    def test_s_past_is_the_keys_before_the_first_query(self) -> None:
        cases = (
            (Phase("prefill"), Workload(input_tokens=S), 0),
            (Phase("prefill"), Workload(input_tokens=S, s_prefix_cached=1000),
             1000),
            (Phase(DECODE, cache_len=S), Workload(input_tokens=S), S - 1),
            (Phase(DECODE, cache_len=S), Workload(input_tokens=S, s_verify=4),
             S - 1),
        )
        for phase, workload, past in cases:
            with self.subTest(phase=phase.kind, workload=workload):
                bound = phase.resolve(workload)
                self.assertEqual(bound["s_past"], past)
                self.assertEqual(bound["s_past"],
                                 bound["s_kv"] - bound["s_query"])


class TestSelectedPairsLaw(unittest.TestCase):
    """The closed form equals the sum it names."""

    def test_closed_form_equals_the_sum(self) -> None:
        for past in (0, 1, 3, 127, 128, 1000):
            for queries in (1, 2, 5, 128, 129, 700):
                for cap, ratio in ((None, None), (64, None), (1, None),
                                   (None, 4), (None, 128), (16, 4), (3, 7)):
                    d = {"s_past": past, "s_query": queries}
                    self.assertEqual(
                        _selected_pairs(d, cap=cap, ratio=ratio),
                        _brute(past, queries, cap, ratio),
                        (past, queries, cap, ratio))

    def test_uncapped_unpooled_is_the_phase_triangle(self) -> None:
        for phase, workload in (
                (Phase("prefill"), Workload(input_tokens=S)),
                (Phase("prefill"),
                 Workload(input_tokens=S, s_prefix_cached=1000)),
                (Phase(DECODE, cache_len=S), Workload(input_tokens=S)),
                (Phase(DECODE, cache_len=S),
                 Workload(input_tokens=S, s_verify=4))):
            bound = phase.resolve(workload)
            self.assertEqual(_selected_pairs(bound), bound["s_pairs"])

    def test_a_verify_step_scores_each_query_at_its_own_context(self) -> None:
        """Three verify positions over a 5-token cache: the first query sees
        the 5 keys an autoregressive step sees, the next two one key more
        each — 5 + 6 + 7 = 18 pairs."""
        bound = Phase(DECODE, cache_len=5).resolve(
            Workload(input_tokens=5, s_verify=3))
        self.assertEqual((bound["s_query"], bound["s_kv"]), (3, 7))
        self.assertEqual(bound["s_pairs"], 18)


class TestTopKAttention(unittest.TestCase):
    """DeepSeek-V3.2's DSA attention: `min(c, k)` keys per query."""

    MODEL = "deepseek-ai/DeepSeek-V3.2"

    def _geometry(self):
        """Heads, the width a query scores at, and the top-k. Both phases
        score in latent space (`transforms.phase_form`: a sparse MLA
        prefills absorbed), so the width is the cached latent row."""
        dims = _resolved(self.MODEL).dim_values
        return (int(dims["n_heads"]),
                int(dims["d_kv_lora"]) + int(dims["d_qk_rope"]),
                int(dims["n_topk_keys"]))

    def test_prefill_scores_each_querys_selection(self) -> None:
        heads, width, k = self._geometry()
        for prefix in (0, 1000):
            with self.subTest(prefix=prefix):
                analysis = _analysis(self.MODEL, "mla_dsa_attention",
                                     prefix=prefix)
                pairs = _brute(prefix, S - prefix, cap=k)
                self.assertEqual(
                    _flops(analysis, "dsa_attention_score"),
                    2 * heads * width * pairs)
                self.assertEqual(_flops(analysis, "dsa_softmax"),
                                 5 * heads * pairs)

    def test_decode_query_scores_its_top_k(self) -> None:
        heads, _, k = self._geometry()
        analysis = _analysis(self.MODEL, "mla_dsa_attention", phase=DECODE)
        self.assertEqual(_flops(analysis, "dsa_softmax"), 5 * heads * min(S, k))


class TestStreamUnionAttention(unittest.TestCase):
    """DeepSeek-V4: each query sees its window and its closed blocks."""

    MODEL = "deepseek-ai/DeepSeek-V4-Flash"

    def _geometry(self):
        dims = _resolved(self.MODEL).dim_values
        return (int(dims["n_heads"]), int(dims["d_head"]),
                int(dims["s_window"]), int(dims["n_topk_keys"]))

    def test_prefill_prices_each_stream_mask(self) -> None:
        heads, width, window, topk = self._geometry()
        for component, ratio, cap in (("dsv4_hca_attention", 128, None),
                                      ("dsv4_csa_attention", 4, topk)):
            with self.subTest(component=component):
                analysis = _analysis(self.MODEL, component,
                                     compress_ratio=ratio)
                pairs = (_brute(0, S, cap=window)
                         + _brute(0, S, cap=cap, ratio=ratio))
                self.assertEqual(
                    _flops(analysis, "attention_score", "qk_score"),
                    2 * heads * width * pairs)
                self.assertEqual(
                    _flops(analysis, "attention_apply", "attn_apply"),
                    2 * heads * width * pairs)

    def test_a_chunk_after_a_prefix_prices_its_own_contexts(self) -> None:
        """Chunked prefill: the step's queries start after the cached
        prefix, so each sees the window and the blocks closed by its own
        position."""
        heads, width, window, topk = self._geometry()
        prefix = 1000
        for component, ratio, cap in (("dsv4_hca_attention", 128, None),
                                      ("dsv4_csa_attention", 4, topk)):
            with self.subTest(component=component):
                analysis = _analysis(self.MODEL, component, prefix=prefix,
                                     compress_ratio=ratio)
                queries = S - prefix
                pairs = (_brute(prefix, queries, cap=window)
                         + _brute(prefix, queries, cap=cap, ratio=ratio))
                self.assertEqual(
                    _flops(analysis, "attention_score", "qk_score"),
                    2 * heads * width * pairs)
                self.assertEqual(
                    _flops(analysis, "attention_apply", "attn_apply"),
                    2 * heads * width * pairs)

    def test_a_speculative_verify_step_prices_each_drafted_position(self) -> None:
        """Speculative decode verifies `k > 1` positions in one step; each is
        a query at its own context, after the `s_past` cached keys."""
        heads, width, window, topk = self._geometry()
        for component, ratio, cap in (("dsv4_hca_attention", 128, None),
                                      ("dsv4_csa_attention", 4, topk)):
            with self.subTest(component=component):
                resolved = _resolved(self.MODEL)
                deployment = Deployment(Program.of(isolate_resolved(
                    resolved, component, {"compress_ratio": ratio},
                    backbone_call(resolved, component)))).at(
                        batch=1, input_tokens=S, s_verify=4).during(
                            DECODE, cache_len=S)
                analysis = deployment.analyze()
                past = int(analysis.binding["s_past"])
                queries = int(analysis.binding["s_query"])
                self.assertEqual(queries, 4)
                pairs = (_brute(past, queries, cap=window)
                         + _brute(past, queries, cap=cap, ratio=ratio))
                self.assertEqual(
                    _flops(analysis, "attention_score", "qk_score"),
                    2 * heads * width * pairs)

    def test_decode_query_reads_the_union_once(self) -> None:
        heads, width, window, topk = self._geometry()
        for component, ratio, cap in (("dsv4_hca_attention", 128, None),
                                      ("dsv4_csa_attention", 4, topk)):
            with self.subTest(component=component):
                analysis = _analysis(self.MODEL, component, phase=DECODE,
                                     compress_ratio=ratio)
                blocks = S // ratio if cap is None else min(S // ratio, cap)
                self.assertEqual(
                    _flops(analysis, "attention_score", "qk_score"),
                    2 * heads * width * (min(S, window) + blocks))


class TestContextParallelSplitsPairs(unittest.TestCase):
    """A prefill context-parallel rank scores `1/cp` of the step's pairs.

    The per-rank projection evaluates the op at the rank's own queries; the
    selection's live pairs travel as the fraction of the matrix the node
    scores, so a rank holding `s_query/cp` queries gets `1/cp` of them, not
    the pairs of whichever `s_query/cp` positions a law re-summed locally
    would assume."""

    def test_each_rank_scores_its_share(self) -> None:
        """On the whole model: a rank's attention reads the residual stream
        as the mesh placed it, which only the model's own walk states."""
        for model_id, op in (
                ("deepseek-ai/DeepSeek-V3.2", "dsa_attention_score"),
                ("deepseek-ai/DeepSeek-V4-Flash", "attention_score")):
            with self.subTest(model=model_id):
                analysis = (Deployment(Program.of(_resolved(model_id)))
                            .on(cp=8).at(batch=1, input_tokens=S).analyze())
                local = physical_flops(analysis,
                                       physical_kernel_shapes(analysis))
                uids = [n.uid for n in analysis.program.nodes()
                        if n.op == op and n.name == "qk_score"]
                self.assertTrue(uids)
                for uid in uids:
                    self.assertEqual(int(local.values[uid]) * 8,
                                     int(analysis.cost.values[uid].flops))


class TestASelectionIsGatheredQueryByQuery(unittest.TestCase):
    """A top-k attention reads each query's selected rows by index: FlashMLA
    sparse, vLLM `_gqa_sparse_fwd_kernel` and sglang
    `_sparse_gqa_chunk_prefill` run a grid over queries, each loading its
    own list. Its cache read is the step's live pairs, each one cache
    position — the same pairs its FLOPs count — and not one pass over the
    `k` rows a single query selects, which priced a 16K-query prefill
    chunk below even the union of what its queries select. Rows are shared
    across queries only by a prompt several requests share, which the engine
    does not model."""

    MODEL = "Qwen/Qwen3.8-Flash-Next"
    COMPONENT = "qwen_qsa_attention"

    def _cache_reads(self, analysis):
        """`(gathered kv_reads, {cached tensor name: CachedTensor})` of the
        attention that gathers: its kv_reads, summed over its nodes."""
        from backend.arch.eval import cache as cache_mod
        from backend.arch.eval.dim import value_of
        from backend.arch.eval.traffic import derive
        traffic = derive(analysis)
        gathered = sum(
            float(value_of(traffic.values[uid].kv_reads))
            for uid in analysis.gathered)
        tensors = {tensor.name: tensor
                   for entry in cache_mod.derive(analysis).values.values()
                   for tensor in entry.tensors}
        return gathered, tensors

    def _per_pair_bytes(self, tensors):
        """Bytes one live pair reads: one position of K and one of V — the
        two tensors the top-k attention reads (the indexer's pooled key is
        streamed by the indexer, not gathered)."""
        from backend.arch.eval.dim import value_of
        return sum(float(value_of(tensors[name].position_bytes))
                   for name in ("key_concat", "v_split_heads"))

    def test_prefill_reads_its_pairs(self) -> None:
        k = int(_resolved(self.MODEL).dim_values["n_topk_keys"])
        for prefix in (0, 1000):
            with self.subTest(prefix=prefix):
                analysis = _analysis(self.MODEL, self.COMPONENT, prefix=prefix)
                read, tensors = self._cache_reads(analysis)
                pairs = _brute(prefix, S - prefix, cap=k)
                self.assertAlmostEqual(
                    read, pairs * self._per_pair_bytes(tensors),
                    delta=1.0)

    def test_decode_reads_one_querys_selection(self) -> None:
        """One query: its pairs are its read, so the gathered read is the
        read law's bytes at the cache length."""
        from backend.arch.eval.dim import value_of
        analysis = _analysis(self.MODEL, self.COMPONENT, phase=DECODE)
        read, tensors = self._cache_reads(analysis)
        law = sum(float(value_of(tensors[name].bytes_at(S, read=True)))
                  for name in ("key_concat", "v_split_heads"))
        self.assertAlmostEqual(read, law, delta=1.0)

    def test_a_verify_step_reads_each_drafted_querys_selection(self) -> None:
        k = int(_resolved(self.MODEL).dim_values["n_topk_keys"])
        resolved = _resolved(self.MODEL)
        analysis = Deployment(Program.of(isolate_resolved(
            resolved, self.COMPONENT, {},
            backbone_call(resolved, self.COMPONENT)))).at(
                batch=1, input_tokens=S, s_verify=4).during(
                    DECODE, cache_len=S).analyze()
        past = int(analysis.binding["s_past"])
        read, tensors = self._cache_reads(analysis)
        self.assertAlmostEqual(
            read, _brute(past, 4, cap=k) * self._per_pair_bytes(tensors),
            delta=1.0)

    def test_an_indexer_streams_the_keys_it_scores(self) -> None:
        """DeepSeek-V4's indexer scores every pooled key with a dense kernel
        (DeepGEMM `fp8_mqa_logits`): it reads one stream's mask, not an index
        list, so it is not a gathering reader — while the attention over the
        window and the selected blocks is."""
        analysis = _analysis("deepseek-ai/DeepSeek-V4-Flash",
                             "dsv4_csa_attention", compress_ratio=4)
        names = {analysis.program.by_uid(uid).name
                 for uid in analysis.gathered}
        self.assertIn("qk_score", names)
        self.assertNotIn("score", names)

    def test_dense_and_windowed_attention_gather_nothing(self) -> None:
        """A dense causal attention and a tiled sliding window state no
        selection: their kernels stream the context, and the walk records
        no gathering reader."""
        for model_id in ("meta-llama/Llama-3.2-1B", "openai/gpt-oss-20b"):
            with self.subTest(model=model_id):
                analysis = (Deployment(Program.of(_resolved(model_id)))
                            .at(batch=1, input_tokens=S, s_prefix_cached=1000)
                            .analyze())
                self.assertEqual(analysis.gathered, frozenset())


if __name__ == "__main__":
    unittest.main()
