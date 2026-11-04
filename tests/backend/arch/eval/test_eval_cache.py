"""The cache layer, against itself: which law each tensor got, and why.

Four cache lengths, because a single length cannot tell a windowed cache
from a dense one, and the whole point of this layer is that the two differ.
The corpus list is written out rather than globbed so a new snapshot is a
deliberate addition to the sweep, not a silent one.
"""
from __future__ import annotations

import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.values import CACHE_LEN
from tests.snapshots import shared_loader, snapshot_config
from backend.inference.parallel_config import ParallelConfig

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: Lengths that separate the laws. 64 is inside every window in the corpus, so
#: every class still looks dense; 1024 and 8192 put the windows and the
#: compressors on different curves; 131072 is where a 128-token window is three
#: orders of magnitude from the dense answer, which is the error this layer
#: exists to stop reporting.
LENGTHS = (64, 1024, 8192, 131072)

#: Every shipped snapshot whose whole-model KV agrees EXACTLY with the
#: incumbent at all four lengths — 38 of 39.
#: The corpus the law/shape tests sweep. LongCat sits outside it only as a
#: historical artifact of the reconciliation this file once hosted (the
#: incumbent billed half its cache — two attention instances per layer, a
#: per-class enumeration saw one); this layer reads the graph and gets 56.
AGREES = [
    "MiniMaxAI__MiniMax-M2.7", "Qwen__Qwen2.5-72B", "Qwen__Qwen2.5-7B",
    "Qwen__Qwen3-235B-A22B", "Qwen__Qwen3-30B-A3B", "Qwen__Qwen3-32B",
    "Qwen__Qwen3-8B", "Qwen__Qwen3-Coder-Next", "THUDM__glm-4-9b-chat",
    "THUDM__glm-4-9b", "XiaomiMiMo__MiMo-V2-Flash", "XiaomiMiMo__MiMo-V2.5-Pro",
    "XiaomiMiMo__MiMo-V2.5", "deepseek-ai__DeepSeek-V2-Lite",
    "deepseek-ai__DeepSeek-V2", "deepseek-ai__DeepSeek-V3.1",
    "deepseek-ai__DeepSeek-V3.2-Exp", "deepseek-ai__DeepSeek-V3.2",
    "deepseek-ai__DeepSeek-V3", "deepseek-ai__DeepSeek-V4-Pro",
    "google__gemma-2-27b", "google__gemma-2-9b", "google__gemma-4-12B-it",
    "inclusionAI__Ling-2.6-1T", "meta-llama__Llama-2-7b-hf",
    "meta-llama__Llama-3.2-1B", "meta-llama__Llama-3.2-3B",
    "mistralai__Mistral-7B-v0.1", "mistralai__Mistral-Nemo-Base-2407",
    "moonshotai__Kimi-K2-Base", "moonshotai__Kimi-K2-Instruct",
    "moonshotai__Kimi-K2-Thinking",
    "moonshotai__Kimi-K2.5", "moonshotai__Kimi-K3",
    "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "openai__gpt-oss-20b",
    "stepfun-ai__Step-3.7-Flash", "zai-org__GLM-5.1", "zai-org__GLM-5.2",
]




@lru_cache(maxsize=None)
def _analysis(name: str, tp: int = 1, cp: int = 1, dcp: int = 1):
    config = json.loads((SNAPSHOTS / f"{name}.json").read_text())["config"]
    resolved = shared_loader().match(config, model_id=name.replace("__", "/", 1))
    analysis = Deployment(
        Program.of(resolved),
        config=ParallelConfig(
            tp_attention=tp, tp_dense_ffn=tp, tp_moe_expert=tp,
            cp=cp, dcp=dcp),
    ).at(batch=1, input_tokens=128).analyze()
    return resolved, analysis


def _cache_layer(analysis):
    from backend.arch.layout.physical_cache import cache_geometry

    return cache_mod.derive(analysis, geometry=cache_geometry(analysis))


def _backbone_bytes(name: str, tp: int = 1, cp: int = 1, dcp: int = 1,
                    length: int = 8192) -> float:
    _resolved, analysis = _analysis(name, tp, cp, dcp)
    stacks = {s.name: s for s in analysis.program.stacks}
    total = 0.0
    for uid, entry in _cache_layer(analysis).values.items():
        stack = stacks.get(
            analysis.program.stack_of(analysis.program.by_uid(uid)))
        if not entry.persists or (stack is not None and stack.is_aux):
            continue
        total += (float(entry.bytes_at(length))
                  * float(analysis.instances.get(uid, 1)))
    return total


@lru_cache(maxsize=None)
def _mine(name: str) -> Dict[int, float]:
    """Backbone KV bytes per length — the incumbent's scope, which is the main
    stack only. An encoder tower keeps no cache at all; the aux test below is
    what covers the head this then drops."""
    return {length: _backbone_bytes(name, length=length) for length in LENGTHS}


class TestLaws(unittest.TestCase):
    """What the reconciliation cannot see: WHICH law each tensor got."""

    def _laws(self, name: str, component: str) -> Dict[str, Tuple[Any, Any]]:
        _resolved, analysis = _analysis(name)
        for uid, entry in cache_mod.derive(analysis).values.items():
            if analysis.program.by_uid(uid).op == component:
                return {t.name: (sp.sympify(t.stored), sp.sympify(t.read))
                        for t in entry.tensors}
        self.fail(f"{name} has no cached {component!r}")

    def test_a_dense_cache_grows_with_the_context(self):
        laws = self._laws("meta-llama__Llama-3.2-1B", "gqa_attention")
        self.assertEqual(sorted(laws), ["k_rope", "v_split_heads"])
        for stored, read in laws.values():
            self.assertEqual(stored, CACHE_LEN)
            self.assertEqual(read, CACHE_LEN)

    def test_a_window_stops_growing(self):
        laws = self._laws("google__gemma-2-9b", "swa_attention")
        for stored, read in laws.values():
            self.assertEqual(stored, sp.Min(4096, CACHE_LEN))
            self.assertEqual(read, stored)

    def test_a_sparse_read_keeps_everything_it_scores(self):
        """DSA is the case a single per-layer number cannot hold: it reads top-K
        and stores all of it, because the indexer needs the whole cache to pick
        the K."""
        laws = self._laws("deepseek-ai__DeepSeek-V3.2", "mla_dsa_attention")
        stored, read = laws["kv_a_layernorm"]
        self.assertEqual(stored, CACHE_LEN)
        self.assertEqual(read, sp.Min(2048, CACHE_LEN))
        # The indexer's own key cache rides along, at its own dtype, and is
        # reachable from nowhere else: the indexer's only output is positions.
        self.assertIn("indexer.key_concat", laws)

    def test_a_pooled_indexer_caches_one_key_per_pool(self):
        """GLM-5.3-Flash at decode. The sparse attention reads the latent rows
        it selected — index_topk pooled tokens plus the incomplete tail,
        2048 + 4 - 1 — while the k-pool indexer caches one pooled key per
        pool, as vLLM and sglang do: ⌈s_kv/4⌉ rows stored (the pending pool
        owns a slot), ⌊s_kv/4⌋ complete pools read. The pending pool's raw
        K and gate are a per-request ring that does not grow. (HF caches the
        raw per-token state and re-pools it every step; the pooled keys are
        bit-identical once a pool is complete.)"""
        config = json.loads(
            (SNAPSHOTS / "zai-org__GLM-5.3-Flash.json").read_text())["config"]
        resolved = shared_loader().match(config,
                                         model_id="zai-org/GLM-5.3-Flash")
        analysis = (Deployment(Program.of(resolved)).at(batch=1)
                    .during("decode", cache_len=4096).analyze())
        laws = {}
        for uid, entry in cache_mod.derive(analysis).values.items():
            if analysis.program.by_uid(uid).op == "mla_dsa_kpool_nope_attention":
                laws = {t.name: (sp.sympify(t.stored), sp.sympify(t.read))
                        for t in entry.tensors}
        self.assertEqual(laws["kv_a_layernorm"],
                         (CACHE_LEN, sp.Min(2051, CACHE_LEN)))
        self.assertEqual(laws["index_kpool.pooled_out"],
                         (sp.ceiling(CACHE_LEN / 4), sp.floor(CACHE_LEN / 4)))
        for ring in ("index_kpool.key_state", "index_kpool.gate_state"):
            stored, read = laws[ring]
            self.assertEqual(stored, 1)
            self.assertNotIn(CACHE_LEN, sp.sympify(read).free_symbols)

    def test_a_recurrent_state_does_not_grow_at_all(self):
        laws = self._laws("Qwen__Qwen3-Coder-Next",
                          "gated_delta_net_fused_attention")
        for state in ("recurrent_state", "conv_state"):
            stored, read = laws[state]
            self.assertEqual(stored, 1)
            self.assertEqual(read, 1)
            self.assertNotIn(CACHE_LEN, sp.sympify(stored).free_symbols)

    def test_one_layer_can_hold_two_laws_at_once(self):
        """The case that makes the list the representation. A V4-Pro CSA layer
        keeps a windowed concat AND a compressed latent, and no single per-layer
        expression is true of both."""
        laws = self._laws("deepseek-ai__DeepSeek-V4-Pro", "dsv4_csa_attention")
        self.assertEqual(laws["kv_concat"][0], sp.Min(128, CACHE_LEN))
        self.assertEqual(laws["compressor.pooled_out"][0],
                         sp.ceiling(CACHE_LEN / 4))
        self.assertNotEqual(laws["kv_concat"][0],
                            laws["compressor.pooled_out"][0])

    def test_a_compressor_keeps_its_unfilled_block_in_fp32(self):
        """model.py::Compressor keeps kv_state and score_state per request:
        the rows of a block not yet pooled, allocated
        `dtype=torch.float32` (L303-304) whatever the KV dtype — vLLM
        asserts fp32 for them and SGLang defaults to it. They do not grow
        with the context. The POOLED rows take their format from the
        model's `cache_formats` by the kind each holds — MXFP4 index keys,
        FP8 compressed KV."""
        _resolved, analysis = _analysis("deepseek-ai__DeepSeek-V4-Pro")
        for uid, entry in cache_mod.derive(analysis).values.items():
            if analysis.program.by_uid(uid).op != "dsv4_csa_attention":
                continue
            by_name = {t.name: t for t in entry.tensors}
            self.assertEqual(by_name["indexer.compressor.pooled_out"].dtype,
                             "FP4_E2M1")
            self.assertEqual(by_name["compressor.pooled_out"].dtype,
                             "F8_E4M3")
            for prefix in ("compressor", "indexer.compressor"):
                for state in ("kv_state", "score_state"):
                    t = by_name[f"{prefix}.{state}"]
                    self.assertEqual(t.dtype, "F32")
                    self.assertEqual(sp.sympify(t.stored), 1)
                    self.assertEqual(sp.sympify(t.read), 1)
            # [2*ratio, 2*head_dim] at ratio 4: head_dim 512, indexer 128.
            self.assertEqual(int(by_name["compressor.kv_state"].elements),
                             8 * 1024)
            self.assertEqual(
                int(by_name["indexer.compressor.kv_state"].elements), 8 * 256)
            return
        self.fail("no CSA layer found")

    def test_a_compressor_reads_its_ring_once_per_window(self):
        """Decode pools the ring once per window it completes (`reads:`), not
        every token: HCA's 128-row ring is read 1/128 of a step at one token.
        Billed whole every step, it put 65 KB on a multiply whose own work the
        walk had already averaged to a quarter."""
        config = json.loads(
            (SNAPSHOTS / "deepseek-ai__DeepSeek-V4-Pro.json").read_text())["config"]
        resolved = shared_loader().match(config, model_id="deepseek-ai/DeepSeek-V4-Pro")
        analysis = (Deployment(Program.of(resolved)).at(batch=1, input_tokens=1)
                    .during("decode", cache_len=4096).analyze())
        for uid, entry in cache_mod.derive(analysis).values.items():
            if analysis.program.by_uid(uid).op != "dsv4_hca_attention":
                continue
            by_name = {t.name: t for t in entry.tensors}
            for state in ("kv_state", "score_state"):
                t = by_name[f"compressor.{state}"]
                self.assertEqual(sp.Rational(str(value_of(t.read))),
                                 sp.Rational(1, 128))
                self.assertEqual(sp.sympify(t.stored), 1)
            return
        self.fail("no HCA layer found")

    def test_a_compressing_producer_keeps_a_full_width_row(self):
        """The row is the width of what it writes, not of what it read.

        A compressor emitting one row per `m` tokens has an output sequence axis
        of `ceil(S/m)`; dividing its elements by `S` would narrow the row by `m`
        and then multiply it back against `ceil(Lq/m)` positions, under-billing
        by `m` squared. HCA compresses at 128.
        """
        _resolved, analysis = _analysis("deepseek-ai__DeepSeek-V4-Pro")
        for uid, entry in cache_mod.derive(analysis).values.items():
            if analysis.program.by_uid(uid).op != "dsv4_hca_attention":
                continue
            by_name = {t.name: t for t in entry.tensors}
            self.assertEqual(by_name["compressor.pooled_out"].elements,
                             by_name["kv_concat"].elements)
            # No overlap at ratio 128: each state is [ratio, head_dim].
            self.assertEqual(int(by_name["compressor.kv_state"].elements),
                             128 * 512)
            return
        self.fail("no HCA layer found")


_TWO_READERS = """
match: {{ model_type: [_two_readers] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, n_heads: "2", d_head: "8", s_window: "4" }}
blocks:
  attn:
    role: attention
    cache: {{ family: MLA, entries: [{{ node: kv, kind: latent }}] }}
    graph:
      kv:      {{ op: linear, in: x, dims: {{ in: d_model, out: d_head }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      kv_b:    {{ op: head_broadcast, in: kv, dims: {{ n_heads: n_heads }} }}
      q:       {{ op: linear, in: x, dims: {{ in: d_model, out: "n_heads * d_head" }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      q_heads: {{ op: head_split, in: q, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      dense:   {{ op: attention_score, in: [q_heads, kv_b],
                 dims: {{ n_heads: n_heads, d_head: d_head }} }}
      second:  {{ op: {second}, in: [q_heads, kv_b],
                 dims: {{ n_heads: n_heads, d_head: d_head }} }}
      y:       {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [y]
  blk:
    graph:
      self_attn: {{ op: attn, in: x }}
    outputs: [self_attn]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: b, rest: true }}
      map:
        b: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""


class TestReadersThroughAView(unittest.TestCase):
    """A cached tensor with no law of its own takes its readers' law, through
    any view between them. Readers that agree give it; readers that state
    different laws leave none to pick, and declaration order is not one."""

    def _laws(self, second: str):
        import os
        import tempfile
        from backend.arch.loader import SpecLoader
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as f:
            f.write(_TWO_READERS.format(second=second))
            path = f.name
        loader = SpecLoader()
        try:
            spec = loader.load_model_spec(path)
        finally:
            os.unlink(path)
        resolved = loader.resolve(spec, {
            "torch_dtype": "bfloat16", "hidden_size": 16, "num_hidden_layers": 1, "vocab_size": 32,
            "model_type": "_two_readers"})
        analysis = (Deployment(Program.of(resolved)).at(batch=1)
                    .during("decode", cache_len=64).analyze())
        return {t.name: (sp.sympify(t.stored), sp.sympify(t.read))
                for entry in cache_mod.derive(analysis).values.values()
                for t in entry.tensors}

    def test_readers_that_agree_give_the_law(self):
        self.assertEqual(self._laws("attention_score")["kv"],
                         (CACHE_LEN, CACHE_LEN))

    def test_readers_that_disagree_are_refused(self):
        from backend.arch.eval.refusal import SpecRefusal
        with self.assertRaises(SpecRefusal) as caught:
            self._laws("swa_attention_score")
        self.assertIn("read laws", str(caught.exception))


_TWO_SITES = """
match: {{ model_type: [_two_sites] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, n_heads: "2", d_head: "8", s_window: "4" }}
blocks:
  attn:
    role: attention
    params:
      score: {{ kind: component, default: attention_score }}
    cache: {{ family: MLA, entries: [{{ node: kv, kind: latent }}] }}
    graph:
      kv:      {{ op: linear, in: x, dims: {{ in: d_model, out: d_c }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      kv_b:    {{ op: head_broadcast, in: kv, dims: {{ n_heads: n_heads }} }}
      kv_p:    {{ op: linear, in: kv_b, dims: {{ in: d_c, out: d_head }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      q:       {{ op: linear, in: x, dims: {{ in: d_model, out: "n_heads * d_head" }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      q_heads: {{ op: head_split, in: q, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      s:       {{ op: <score>, in: [q_heads, kv_p],
                 dims: {{ n_heads: n_heads, d_head: d_head }} }}
      y:       {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [y]
  blk:
    graph:
      first:  {{ op: attn, in: x, {first} }}
      second: {{ op: attn, in: first, {second} }}
    outputs: [second]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: b, rest: true }}
      map:
        b: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""

#: The two ways the toy's attention is called: a dense read of a `d_head`-wide
#: latent, and a windowed read of one twice as wide.
_DENSE = "dims: { d_c: d_head }"
_WINDOWED = 'dims: { d_c: "2 * d_head" }, params: { score: swa_attention_score }'


class TestEachCallSiteKeepsItsOwnCache(unittest.TestCase):
    """One component called at two sites of a layer, each binding a different
    cached width and a different read: each site's cache is billed by its own
    call — stored and read bytes alike — whichever site comes first."""

    def _by_site(self, first: str, second: str):
        import os
        import tempfile
        from backend.arch.loader import SpecLoader
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as f:
            f.write(_TWO_SITES.format(first=first, second=second))
            path = f.name
        loader = SpecLoader()
        try:
            spec = loader.load_model_spec(path)
        finally:
            os.unlink(path)
        resolved = loader.resolve(spec, {
            "torch_dtype": "bfloat16", "hidden_size": 16, "num_hidden_layers": 1,
            "vocab_size": 32, "model_type": "_two_sites"})
        analysis = (Deployment(Program.of(resolved)).at(batch=1)
                    .during("decode", cache_len=64).analyze())
        return {analysis.program.by_uid(uid).name: entry
                for uid, entry in cache_mod.derive(analysis).values.items()}

    def _assert_site(self, entry, *, width: int, law):
        (tensor,) = entry.tensors
        self.assertEqual((sp.sympify(tensor.stored), sp.sympify(tensor.read)),
                         (law, law))
        for length in (64, 1024):
            want = float(law.subs(CACHE_LEN, length)) * width * 2   # bf16
            self.assertEqual(float(entry.bytes_at(length)), want)
            self.assertEqual(float(entry.bytes_at(length, read=True)), want)

    def test_each_site_is_billed_by_its_own_declaration(self):
        dense, windowed = CACHE_LEN, sp.Min(4, CACHE_LEN)
        sites = self._by_site(_DENSE, _WINDOWED)
        self.assertEqual(set(sites), {"first", "second"})
        self._assert_site(sites["first"], width=8, law=dense)
        self._assert_site(sites["second"], width=16, law=windowed)

        swapped = self._by_site(_WINDOWED, _DENSE)
        self._assert_site(swapped["first"], width=16, law=windowed)
        self._assert_site(swapped["second"], width=8, law=dense)


class TestScope(unittest.TestCase):
    def test_an_encoder_tower_keeps_no_cache(self):
        """Step-3.7-Flash's perception encoder runs once over an image, in the
        encode phase, not once per token. Its attention declares a cache
        (`family: ViT`), yet nothing of it is kept for a later step: neither
        the language model's analysis nor the encoder's own derives one.
        Billing it per request adds a gigabyte no server holds."""
        _resolved, analysis = _analysis("stepfun-ai__Step-3.7-Flash")
        program = analysis.program
        towers = [n for n in program.nodes()
                  if n.op == "perception_encoder_attention"
                  and n.uid in program.encoder_uids]
        self.assertTrue(towers, "Step-3.7-Flash's encoder has its attention")
        self.assertTrue(program.spec.get_block(
            "perception_encoder_attention").cache.entries,
            "and that attention declares a cache")
        self.assertFalse(program.encoder_uids
                         & set(cache_mod.derive(analysis).values))
        encode = Deployment(program).at(
            batch=1, media={"pixel_values": {"n_img": 1},
                            "patch_pixel_values": {"n_crops": 0}}).during(
            "encode", stack="vision").analyze()
        self.assertTrue([n for n in towers if n.uid in encode.cost.values],
                        "the encode phase runs the attention")
        self.assertEqual(dict(cache_mod.derive(encode).values), {})

    def test_a_speculative_head_does_persist_one(self):
        _resolved, analysis = _analysis("deepseek-ai__DeepSeek-V3")
        stacks = {s.name: s for s in analysis.program.stacks}
        mtp = [entry for uid, entry
               in cache_mod.derive(analysis).values.items()
               if getattr(stacks.get(analysis.program.stack_of(
                   analysis.program.by_uid(uid))), "aux_kind", None) == "mtp"]
        self.assertTrue(mtp, "V3 ships an MTP layer with its own attention")
        self.assertTrue(all(e.persists for e in mtp))

    def test_a_sub_cache_is_not_also_counted_on_its_own(self):
        """The indexer's cache belongs to the layer that invoked it. Keyed on
        its own uid as well, every byte of it would be counted twice."""
        _resolved, analysis = _analysis("deepseek-ai__DeepSeek-V3.2")
        ops = [analysis.program.by_uid(uid).op
               for uid in cache_mod.derive(analysis).values]
        self.assertNotIn("dsa_indexer", ops)
        self.assertIn("mla_dsa_attention", ops)


class TestProducingUidsAreTheLedger(unittest.TestCase):
    """`cached_node_uids` is the same set `derive` writes on CachedTensor.node.

    Declaration names are lookup keys. A declared child that is itself a
    cached component is not a tensor — CSA's indexer emits top-k indices;
    the cache is the compressed keys inside. A declared child that is a
    wrapper is the tensor: GQA's `v_split_heads` is what `attn_apply`
    reads; the permute inside is geometry. Every reader (write endpoint,
    traffic, fusion preserve) has to see that set, not the names.
    """

    def _billed(self, analysis):
        return {
            tensor.node
            for entry in cache_mod.derive(analysis).values.values()
            for tensor in entry.tensors
            if not tensor.recurrent
        }

    def test_uids_match_derive_across_cache_families(self):
        for name in (
            "meta-llama__Llama-3.2-1B",
            "google__gemma-4-12B-it",
            "deepseek-ai__DeepSeek-V3.2",
            "deepseek-ai__DeepSeek-V4-Pro",
        ):
            with self.subTest(model=name):
                _resolved, analysis = _analysis(name)
                self.assertEqual(
                    self._billed(analysis),
                    cache_mod.cached_node_uids(analysis.program),
                )

    def test_v4_pro_indexer_is_not_a_cached_tensor(self):
        _resolved, analysis = _analysis("deepseek-ai__DeepSeek-V4-Pro")
        ops = {
            analysis.program.by_uid(uid).op
            for uid in cache_mod.cached_node_uids(analysis.program)
        }
        self.assertNotIn("dsv4_indexer", ops)
        self.assertNotIn("topk", ops)

    def test_a_declared_wrapper_is_the_cached_tensor(self):
        """GQA publishes V as `v_split_heads`. The permute inside is
        geometry, not identity — `attn_apply` reads the wrapper."""
        _resolved, analysis = _analysis("meta-llama__Llama-3.2-1B")
        ops = {
            analysis.program.by_uid(uid).op
            for uid in cache_mod.cached_node_uids(analysis.program)
        }
        self.assertIn("head_split", ops)
        self.assertNotIn("transpose", ops)


class TestPerRank(unittest.TestCase):
    """What a rank holds, with no divisor anywhere.

    A cache shards two ways and both arrive from the layout-carrying
    analysis. Its WIDTH is the physical shape of the cached tensor — a
    GQA layer's K/V heads come out narrower and an MLA latent, whose
    axes no TP world owns, does not — and its POSITIONS by
    `positions_over` from the DCP cache family. A recurrent state has
    no such axis and remains replicated.

    The `kv_tp_shardable:` flag this proved redundant is deleted — schema
    field, yaml lines and all; the shapes were always the answer.
    """

    def _per_component(self, name: str, tp: int) -> Dict[str, float]:
        _resolved, analysis = _analysis(name, tp)
        out: Dict[str, float] = {}
        for uid, entry in _cache_layer(analysis).values.items():
            op = analysis.program.by_uid(uid).op
            out[op] = max(out.get(op, 0.0),
                          float(entry.bytes_at(8192)))
        return out

    def test_decode_context_parallelism_divides_the_positions(self):
        """It reaches the cache through no shape: the length is a workload fact
        the consumer substitutes, so `positions_over` is where it lands.

        `dcp`, not `cp`: splitting the CACHE is what decode context parallelism
        IS. It adds no ranks — it re-cuts the attention TP ones — so the
        comparison is against the same TP width, not against a one-rank
        baseline. MHA has no replica cell for DCP to sit in, so the models
        here are GQA and MLA."""
        for name, tp in (
            ("Qwen__Qwen2.5-7B", 8),
            ("deepseek-ai__DeepSeek-V3", 8),
        ):
            with self.subTest(model=name):
                self.assertEqual(_backbone_bytes(name, tp=tp, dcp=1),
                                 _backbone_bytes(name, tp=tp, dcp=2) * 2)

    def test_dcp_does_not_divide_recurrent_state_in_a_hybrid_backbone(self):
        """Position ownership is a tensor fact, not an attention-class fact.

        These shipped hybrids keep ordinary K/V positions and a recurrent
        state in the same backbone. DCP splits only the former; applying one
        class-level divisor to both makes the constant state floor too small.
        """
        names = (
            "Qwen__Qwen3-Coder-Next",
            "inclusionAI__Ling-2.6-1T",
            "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        )

        def totals(name: str, dcp: int) -> Tuple[float, float]:
            _resolved, analysis = _analysis(name, tp=8, dcp=dcp)
            positioned = 0.0
            recurrent = 0.0
            for uid, entry in _cache_layer(analysis).values.items():
                count = float(analysis.instances.get(uid, 1))
                for tensor in entry.tensors:
                    amount = float(tensor.bytes_at(8192)) * count
                    if tensor.recurrent:
                        recurrent += amount
                    else:
                        positioned += amount
            return positioned, recurrent

        for name in names:
            with self.subTest(model=name):
                whole_positions, whole_state = totals(name, 1)
                local_positions, local_state = totals(name, 2)
                self.assertGreater(whole_positions, 0)
                self.assertGreater(whole_state, 0)
                self.assertEqual(whole_positions, local_positions * 2)
                self.assertEqual(whole_state, local_state)

    def test_dcp_divides_dynamic_scales_but_not_static_scales(self):
        """Position ownership cannot divide metadata with no position axis."""
        from backend.arch.cache_formats import CacheFormat
        from backend.arch.eval.cache_view import CacheClass, CacheView

        def view(granularity):
            fmt = CacheFormat.parse(
                {"dtype": "F8_E4M3",
                 "scale": {"granularity": granularity, "dtype": "F32"}}, "t")
            tensor = cache_mod.CachedTensor(
                node=1, name="kv", elements=8, inner=8, rows=1,
                positions_over=2, format=fmt)
            return CacheView(entries=(CacheClass(
                cache_mod.Cache(component=1, tensors=(tensor,)), 1),))

        # Static scales: whole on every owner. Data: 8 FP8 bytes per
        # position over 2 owners = 4 per token.
        self.assertEqual(view("per-tensor").static_scale_bytes(), 4.0)
        self.assertEqual(view("per-channel").static_scale_bytes(), 32.0)
        self.assertEqual(view("per-tensor").growth_and_floor(), (4.0, 0.0))
        # A dynamic scale rides its position and is divided with it:
        # (8 data + 1 scale × 4 B) / 2 owners per token.
        self.assertEqual(view("per-group").static_scale_bytes(), 0.0)
        self.assertEqual(view("per-group").growth_and_floor(), (6.0, 0.0))

    def test_only_tensors_stored_quantised_carry_scales(self):
        """Scales are a tensor's own format's: a state stored FP8 with a
        per-tensor scale carries one per layer, a state stored F32 none."""
        from backend.arch.cache_formats import CACHE_FORMATS, CacheFormat
        from backend.arch.eval.cache_view import CacheClass, CacheView

        def state(name, fmt):
            return cache_mod.CachedTensor(
                node=1, name=name, elements=256, inner=128, rows=2,
                stored=sp.Integer(1), read=sp.Integer(1), recurrent=True,
                format=fmt)

        def view(*tensors):
            return CacheView(entries=(CacheClass(
                cache_mod.Cache(component=1, tensors=tensors), 3),))

        fp8 = state("conv_state", CACHE_FORMATS["fp8_per_tensor"])
        fp32 = state("kv_state", CacheFormat.plain("F32"))
        # per-tensor: one static scale per quantised tensor per layer.
        self.assertEqual(view(fp8).static_scale_bytes(), 12.0)
        self.assertEqual(view(fp32).static_scale_bytes(), 0.0)
        self.assertEqual(view(fp8, fp32).static_scale_bytes(), 12.0)
        # per-group: the dynamic scales of a state ride the constant floor —
        # 2 rows × 1 group × 4 B × 3 layers.
        grouped = state("conv_state", CACHE_FORMATS["fp8_group128"])
        data = 3 * 256 * 1.0 + 3 * 256 * 4.0
        self.assertEqual(view(grouped, fp32).growth_and_floor(),
                         (0.0, data + 2 * 4.0 * 3))

    def test_prefill_context_parallelism_divides_no_position(self):
        """PCP splits the prefill COMPUTE and replicates the cache: vLLM
        all_gathers each rank's prefill tail across the PCP group before the
        write (`attention/pcp.py::_gather_prefill_cache_inputs`), so every rank
        ends the step holding the whole thing.

        This is the law the memory budget cares about, and it is the opposite
        of the one above — a deployment that bought `cp=8` for prefill latency
        did not buy 8x the KV capacity, and pricing it as if it had would let a
        sweep recommend a context length the cluster cannot hold."""
        for name in ("meta-llama__Llama-3.2-1B", "deepseek-ai__DeepSeek-V3"):
            with self.subTest(model=name):
                self.assertEqual(_backbone_bytes(name, cp=1),
                                 _backbone_bytes(name, cp=2))

    def test_a_latent_cache_ignores_tensor_parallelism(self):
        """MLA has no KV head axis to split, so every rank stores the same
        latent. Derived: `d_kv_lora` is in no policy, so nothing divides it."""
        self.assertEqual(_backbone_bytes("deepseek-ai__DeepSeek-V3", tp=1),
                         _backbone_bytes("deepseek-ai__DeepSeek-V3", tp=8))

    def test_a_degree_that_does_not_divide_the_heads_does_not_pretend_to(self):
        """Qwen2.5-7B holds 4 KV heads. Over 8 ranks it shards by 4 and
        replicates the rest, which is what a deployment does — the incumbent's
        `kv_shard_divisor` returns `tp * cp` unconditionally and so reports half
        the per-rank KV, and a max-batch computed from it is twice too
        optimistic."""
        whole = _backbone_bytes("Qwen__Qwen2.5-7B")
        self.assertEqual(whole / 4, _backbone_bytes("Qwen__Qwen2.5-7B", tp=4))
        self.assertEqual(whole / 4, _backbone_bytes("Qwen__Qwen2.5-7B", tp=8))

    def test_a_hybrid_backbone_has_no_single_divisor(self):
        """The structural reason the scalar had to go. gemma-4 mixes classes
        whose head counts saturate at different degrees, so the whole-model
        ratio is not an integer at all and no `kv_div` can be right for both."""
        whole = _backbone_bytes("google__gemma-4-12B-it")
        ratio = whole / _backbone_bytes("google__gemma-4-12B-it", tp=2)
        self.assertNotEqual(ratio, int(ratio))
        self.assertLess(ratio, 2.0)


class TestGrowthFormulaFromLeaves(unittest.TestCase):
    """The dashboard formula is the leaf Dim identities, composed up.

    Not node names, not grounded integers, not a folded class tag. Each
    cached tensor that adds positions per token contributes
    `expr_of(elements)` at its rate (a pool of `m` tokens per row is
    `/ m`; a full window adds nothing and sits in the floor); a layer sums
    those leaves; identical Σs factor `n_layers`; distinct Σs stay a sum.
    """

    def _gf(self, name: str) -> Dict[str, Any]:
        from backend.arch.eval.cache_view import cache_view_from_analysis
        _resolved, analysis = _analysis(name)
        return cache_view_from_analysis(analysis).growth_formula

    def test_mla_keeps_the_original_latent_plus_rope_law(self):
        gf = self._gf("deepseek-ai__DeepSeek-V3")
        self.assertEqual(
            gf["symbolic"],
            "n_layers × (d_kv_lora + d_qk_rope) × bpe",
        )
        self.assertEqual(gf["substituted"], "(3 + 58) × (512 + 64) × 2")

    def test_gqa_sums_the_two_cached_leaves(self):
        gf = self._gf("meta-llama__Llama-3.2-3B")
        self.assertEqual(
            gf["symbolic"],
            "n_layers × (d_head × n_kv_heads + d_head × n_kv_heads) × bpe",
        )
        self.assertEqual(gf["substituted"], "28 × (1024 + 1024) × 2")

    def test_a_full_window_adds_nothing_per_token(self):
        """Step-3.7: 12 full-attention layers grow per token; the 33 windowed
        ones hold their window whatever the length — the floor, not the
        slope. Counting them per token billed a 128-token window as if it
        held every token."""
        name = "stepfun-ai__Step-3.7-Flash"
        gf = self._gf(name)
        self.assertEqual(
            gf["symbolic"],
            "n_layers × (n_kv_heads × (d_nope + d_rope) + d_head × n_kv_heads) × bpe",
        )
        self.assertEqual(gf["substituted"], "(1 + 11) × (1024 + 1024) × 2")
        self._assert_affine(name)

    def test_a_compressed_pool_grows_at_its_ratio(self):
        """DeepSeek-V4: a compressor keeps one row per `m` tokens, so a
        token adds `1/m` of a row; the 128-token windowed concat and the
        compressors' running states are the per-request floor."""
        name = "deepseek-ai__DeepSeek-V4-Pro"
        gf = self._gf(name)
        data, scales = gf["subterms"]
        self.assertEqual(
            data["symbolic"],
            "n_layers × (d_head / 128) × bpe"
            " + n_layers × (d_head / 4 + d_indexer_head / 4) × bpe",
        )
        # A row is 448 FP8 channels + a 64-channel BF16 RoPE tail (576 B
        # over 512); an index key is MXFP4.
        self.assertEqual(data["result_bytes"], 31 * 576 / 128
                         + 30 * (576 / 4 + 128 * 0.5 / 4))
        # Beside each row the format stores 7 one-byte scales (448 / 64),
        # beside each index key 4 (128 / 32): stored per token like the
        # data, so the slope carries them.
        self.assertEqual(scales["result_bytes"], 31 * 7 / 128
                         + 30 * (7 / 4 + 4 / 4))
        self.assertAlmostEqual(gf["result_bytes"],
                               data["result_bytes"] + scales["result_bytes"])
        self._assert_affine(name)

    def _assert_affine(self, name: str) -> None:
        """Past every window, per-token growth (data and the scales that
        ride each position) × L + floor IS the request's stored bytes: the
        per-token figure is that law's slope, one fact."""
        from backend.arch.eval.cache_view import cache_view_from_analysis
        _resolved, analysis = _analysis(name)
        view = cache_view_from_analysis(analysis)
        growth, floor = view.growth_and_floor()
        self.assertGreater(floor, 0)
        for length in (8192, 131072):
            self.assertAlmostEqual(growth * length + floor,
                                   view.storage_bytes_at(length), places=3)

if __name__ == "__main__":
    unittest.main()


class TestRequestFormula(unittest.TestCase):
    """One request's cache, written out: terms that add up to
    `storage_bytes_at`, whatever the mix of laws."""

    def _children_sum(self, tree: Dict[str, Any]) -> None:
        subs = tree.get("subterms") or []
        if subs:
            self.assertAlmostEqual(
                sum(st["result_bytes"] for st in subs), tree["result_bytes"],
                delta=1e-6 * tree["result_bytes"], msg=tree["label"])
        for st in subs:
            self._children_sum(st)

    def test_terms_add_up_to_the_stored_bytes(self):
        from backend.arch.eval.cache_view import cache_view_from_analysis
        for name in ("meta-llama__Llama-3.2-3B", "deepseek-ai__DeepSeek-V3",
                     "openai__gpt-oss-20b", "deepseek-ai__DeepSeek-V4-Pro"):
            with self.subTest(name=name):
                view = cache_view_from_analysis(_analysis(name)[1])
                for length in (8192, 1048576):
                    tree = view.request_formula(length)
                    self.assertEqual(tree["result_bytes"],
                                     view.storage_bytes_at(length))
                    self._children_sum(tree)

    def test_each_class_line_is_the_leaf_law_and_its_values_compute(self):
        """A class line is written from the leaves (`n_layers × (Σ leaf) ×
        bpe × law`), not from a lumped per-token figure, and its values
        line is arithmetic that computes the bytes it explains."""
        from backend.arch.eval.cache_view import cache_view_from_analysis
        for name in ("openai__gpt-oss-20b", "deepseek-ai__DeepSeek-V4-Pro"):
            view = cache_view_from_analysis(_analysis(name)[1])
            for v in (view, view.on_layer_split(4)) if not view.mixed_growth_laws else (view,):
                def walk(tree):
                    for st in tree.get("subterms") or []:
                        with self.subTest(name=name, line=st["label"]):
                            self.assertTrue(st["symbolic"].startswith("n_layers × "))
                            self.assertNotIn("per_token", st["symbolic"])
                            got = eval(st["substituted"].replace("×", "*"),
                                       {"__builtins__": {}})
                            self.assertAlmostEqual(got, st["result_bytes"],
                                                   delta=1e-9 * st["result_bytes"])
                        walk(st)
                walk(v.request_formula(262144))
        gpt = cache_view_from_analysis(_analysis("openai__gpt-oss-20b")[1])
        swa = gpt.request_formula(262144)["subterms"][0]
        self.assertEqual(swa["symbolic"], "n_layers × (d_head × n_kv_heads + "
                         "d_head × n_kv_heads) × bpe × Min(128, s_kv)")
        self.assertEqual(swa["substituted"], "12 × (512 + 512) × 2 × 128")

    def test_a_dense_cache_is_the_per_token_law_times_the_length(self):
        from backend.arch.eval.cache_view import cache_view_from_analysis
        view = cache_view_from_analysis(_analysis("deepseek-ai__DeepSeek-V3")[1])
        tree = view.request_formula(1048576)
        self.assertEqual(tree["symbolic"],
                         "(n_layers × (d_kv_lora + d_qk_rope) × bpe) × s_kv")
        self.assertEqual(tree["substituted"],
                         "((3 + 58) × (512 + 64) × 2) × 1048576")

    def test_classes_the_same_per_layer_fold_into_one_term(self):
        """One term per distinct per-layer storage: DSv4's invocations of
        one class fold, gpt-oss's GQA and SWA (different laws) stay two."""
        from backend.arch.eval.cache_view import cache_view_from_analysis
        for name, terms in (("deepseek-ai__DeepSeek-V4-Pro", 2),
                            ("openai__gpt-oss-20b", 2)):
            with self.subTest(name=name):
                view = cache_view_from_analysis(_analysis(name)[1])
                tree = view.request_formula(1048576)
                self.assertEqual(len(tree["subterms"]), terms)
                self.assertEqual(len(tree["subterms"]), len(
                    {view._storage_sig(e.cache) for e in view.entries}))
                self._children_sum(tree)

    def test_a_different_format_is_a_different_class(self):
        """Same law, same leaves, one layer's cache stored in another
        format: not the same class, so not folded — and the growth law
        splits too, whatever the labels say."""
        from dataclasses import replace

        from backend.arch.cache_formats import CacheFormat
        from backend.arch.eval.cache_view import cache_view_from_analysis
        view = cache_view_from_analysis(_analysis("deepseek-ai__DeepSeek-V3")[1])
        self.assertEqual(len(view.request_formula(8192)["subterms"]), 1)
        first, *rest = view.entries
        fp8 = replace(first.cache, tensors=tuple(
            replace(t, format=CacheFormat(dtype="float8_e4m3fn"))
            for t in first.cache.tensors))
        mixed = replace(view, entries=(first._replace(cache=fp8), *rest))
        tree = mixed.request_formula(8192)
        self.assertEqual(len(tree["subterms"]), 2)
        self.assertTrue(mixed.mixed_growth_laws)
        self._children_sum(tree)

    def test_terms_add_up_when_layers_are_split(self):
        """Under a layer split the rank holds a fraction of the layers; the
        merged terms take the same fold as the total."""
        from backend.arch.eval.cache_view import cache_view_from_analysis
        view = cache_view_from_analysis(
            _analysis("deepseek-ai__DeepSeek-V3")[1]).on_layer_split(4)
        tree = view.request_formula(8448)
        self.assertEqual(tree["result_bytes"], view.storage_bytes_at(8448))
        self._children_sum(tree)


class TestStorageIdentity(unittest.TestCase):
    """`CacheView.storage_identity`: equal exactly when the caches are."""

    def _id(self, name: str) -> str:
        from backend.arch.eval.cache_view import cache_view_from_analysis
        return cache_view_from_analysis(_analysis(name)[1]).storage_identity

    def test_checkpoints_of_one_cache_share_it(self):
        self.assertEqual(self._id("deepseek-ai__DeepSeek-V3"),
                         self._id("deepseek-ai__DeepSeek-R1"))

    def test_one_spec_with_another_cache_does_not(self):
        """GLM-5.1 and GLM-5.2 resolve to the same spec, and their caches
        still differ: the identity is the cache's, not the spec's."""
        self.assertNotEqual(self._id("zai-org__GLM-5.1"),
                            self._id("zai-org__GLM-5.2"))

    def test_another_format_or_layer_split_is_another_cache(self):
        from dataclasses import replace

        from backend.arch.cache_formats import CacheFormat
        from backend.arch.eval.cache_view import cache_view_from_analysis
        view = cache_view_from_analysis(_analysis("deepseek-ai__DeepSeek-V3")[1])
        first, *rest = view.entries
        fp8 = replace(first.cache, tensors=tuple(
            replace(t, format=CacheFormat(dtype="float8_e4m3fn"))
            for t in first.cache.tensors))
        self.assertNotEqual(view.storage_identity, replace(
            view, entries=(first._replace(cache=fp8), *rest)).storage_identity)
        self.assertNotEqual(view.storage_identity,
                            view.on_layer_split(4).storage_identity)


class TestStorageDtypes(unittest.TestCase):
    """`CacheView.storage_dtypes`: the cache's element dtypes, most bytes
    first, read off each tensor's format."""

    def _dtypes(self, name: str):
        from backend.arch.eval.cache_view import cache_view_from_analysis
        return cache_view_from_analysis(
            _analysis(name)[1]).storage_dtypes(1048576)

    def test_a_bf16_cache_is_bf16(self):
        self.assertEqual(self._dtypes("deepseek-ai__DeepSeek-V3"), ["BF16"])

    def test_a_mixed_cache_lists_every_dtype_by_bytes(self):
        """DeepSeek-V4: the FP8 latent pool outweighs the FP4 index keys,
        and the compressors' FP32 running states are a constant."""
        self.assertEqual(self._dtypes("deepseek-ai__DeepSeek-V4-Pro"),
                         ["FP8 (E4M3)", "FP4", "FP32"])


class TestSlotKindsAndDtypes(unittest.TestCase):
    """Each cached tensor carries the kind its entry declares and the dtype
    that entry resolves to for this checkpoint — the model's own when it
    declares none."""

    def _tensors(self, name: str) -> Dict[str, Tuple[Any, Any]]:
        _resolved, analysis = _analysis(name)
        out: Dict[str, Tuple[Any, Any]] = {}
        for entry in cache_mod.derive(analysis).values.values():
            for tensor in entry.tensors:
                out[tensor.name] = (tensor.kind, tensor.dtype)
        return out

    def test_a_state_dtype_bound_to_the_config_reads_it(self):
        """Nemotron-3 declares `mamba_ssm_cache_dtype: float32`; the SSM state
        is stored at it and the conv window at the model's BF16."""
        name = "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        snapshot_config(name)
        tensors = self._tensors(name)
        self.assertEqual(tensors["recurrent_state"], ("recurrent_state", "F32"))
        self.assertEqual(tensors["conv_state"], ("conv_state", "BF16"))
        self.assertEqual(tensors["k_split_heads"], ("kv", "BF16"))

    def test_a_delegating_entry_names_what_its_subcache_holds_here(self):
        """DeepSeek-V4's indexer caches its compressor's pooled rows as its
        index key; the attention's own compressor pools KV. The compressors'
        running buffers keep their own kind under either."""
        name = "deepseek-ai__DeepSeek-V4-Flash"
        snapshot_config(name)
        tensors = self._tensors(name)
        self.assertEqual(tensors["compressor.pooled_out"],
                         ("compressed_kv", "F8_E4M3"))
        self.assertEqual(tensors["indexer.compressor.pooled_out"],
                         ("index_key", "FP4_E2M1"))
        # The running buffer keeps the fp32 the forward allocates — not the
        # pooled row's format beside it.
        self.assertEqual(tensors["indexer.compressor.kv_state"],
                         ("compressor_state", "F32"))
        self.assertEqual(tensors["kv_concat"], ("kv", "F8_E4M3"))


class TestStoredAsTheForwardQuantisesIt(unittest.TestCase):
    """A cache slot is billed at the format the forward quantises its write
    to, not the dtype of the buffer it lands in. DeepSeek's V4.1 paper
    reports "global KV cache per token" for the family; these are the
    figures that follow from each forward's own quantisation."""

    @staticmethod
    def _per_token(name: str, profile: Any = None) -> float:
        from backend.arch.eval.cache_view import cache_view_from_analysis
        config = snapshot_config(name)
        resolved = shared_loader().match(
            config, model_id=name.replace("__", "/", 1))
        deployment = Deployment(Program.of(resolved))
        if profile is not None:
            deployment = deployment.with_precision(profile)
        analysis = deployment.at(batch=1, input_tokens=1).analyze()
        length = 1 << 40   # past every window: the per-token slope
        return cache_view_from_analysis(analysis).storage_bytes_at(
            length) / length

    def test_v41_flash_stores_nvfp4_kv_and_mxfp4_index_keys(self):
        """Paper: 890 B. Four KV sources (three at ratio 2, one at 1), each
        an NVFP4 latent (256 + 32 scale bytes) and an MXFP4 key (64 + 4)."""
        self.assertAlmostEqual(
            self._per_token("deepseek-ai__DeepSeek-V4.1-Flash"), 890.0, places=3)

    def test_v4_flash_keeps_the_rope_tail_bf16(self):
        """448 FP8 channels + 7 E8M0 scales + a 64-channel BF16 RoPE tail =
        583 per row, MXFP4 index keys 68. Paper: 3514 B — FlashMLA pads the
        seven scales to eight bytes, a kernel layout the forward does not
        state."""
        self.assertAlmostEqual(
            self._per_token("deepseek-ai__DeepSeek-V4-Flash"),
            21 * (583 + 68) / 4 + 20 * 583 / 128, places=3)

    def test_v32_simulates_its_fp8_latent_before_the_write(self):
        """V3.2's forward quantises its latent to FP8 (one F32 scale per
        128) and dequantises it back into a BF16 buffer; the RoPE key stays
        BF16. 512 + 16 + 128 + an FP8 index key (128 + 4) = 788 a layer.
        Paper: 48,068 B."""
        self.assertAlmostEqual(
            self._per_token("deepseek-ai__DeepSeek-V3.2"), 48068.0,
            places=3)

    def test_a_tail_carries_no_scale_and_keeps_its_width(self):
        from backend.arch.cache_formats import CacheFormat
        fmt = CacheFormat.parse({
            "dtype": "F8_E4M3",
            "scale": {"granularity": "per-group", "group": 64,
                      "dtype": "F8_E8M0"},
            "unquantized_tail": {"channels": "d_qk_rope", "dtype": "BF16"},
        }, "test")
        with self.assertRaises(ValueError):
            fmt.bytes_per_element(512)   # unbound: no frame named d_qk_rope
        bound = fmt.bound(64)
        self.assertEqual(bound.bytes_per_element(512) * 512, 448 + 128)
        self.assertEqual(bound.scale_counts(512, 512, 1), (7.0, 0.0))
        with self.assertRaises(ValueError):
            bound.bytes_per_element(32)  # a tail wider than the row
