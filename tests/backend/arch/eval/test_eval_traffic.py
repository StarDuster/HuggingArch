"""`Traffic` — HBM bytes, derived from layers that already exist.

Falsification targets:
  - the weight term must equal the streamed parameters times their storage
    width, plus one row per lookup for a gathered table. It is the one term
    with an independent anchor (the checkpoint), so if it drifts nothing else
    can be trusted;
  - the cache term must equal the per-token footprint times `Lq` times the
    layer count, computed from first principles rather than from the code under
    test;
  - the per-token footprint must be layout-INDEPENDENT. The schema says "with
    B = S = 1", which is a division; slicing two leading axes assumes `[B, S,
    ...]` and a K tensor is `[B, N, S, D]`, so it silently drops the head count
    instead and the cache comes out `n_kv_heads` times small;
  - a fusion must change activation traffic and NOT weight traffic. A fused QKV
    still reads three weight matrices; if the weight term moves, the fused node
    lost its origins' storage.
"""
from __future__ import annotations

import json
import sys
import unittest
import pytest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.fusion import Framework
from backend.arch.eval.program import Program
from backend.arch.eval.dim import value_of
from backend.arch.eval.traffic import (
    Traffic, derive,
)
from backend.arch.eval.transforms import (
    DECODE, Deployment,
)
from tests.snapshots import shared_program, shared_walk
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.arch.eval.precision import PrecisionProfile

pytestmark = pytest.mark.regression  # walks every snapshot


def _framework(name: str) -> Framework:
    binding = get_binding(name)
    if binding is None:
        raise KeyError(f"no framework binding for {name}")
    return Framework.of(binding, FUSION_SCHEMES)


def _single_rank_rule():
    """The rule a bare `Evaluator` walk runs under: one rank, no worlds."""
    from backend.arch.layout.parallel_config import ParallelConfig
    from backend.arch.layout.propagation import LayoutPropagation

    return LayoutPropagation.of(ParallelConfig(), "prefill")

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _program(name: str) -> Program:
    return shared_program(name.replace('__', '/', 1))


#: An attention graph that declares no `cache:`, for the one proposition
#: here that is about a spec rather than about a model: every op a cache
#: could be guessed from, and no declaration. `d_head` is bound off
#: `d_model` and `n_heads` rather than off config, because a spec's `dims:`
#: resolve in order against what is already resolved.
_CACHELESS_SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  n_heads: config.num_attention_heads
  n_kv_heads: config.num_key_value_heads
  d_head: "d_model // n_heads"
  n_vocab: config.vocab_size
blocks:
  attn_blk:
    inputs: [x]
    graph:
      q_proj: {{ op: linear, in: x, dims: {{ in: d_model, out: "n_heads * d_head" }},
                weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      k_proj: {{ op: linear, in: x, dims: {{ in: d_model, out: "n_kv_heads * d_head" }},
                weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      v_proj: {{ op: linear, in: x, dims: {{ in: d_model, out: "n_kv_heads * d_head" }},
                weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      q_split_heads: {{ op: head_split, in: q_proj, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      k_split_heads: {{ op: head_split, in: k_proj, dims: {{ n_heads: n_kv_heads, d_head: d_head }} }}
      v_split_heads: {{ op: head_split, in: v_proj, dims: {{ n_heads: n_kv_heads, d_head: d_head }} }}
      qk_score: {{ op: attention_score, in: [q_split_heads, k_split_heads],
                  dims: {{ n_heads: n_heads, d_head: d_head }} }}
      softmax: {{ op: softmax, in: qk_score, dims: {{ n_heads: n_heads }} }}
      attn_apply: {{ op: attention_apply, in: [softmax, v_split_heads],
                    dims: {{ n_heads: n_heads, d_head: d_head }} }}
      o_merge_heads: {{ op: head_merge, in: attn_apply, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      o_proj: {{ op: linear, in: o_merge_heads, dims: {{ in: "n_heads * d_head", out: d_model }},
                weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      o: {{ op: add, in: [x, o_proj], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: layers, repeat: "2", template: attn_blk, inputs: [embed_tokens],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: norm, op: rmsnorm, dim: d_model, inputs: [layers],
      weight_axes: [{{ replicated: true }}] }}
  - {{ name: lm_head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _cacheless_attention_program() -> Program:
    """`_CACHELESS_SPEC` resolved and built, over its own loader.

    The real component library is copied in because the graph calls library
    components (`head_split`, `head_merge`); the harness is the one
    `test_concat_axis` and `test_ep_as_count.toy_resolved` use."""
    import shutil
    import tempfile
    from backend.arch.loader import SpecLoader
    name = "cacheless_attention_toy"
    components = PROJECT_ROOT / "backend" / "arch" / "components"
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        shutil.copytree(components / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(
            _CACHELESS_SPEC.format(name=name), encoding="utf-8")
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8,
             "num_attention_heads": 2, "num_key_value_heads": 2,
             "vocab_size": 32}, model_id=name)
    return Program.of(resolved)


def _leaf_traffic(analysis: Any) -> Traffic:
    layer = derive(analysis)
    return Traffic.sum(
        value for uid, value in layer.values.items()
        if not analysis.program.by_uid(uid).children)


def _model_traffic(analysis: Any) -> Traffic:
    """The model's accounting roots, each scaled by how many of it there are.

    The same roots `Analysis.total` sums, with `Traffic`'s combine instead of
    `Cost`'s — which is the point of `Program.contributions` handing out a count
    rather than a scaled value. This function used to spell the fold out a third
    time and multiply the five terms by hand.
    """
    layer = derive(analysis)
    return Traffic.sum(
        layer.values[part.node.uid].scaled(part.count)
        for part in analysis.program.contributions
        if part.node.uid in layer.values)


class TestWeightTermIsAnchored(unittest.TestCase):
    def test_weights_are_read_once_and_a_lookup_reads_its_rows(self):
        """The only term with an outside anchor: a decode step streams every
        projection weight once — and a TIED tensor once per reader — while a
        lookup table is gathered, not streamed: it reads one row per lookup
        (vLLM `vocab_parallel_embedding.py` `F.embedding(input_, weight)`).
        Llama ties its LM head to the embedding, so the table appears twice:
        whole for the projection that streams it (`Program.read_aliases` is
        the declaration), and as B rows for the gather."""
        program = _program("meta-llama__Llama-3.2-1B")
        batch = 3
        d = Deployment(program).at(
            batch=batch, input_tokens=2048).during(DECODE, cache_len=2048)
        analysis = shared_walk(d)
        traffic = _model_traffic(analysis)
        params = int(analysis.total().params)
        binding = analysis.binding
        d_model, n_vocab = int(binding["d_model"]), int(binding["n_vocab"])
        table = n_vocab * d_model
        tied = sum(
            int(analysis.cost.values[src].params)
            for sources in program.read_aliases.values()
            for src in sources)
        self.assertEqual(tied, table, "Llama ties its head to the table")
        self.assertEqual(int(traffic.weight_reads),
                         (params - table + tied + batch * d_model) * 2)

    def test_a_quantised_model_reads_fewer_bytes_than_parameters(self):
        d = Deployment(_program("deepseek-ai__DeepSeek-V3")).at(
            batch=1, input_tokens=128)
        analysis = shared_walk(d)
        traffic = _model_traffic(analysis)
        params = int(analysis.total().params)
        self.assertLess(int(traffic.weight_reads), params * 2,
                        "FP8 weights are one byte, not two")
        self.assertGreater(int(traffic.metadata_reads), 0,
                           "and their scales are read too")


class TestCacheTerm(unittest.TestCase):
    def test_matches_first_principles(self):
        """`2 (K and V) x n_kv_heads x d_head x width` per token per layer,
        times `Lq`, times the layer count. Computed here from the binding, not
        from the module under test."""
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            batch=1, input_tokens=2048, output_tokens=256).during(DECODE)
        analysis = shared_walk(d)
        binding = analysis.binding
        layers = sum(t.count for t in analysis.program.blocks)
        per_token = 2 * int(binding["n_kv_heads"]) * int(binding["d_head"]) * 2
        expected = per_token * layers * int(binding["s_kv"])
        self.assertEqual(int(_model_traffic(analysis).kv_reads), expected)

    def test_the_per_token_footprint_is_layout_independent(self):
        """Doubling the batch doubles the cache read and nothing else about the
        per-token footprint. A slice-based implementation would depend on where
        `B` and `S` sit."""
        base = Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            input_tokens=1024).during(DECODE, cache_len=1024)
        one = int(_model_traffic(shared_walk(base.at(batch=1, input_tokens=1024))).kv_reads)
        four = int(_model_traffic(shared_walk(base.at(batch=4, input_tokens=1024))).kv_reads)
        self.assertEqual(four, one * 4)

    def test_prefill_reads_the_cache_on_the_edge_that_reads_it(self):
        """A prefill that attends exactly what it just wrote lands NOTHING in
        `kv_reads`, and that is the rule rather than an omission: attention's
        edge spans what the K/V leaves materialised this step, so the edge
        carries every byte at its own price. `kv_reads` is the COMPLEMENT of
        that — the cache an edge price cannot express — and it fills in only
        once a step reads more context than it wrote.

        The two are the same bytes either way, which the totals show: 1024
        positions attended cost the same whether they were all written now or
        768 of them came back from a prefix cache.
        """
        fresh = Deployment(_program("meta-llama__Llama-3.2-1B")).at(
            batch=1, input_tokens=1024)
        reused = fresh.at(s_prefix_cached=768)
        self.assertEqual(int(_model_traffic(shared_walk(fresh)).kv_reads), 0)
        self.assertGreater(int(_model_traffic(shared_walk(reused)).kv_reads), 0)

        def _attention_bytes(deployment):
            analysis = shared_walk(deployment)
            binding = analysis.binding
            self.assertEqual(int(binding["s_kv"]), 1024)
            traffic = _model_traffic(analysis)
            return int(traffic.act_reads) + int(traffic.kv_reads)

        # Not equal — the reusing step projects Q for 256 positions, not
        # 1024 — but the CACHE half of it is, and it is the only part that
        # the two terms disagree about where to put.
        layers = sum(t.count for t in shared_walk(fresh).program.blocks)
        binding = shared_walk(fresh).binding
        per_token = 2 * int(binding["n_kv_heads"]) * int(binding["d_head"]) * 2
        self.assertEqual(int(_model_traffic(shared_walk(reused)).kv_reads),
                         per_token * layers * 1024)
        self.assertGreater(_attention_bytes(fresh), _attention_bytes(reused))

    def test_a_spec_that_declares_no_cache_reads_none(self):
        """`cache:` is a declaration; a component that does not make it has
        no cache, and nothing is inferred from op names.

        Written against a SPEC that never declares `cache:`, which is what
        the proposition is about. It used to be written by taking the real
        Llama — which DOES declare one — and clearing `kv_cached` /
        `kv_states` / `kv_cache_dtype` off every node, which asserted only
        that erased facts cannot be read: a tautology that tested the edit
        rather than the declaration. It also mutated a program other tests
        held, which is how `test_matches_first_principles` came to report 0
        instead of 71303168 cache bytes.

        The teeth are that the graph is full of ops whose NAMES invite the
        inference — `attention_score`, `attention_apply`, `head_split` —
        and declares no cache, so a walk that read one would have had to
        guess from the names.

        ASKED AT DECODE AND AT A CACHED PREFIX, NOT AT PLAIN PREFILL, and
        that is the whole difference between this test and a vacuous one: a
        prefill step with no cached prefix reads no KV cache whether one is
        declared or not, so `kv_reads == 0` there is a property of the
        workload and holds for Llama too. The control below is what pins
        that down — the same deployments on a model that DOES declare a
        cache must read something — so zero for the cacheless spec is a
        result. Move this back to plain prefill and the control fails."""
        cacheless = _cacheless_attention_program()
        declared = _program("meta-llama__Llama-3.2-1B")
        for label, at in (
                ("decode", lambda d: d.at(batch=1, input_tokens=512)
                 .during(DECODE, cache_len=512)),
                ("prefill over a cached prefix",
                 lambda d: d.at(batch=1, input_tokens=512,
                                s_prefix_cached=256))):
            with self.subTest(phase=label):
                self.assertGreater(
                    int(_model_traffic(
                        shared_walk(at(Deployment(declared)))).kv_reads), 0,
                    "control: a declared cache must read at this deployment, "
                    "or zero below means nothing")
                self.assertEqual(
                    int(_model_traffic(
                        shared_walk(at(Deployment(cacheless)))).kv_reads), 0)
        self.assertTrue(
            any(node.op == "attention_score" for node in cacheless.nodes()),
            "the spec must contain the ops whose names could be guessed from")

    def test_dspark_charges_each_sequence_branch_by_its_own_law(self):
        """The mixed union is not all-cache and not all-activation: its three
        MTP layers gather 1,552/170 selected window pairs per step, while
        the live rows follow the token axis that arrived and gather query by
        row. Both facts come from `Analysis.gathered`; traffic has no second
        mixed-branch switch."""
        from backend.arch.eval import cache as cache_mod

        def walk(prefix=32, *, decode=False):
            deployment = Deployment(
                _program("deepseek-ai__DeepSeek-V4.1-Flash"))
            if decode:
                return shared_walk(deployment.at(
                    batch=1, input_tokens=32, s_verify=5).during(
                    DECODE, cache_len=32))
            return shared_walk(deployment.at(
                batch=1, input_tokens=64, s_prefix_cached=prefix).during(
                "prefill"))

        # Unfused, the score and the apply each gather the selected pairs:
        # twice one gather's bytes.
        for analysis, expected in ((walk(32), 2 * 2458368),
                                   (walk(decode=True), 2 * 269280),
                                   (walk(0), 2 * 3294720)):
            with self.subTest(expected=expected):
                traffic = derive(analysis)
                readers = [(node, traffic[node.uid])
                           for node in analysis.program.nodes()
                           if "dsv41_dspark_attention" in node.display_path
                           and node.op in ("attention_score",
                                           "attention_apply")]
                self.assertEqual(sum(int(value_of(value.kv_reads))
                                     for _n, value in readers), expected)

        analysis = walk(32)
        coverage = cache_mod.context_sources(cache_mod.Edges(analysis))
        all_cache = [node for node in analysis.program.nodes()
                     if node.op == "kv_stream_concat" and node.uid in coverage
                     and "csa2_attention" in node.display_path]
        self.assertTrue(all_cache)
        for node in all_cache:
            found = coverage[node.uid]
            self.assertTrue(found.disjoint)
            self.assertTrue(found.branches)
            self.assertTrue(all(branch.seeds for branch in found.branches))


class TestFusionShowsUpInBytesAndOnlyThere(unittest.TestCase):
    #: A dense model, an MLA one whose rotation is partial, and the two
    #: DeepSeek-V4 generations — the only specs in the corpus that reach the
    #: launches which rotate into a buffer of their own (`fused_inv_rope`,
    #: `fused_qnorm_rope_kv_insert`, the indexer's two writings of one
    #: rotation). Without them the whole quantising half of the rotation
    #: family was unmeasured here.
    MODELS = ("meta-llama__Llama-3.2-1B", "deepseek-ai__DeepSeek-V3",
              "deepseek-ai__DeepSeek-V4-Flash",
              "deepseek-ai__DeepSeek-V4.1-Flash")

    def test_activation_traffic_falls_and_weight_traffic_does_not(self):
        framework = _framework("vllm")
        for name in self.MODELS:
            with self.subTest(model=name):
                d = Deployment(_program(name)).at(batch=1, input_tokens=2048)
                naive = _leaf_traffic(shared_walk(d))
                fused = _leaf_traffic(shared_walk(d.fused(framework)))
                self.assertEqual(naive.weight_reads, fused.weight_reads,
                                 "a fused QKV still reads three matrices")
                self.assertLess(fused.act_reads + fused.act_writes,
                                naive.act_reads + naive.act_writes)

    def test_the_saving_needs_no_arithmetic(self):
        """Nothing subtracts an eliminated buffer. The fused node's traffic is
        computed from its own boundary, and the difference is the saving."""
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=2048)
        naive = _leaf_traffic(shared_walk(d))
        fused = _leaf_traffic(shared_walk(d.fused(_framework("vllm"))))
        saved = ((naive.act_reads + naive.act_writes)
                 - (fused.act_reads + fused.act_writes))
        self.assertGreater(saved / (naive.act_reads + naive.act_writes), 0.2)

    #: Every rotation the op registry knows. A rotation preserves its input's
    #: shape, which is what lets a kernel do it in place.
    ROTATIONS = ("rope", "rope_2d", "rope_emb", "rope_emb_inverse", "mrope",
                 "xdrope")

    def _rotations(self, name: str):
        """Every fused launch of `name` that absorbed a rotation, with what
        the naive walk says about it: the fused node, its traffic row, the
        elements its rope members produced, whether rotating is ALL the
        launch does, and the tensors its boundary publishes on each side."""
        from backend.arch.eval.fusion import _moves_values
        from backend.arch.eval.values import product

        d = Deployment(_program(name)).at(batch=1, input_tokens=2048)
        naive = shared_walk(d)
        analysis = shared_walk(d.fused(_framework("vllm")))
        traffic = derive(analysis)
        for node in analysis.program.nodes():
            origins = analysis.rewrite.sources(node.uid)
            members = [member for uid in origins
                       if (member := naive.program.by_uid(uid)) is not None]
            rotations = [m for m in members if m.op in self.ROTATIONS]
            if not origins or not rotations:
                continue
            # What else the launch computes. A match whose only computing
            # members are rotations is a launch that rotates and nothing
            # else; one that also norms, compresses or stores moves the
            # operands of those too, and the rotated lanes are no longer
            # the whole of what it touches.
            movers = [m for m in members if _moves_values(m.op)]
            shape = analysis.shapes.values[node.uid]
            yield node, traffic.values[node.uid], {
                "rotated": sum(
                    value_of(product(naive.shapes.values[m.uid].out))
                    for m in rotations),
                "rotation_only": all(m.op in self.ROTATIONS for m in movers),
                "published_in": sum(value_of(product(dims))
                                    for dims in shape.ins),
                "published_out": sum(
                    value_of(product(dims)) for dims in
                    (shape.out, *(shape.ports or {}).values())),
                "shape": shape,
                "naive": naive,
            }

    def test_a_rotation_in_place_reads_the_lanes_it_rotates_and_no_others(self):
        """An in-place rope launch READS its rotated lanes, and nothing else.

        The boundary a partial rotation publishes is wider than the kernel:
        `fused_qk_rope_partial_q` absorbs the `slice` that cuts the rope
        lanes out of q and the `concat` that puts them back, so the whole of
        q crosses the node on both sides while the kernel touches only each
        head's last `d_rope` of it. Priced at the published tensor, the node
        billed three times the bytes it moves at a dense MLA shape, and
        disagreed with its own FLOPs, which were the lane's all along.

        The anchor is the NAIVE walk's rope outputs — the lanes the matched
        rope members produced, read off the pre-fusion shapes rather than
        off the boundary under test. Llama's full rotation is the same
        identity with nothing sliced away, so one assertion covers both the
        case that moved and the case that must not.

        Read and write are two assertions because they are two
        derivations: the read subtracts a region per boundary input
        (`Shape.read_elements`), the write recurses through the join that
        re-published it (`fusion._passed_through`'s `unwritten`). One
        passing has never implied the other.
        """
        seen = set()
        for name in self.MODELS:
            for node, row, facts in self._rotations(name):
                if (node.allocates_output or node.output_dtype
                        or not facts["rotation_only"]):
                    continue
                with self.subTest(model=name, op=node.op):
                    self.assertEqual(
                        sum(value_of(edge.elements) for edge in row.in_edges),
                        facts["rotated"],
                        f"{node.op} reads other lanes than the ones it rotates")
                    seen.add(node.op)
        self.assertEqual(seen, {"fused_qk_rope", "fused_qk_rope_partial_q"})

    def test_a_rotation_in_place_writes_the_lanes_it_rotates_and_no_others(self):
        """The same launch WRITES its rotated lanes, and nothing else.

        The write side is the other derivation: the published output is a
        `concat`, and what nobody wrote is whatever its branches' own
        producers left unwritten. Asserted apart from the read so that a
        boundary which recovered one side and not the other is red.
        """
        seen = set()
        for name in self.MODELS:
            for node, row, facts in self._rotations(name):
                if (node.allocates_output or node.output_dtype
                        or not facts["rotation_only"]):
                    continue
                with self.subTest(model=name, op=node.op):
                    self.assertEqual(
                        sum(value_of(term.elements)
                            for term in row.out_terms),
                        facts["rotated"],
                        f"{node.op} writes other lanes than the ones it rotates")
                    seen.add(node.op)
        self.assertEqual(seen, {"fused_qk_rope", "fused_qk_rope_partial_q"})

    def test_a_rotation_into_a_new_buffer_moves_the_whole_tensor(self):
        """A rope launch that writes a buffer of its OWN moves all of it.

        The region a match passes through is a claim about an in-place
        rewrite, and a plan whose kernel allocates its output is making the
        opposite claim — it reads every lane to convert it and writes every
        lane of the new tensor. DeepSeek-V4's `fused_inv_rope` quantises the
        un-rotated o for the wo_a einsum and `fused_qnorm_rope_kv_insert`
        quantises the latent KV row into the paged cache; billed as
        in-place rotations they moved an eighth of what they move, and the
        access popover said so while the row charged the eighth
        (`tests/backend/inference/test_access_formula_states_the_charge`).

        What marks them is the plan's own `allocates_output`, not their names.
        """
        seen = set()
        for name in self.MODELS:
            for node, row, facts in self._rotations(name):
                if not (node.allocates_output or node.output_dtype):
                    continue
                with self.subTest(model=name, op=node.op):
                    self.assertEqual(
                        sum(value_of(edge.elements) for edge in row.in_edges),
                        facts["published_in"],
                        f"{node.op} reads less than the tensors it is handed")
                    self.assertEqual(
                        sum(value_of(term.elements)
                            for term in row.out_terms),
                        facts["published_out"],
                        f"{node.op} writes less than the tensors it publishes")
                    seen.add(node.op)
        self.assertEqual(seen, {"fused_inv_rope", "fused_qnorm_rope_kv_insert",
                                "fused_indexer_q_rope_quant",
                                "fused_indexer_q_rope_quant_split"})

    def test_a_narrowed_price_leaves_the_published_tensor_whole(self):
        """Pricing a region never narrows the tensor the node publishes.

        Two consumers read the boundary for two different things. The byte
        price wants the region the launch moved; shape propagation and the
        measurement harness want the tensor the launch is HANDED — the
        harness allocates it and the kernel strides over it
        (`kernel_bench/launch_trace.disagreement` refuses a cell whose real
        launch moves less than half the elements the walk declared, so a
        boundary narrowed to the rope lanes would have every `d_head /
        d_rope = 3` cell declared at a third of the buffer the kernel
        builds).

        So every narrowed side must still publish exactly what the member
        it came from published before the fusion, and strictly more than it
        is billed for.
        """
        from backend.arch.eval.values import product

        narrowed = 0
        for name in self.MODELS:
            for node, _row, facts in self._rotations(name):
                shape, naive = facts["shape"], facts["naive"]
                moved = shape.moved
                if moved is None:
                    continue
                with self.subTest(model=name, op=node.op):
                    for index, region in moved.reads.items():
                        self.assertLess(
                            value_of(region),
                            value_of(product(shape.ins[index])),
                            f"{node.op} publishes only the region it prices")
                        narrowed += 1
                    for port, region in moved.writes.items():
                        self.assertLess(
                            value_of(region),
                            value_of(product(shape.axes_for(port))),
                            f"{node.op} publishes only the region it prices")
                        narrowed += 1
                    # And the published tensors are the pre-fusion ones.
                    for uid, port in node.output_ports.items():
                        self.assertEqual(
                            tuple(shape.axes_for(port)),
                            tuple(naive.shapes.values[uid].out),
                            f"{node.op} republished {port!r} at another shape")
        self.assertTrue(narrowed, "no boundary was narrowed at all")


#: Every edge at F32: an activation rule of the precision profile.
_ACTIVATIONS_F32 = PrecisionProfile.canonicalize(
    {"activations": [{"format": {"dtype": "F32"}}]})


class TestDtypesAreADeploymentChoice(unittest.TestCase):
    def test_collective_preserves_index_tensor_width(self):
        from backend.arch.eval.traffic import declared_input, declared_output
        from backend.arch.spec_tree import InputRef, SpecIRNode

        index = SpecIRNode(
            name="indices", op="dsa_topk_mask", is_component=False
        )
        collective = SpecIRNode(
            name="gather",
            op="all_gather",
            is_component=False,
            provenance="comm",
            inputs_obj=(InputRef(src=index),),
        )
        from backend.arch.eval.traffic import Written
        self.assertEqual(declared_input(collective), Written(4.0, "I32"))
        self.assertEqual(declared_output(collective), Written(4.0, "I32"))

    def test_activation_width_scales_activation_traffic(self):
        """On a node whose sides ARE activations. An op that declares an
        index-dtype override (embedding's int32 IDs) is priced at the
        declaration, so it is excluded by the same registry fact that
        excludes it in the derivation — not by name."""
        from backend.arch.ops import OP_REGISTRY
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=512)
        analysis = shared_walk(d)
        wide = derive(shared_walk(d.with_precision(_ACTIVATIONS_F32)))
        narrow = derive(analysis)
        uid = next(
            u for u in wide.values
            if (node := analysis.program.by_uid(u)) is not None
            and not node.children
            and (spec := OP_REGISTRY.get(node.op or "")) is not None
            and spec.input_dtype is None
            and wide.values[u].act_reads)
        self.assertEqual(wide.values[uid].act_reads,
                         narrow.values[uid].act_reads * 2)

    def test_an_index_side_keeps_its_declared_width(self):
        """The side an op declares as indices does not move at the activation
        dtype: token IDs are int32 at BF16 and int32 at FP8. The other side
        still scales — the override is per side, not per node."""
        from backend.arch.ops import OP_REGISTRY
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=512)
        analysis = shared_walk(d)
        wide = derive(shared_walk(d.with_precision(_ACTIVATIONS_F32)))
        narrow = derive(analysis)
        uid = next(
            u for u in wide.values
            if (node := analysis.program.by_uid(u)) is not None
            and not node.children
            and (spec := OP_REGISTRY.get(node.op or "")) is not None
            and spec.input_dtype is not None
            and wide.values[u].act_reads)
        self.assertEqual(wide.values[uid].act_reads,
                         narrow.values[uid].act_reads)
        self.assertEqual(wide.values[uid].act_writes,
                         narrow.values[uid].act_writes * 2)

    def test_weights_do_not_take_their_width_from_here(self):
        """`Formats` answers that per tensor. A second answer is how a read path
        came to assume a uniform FP8 while storage used the native BF16."""
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=512)
        analysis = shared_walk(d)
        a = _model_traffic(analysis).weight_reads
        layer = derive(shared_walk(d.with_precision(_ACTIVATIONS_F32)))
        b = Traffic.sum(layer.values[n.uid]
                        for n in analysis.program.ir.top_level
                        if n.uid in layer.values).weight_reads
        top = Traffic.sum(derive(analysis).values[n.uid]
                          for n in analysis.program.ir.top_level
                          if n.uid in derive(analysis).values).weight_reads
        self.assertEqual(b, top)
        self.assertGreater(int(a), 0)


class TestBroadcastIsNotAMaterialisation(unittest.TestCase):
    """A `head_broadcast` is a compile-time stride-0 expansion: the consumer's
    shape carries the post-broadcast extent but the kernel loads the source
    once. Counting the shape is how absorbed MLA came out reading `n_heads`
    times the bytes it does — and absorbed is the form every system uses at
    decode, so the model said the wrong one was cheaper."""

    def _decode_bytes(self, absorbed: bool) -> int:
        from backend.arch.eval.evaluator import Evaluator

        program = _program("deepseek-ai__DeepSeek-V3")
        d = Deployment(program).at(
            batch=1, input_tokens=4096, output_tokens=512).during(DECODE)
        target = d.phase.substitute(program) if absorbed else program
        analysis = Evaluator(target, rule=_single_rank_rule(),
                             workload=d.workload).run(d.binding())
        layer = derive(analysis)
        total = 0
        for template in analysis.program.blocks:
            per = Traffic.sum(layer.values[n.uid] for n in template.roots
                              if n.uid in layer.values)
            total += int(per.total) * template.count
        return total

    def test_absorbed_mla_is_the_cheaper_form_at_decode(self):
        """The whole reason it exists. It costs MORE flops — this is a trade of
        compute for bandwidth — so only bytes can show it."""
        self.assertLess(self._decode_bytes(absorbed=True),
                        self._decode_bytes(absorbed=False))

    def test_a_mixed_concat_drops_nothing(self):
        """Prefill MLA concatenates a real head-split with a broadcast rope.
        `concat` takes the minimum across branches, so a fan-out only counts
        when every branch has it — dropping on the mixed case would undercount
        the materialised K."""
        from backend.arch.eval.cache import Edges
        from backend.arch.eval.traffic import _fanout

        program = _program("deepseek-ai__DeepSeek-V3")
        d = Deployment(program).at(batch=1, input_tokens=1024)
        analysis = shared_walk(d)
        edges = Edges(analysis)
        factors = [
            _fanout(node, analysis.shapes.values, edges=edges)
            for node in analysis.program.nodes()
            if (getattr(node, "op", "") or "") == "concat"
        ]
        self.assertTrue(factors, "DeepSeek-V3 concatenates its K")
        self.assertTrue(all(f == 1 for f in factors),
                        "prefill's k_concat mixes a real head dim with a "
                        "broadcast, so nothing may be dropped")


class TestSeveralReadersOfOneStream(unittest.TestCase):
    """Two unfused kernels that read one stream are two launches, and each
    streams it from HBM: each is billed its own read, and nothing is split
    between them. Fusing them into one launch is what saves the second
    read."""

    def _decode_leaves(self, model: str) -> Dict[str, float]:
        analysis = (shared_walk(Deployment(_program(model))
                    .at(batch=1, input_tokens=1, output_tokens=1)
                    .during(DECODE, cache_len=4096)))
        layer = derive(analysis)
        rows: Dict[str, float] = {}
        for uid, value in layer.values.items():
            node = analysis.program.by_uid(uid)
            if node is None or node.children or not value.kv_reads:
                continue
            rows[node.name] = rows.get(node.name, 0.0) + float(
                value_of(value.kv_reads))
        return rows

    def test_a_sparse_reader_pays_for_what_it_reads(self):
        """Both DSA contractions read only the selected top-k positions.

        Score reads the latent plus the positional feature for each selected
        key, while apply reads the latent value width.  The two bills should
        consequently be of the same order, with score modestly wider; dropping
        the feature width made score look hundreds of times cheaper.
        """
        rows = self._decode_leaves("deepseek-ai__DeepSeek-V3.2-Exp")
        self.assertGreater(rows["qk_score"], rows["attn_apply_latent"])
        self.assertLess(rows["qk_score"], rows["attn_apply_latent"] * 2)

    def test_each_reader_pays_the_law_it_reads(self):
        """Every leaf's `kv_reads` against the laws of the streams it reads,
        on the model whose three streams — latent, positional, indexer — are
        read by three different ops: the latent by the score and the apply,
        the positional half by the score alone, the indexer keys by the
        indexer. At one query a step a gathering kernel's read is the law."""
        from backend.arch.eval.traffic import _kv_reads

        analysis = (shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V3.2-Exp"))
                    .at(batch=1, input_tokens=1, output_tokens=1)
                    .during(DECODE, cache_len=4096)))
        laws: Dict[str, float] = {}
        for uid, law in _kv_reads(analysis).items():
            name = analysis.program.by_uid(uid).name
            laws[name] = laws.get(name, 0.0) + float(value_of(law))
        rows = self._decode_leaves("deepseek-ai__DeepSeek-V3.2-Exp")
        latent, positional = laws["kv_a_layernorm"], laws["k_pe_rope"]
        self.assertAlmostEqual(rows["attn_apply_latent"], latent, places=3)
        self.assertAlmostEqual(rows["qk_score"], latent + positional,
                               places=3)
        self.assertAlmostEqual(rows["indexer_logits"],
                               laws["indexer.key_concat"], places=3)

    def test_two_unfused_readers_each_read_the_latent(self):
        """Absorbed MLA reads the compressed latent twice, once in the score
        and once in the apply, so the latent's law is billed to each. The
        score arrives through a concat and reads the positional half as well;
        that is the OTHER stream, and the score is alone on it."""
        from backend.arch.eval.traffic import _kv_reads

        analysis = (shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V3"))
                    .at(batch=1, input_tokens=1, output_tokens=1)
                    .during(DECODE, cache_len=4096)))
        # Both sides summed over the block templates that share a node name,
        # so the comparison is one model against itself.
        laws: Dict[str, float] = {}
        for uid, law in _kv_reads(analysis).items():
            name = analysis.program.by_uid(uid).name
            laws[name] = laws.get(name, 0.0) + float(value_of(law))
        rows = self._decode_leaves("deepseek-ai__DeepSeek-V3")
        latent, positional = laws["kv_a_layernorm"], laws["k_pe_rope"]
        self.assertAlmostEqual(rows["attn_apply_latent"], latent, places=3)
        self.assertAlmostEqual(rows["qk_score"], latent + positional,
                               places=3)


class TestEveryStreamsLawIsPaidByAName(unittest.TestCase):
    """A declared cache stream is read by every kernel that reads it, and
    each of those kernels is charged its read. Conservation over the whole corpus is the gate, because
    the ways a law goes unpaid are not ones a single model exhibits: a
    recurrent state is read by no edge at all, and a decompressing MLA is read
    by an edge that spans less than the stream.

    The state half was live under-counting until 2026-08 — a Mamba layer's
    decode IS that read, and it was free — and no test could see it, because
    every test asked one model whether its own numbers were self-consistent.

    Each kernel pays its own read: unfused, attention's score and apply are
    two launches that each stream the cache, and a cache another layer
    shares (DeepSeek-V4.1's compressed KV, written by the source layers and
    read by every layer after) is read by each reading layer's kernels. So
    the stream's law counts once per leaf that reads it (`kv_access`), and
    once when nothing does — a recurrent state, read by its carrier. At one
    query a step a gathering kernel's read is the law too.
    """

    def _shortfall(self, name: str) -> float:
        from backend.arch.eval.cache import kv_access, resident
        from backend.arch.eval.traffic import _kv_reads

        analysis = (shared_walk(Deployment(_program(name))
                    .at(batch=1, input_tokens=1, output_tokens=1)
                    .during(DECODE, cache_len=4096)))
        layer = derive(analysis)
        billed = sum(float(value_of(v.kv_reads))
                     for uid, v in layer.values.items()
                     if (node := analysis.program.by_uid(uid)) is not None
                     and not node.children)
        held = resident(analysis)
        reading: Dict[int, set] = {}
        for reader, access in kv_access(analysis).values.items():
            for ref in access.reads:
                if ref in held:
                    reading.setdefault(held[ref][2].node, set()).add(reader)
        laws = sum(float(value_of(law)) * max(1, len(reading.get(node, ())))
                   for node, law in _kv_reads(analysis).items())
        return laws - billed

    def test_the_corpus_conserves(self):
        names = sorted(p.stem for p in SNAPSHOTS.glob("*.json")
                       if "config" in json.loads(p.read_text()))
        self.assertGreater(len(names), 30, "the sweep found no corpus")
        for name in names:
            with self.subTest(name):
                short = self._shortfall(name)
                self.assertAlmostEqual(
                    short, 0.0, delta=1.0,
                    msg=f"{name} bills {short:,.0f} bytes a step a layer "
                        f"less than its cache streams declare. Either an "
                        f"op that reads a stream stopped being charged for "
                        f"it, or a new stream reaches no reader.")


class TestIntensity(unittest.TestCase):
    def test_flops_per_byte(self):
        self.assertEqual(Traffic(act_reads=100).intensity(400), 4)
        self.assertEqual(Traffic().intensity(400), 0)


class TestPairStored(unittest.TestCase):
    def test_short_formats_leave_rep_none_rather_than_guessing(self):
        from backend.arch.eval.analysis import TensorRow, pair_stored
        from backend.arch.eval.formats import (
            Representation, Representations, TensorFormat)

        class _T:
            def __init__(self, name):
                self.name = name

        weight, bias = _T("w"), _T("bias")
        fmt = Representation(saved=TensorFormat(storage_dtype="BF16"))
        rows = list(pair_stored(7, (weight, bias),
                                (Representations.of(fmt),), count=3))
        self.assertEqual(rows[0], TensorRow(7, weight, fmt, 3))
        self.assertIsNone(rows[1].rep)
        self.assertIs(rows[1].tensor, bias)

    def test_a_layer_split_tensor_is_one_row_per_segment(self):
        """`count` already means "this tensor, at this format, x this many
        copies", so a tensor stored two ways across its node's layers is two
        rows whose counts sum to the node's."""
        from backend.arch.eval.analysis import pair_stored
        from backend.arch.eval.formats import (
            Representation, Representations, TensorFormat)

        class _T:
            def __init__(self, name):
                self.name = name

        reps = Representations((
            Representation(saved=TensorFormat(storage_dtype="F8_E4M3"),
                           layers=tuple(range(3, 88))),
            Representation(saved=TensorFormat(storage_dtype="BF16"),
                           layers=(88, 89, 90)),
        ))
        rows = list(pair_stored(7, (_T("w"),), (reps,), count=88))
        self.assertEqual([(row.count, row.storage_dtype) for row in rows],
                         [(85, "F8_E4M3"), (3, "BF16")])
        self.assertEqual(sum(row.count for row in rows), 88)
        self.assertEqual({row.tensor_index for row in rows}, {0})
        # Each row says which segment it is, so a per-rank projection takes
        # that segment's own placement and copies (`layout.persistent` keys
        # its artifacts `(uid, tensor_index, segment_index)`).
        self.assertEqual([row.segment_index for row in rows], [0, 1])

    def test_a_single_copy_bill_weights_each_segment_by_its_share(self):
        """A caller that bills ONE copy and multiplies by its own count (a
        fused kernel reading its origin's weights) gets each segment's share
        of that copy: the shares sum to 1, so the bill is the per-copy
        average and exact once multiplied."""
        from fractions import Fraction

        from backend.arch.eval.analysis import pair_stored
        from backend.arch.eval.formats import (
            Representation, Representations, TensorFormat)

        class _T:
            def __init__(self, name):
                self.name = name

        reps = Representations((
            Representation(saved=TensorFormat(storage_dtype="F8_E4M3"),
                           layers=tuple(range(3, 88))),
            Representation(saved=TensorFormat(storage_dtype="BF16"),
                           layers=(88, 89, 90)),
        ))
        counts = [row.count for row in pair_stored(7, (_T("w"),), (reps,))]
        self.assertEqual(counts, [Fraction(85, 88), Fraction(3, 88)])
        self.assertEqual(sum(counts), 1)


class TestQuantisedOutputScaleTile(unittest.TestCase):
    """The F8_128x4 tile padding of the record vLLM's q-quantising a-norm
    writes is a machine-readable declaration (query_quant.py:31-33: token
    rows rounded to 128, 32-value groups to 4, the padded positions stored
    as zeros)."""

    def test_the_generic_group_law_underbills_the_tiled_record(self):
        from backend.arch.activation_formats import ACTIVATION_FORMATS
        from backend.arch.eval.traffic import _format_width
        fmt = ACTIVATION_FORMATS["fp8_token_group32_ue8m0"]
        # The smoke cell: 64 tokens x 128-wide q (4 groups). The launch
        # writes ceil(64/128)*128 * ceil(4/4)*4 = 512 scale bytes; the plain
        # per-32 share bills 256.
        plain = _format_width(fmt, 64 * 128, 128)
        tiled = _format_width(fmt, 64 * 128, 128, (128, 4))
        self.assertEqual(value_of(plain - 1) * 64 * 128, 256)
        self.assertEqual(value_of(tiled - 1) * 64 * 128, 512)
        # A 32-wide q row (one group, padded to four): 512, not 64.
        wide32 = _format_width(fmt, 64 * 32, 32, (128, 4))
        self.assertEqual(value_of(wide32 - 1) * 64 * 32, 512)
        # A row that is no whole number of groups still takes a scale for
        # its partial last group: 72 wide is 3 groups of 32, padded to 4 by
        # the tile, so 128 x 4 scale bytes; untiled, 64 x 3.
        self.assertEqual(
            value_of(_format_width(fmt, 64 * 72, 72, (128, 4)) - 1) * 64 * 72,
            512)
        self.assertEqual(
            value_of(_format_width(fmt, 64 * 72, 72) - 1) * 64 * 72, 64 * 3)
        f32 = ACTIVATION_FORMATS["fp8_token_group128"]
        # 200 wide at groups of 128 is 2 F32 scales a row, not 200/128.
        self.assertEqual(
            value_of(_format_width(f32, 3 * 200, 200) - 1) * 3 * 200,
            3 * 2 * 4)
        # A cell already on the tile boundary is unchanged.
        self.assertEqual(float(_format_width(fmt, 128 * 128, 128, (128, 4))),
                         float(_format_width(fmt, 128 * 128, 128)))

    def test_the_plan_declares_the_tile_and_the_fused_node_carries_it(self):
        """On the serving graph, where the q cast is a node the plan takes
        in as a required member, the launch carries the plan's tile."""
        from backend.inference.serving import rewrite_serving_graph
        self.assertEqual(
            FUSION_SCHEMES["fused_q_kv_rmsnorm_quant"].output_scale_tile,
            {"rows": 128, "groups": 4})
        walk = shared_walk(Deployment(_program(
            "deepseek-ai__DeepSeek-V4.1-Flash")).at(input_tokens=64))
        from backend.inference.framework_bindings import ServingDeployment
        serving = rewrite_serving_graph(
            walk, framework_binding=get_binding("vllm").deployed(
                ServingDeployment.at(gpu_sku="B200",
                                     activation_dtype="BF16")))
        nodes = [n for n in serving.program.nodes()
                 if n.op == "fused_q_kv_rmsnorm_quant"]
        self.assertTrue(nodes)
        self.assertEqual({n.output_scale_tile for n in nodes}, {(128, 4)})
        for node in nodes:
            self.assertIn("quantise_activation",
                          {serving.pre_program.by_uid(uid).op
                           for uid in node.origins})

    def test_the_launch_writes_the_tiled_record(self):
        """The tile is priced where the launch's write is: its q record at
        64 tokens is the FP8 elements plus the scale sidecar padded to
        128 rows x a multiple of 4 groups, beside the kv it writes in its
        own dtype. Counted by hand, not by `_format_width`."""
        from backend.inference.framework_bindings import ServingDeployment
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        from tests.snapshots import resolve_snapshot
        result = analyze(
            resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash"),
            parallel=ParallelConfig(),
            framework_binding=get_binding("vllm").deployed(
                ServingDeployment.at(gpu_sku="B200",
                                     activation_dtype="BF16")),
            batch_size=1, avg_input_tokens=64, avg_output_tokens=0,
            breakdown=True)
        rows = [r for r in result.per_op_timing
                if r["op"] == "fused_q_kv_rmsnorm_quant"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["output_activation_format"],
                             "fp8_token_group32_ue8m0")
            self.assertEqual(row["shape_out"], [1, 64, 1280])
            self.assertEqual(row["shape_outs"], [[1, 64, 512]])
            q = 64 * 1280 * 1                 # E4M3 elements
            scales = 128 * 40 * 1             # 64 rows -> 128; 40 groups
            kv = 64 * 512 * 2                 # BF16
            self.assertEqual(row["act_out_bytes_per_rank"],
                             (q + scales + kv) * int(row["count_per_step"]))


if __name__ == "__main__":
    unittest.main()


def test_a_join_with_one_non_empty_operand_moves_nothing():
    """A concat whose other operands hold no element hands its one operand
    on uncopied (vLLM v0.30.0 step3_vl.py:704-707: no `torch.cat` for an
    image with no crops); with two non-empty operands it copies both."""
    program = shared_program("stepfun-ai/Step-3.7-Flash")

    def rows_join(crops):
        (encode,) = Deployment(program).at(batch=1, media={
            "pixel_values": {"n_img": 1},
            "patch_pixel_values": {"n_crops": crops}}).encoders()
        analysis = shared_walk(encode)
        (join,) = [n for n in analysis.program.walked_nodes()
                   if n.name == "vision.rows"]
        return derive(analysis).values[join.uid]

    none, two = rows_join(0), rows_join(2)
    assert (value_of(none.act_reads), value_of(none.act_writes)) == (0, 0)
    assert value_of(two.act_reads) > 0 and value_of(two.act_writes) > 0
