"""Tests for the /inference/compute-graph endpoint — the final
post-passes IR (parallelism comm injection + kernel fusion) serialized
for the inference page's ELK visualization.

Covers:
  * Endpoint shape: {model, framework, parallel, graph:{nodes:[...]}}.
  * Node kinds: compute / fused / comm, with fused_op on fused nodes.
  * TP=1 vs TP=8 graph difference: comm nodes appear at TP>1, and the
    norm fusion node switches fused_add_rmsnorm → allreduce_fused_rmsnorm.
  * Error paths: missing model, bad framework, invalid parallel config.
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.api import app

# -- the endpoint's answer, once per distinct request ----------------------
#
# This file asked /inference/compute-graph 57 times for 31 distinct requests:
# a dozen tests each check one thing about the same TP=8 Llama graph, and
# building one costs seconds (the walk, the comm graft, the fusion rewrite
# and the serialization). The answer is a pure function of the query — the
# handler reads nothing else — and a response is read-only to a test:
# `.json()` parses the body afresh on every call, so no test can hand
# another a document it edited. So the module keeps each distinct answer and
# this file's tests share it.
#
# The memo lives exactly as long as this module and is reachable only from
# it; no other file sees a shared object.

_CLIENT: Optional[TestClient] = None
_ANSWERS: Dict[Tuple[Tuple[str, Any], ...], Any] = {}


def compute_graph(**params) -> Any:
    """The endpoint's answer to this request, built once per distinct one."""
    global _CLIENT
    params.setdefault("model_id", "meta-llama/Llama-2-7b-hf")
    key = tuple(sorted(params.items()))
    if key not in _ANSWERS:
        if _CLIENT is None:
            _CLIENT = TestClient(app)
        _ANSWERS[key] = _CLIENT.get("/inference/compute-graph", params=params)
    return _ANSWERS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_answers():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the answers it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    global _CLIENT
    yield
    _ANSWERS.clear()
    _CLIENT = None


class TestComputeGraphEndpoint(unittest.TestCase):
    def _graph(self, **params):
        return compute_graph(**params)

    @staticmethod
    def _flatten(nodes):
        """Depth-first flatten of nested compound nodes."""
        out = []
        for n in nodes:
            out.append(n)
            out.extend(TestComputeGraphEndpoint._flatten(n.get("children", [])))
        return out

    @staticmethod
    def _all_nodes(data):
        """Every node in the whole-model graph, flattened through compounds."""
        return TestComputeGraphEndpoint._flatten(data["graph"]["nodes"])

    def test_payload_carries_no_transport_keys(self):
        """The keys `_node_build.strip_transport_keys` removes are read
        between passes and stripped once the edges are built; at TP=2 the
        graph also holds grafted comm nodes and fused expansions, which must
        be stripped the same."""
        TRANSPORT_KEYS = ("port_labels", "output_kind", "identity_source")
        r = self._graph(tp_attention=2, tp_dense_ffn=2)
        self.assertEqual(r.status_code, 200, r.text[:300])
        found = []

        # A declaration card is wire format (`views.diff.declared`), even
        # where one of its facts spells the same as a transport key. Only
        # renderer-node keys are transport; the card's contents are payload.
        found.extend((n["id"], key)
                     for n in self._all_nodes(r.json())
                     for key in TRANSPORT_KEYS if key in n)
        self.assertEqual(found, [])

    def test_returns_whole_model_backbone(self):
        r = self._graph(tp_attention=1, tp_dense_ffn=1)
        self.assertEqual(r.status_code, 200, r.text[:300])
        data = r.json()
        self.assertEqual(data["model"], "meta-llama/Llama-2-7b-hf")
        top = data["graph"]["nodes"]
        ids = {n["id"] for n in top}
        # Whole-model backbone present at the top level.
        self.assertIn("embed_tokens", ids)
        self.assertIn("norm", ids)
        self.assertIn("lm_head", ids)
        # At least one collapsed block stack.
        block_nodes = [n for n in top if n["op"] == "block_stack"]
        self.assertTrue(block_nodes, "no block-stack compound emitted")
        for b in block_nodes:
            self.assertTrue(b["expandable"])
            self.assertTrue(b["children"])
            self.assertIn("count", b)
        # Backbone is wired: norm/lm_head reference upstream producers.
        norm = next(n for n in top if n["id"] == "norm")
        self.assertTrue(norm["inputs"], "norm has no backbone input edge")

    def test_emits_top_level_edges_for_renderer(self):
        # dag.js consumes `graph.edges[]` as the authoritative root-edge
        # source — re-deriving from `inputs[]` in the renderer was
        # removed as part of the single-source-of-truth refactor.
        # Regression guard: without this list the whole-model overview
        # rendered every top-level node disconnected (embed_tokens /
        # layer-stacks / norm / lm_head floating with no arrows).
        data = self._graph(tp_attention=1, tp_dense_ffn=1).json()
        edges = data["graph"].get("edges")
        self.assertIsInstance(edges, list, "graph.edges[] missing")
        self.assertTrue(edges, "no top-level edges emitted")
        for e in edges:
            self.assertIn("src", e)
            self.assertIn("dst", e)
            self.assertIn("id", e)
        ids = {n["id"] for n in data["graph"]["nodes"]}
        # Every edge endpoint must reference an existing top-level node.
        for e in edges:
            self.assertIn(e["src"], ids, f"dangling src {e['src']!r}")
            self.assertIn(e["dst"], ids, f"dangling dst {e['dst']!r}")
        # Backbone chain present: embed_tokens → block_stack → norm → lm_head.
        edge_pairs = {(e["src"], e["dst"]) for e in edges}
        block_id = next(n["id"] for n in data["graph"]["nodes"]
                        if n["op"] == "block_stack")
        self.assertIn(("embed_tokens", block_id), edge_pairs)
        self.assertIn((block_id, "norm"), edge_pairs)
        self.assertIn(("norm", "lm_head"), edge_pairs)

    def test_nodes_use_inspector_format(self):
        # Reuse contract: nodes arrive in the Inspector's dag.js dict
        # shape (`style`, not a bespoke `kind`), produced by the shared
        # NodeBuilder. A regression to a parallel format breaks this.
        data = self._graph(tp_attention=8, tp_dense_ffn=8).json()
        for n in self._all_nodes(data):
            self.assertIn("style", n)
            self.assertIn("id", n)
            self.assertIn("op", n)
            self.assertNotIn("kind", n)  # the old bespoke field is gone

    def test_node_ids_globally_unique(self):
        # DSv3 puts dense / MoE / MTP blocks in ONE graph; each block
        # repeats self_attn / mlp / input_layernorm with the same
        # spec_id. Block-internal ids MUST be namespaced so dag.js's
        # id-keyed maps don't collapse the duplicates (which broke
        # expand-clicks and tangled edges). Lock global uniqueness.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        ids = [n["id"] for n in self._all_nodes(data)]
        dupes = [i for i, cnt in Counter(ids).items() if cnt > 1]
        self.assertEqual(dupes, [], f"duplicate node ids: {dupes[:5]}")

    def test_dsv3_dense_then_moe_then_mtp_branch(self):
        # DSv3's `layers` stack splits into dense (×3) + MoE (×58)
        # blocks, chained dense → MoE; the MTP head branches off the
        # backbone, at the point its spec declares.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        top = data["graph"]["nodes"]
        blocks = [n for n in top if n["op"] == "block_stack"]
        # 3 block stacks: dense, MoE, MTP.
        self.assertGreaterEqual(len(blocks), 3)
        aux = [n for n in blocks if n.get("aux_kind")]
        self.assertTrue(aux, "MTP aux block missing")
        # Where the MTP reads is the spec's `inputs:` to state, not this
        # test's to guess: DSv3 declares `inputs: [norm]`, and that is what
        # both frameworks do — vLLM and sglang each return the post-`norm`
        # hidden state from the backbone and hand THAT to the MTP module as
        # `previous_hidden_states` (which applies its own `hnorm` on top).
        # The assertion is therefore that the edge is the declared one, not
        # that it lands on a `__block__` id; an earlier version required the
        # last main block, which was the pre-`aux_kind` heuristic and drew a
        # tensor nobody passes.
        # Formal 0 is that hidden state; the head's other formal is the token
        # ids it embeds (`inputs: [norm, input_ids]` on its layer), one input
        # per formal in formal order, each bound to its own input node.
        mtp = aux[0]
        self.assertEqual(["norm", "input_ids"], mtp["inputs"])
        bound = {c["formal"]: c["id"].rsplit("/", 1)[-1]
                 for c in mtp["children"] if "formal" in c}
        self.assertEqual({0: "x", 1: "input_ids"}, bound)
        # And the declared edge has to LAND: every input the aux block names
        # must be a node the top-level graph published, or the renderer draws
        # an arrow from nothing.
        top_ids = {n["id"] for n in top}
        for pid in mtp["inputs"]:
            self.assertIn(pid, top_ids)

    def test_tp1_has_fused_no_comm(self):
        data = self._graph(tp_attention=1, tp_dense_ffn=1).json()
        nodes = self._all_nodes(data)
        styles = {n.get("style") for n in nodes}
        self.assertIn("fused", styles)
        self.assertNotIn("comm", styles)
        fused_ops = {n["op"] for n in nodes if n.get("style") == "fused"}
        self.assertIn("fused_add_rmsnorm", fused_ops)
        self.assertNotIn("allreduce_fused_rmsnorm", fused_ops)

    def test_tp8_introduces_comm_and_switches_norm_fusion(self):
        data = self._graph(tp_attention=8, tp_dense_ffn=8).json()
        nodes = self._all_nodes(data)
        styles = {n.get("style") for n in nodes}
        self.assertIn("comm", styles)
        comm_ops = {n["op"] for n in nodes if n.get("style") == "comm"}
        self.assertTrue(comm_ops <= {
            "all_reduce", "reduce_scatter", "all_gather", "all_to_all",
        })
        fused_ops = {n["op"] for n in nodes if n.get("style") == "fused"}
        self.assertIn("allreduce_fused_rmsnorm", fused_ops)
        self.assertNotIn("fused_add_rmsnorm", fused_ops)

    def test_fused_ops_are_compounds_of_their_subgraph(self):
        # A fused op is modeled as a compound that REPLACED its subgraph:
        # collapsed it's the fused kernel, expanded it reveals the
        # pre-fusion nodes it absorbed. Lock the headline cases.
        data = self._graph(tp_attention=8, tp_dense_ffn=8).json()
        fused = {
            n["op"]: n for n in self._all_nodes(data)
            if n.get("style") == "fused"
        }
        # flash_attention expands to score → softmax → apply.
        fa = fused.get("flash_attention")
        self.assertIsNotNone(fa)
        self.assertTrue(fa.get("fused"))
        self.assertTrue(fa.get("expandable"))
        child_ops = {c["op"] for c in fa["children"]}
        self.assertEqual(
            child_ops, {"attention_score", "softmax", "attention_apply"},
        )
        # allreduce_fused_rmsnorm expands to all_reduce → add → rmsnorm —
        # the injected all_reduce MUST appear (resolved via comm_tree,
        # not the raw spec pre_tree which lacks comm nodes).
        ar = fused.get("allreduce_fused_rmsnorm")
        self.assertIsNotNone(ar)
        ar_ops = {c["op"] for c in ar["children"]}
        self.assertEqual(ar_ops, {"all_reduce", "add", "rmsnorm"})

    def test_block_input_reaches_fused_residual_add(self):
        # Regression: the post-attention residual `add` consumes the block
        # input residual `x`. Under TP>1 it fuses into
        # allreduce_fused_rmsnorm (all_reduce → add → rmsnorm), so the add
        # gains an in-scope producer (the all_reduce) and is no longer a
        # scope entry — without an explicit producer for `x`, its residual
        # input dangled (the "add has only one input" bug). Every
        # block_stack must expose a resolvable `x` input node, and that
        # node must feed the fused post_attention_layernorm compound so the
        # frontend can fan the residual into the nested add.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        stacks = [n for n in data["graph"]["nodes"]
                  if n.get("op") == "block_stack"]
        self.assertTrue(stacks)
        checked_compound = False
        for bs in stacks:
            xs = [c for c in bs["children"] if c.get("op") == "input"]
            self.assertTrue(
                xs, f"block {bs['id']!r} has no block-input node")
            # Input nodes must be shaped (build_edges refuses shapeless
            # arrows).
            for x in xs:
                self.assertTrue(
                    x.get("tensor_out"),
                    f"block-input {x['id']!r} unshaped")
            x_ids = {x["id"] for x in xs}
            # The residual must produce a real edge into the fused
            # post_attention_layernorm compound (whichever child_edge src
            # is a block-input node).
            comp = next(
                (c for c in bs["children"]
                 if c.get("op") == "allreduce_fused_rmsnorm"), None)
            if comp is None:
                continue
            checked_compound = True
            srcs_into_comp = {
                e["src"] for e in (bs.get("child_edges") or [])
                if e["dst"] == comp["id"]
            }
            self.assertTrue(
                x_ids & srcs_into_comp,
                f"residual x does not feed fused compound {comp['id']!r} "
                f"in block {bs['id']!r}; add would render input-starved",
            )
            # And the absorbed add still declares two inputs (residual +
            # all_reduce) — the connectivity we're protecting.
            add = next(c for c in comp["children"] if c["op"] == "add")
            self.assertEqual(
                len(add.get("inputs") or []), 2,
                "fused residual add lost an input",
            )
        self.assertTrue(checked_compound,
                        "no allreduce_fused_rmsnorm compound exercised")

    def _assert_no_dangling(self, data):
        """Every node input ref must resolve to a real producer id
        somewhere in the graph (exact id, or the `.name` suffix form
        `build_edges` accepts). A dangling ref renders as a missing edge
        or, worse, silently drops a data dependency."""
        nodes = self._all_nodes(data)
        ids = {n["id"] for n in nodes}

        def resolvable(ref, port):
            base = ref[: -(len(port) + 1)] if port and ref.endswith("." + port) else ref
            if base in ids or ref in ids:
                return True
            return any(i.endswith("." + base) for i in ids)

        dangling = []
        for n in nodes:
            ins = n.get("inputs") or []
            ports = n.get("input_ports") or [""] * len(ins)
            for i, r in enumerate(ins):
                p = ports[i] if i < len(ports) else ""
                if not resolvable(r, p):
                    dangling.append((n["op"], n["id"].split("/")[-1], r.split("/")[-1]))
        self.assertEqual(dangling, [], f"dangling input refs: {dangling}")

    def test_no_dangling_refs_across_architectures(self):
        # Whole-graph connectivity guard. Covers the three fusion/render
        # ref-rewrite regressions:
        #   A. fused_qk_rope named ports — `attention_score` reading the
        #      pre-fusion `q_rope`/`k_rope` (GQA/MHA models).
        #   B. `repeat: 0` shared_expert — `moe_out` reading a dropped node
        #      (Qwen3 has no shared expert).
        #   C. router primary-fold — `softmax_scores` reading `gate.linear`
        #      after the wrapper absorbed the primary linear.
        for model in ("deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-30B-A3B",
                      "meta-llama/Llama-2-7b-hf"):
            for tp in (1, 8):
                for phase in ("prefill", "decode"):
                    data = self._graph(model_id=model, tp_attention=tp,
                                       tp_dense_ffn=tp, ep_moe=tp,
                                       phase=phase).json()
                    with self.subTest(model=model, tp=tp, phase=phase):
                        self._assert_no_dangling(data)

    def test_collision_victim_block_renders_fused(self):
        # DSv3 dense + MoE blocks share the template-local spec_id
        # `self_attn.attn_apply`; the framework rewrite changes only
        # one (cost is still correct — count_per_step aggregates across
        # both). The graph must still render the MoE block's attention as
        # the fused compound, NOT leave qk_score/softmax as standalone
        # siblings duplicated alongside the fused node.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        # No standalone attention_score / softmax anywhere — every
        # attention is collapsed into a flash_attention compound.
        def unfused_attention(nodes, inside_fused=False):
            bare = []
            for node in nodes:
                nested = inside_fused or node.get("style") == "fused"
                if node["op"] in ("attention_score", "softmax") and not nested:
                    bare.append(node)
                bare.extend(unfused_attention(
                    node.get("children", []), nested,
                ))
            return bare

        bare = unfused_attention(data["graph"]["nodes"])
        self.assertEqual(bare, [])
        # The only legitimate occurrences are inside fused compounds'
        # children; assert none appear as a fused-sibling duplicate by
        # checking no attention_score is both present as a node AND has a
        # fused flash_attention sibling sharing its parent. Simplest:
        # count standalone (non-child-of-fused) score nodes is 0 at the
        # block level — verified structurally below.
        fused_attn = [n for n in self._all_nodes(data)
                      if n.get("style") == "fused" and n["op"] == "flash_attention"]
        self.assertTrue(fused_attn)
        for fa in fused_attn:
            child_ops = {c["op"] for c in fa.get("children", [])}
            self.assertEqual(
                child_ops, {"attention_score", "softmax", "attention_apply"},
            )

    def test_view_ops_absorbed_into_elementwise_not_attention(self):
        # vLLM convention: a zero-cost view op is absorbed into the
        # adjacent elementwise/materialize op that executes it (rope /
        # norm / concat), NOT into a GEMM or the attention kernel. So
        # the head-split feeding rope is absorbed by its rope host
        # (standalone `rope`, or the `fused_qk_rope` / `fused_qk_norm_rope`
        # compound that wraps it when the framework binding fuses Q/K
        # rope into one kernel). The head-reshape feeding/following
        # attention/o_proj stays a standalone view node (a free
        # reinterpret in the model forward).
        data = self._graph(tp_attention=8, tp_dense_ffn=8).json()  # llama
        nodes = self._all_nodes(data)
        by_op = {}
        for n in nodes:
            by_op.setdefault(n["op"], []).append(n)
        # On Llama under the vLLM binding, fused_qk_rope fires and wraps both
        # rope nodes; the head_split feeders are absorbed into the
        # fused compound. (For a model/framework combo without the
        # fusion, the absorption would land on the standalone rope.)
        rope_host_ops = ("rope", "fused_qk_rope", "fused_qk_norm_rope")
        hosts_with_absorbed_split = [
            n for n in nodes
            if n["op"] in rope_host_ops
            and any(c["op"] == "head_split" for c in (n.get("children") or []))
        ]
        self.assertTrue(
            hosts_with_absorbed_split,
            "a rope-family host should absorb the head-split feeding it",
        )
        # The head_merge feeding o_proj (a GEMM) is NOT absorbed — it
        # remains a standalone node, not a child of o_proj.
        for op in ("linear", "qkv_parallel_linear"):
            for n in by_op.get(op, []):
                child_ops = {c["op"] for c in n.get("children", [])}
                self.assertNotIn("head_merge", child_ops)
                self.assertNotIn("head_split", child_ops)

    def test_view_absorption_preserves_input_ports(self):
        # `inputs` and `input_ports` are parallel arrays (see
        # rendering/_node_build.py:161-166). When _absorb_view_ops
        # rewrites a host's `inputs` to skip absorbed views and inline
        # their producers, it must rewrite `input_ports` in lockstep.
        # Otherwise port info on the absorbed view's source — e.g.
        # `kv_a_proj_with_mqa.output_0` feeding `kv_latent_slice` which
        # feeds `kv_a_layernorm` — gets silently downgraded to '',
        # and the frontend's `expandDst` can't route the cross-scope
        # edge to the right multi-output port.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        for n in self._all_nodes(data):
            ports = n.get("input_ports")
            self.assertIsNotNone(
                ports,
                f"node {n['id']!r} missing input_ports parallel array",
            )
            self.assertEqual(
                len(ports), len(n["inputs"]),
                f"node {n['id']!r}: input_ports length {len(ports)} != "
                f"inputs length {len(n['inputs'])}",
            )
            # Each input ref ending in `.output_<N>` (multi-output port)
            # must carry that port string in the parallel array — the
            # core Bug-A invariant. `_absorb_view_ops` regressed this
            # by rewriting `inputs` without `input_ports`.
            for ref, port in zip(n["inputs"], ports, strict=True):
                if "." in ref and ref.rsplit(".", 1)[-1].startswith("output_"):
                    self.assertEqual(
                        port, ref.rsplit(".", 1)[-1],
                        f"node {n['id']!r}: input ref {ref!r} carries "
                        f"port suffix but input_ports has {port!r}",
                    )
        # And concretely: the kv_a_layernorm consumes
        # kv_a_proj_with_mqa.output_0 after kv_latent_slice absorption.
        kv_a_ln = next(
            (n for n in self._all_nodes(data)
             if n["id"].endswith("self_attn.kv_a_layernorm")),
            None,
        )
        self.assertIsNotNone(kv_a_ln, "kv_a_layernorm missing from DSv3 graph")
        self.assertEqual(len(kv_a_ln["inputs"]), 1)
        self.assertEqual(
            kv_a_ln["input_ports"], ["output_0"],
            f"kv_a_layernorm must carry the kv_a_proj_with_mqa.output_0 "
            f"port through view absorption; got {kv_a_ln['input_ports']!r}",
        )

    def test_broadcast_absorbed_into_concat(self):
        # DSv3 MLA: the k_pe head-broadcast (GQA-MQA KV expansion) is
        # absorbed into the K-reconstruction concat — matching vLLM's
        # `k = k.new_empty(); k[.., nope:] = k_pe` in-place broadcast
        # write into the materialized K buffer.
        data = self._graph(model_id="deepseek-ai/DeepSeek-V3",
                            tp_attention=8, tp_dense_ffn=8).json()
        concats = [n for n in self._all_nodes(data) if n["op"] == "concat"]
        # At least one concat absorbed a head_broadcast.
        absorbed_broadcast = any(
            any(c["op"] == "head_broadcast" for c in n.get("children", []))
            for n in concats
        )
        self.assertTrue(
            absorbed_broadcast,
            "k_pe head_broadcast should be absorbed into the K concat",
        )

    def test_components_are_expandable_compounds(self):
        # Inside a block compound, nested components (self_attn / mlp)
        # must themselves be expandable compounds — the recursive
        # collapse the Inspector provides.
        data = self._graph(tp_attention=1, tp_dense_ffn=1).json()
        compounds = [
            n for n in self._all_nodes(data)
            if n.get("expandable") and n.get("children")
        ]
        # At least the block stack + inner components.
        self.assertGreaterEqual(len(compounds), 2)

    def test_node_inputs_are_id_lists(self):
        data = self._graph(tp_attention=8, tp_dense_ffn=8).json()
        for n in self._all_nodes(data):
            self.assertIsInstance(n["inputs"], list)
            for pid in n["inputs"]:
                self.assertIsInstance(pid, str)

    def test_missing_model_is_400(self):
        r = compute_graph(model_id="")
        self.assertEqual(r.status_code, 400)

    def test_unknown_framework_is_404(self):
        r = self._graph(framework="noframework")
        self.assertEqual(r.status_code, 404)

    def test_invalid_parallel_config_is_400(self):
        # tp_attention × cp × dp must equal per-stage; a lone mismatched
        # knob combination triggers ParallelConfig.validate errors.
        # tp_attention=3 with tp_dense_ffn=2 → group mismatch.
        r = self._graph(tp_attention=3, tp_dense_ffn=2)
        self.assertEqual(r.status_code, 400, r.text[:200])

    def test_phase_defaults_to_prefill(self):
        # Backward compat: existing callers don't pass `phase`, so the
        # default response must carry "prefill" and match the explicit
        # `?phase=prefill` topology.
        r = self._graph(tp_attention=1, tp_dense_ffn=1)
        self.assertEqual(r.status_code, 200, r.text[:200])
        self.assertEqual(r.json()["phase"], "prefill")

    def test_invalid_phase_is_400(self):
        r = self._graph(phase="midfill")
        self.assertEqual(r.status_code, 400, r.text[:200])

    # -- the workload the graph is analysed at ---------------------------

    def _kv_nodes(self, data):
        return [n for n in self._all_nodes(data) if n.get("kv")]

    def test_request_without_workload_is_one_token_at_batch_one(self):
        """A request that names no workload is the one-token, batch-one step
        it always was: the same response as naming those defaults."""
        for phase in ("prefill", "decode"):
            bare = self._graph(phase=phase)
            named = self._graph(
                phase=phase, avg_input_tokens=1, avg_output_tokens=0,
                prefix_cached_tokens=0, speculative_draft_tokens=0, batch_size=1)
            self.assertEqual(bare.status_code, 200, bare.text[:200])
            self.assertEqual(bare.json(), named.json())
            kv = self._kv_nodes(bare.json())
            self.assertTrue(kv)
            self.assertEqual({(n["kv"]["cache_len"], n["kv"]["batch"]) for n in kv},
                             {(1, 1)})

    def test_kv_amounts_follow_the_requested_step(self):
        """Prefill of N tokens writes N positions per request; decode reads
        the estimate's average cache (input + output/2) for every request of
        the batch."""
        one = self._kv_nodes(self._graph().json())[0]["kv"]
        pre = self._graph(avg_input_tokens=1024, batch_size=4)
        self.assertEqual(pre.status_code, 200, pre.text[:200])
        kv = self._kv_nodes(pre.json())[0]["kv"]
        self.assertEqual((kv["cache_len"], kv["batch"]), (1024, 4))
        for w1, w in zip(one["writes"], kv["writes"], strict=True):
            self.assertEqual(w["bytes_per_request"], 1024 * w1["bytes_per_request"])
        self.assertEqual(kv["read_bytes_per_step"], 4 * 1024 * one["read_bytes_per_step"])

        dec = self._graph(phase="decode", avg_input_tokens=1024,
                          avg_output_tokens=1024, batch_size=8)
        self.assertEqual(dec.status_code, 200, dec.text[:200])
        kv = self._kv_nodes(dec.json())[0]["kv"]
        self.assertEqual((kv["cache_len"], kv["batch"]), (1536, 8))
        self.assertEqual(kv["read_bytes_per_step"], 8 * 1536 * one["read_bytes_per_step"])

    def test_cp_prefill_draws_at_a_length_cp_divides(self):
        r = self._graph(cp=2, avg_input_tokens=1024)
        self.assertEqual(r.status_code, 200, r.text[:200])
        self.assertEqual(r.json()["phase"], "prefill")
        kv = self._kv_nodes(r.json())
        self.assertTrue(kv)
        self.assertEqual(kv[0]["kv"]["cache_len"], 1024)

    def test_cp_prefill_at_a_length_cp_cannot_split_names_the_reason(self):
        """The failure is a 400 whose body says, as a code, that the length
        does not split over the sequence world, and `detail` names the model
        input whose rows do not split."""
        for tokens in (1, 1023):
            params = {"cp": 2} if tokens == 1 else {"cp": 2, "avg_input_tokens": tokens}
            r = self._graph(**params)
            self.assertEqual(r.status_code, 400, r.text[:200])
            body = r.json()
            self.assertEqual(body["reason"], "sequence_not_divisible")
            self.assertEqual((body["extent"], body["world"], body["degree"]),
                             (tokens, "cp", 2))
            self.assertEqual(
                body["detail"],
                f"input_ids: token extent {tokens} does not divide the "
                f"sequence world degree 2")

    def test_other_failures_carry_no_reason(self):
        r = self._graph(tp_attention=3, tp_dense_ffn=2)
        self.assertEqual(r.status_code, 400, r.text[:200])
        self.assertNotIn("reason", r.json())

    def test_mla_decode_swaps_to_absorbed_topology(self):
        # Decode-time serving stacks (vLLM / SGLang / TRT-LLM) bind the
        # absorbed MLA component so kv_b_proj is folded into per-head
        # Q/O matmuls and KV cache stays in the latent (R+P) space.
        # Prefill keeps the decompressed form (explicit kv_b_proj
        # linear). DSv3 is the canonical MLA-using model.
        pre = self._graph(
            model_id="deepseek-ai/DeepSeek-V3",
            tp_attention=8, tp_dense_ffn=8, phase="prefill",
        ).json()
        dec = self._graph(
            model_id="deepseek-ai/DeepSeek-V3",
            tp_attention=8, tp_dense_ffn=8, phase="decode",
        ).json()
        self.assertEqual(pre["phase"], "prefill")
        self.assertEqual(dec["phase"], "decode")

        def _ops(data):
            return {n["op"] for n in self._all_nodes(data)}

        def _spec_ids(data):
            return {n.get("id", "") for n in self._all_nodes(data)}

        pre_ids = _spec_ids(pre)
        dec_ids = _spec_ids(dec)

        def _kv_b_linears(data):
            return [n for n in self._all_nodes(data)
                    if "kv_b_proj" in n.get("id", "") and n["op"] == "linear"]

        # Prefill runs kv_b_proj as a linear; decode keeps the stored
        # kv_b_proj and runs it as the q_absorbed_nope + v_absorbed
        # per-head matmuls instead (`module: kv_b_proj`).
        self.assertTrue(_kv_b_linears(pre),
                        "prefill MLA graph should run kv_b_proj as a linear")
        self.assertFalse(_kv_b_linears(dec),
                         "decode MLA graph should not run kv_b_proj as a linear")
        # Absorbed form's signature ops only appear in decode.
        self.assertTrue(
            any("q_absorbed_nope" in sid for sid in dec_ids),
            "decode MLA graph should contain q_absorbed_nope",
        )
        self.assertTrue(
            any("v_absorbed" in sid for sid in dec_ids),
            "decode MLA graph should contain v_absorbed",
        )
        # And prefill must NOT carry those absorbed-only leaves.
        self.assertFalse(
            any("q_absorbed_nope" in sid for sid in pre_ids),
        )

    def test_non_mla_model_unchanged_across_phases(self):
        # Llama-style attention has no decode-time topology swap; the
        # graph must be identical in both phases (sanity that the swap
        # is gated on MLA presence, not applied unconditionally).
        pre = self._graph(tp_attention=8, tp_dense_ffn=8, phase="prefill").json()
        dec = self._graph(tp_attention=8, tp_dense_ffn=8, phase="decode").json()
        pre_ids = sorted(n.get("id", "") for n in self._all_nodes(pre))
        dec_ids = sorted(n.get("id", "") for n in self._all_nodes(dec))
        self.assertEqual(pre_ids, dec_ids)

    def test_nodes_carry_tensor_shapes_for_edge_labels(self):
        # The frontend ELK renderer derives each edge's `dim:` label from
        # the SOURCE node's `tensor_out` (dag.js:749). Without this field
        # on the wire the graph draws naked arrows. Regression: backend
        # used to call NodeBuilder.build_roots without splatting engine
        # shapes onto the resulting dicts, so every edge label was blank.
        data = self._graph(tp_attention=1, tp_dense_ffn=1).json()
        flat = self._all_nodes(data)
        # Compute leaves should carry shape — pick concrete attention/MLP
        # operands so a regression to "no shapes anywhere" trips here.
        kinds_with_shape = {"linear", "rmsnorm", "embedding",
                             "silu_and_mul", "activation"}
        leaves_with_shape = [
            n for n in flat
            if n.get("op") in kinds_with_shape and n.get("tensor_out")
        ]
        self.assertGreater(
            len(leaves_with_shape), 5,
            "expected several leaf compute nodes to carry tensor_out; got "
            f"{len(leaves_with_shape)}. Sample missing: "
            + repr([
                n["id"] for n in flat
                if n.get("op") in kinds_with_shape and not n.get("tensor_out")
            ][:5]),
        )
        # embed_tokens is always at the top of the backbone; its tensor_out
        # is the residual stream width — a stable anchor.
        embed = next((n for n in flat if n.get("id") == "embed_tokens"), None)
        self.assertIsNotNone(embed)
        self.assertTrue(embed.get("tensor_out"))
        self.assertTrue(embed["tensor_out"].startswith("["))

    def test_multi_output_fused_leaves_emit_tensor_outs(self):
        # vLLM-fused merged_column_parallel_linear has multiple output
        # ports (kv_a / q_a stacked) — its main shape_out is None but
        # `shape_outs[port]` carries the per-port shapes the frontend
        # needs for edges that consume a specific port. Make sure the
        # multi-output channel survives serialization.
        data = self._graph(
            model_id="deepseek-ai/DeepSeek-V3",
            tp_attention=8, tp_dense_ffn=8, ep_moe=8,
        ).json()
        flat = self._all_nodes(data)
        mcpl = [n for n in flat if n.get("op") == "merged_column_parallel_linear"]
        self.assertTrue(mcpl, "expected merged_column_parallel_linear nodes")
        with_ports = [n for n in mcpl if n.get("tensor_outs")]
        self.assertTrue(
            with_ports,
            "merged_column_parallel_linear should carry tensor_outs per port",
        )
        sample = with_ports[0]["tensor_outs"]
        self.assertTrue(all(v.startswith("[") for v in sample.values()))

    def test_a_compound_and_the_nodes_it_absorbed_carry_their_own_shapes(self):
        data = self._graph(model_id="google/gemma-4-12B-it").json()
        compounds = [n for n in self._all_nodes(data)
                     if n.get("op") == "fused_qk_rope"]
        self.assertTrue(compounds, "gemma-4 should fuse q/k rope at vLLM")
        for c in compounds:
            self.assertIsNone(
                c.get("tensor_out"),
                f"{c['id']} publishes only {sorted(c.get('tensor_outs') or ())}"
                f" — it has no main output to report",
            )
            self.assertEqual(
                sorted(c.get("tensor_outs") or ()), ["k_out", "q_out"])
            absorbed = c.get("children") or []
            self.assertTrue(absorbed, f"{c['id']} absorbed nothing")
            for child in absorbed:
                self.assertTrue(
                    child.get("tensor_out"),
                    f"absorbed {child['id']} lost its pre-fusion shape",
                )
            self.assertEqual(
                {child["tensor_out"] for child in absorbed},
                set(c["tensor_outs"].values()),
                "the compound's ports are exactly what its members produce",
            )

    def test_parallel_fused_linears_expand_with_per_rank_weight_dims(self):
        data = self._graph(
            model_id="meta-llama/Llama-2-7b-hf",
            tp_attention=8, tp_dense_ffn=8,
        ).json()
        qkvs = [n for n in self._all_nodes(data)
                if n.get("op") == "qkv_parallel_linear"]
        self.assertTrue(qkvs, "llama-2-7b should fuse qkv at vLLM")
        self.assertTrue(all(n.get("style") == "fused" for n in qkvs), qkvs)
        kids = [
            c for n in qkvs for c in (n.get("children") or [])
            if c.get("op") == "linear" and str(c.get("id") or "").endswith("q_proj")
        ]
        self.assertTrue(kids, "fused qkv absorbed no q_proj")
        for kid in kids:
            self.assertTrue((kid.get("detail") or "").startswith("W:"), kid)
            self.assertNotIn(
                "→4,096", kid.get("detail") or "",
                f"absorbed {kid['id']} still shows the unsharded out dim",
            )

    def test_tp_linears_show_per_rank_weight_dims(self):
        # Overlay is keyed by uid from the walk's per-rank shapes.
        # o_proj is row-parallel: TP=8 shards the 4,096 input, not the
        # output. The unsharded label would be W:4,096→4,096.
        data = self._graph(
            model_id="meta-llama/Llama-2-7b-hf",
            tp_attention=8, tp_dense_ffn=8,
        ).json()
        o_proj = [
            n for n in self._all_nodes(data)
            if n.get("op") == "linear" and str(n.get("id")).endswith("o_proj")
        ]
        self.assertTrue(o_proj, "llama-2-7b block has no o_proj")
        for n in o_proj:
            self.assertEqual(
                n.get("detail"), "W:512→4,096",
                f"{n['id']} is not showing the TP=8 weight",
            )

    def test_serialize_post_tree_requires_the_pre_fusion_walk(self):
        from backend.rendering._compute_graph import serialize_post_tree
        with self.assertRaises(TypeError):
            serialize_post_tree(object(), object())


class TestAbsorbHostDeclaration(unittest.TestCase):
    def test_view_hosts_follow_the_declarations(self):
        """`ops.hosts_views` derives which kernels take an adjacent view
        into their prologue/epilogue. It still answers yes for every op and
        fused op the hand-kept lists it replaced named, and no for a GEMM,
        the attention kernel and a fused op built around either."""
        from backend.arch.ops import OP_REGISTRY, hosts_views
        from backend.inference.fusion_plans import FUSION_SCHEMES

        rotations = [name for name, spec in OP_REGISTRY.items()
                     if spec.rotary_position_axes]
        for op in ("rmsnorm", "layernorm", "silu", "gelu", "gelu_tanh",
                   "mul", "add", "sigmoid", "concat", *rotations):
            self.assertTrue(hosts_views((op,)), op)
        for op in ("linear", "attention_score", "attention_apply",
                   "transpose", "all_reduce"):
            self.assertFalse(hosts_views((op,)), op)
        for fused in ("silu_and_mul", "gelu_and_mul", "fused_qk_rope",
                      "fused_qk_norm_rope", "fused_add_rmsnorm",
                      "allreduce_fused_rmsnorm"):
            self.assertTrue(
                hosts_views(FUSION_SCHEMES[fused].naive_pattern.ops()), fused)
        for fused in ("flash_attention", "fused_moe", "gated_act_ffn"):
            self.assertFalse(
                hosts_views(FUSION_SCHEMES[fused].naive_pattern.ops()), fused)


if __name__ == "__main__":
    unittest.main()


class TestEveryRefNamesItsProducer(unittest.TestCase):
    """A ref resolves by identity — the fused node's port, a stack's last
    block group, a stage across a pipeline cut — or the graph refuses to
    build. Nothing is dropped and nothing falls back to the backbone."""

    def _graph(self, model_id, **params):
        r = compute_graph(model_id=model_id, **params)
        self.assertEqual(r.status_code, 200, r.text[:300])
        return r.json()["graph"]

    @staticmethod
    def _flat(nodes):
        return TestComputeGraphEndpoint._flatten(nodes)

    def test_swiglu_reads_both_ports_of_the_merged_projection(self):
        for model in ("zai-org/GLM-5", "meta-llama/Llama-3.2-3B"):
            nodes = self._flat(self._graph(model)["nodes"])
            merged = {n["id"] for n in nodes
                      if n.get("op") == "merged_column_parallel_linear"}
            gated = [n for n in nodes if n.get("op") == "silu_and_mul"]
            self.assertTrue(gated, model)
            for fused in gated:
                ports = {}
                for child in fused["children"]:
                    for ref in child.get("inputs") or ():
                        base, _, port = ref.rpartition(".")
                        if base in merged:
                            ports[child["op"]] = port
                with self.subTest(model=model, node=fused["id"]):
                    self.assertEqual(ports, {"silu": "output_0",
                                             "mul": "output_1"})

    def test_the_step_receives_what_the_vision_encoder_hands_over(self):
        """The vision tower runs in its own encode walk; the step reads its
        merger output as a shaped boundary input."""
        graph = self._graph("Qwen/Qwen3.8-27B")
        nodes = self._flat(graph["nodes"])
        handed = next(n for n in nodes
                      if n["id"] == "model.visual")
        self.assertEqual(handed["op"], "input")
        self.assertTrue(handed.get("tensor_out"))
        self.assertFalse([n for n in nodes if n.get("aux_kind") == "vision"])
        self.assertTrue([e for e in graph["edges"]
                         if e["src"] == handed["id"] and e.get("shape")])

    def test_a_stack_is_read_at_its_last_block_group(self):
        graph = self._graph("deepseek-ai/DeepSeek-V4-Flash")
        stacks = [n["id"] for n in graph["nodes"]
                  if n.get("op") == "block_stack" and not n.get("aux_kind")]
        self.assertGreater(len(stacks), 1)
        readers = [n for n in self._flat(graph["nodes"])
                   if n["id"].startswith("hc_head")
                   and any(ref in stacks for ref in n.get("inputs") or ())]
        self.assertTrue(readers)
        for n in readers:
            self.assertEqual(
                [ref for ref in n["inputs"] if ref in stacks], [stacks[-1]],
                n["id"])

    def test_a_later_stage_receives_what_an_earlier_one_sent(self):
        graph = self._graph("meta-llama/Llama-3.2-3B", pp=2, pp_rank=1)
        recv = [n for n in graph["nodes"] if n.get("op") == "recv"]
        self.assertEqual([n["label"] for n in recv], ["recv from stage 0"])
        ids = {n["id"] for n in graph["nodes"]}
        for n in graph["nodes"]:
            for ref in n.get("inputs") or ():
                self.assertIn(ref.rsplit(".", 1)[0] if ref not in ids else ref,
                              ids, n["id"])


class TestAFusedModelLevelSpanReadsItsInputs(unittest.TestCase):
    """Qwen3.8-Flash-Next's final hyper-connection mixer is written on the
    model spine, and vLLM's `hc_silu` absorbs two of its ops. The fused node
    reads what the first absorbed op read; no ref names the absorbed one."""

    def test_the_final_mixer_kernel_reads_the_down_projection(self):
        r = compute_graph(model_id="Qwen/Qwen3.8-Flash-Next",
                          framework="vllm")
        self.assertEqual(r.status_code, 200, r.text[:300])
        nodes = TestComputeGraphEndpoint._flatten(r.json()["graph"]["nodes"])
        prefix = "model.language_model.hyper_connection_mixer."
        kernel = [n for n in nodes if n.get("op") == "hc_silu"
                  and n["id"].startswith(prefix)]
        self.assertEqual(len(kernel), 1, [n["id"] for n in kernel])
        self.assertEqual(kernel[0]["inputs"],
                         [prefix + "input_mix_weight_down"])
        for n in nodes:
            self.assertNotIn(prefix + "mix_scale", n.get("inputs") or ())
