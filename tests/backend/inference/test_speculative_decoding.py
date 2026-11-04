"""Speculative decoding: target-verify modeling.

By default the model scopes speculative decoding to the *target
verify* step — the draft model (MTP head) is NOT separately costed
unless the request asks for it (`speculative_draft`, pinned in
test_speculative_draft.py). The default is the memory-bound-decode
insight made explicit: a verify forward
over K + 1 positions reads the same weights as a single decode step (≈ free
in the weight-bound regime) yet emits 1 + αK accepted tokens on average.

Concretely, with `speculative_draft_tokens = K > 0` draft tokens:
  • the decode binding sets Sq = K + 1, Lq = cache + K — every backbone
    attention op sees a (K+1)-wide query block over the (slightly longer)
    cache (`phase_binding`, phases.py).
  • throughput amortizes draft+accept into one rate via
    effective_tokens = (1 + αK)·B (`phases.Speculation`).
  • MTP / aux blocks stay filtered out of per_op_timing — by default the
    draft cost is unmodeled, so no MTP rows appear at any K.

These tests pin that contract: verify-shape reaches the per-op rows,
the binding math is exact, and the draft head is never costed.
"""
from __future__ import annotations

import sys
import unittest

from tests.snapshots import snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.phases import phase_binding

# ---------------------------------------------------------------------------
# 1. Phase binding — the Sq / Lq / tokens_per_step math that drives both
#    the verify-shape and the throughput amortization.
# ---------------------------------------------------------------------------

class TestDecodeBindingSpecVerify(unittest.TestCase):

    def test_ar_baseline_k1(self):
        """K=1 is plain autoregressive decode: one query, full cache,
        every pair live (s_pairs = s_kv), one token per step."""
        b = phase_binding(
            phase="decode", avg_input_tokens=4096, avg_output_tokens=256,
            s_verify=1,
        )
        self.assertEqual(b["s_query"], 1)
        # cache_len = avg_input + avg_output/2 = 4096 + 128 = 4224.
        self.assertEqual(b["s_kv"], 4224)
        self.assertEqual(b["tokens_per_step"], 1)
        self.assertEqual(b["s_verify"], 1)
        self.assertEqual(b["s_pairs"], b["s_kv"])

    def test_spec_verify_k3_sets_verify_shape(self):
        """K=3 verify: Sq widens to K, Lq extends by K−1 so the K queries
        attend to the cache plus the in-flight verify block."""
        b = phase_binding(
            phase="decode", avg_input_tokens=4096, avg_output_tokens=256,
            s_verify=3,
        )
        self.assertEqual(b["s_query"], 3)
        self.assertEqual(b["s_kv"], 4224 + (3 - 1))  # 4226
        self.assertEqual(b["tokens_per_step"], 3)
        self.assertEqual(b["s_verify"], 3)
        # The verify block's pairs: each query sees the cache and the
        # queries before it — less than the full K-wide matrix.
        self.assertEqual(b["s_pairs"], 3 * 4223 + 3 * 4 // 2)
        self.assertLess(b["s_pairs"], 3 * b["s_kv"])


# ---------------------------------------------------------------------------
# 2. End-to-end: verify-shape reaches per_op_timing, draft never costed.
# ---------------------------------------------------------------------------

class TestSpecDecodeEndToEnd(unittest.TestCase):

    def _estimate(self, model_id, K, alpha=0.7):
        from fastapi.testclient import TestClient
        from backend.api import app
        client = TestClient(app)
        resp = client.post(
            "/inference/estimate",
            json={
                "model_id": model_id,
                "gpu_sku": "H100-SXM",
                # Eight GPUs, stated as the mesh they form: the GPU count
                # is the parallel config's rank count, and a bare
                # `num_gpus` beside a one-rank mesh is refused.
                "parallel": {"tp_attention": 8, "tp_dense_ffn": 8, "ep_moe": 8},
                "avg_input_tokens": 4096,
                "avg_output_tokens": 256,
                "speculative_draft_tokens": K,
                "acceptance_rate": alpha,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    @staticmethod
    def _attention_query_axes(per_op_rows):
        """Query-position axis (shape_in[0][2]) for every attention-family
        op row that carries a 4-D [B, N, Sq, D] activation input."""
        out = []
        for r in per_op_rows:
            op = r["op"]
            if not ("attention" in op or "flash" in op or "score" in op):
                continue
            si = r.get("shape_in") or []
            if si and isinstance(si[0], list) and len(si[0]) == 4:
                out.append((op, si[0][2]))
        return out

    def test_mimo_verify_shape_widens_with_k(self):
        snapshot_config("XiaomiMiMo__MiMo-V2.5.json")
        j0 = self._estimate("XiaomiMiMo/MiMo-V2.5", 0)
        j2 = self._estimate("XiaomiMiMo/MiMo-V2.5", 2)

        axes0 = self._attention_query_axes(j0["per_op_timing_decode"])
        axes2 = self._attention_query_axes(j2["per_op_timing_decode"])
        self.assertTrue(axes0, "no attention rows found at K=0")
        self.assertTrue(axes2, "no attention rows found at K=2")
        # AR baseline: every attention query axis = 1.
        for op, q in axes0:
            self.assertEqual(q, 1, f"{op} query axis should be 1 at K=0")
        # Spec verify: every attention query axis = K + 1 = 3.
        for op, q in axes2:
            self.assertEqual(q, 3, f"{op} query axis should be 3 at K=2")

    def test_mtp_draft_never_costed(self):
        """MTP (aux_kind=mtp) carries the draft head. The model does NOT
        cost the draft, so no MTP row appears in per_op_timing at any K —
        including K>0 where a naive implementation might let it leak in."""
        snapshot_config("XiaomiMiMo__MiMo-V2.5.json")
        layers = set()
        for K in (0, 2):
            j = self._estimate("XiaomiMiMo/MiMo-V2.5", K)
            layers.add(j["kv_cache"]["num_layers"])
            mtp_rows = [
                r for r in j["per_op_timing_decode"]
                if "mtp" in (r.get("template_name") or "").lower()
                or "mtp" in r["op"].lower()
            ]
            self.assertEqual(
                mtp_rows, [],
                f"MTP draft must not be costed (K={K}); leaked {len(mtp_rows)} rows",
            )
        # Nor is its cache held: a verify without a draft keeps the
        # backbone's cache only, at every K.
        self.assertEqual(len(layers), 1)

    def test_spec_decode_independent_of_mtp_presence(self):
        """A model with no MTP head (Llama-3.2-3B) still produces a valid
        verify-shape decode at K>0 — spec decode is a workload knob, not a
        capability gated on the spec carrying a draft module."""
        snapshot_config("meta-llama__Llama-3.2-3B.json")
        j = self._estimate("meta-llama/Llama-3.2-3B", 2, alpha=0.7)
        axes = self._attention_query_axes(j["per_op_timing_decode"])
        self.assertTrue(axes, "Llama produced no attention decode rows at K=2")
        for op, q in axes:
            self.assertEqual(q, 3, f"{op} query axis should be 3 at K=2")

    def test_walk_formula_carries_the_verify_pairs(self):
        """The serving walk binds n_verify as the query axis and the verify
        block's live pairs as `s_pairs`, so a spec-decode attention formula
        names them instead of a folded number."""
        from backend.arch.eval.dim import expr_of, names_of, value_of
        from backend.inference.phases import evaluate_phase_step
        from tests.snapshots import resolve_snapshot

        resolved = resolve_snapshot("meta-llama__Llama-3.2-3B")
        step, bind = evaluate_phase_step(
            resolved, phase="decode", batch=1,
            avg_input_tokens=1024, avg_output_tokens=256, s_verify=3)
        self.assertEqual(value_of(step.analysis.binding["n_verify"]), 3)
        self.assertTrue(step.analysis.binding["s_query"].is_("n_verify"))
        score = next(n for n in step.analysis.program.nodes()
                     if n.op == "attention_score")
        flops = step.analysis.cost.values[score.uid].flops
        self.assertIn("s_pairs", names_of(flops))
        self.assertNotIn(str(int(bind["s_pairs"])),
                         str(expr_of(flops)).replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
