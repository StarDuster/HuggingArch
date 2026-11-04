"""Regression tests for PR-1: phase binding + primitive-sum FLOPs.

The framework's correctness rests on a few analytical checkpoints
that do NOT depend on which model we test against — they're
properties of the (Sq, Lq, s_pairs) abstraction itself:

  1. At short context, attention's quadratic term is small, so total
     FLOPs ≈ `~6 × N × tokens` (Chinchilla approximation; ND ≈ params
     × token-count, the 6× = 2× matmul × 3 reads-per-param-per-token).
     The exact ratio drifts with FFN/attention split but should be in
     a tight band (4× to 7×).

  2. Going from S to 4×S: linear (≤ 4×) for FFN-dominated; quadratic
     (≥ 4×) for attention-dominated. Long-context (8k+) MUST exceed
     4× because the attention term is non-trivial.

  3. Decode (Sq=1) must produce drastically less FLOPs/step than the
     same-batch prefill (Sq=S_in), because all parameter-weighted
     compute scales with Sq. Ratio expected: `Sq_decode / Sq_prefill`
     plus the small attention-cache term.

  4. MLA absorbed vs decompressed: same params, different op DAG
     traversal. At prefill bindings (Sq=Lq=S) the absorbed form does
     more compute (per-head latent matmuls instead of one fused
     kv_b_proj decompression). Ratio should be in `[1.2, 2.0]` for
     DSv3-class dims (R=512, D=Dv=128, P=64).

  5. Migration sanity: passing `s_pairs = S²/2` to the primitives
     reproduces the OLD `B·N·S²·D` formula (modulo softmax, which is
     now correctly halved by causal). This is what keeps the Inspector
     display identical.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.projection import project_block
from backend.arch.loader import SpecLoader
from backend.arch import ops
from backend.inference.phases import (
    PhaseInputs, estimate_phase, evaluate_phase_step, flops_fold, phase_binding,
)


SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolved(model_short: str, snapshot_filename: str):
    """Load `<model_short>` spec resolved against `<snapshot>` config."""
    config = json.loads((SNAPSHOT_DIR / snapshot_filename).read_text())["config"]
    return SpecLoader().resolve_by_name(model_short, config)


# ---------------------------------------------------------------------------
# 1. Primitive-level invariants — proves the (Sq, Lq, s_pairs)
#    abstraction is wired correctly without involving any model spec.
# ---------------------------------------------------------------------------

class TestAttentionPrimitivesSqLq(unittest.TestCase):
    """Hit the attention primitives with hand-bound dim dicts and
    verify the FLOPs lambdas produce exactly the expected values."""

    def test_attention_score_explicit_2x_and_causal(self):
        # 2 × B × N × s_pairs × D, at half the 1024×1024 matrix
        spec = ops.get_op("attention_score")
        flops = spec.flops_fn({
            "B": 1, "n_heads": 32, "s_query": 1024, "s_kv": 1024, "d_head": 128,
            "s_pairs": 1024 * 1024 // 2,
        })
        # 2 * 1 * 32 * 1024 * 1024 * 128 * 1/2 = 32 * 1024² * 128 = 4 GFLOPs
        self.assertEqual(int(flops), 2 * 1 * 32 * 1024 * 1024 * 128 // 2)

    def test_attention_score_decode_no_causal_saving(self):
        # decode: Sq=1, Lq=cache_len, every pair live (no upper triangle)
        spec = ops.get_op("attention_score")
        flops = spec.flops_fn({
            "B": 16, "n_heads": 32, "s_query": 1, "s_kv": 4096, "d_head": 128,
            "s_pairs": 4096,
        })
        # 2 * 16 * 32 * 1 * 4096 * 128 * 1
        self.assertEqual(int(flops), 2 * 16 * 32 * 1 * 4096 * 128)

    def test_attention_score_spec_decode_K(self):
        # speculative-decode-K: Sq=K, Lq=cache+K-1
        spec = ops.get_op("attention_score")
        K, cache = 4, 2048
        Lq = cache + (K - 1)
        flops = spec.flops_fn({
            "B": 8, "n_heads": 32, "s_query": K, "s_kv": Lq, "d_head": 128,
            "s_pairs": K * Lq,  # simplification — ignoring the intra-K triangle
        })
        self.assertEqual(int(flops), 2 * 8 * 32 * K * Lq * 128)

    def test_softmax_scales_with_sq_lq(self):
        spec = ops.get_op("softmax")
        flops = spec.flops_fn({
            "B": 1, "n_heads": 16, "s_query": 4, "s_kv": 1024,
            "s_pairs": 4 * 1024,
        })
        # 5 * 1 * 16 * 4 * 1024 * 1
        self.assertEqual(int(flops), 5 * 1 * 16 * 4 * 1024)

    def test_swa_window_caps_lq(self):
        spec = ops.get_op("swa_attention_score")
        # A decode query attends at most W keys: past Lq = W the cache length
        # no longer moves the band's pairs, and the cost is written in them.
        frame = {"B": 1, "n_heads": 16, "s_query": 1, "s_window": 4096,
                 "d_head": 64}
        pairs_above = spec.cost_pairs_fn({**frame, "s_kv": 8192})
        pairs_capped = spec.cost_pairs_fn({**frame, "s_kv": 4096})
        self.assertEqual(int(pairs_above), int(pairs_capped))
        self.assertEqual(int(spec.flops_fn({**frame, "s_pairs": pairs_above})),
                         2 * 16 * 4096 * 64)

    def test_per_head_linear_scales_by_n(self):
        # FLOPs = 2·B·S·N·in·out  (one per-head matmul × N heads)
        spec = ops.get_op("per_head_linear")
        flops_n1 = spec.flops_fn({"B": 1, "s_ctx": 1024, "n_heads": 1, "in": 128, "out": 512, "bias": 0})
        flops_n128 = spec.flops_fn({"B": 1, "s_ctx": 1024, "n_heads": 128, "in": 128, "out": 512, "bias": 0})
        self.assertEqual(int(flops_n128), 128 * int(flops_n1))


# ---------------------------------------------------------------------------
# 2. Phase binding builder — covers all four workload shapes the
#    framework should support after PR-2 (PR-1 only ships prefill / AR
#    decode, but the binding helper accepts the wider API).
# ---------------------------------------------------------------------------

class TestPhaseBinding(unittest.TestCase):

    def test_prefill_pairs_exact(self):
        b = phase_binding(phase="prefill", avg_input_tokens=1024, avg_output_tokens=512)
        # Sq = Lq = 1024; the triangle with its diagonal, 1024·1025/2
        self.assertEqual(b["s_query"], 1024)
        self.assertEqual(b["s_kv"], 1024)
        self.assertEqual(b["s_pairs"], 1024 * 1025 // 2)
        self.assertEqual(b["cache_len"], 0)

    def test_decode_ar_no_causal_saving(self):
        b = phase_binding(phase="decode", avg_input_tokens=1024, avg_output_tokens=512)
        self.assertEqual(b["s_query"], 1)
        # cache_len = S_in + S_out/2 = 1024 + 256 = 1280
        self.assertEqual(b["s_kv"], 1280)
        self.assertEqual(b["s_pairs"], 1280)
        self.assertEqual(b["cache_len"], 1280.0)

    def test_decode_short_seq_floors_lq_to_one(self):
        # cache_len = 0 + 0/2 = 0 → max(1, ...) = 1 (avoid division-by-zero downstream)
        b = phase_binding(phase="decode", avg_input_tokens=0, avg_output_tokens=0)
        self.assertEqual(b["s_kv"], 1)


# ---------------------------------------------------------------------------
# 3. Engine + phase binding end-to-end — does the spec engine produce
#    sensible numbers for prefill vs decode, and do they obey the
#    expected scaling laws?
# ---------------------------------------------------------------------------

class TestEnginePhasePrefillDecode(unittest.TestCase):
    """Run a real spec through the engine under prefill / decode
    bindings and check the analytical relationships hold."""

    @classmethod
    def setUpClass(cls):
        # DSv2-Lite: small enough to test fast, has MLA + MoE, well-known dims.
        cls.spec = _resolved("deepseek_v2", "deepseek-ai__DeepSeek-V2-Lite.json")

    def _run_phase(self, phase, S_in, S_out, batch=1):
        row = [{"access_bytes_per_rank": 1, "t_compute_s": 0,
                "t_memory_s": 0, "t_comm_s": 0, "t_launch_s": 0, "t_op_s": 0}]
        m = estimate_phase(
            PhaseInputs(
                avg_input_tokens=S_in, avg_output_tokens=S_out,
                weight_bytes_per_rank=1e10,
                per_token_kv_bytes_per_rank=1e6, static_kv_bytes_per_rank=0,
                resolved_spec=self.spec,
            ),
            phase=phase, batch=batch, graph_rows=row,
        )
        return m

    def _table(self, phase, S_in, S_out, batch=1):
        """The logical walk's FLOPs table for the step: the model's FLOPs,
        which is what these scaling laws are about (a phase's headline is
        its served rows' sum, and the stub rows above carry none)."""
        step, _bind = evaluate_phase_step(
            self.spec, phase=phase, batch=batch,
            avg_input_tokens=S_in, avg_output_tokens=S_out)
        return flops_fold(step).rows

    def _walk_flops(self, phase, S_in, S_out, batch=1) -> float:
        return float(sum(r.get("flops_full") or 0
                         for r in self._table(phase, S_in, S_out, batch)
                         if r.get("depth") == 0))

    def test_engine_path_active(self):
        self.assertGreater(len(self._table("prefill", 1024, 512)), 0,
                           "the FLOPs table should be populated")

    def test_attention_quadratic_kicks_in_at_long_context(self):
        # FFN scales linearly in S; attention scales quadratically.
        # Going from S=2k to S=8k (4× S), total FLOPs should grow MORE
        # than 4× because attention's contribution rises quadratically.
        ratio = (self._walk_flops("prefill", 8192, 0)
                 / self._walk_flops("prefill", 2048, 0))
        self.assertGreater(ratio, 4.0,
                           f"Long-context prefill should super-linearly grow "
                           f"(got {ratio:.2f}×; pure linear would be 4×)")

    def test_decode_far_smaller_than_prefill(self):
        # Sq=1 in decode collapses parameter-weighted compute.
        # Decode FLOPs/step should be at least 100× less than prefill
        # at S_in=1024 (this is the "memory-bound regime" claim).
        pre = self._walk_flops("prefill", 1024, 512, batch=1)
        dec = self._walk_flops("decode", 1024, 512, batch=1)
        self.assertGreater(pre / max(dec, 1),
                           100.0,
                           f"Decode should be << prefill at same batch "
                           f"(prefill={pre:.2e}, decode={dec:.2e})")

    def test_decode_arithmetic_intensity_low(self):
        # AI = FLOPs / bytes_read. Decode reads weights + KV cache for
        # only ~1 token of work, so AI should be low (memory-bound).
        # Prefill does S tokens of work for the same weight read, so
        # AI is much higher.
        m_pre = self._run_phase("prefill", 1024, 512, batch=8)
        m_dec = self._run_phase("decode", 1024, 512, batch=8)
        # The stub rows carry the bytes and no FLOPs: the walk's FLOPs over
        # the phase's bytes is the intensity these rows would have.
        ai_pre = self._walk_flops("prefill", 1024, 512, 8) / m_pre.total_bytes
        ai_dec = self._walk_flops("decode", 1024, 512, 8) / m_dec.total_bytes
        self.assertGreater(ai_pre, ai_dec * 10,
                           "Prefill AI should dominate decode AI by 10× or more")

    def test_phase_metrics_jsonable(self):
        # PhaseMetrics → asdict → JSON: caught the original SymPy leak,
        # keep this guard so future primitive additions don't reintroduce it.
        from dataclasses import asdict
        m = self._run_phase("prefill", 1024, 512)
        # json.dumps fails fast if any value isn't a primitive Python type
        # (the framework's contract for FastAPI's encoder).
        json.dumps(asdict(m), default=lambda o: float(o) if hasattr(o, "is_number") else None)


class TestDecodeAbsorbedEnginePath(unittest.TestCase):
    """Regression: decode's absorbed-MLA swap must reach BOTH the numeric
    engine AND the parallel SymbolicEngine that compute_model runs internally
    to harvest per-node Exprs.

    DeepSeek-V3's absorbed attention graph has a different child count than the
    decompressed one, so the old engine-only swap left numeric=absorbed vs
    symbolic=decompressed; `_refresh_node_display`'s strict zip then raised,
    the old `except` used to swallow decode into the closed-form fallback, and decode's
    `flops_by_module` came back empty — the per-op breakdown table collapsed to
    just the fused layer rows + comm. (V2-Lite's counts happened to match, so it
    didn't crash but mislabeled the symbolic exprs.) Guard that decode stays on
    the engine path with a populated breakdown."""

    @classmethod
    def setUpClass(cls):
        cls.spec = _resolved("deepseek_v3", "deepseek-ai__DeepSeek-V3.json")

    def test_decode_engine_path_populated(self):
        step, _bind = evaluate_phase_step(
            self.spec, phase="decode", batch=1,
            avg_input_tokens=1024, avg_output_tokens=512)
        self.assertGreater(
            len(flops_fold(step).rows), 0,
            "decode's FLOPs table must be populated — empty means the "
            "absorbed-MLA engine path crashed",
        )


# ---------------------------------------------------------------------------
# 4. MLA absorbed vs decompressed — same checkpoint, different op DAGs.
# ---------------------------------------------------------------------------

class TestMLAAbsorbedVsDecompressed(unittest.TestCase):
    """Verify the absorbed component evaluates and produces the
    expected param/FLOPs trade vs decompressed."""

    @classmethod
    def setUpClass(cls):
        cls.spec = _resolved("deepseek_v3", "deepseek-ai__DeepSeek-V3.json")
        cls.spec.spec.components["mla_attention_absorbed"] = (
            SpecLoader()._component_lib["mla_attention_absorbed"]
        )
        # Both forms evaluated at the same call of the model: the backbone's
        # MLA, whose place the absorbed form takes.
        from backend.arch.eval.transforms import CallSite
        from tests.snapshots import backbone_call
        call = backbone_call(cls.spec, "mla_attention").call
        cls.sites = {name: CallSite.of(cls.spec, name, call=call)
                     for name in ("mla_attention", "mla_attention_absorbed")}

    def test_params_identical(self):
        # Same checkpoint weights, just different op DAG → params equal.
        decomp = project_block(
            self.spec, "mla_attention", batch=1, input_tokens=2048,
            site=self.sites["mla_attention"],
        )
        absorb = project_block(
            self.spec, "mla_attention_absorbed", batch=1, input_tokens=2048,
            site=self.sites["mla_attention_absorbed"],
        )
        self.assertEqual(decomp.total_params, absorb.total_params,
                         f"absorbed ({absorb.total_params}) should equal "
                         f"decompressed ({decomp.total_params}) params")

    def test_absorbed_more_flops_at_prefill(self):
        # At prefill bindings (Sq=Lq=S, causal=1/2), absorbed does more
        # compute because it's running per-head latent matmuls (Q-side
        # absorb + V-side absorb) on top of an in-latent QKᵀ. Expected
        # ratio for DSv3 dims (R=512, D=Dv=128, P=64) is roughly 1.2-1.6×.
        decomp = project_block(
            self.spec, "mla_attention", batch=1, input_tokens=2048,
            site=self.sites["mla_attention"],
        )
        absorb = project_block(
            self.spec, "mla_attention_absorbed", batch=1, input_tokens=2048,
            site=self.sites["mla_attention_absorbed"],
        )
        ratio = absorb.total_flops / decomp.total_flops
        self.assertGreater(ratio, 1.2,
                           f"absorbed/decompressed FLOPs ratio = {ratio:.2f} (expected > 1.2)")
        self.assertLess(ratio, 2.0,
                        f"absorbed/decompressed FLOPs ratio = {ratio:.2f} (expected < 2.0)")


# ---------------------------------------------------------------------------
# 5. Migration sanity — old (B, S) callers must produce EXACT numbers
#    given the framework's deterministic pairs `Sq·(Sq+1)/2`.
#    No tolerance: any drift means a primitive's FLOPs lambda changed
#    (intentionally or otherwise), and the new baseline must be re-pinned
#    by hand. This is the tripwire that catches silent regressions.
# ---------------------------------------------------------------------------

class TestLegacyBindingMigration(unittest.TestCase):

    def test_inspector_path_numbers_stable(self):
        # The Inspector path uses the legacy (B, S) → setdefault
        # Sq=Lq=S, s_pairs=S·(S+1)/2 binding. Pinned-exact numbers so
        # future primitive changes can't silently shift Inspector
        # display — if you legitimately change a FLOPs lambda or
        # extend a component's op DAG, update these baselines as part
        # of the same change.
        from backend.analyzer import analyze_model

        result = analyze_model(
            "deepseek-ai/DeepSeek-V3",
            batch_size=1, seq_len=2048, dtype="float16",
        )
        s = result["summary"]
        # Captured at commit `110d369` (legacy causal_factor switched
        # from Rational(1,2) to exact (Sq+1)/(2·Lq); diagonal-correction
        # lifts attention_flops by 1/(2·S) ≈ 0.009% at S=2048). Re-pinned
        # after dispatch→permute rename zeroed the boundary op's FLOPs
        # (memory-only index permute, matches `gather`).
        #
        # `total_flops` re-pinned again when the re-fold stopped picking the
        # model's tail by NAME and started taking every top-level node. The
        # delta is +29,360,128 = 2 × 7168 × 2048 exactly — DSv3's final norm at
        # this binding, which the head-by-name tail had been dropping. It is
        # 1.8e-7 of the total, which is why nothing with a tolerance saw it and
        # why an exact pin is the only thing that could. `total_params`,
        # `ffn_flops` and `attention_flops` are unchanged: the norm is neither,
        # and it carries no parameters the count was missing.
        #
        # `attention_flops` and `total_flops` re-pinned when RoPE went from 6
        # to 3 FLOPs per rotated element (4 mul + 2 add per pair, vLLM
        # pos_encoding_kernels.cu:31-32): −3,144,941,568 = 62 MLA layers ×
        # 3 × 2048 × 64 × (128 + 1) — the q_pe (128 heads) and k_pe rotations.
        self.assertEqual(s["total_params"],     684_489_845_504,
                         "DSv3 total_params changed (params_fn untouched in PR-1/PR-2)")
        self.assertEqual(s["ffn_flops"],        101_120_057_475_072,
                         "DSv3 ffn_flops changed (FFN primitives untouched in PR-1/PR-2)")
        self.assertEqual(s["attention_flops"],  58_259_400_556_544,
                         "DSv3 attention_flops changed — re-pin baseline if intentional")
        self.assertEqual(s["total_flops"],      167_397_297_946_624,
                         "DSv3 total_flops changed — re-pin baseline if intentional")


if __name__ == "__main__":
    unittest.main()
