"""Tests for declarative A-class axioms (schema.AxiomSet + validate/_axioms.py).

The axiom is a JUDGE: it asserts the spec's computed shape/compute/KV matches
the source-derived value declared in the component. Two halves:
  • a real component's axioms hold (DSv4 indexer, source-locked);
  • a wrong declaration FIRES (a judge that can't go red proves nothing).
"""
from __future__ import annotations

import json
import sys
import unittest

from tests.snapshots import shared_loader, snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.validate._axioms import run_axiom_checks

SNAP = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
        / "deepseek-ai__DeepSeek-V4-Pro.json")
MID = "deepseek-ai/DeepSeek-V4-Pro"


def test_a_claimed_count_is_a_number_or_it_raises():
    """A per-position count is compared as a number. An element count the
    call's dims leave unbound is not silently taken as some integer."""
    import pytest
    import sympy as sp
    from types import SimpleNamespace
    from backend.arch.eval.cache import CachedTensor
    from backend.arch.eval.dim import Dim
    from backend.arch.validate._axioms import _Call

    ax = SimpleNamespace(kind="kv_growth", storage="per_token", node=None,
                         src="test", bound=None, expect="d")

    def judged(elements):
        tensor = CachedTensor(node=0, name="k", elements=elements,
                              stored=sp.Symbol("s_kv"), read=sp.Symbol("s_kv"))
        return _Call(template="m", path="m", role="attention", label=None,
                     axioms=None, tensors=(tensor,),
                     dims={"d": Dim(sp.Symbol("d"), 8)})._kv_growth(ax)

    assert judged(Dim(sp.Symbol("d"), 8)) == []
    assert [f.kind for f in judged(16)] == ["axiom_kv_violation"]
    with pytest.raises((TypeError, ValueError)):
        judged(sp.Symbol("s_draft_block"))


def test_a_window_over_a_compressed_cache_is_bounded():
    """DSA's top-k over a 4x compressed cache stores `Min(512, ⌈s_kv/4⌉)`
    positions: as the sequence grows it stops at 512, and a bound claim of
    512 holds while one of 4 (a number that merely appears in it) does not."""
    import sympy as sp
    from types import SimpleNamespace
    from backend.arch.eval.cache import CachedTensor
    from backend.arch.validate._axioms import _Call, _storage_is_bounded

    s_kv = sp.Symbol("s_kv")
    stored = sp.Min(512, sp.ceiling(s_kv / 4))
    assert _storage_is_bounded(stored) is True
    assert _storage_is_bounded(s_kv) is False

    def judged(bound):
        ax = SimpleNamespace(kind="kv_growth", storage="windowed", node=None,
                             src="test", bound=bound, expect=None)
        tensor = CachedTensor(node=0, name="k", elements=1,
                              stored=stored, read=stored)
        return [f.kind for f in _Call(
            template="m", path="m", role="attention", label=None, axioms=None,
            tensors=(tensor,), dims={})._kv_growth(ax)]

    assert judged("512") == []
    assert judged("4") == ["axiom_kv_growth_violation"]


def test_an_undecidable_window_is_unjudged_in_the_config_check():
    """`Min(W, s_kv)` with `W` left free may or may not cap: the config's
    sliding window is not judged against it, and the call says so rather
    than the validator crashing or calling the window unmodelled."""
    import sympy as sp
    from types import SimpleNamespace
    from backend.arch.eval.cache import CachedTensor
    from backend.arch.validate._axioms import _Call, _check_config_windowing

    stored = sp.Min(sp.Symbol("W"), sp.Symbol("s_kv"))
    call = _Call(template="swa", path="layers.0.self_attn", role="attention",
                 label=None, axioms=None, dims={},
                 tensors=(CachedTensor(node=0, name="k", elements=1,
                                       stored=stored, read=stored),))
    resolved = SimpleNamespace(
        config={"sliding_window": 4096},
        spec=SimpleNamespace(validation_exemptions=None))
    (finding,) = _check_config_windowing(resolved, [call])
    assert finding.kind == "axiom_unjudged"
    assert finding.node == "layers.0.self_attn"


def _kinds(ws):
    return {w.kind for w in ws}


class TestAxioms(unittest.TestCase):
    def setUp(self):
        snapshot_config(SNAP.name)
        cfg = json.loads(SNAP.read_text())["config"]
        self.resolved = shared_loader().match(cfg, model_id=SNAP.stem.replace("__", "/", 1))
        self.ix = self.resolved.spec.components["dsv4_indexer"]

    def test_axioms_loaded(self):
        # loader must carry `axioms:` through (BlockGraph is built field-by-field).
        self.assertIsNotNone(self.ix.axioms)
        self.assertTrue(self.ix.axioms.checks)

    def test_indexer_axioms_hold(self):
        # The declared source facts match the engine's computed values.
        ws = run_axiom_checks(MID, self.resolved)
        self.assertEqual(ws, [], f"unexpected axiom violations: {ws}")

    def test_wrong_kv_axiom_fires(self):
        # Break the indexer KV per-position size (Ih → In, the MQA trap) → red.
        kv_ax = next(a for a in self.ix.axioms.checks if a.kind == "kv_growth")
        original = kv_ax.expect
        kv_ax.expect = "n_indexer_heads"  # wrong: cache carries index_head_dim, not index_n_heads
        try:
            ws = run_axiom_checks(MID, self.resolved)
            self.assertIn("axiom_kv_violation", _kinds(ws))
        finally:
            kv_ax.expect = original


class TestMlaFamilyAxioms(unittest.TestCase):
    """Axioms on the shared MLA components hold for every model that uses them,
    and fire on the exact cross-model mistakes they guard against."""

    def _load(self, snap_name, mid):
        p = PROJECT_ROOT / "backend" / "arch" / "snapshots" / snap_name
        snapshot_config(snap_name)
        cfg = json.loads(p.read_text())["config"]
        return mid, shared_loader().match(cfg, model_id=mid)

    def test_mla_axioms_hold_v3_and_kimi(self):
        for snap, mid in (("deepseek-ai__DeepSeek-V3.json", "deepseek-ai/DeepSeek-V3"),
                          ("moonshotai__Kimi-K2-Instruct.json", "moonshotai/Kimi-K2-Instruct")):
            mid, r = self._load(snap, mid)
            self.assertEqual(run_axiom_checks(mid, r), [], f"{mid}: unexpected violations")

    def test_v32_indexer_full_holds(self):
        mid, r = self._load("deepseek-ai__DeepSeek-V3.2.json", "deepseek-ai/DeepSeek-V3.2")
        self.assertEqual(run_axiom_checks(mid, r), [])

    def test_mla_latent_kv_per_head_trap_fires(self):
        # Latent KV is MQA (R+P), not per-head (N*(R+P)) — the classic MLA trap.
        mid, r = self._load("deepseek-ai__DeepSeek-V3.json", "deepseek-ai/DeepSeek-V3")
        kv = next(a for a in r.spec.components["mla_attention"].axioms.checks
                  if a.kind == "kv_growth")
        original = kv.expect
        kv.expect = "n_heads * (d_kv_lora + d_qk_rope)"
        try:
            self.assertIn("axiom_kv_violation", _kinds(run_axiom_checks(mid, r)))
        finally:
            kv.expect = original


class TestComposedAttention(unittest.TestCase):
    """A modern attention layer is a COMPOSITION of KV streams with different
    growth laws (V4 CSA: windowed SWA + growing compressor). The law is
    per-stream; a single component label can't describe it."""

    def _v4(self):
        p = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
             / "deepseek-ai__DeepSeek-V4-Pro.json")
        snapshot_config(p.name)
        return "deepseek-ai/DeepSeek-V4-Pro", shared_loader().match(
            json.loads(p.read_text())["config"], model_id="deepseek-ai/DeepSeek-V4-Pro")

    def test_csa_per_stream_holds(self):
        # kv_concat=windowed + compressor=per_token, declared per-stream → green.
        mid, r = self._v4()
        self.assertEqual(run_axiom_checks(mid, r), [])

    def test_uniform_claim_fires_on_mixed_component(self):
        # Declaring the whole CSA component per_token (node-absent) must fire:
        # kv_concat CAPS, so a uniform claim is wrong — the check tells you to
        # split it per-stream.
        from backend.arch.schema import Axiom
        mid, r = self._v4()
        csa = r.spec.components["dsv4_csa_attention"]
        csa.axioms.checks.append(Axiom(kind="kv_growth", storage="per_token",
                                       src="uniform claim on a mixed component"))
        try:
            self.assertIn("axiom_kv_growth_violation", _kinds(run_axiom_checks(mid, r)))
        finally:
            csa.axioms.checks.pop()


class TestKvGrowthLaw(unittest.TestCase):
    """The kv_growth facet describes the cache GROWTH LAW (per_token / windowed
    / constant_state), not a universal "per token" frame — so SWA and linear
    are correct, and the overcounts that birth the 128TB-class bugs fire."""

    def _load(self, snap_name, mid):
        p = PROJECT_ROOT / "backend" / "arch" / "snapshots" / snap_name
        snapshot_config(snap_name)
        return mid, shared_loader().match(json.loads(p.read_text())["config"], model_id=p.stem.replace("__", "/", 1))

    def test_linear_constant_state_holds(self):
        # Qwen3-Next gated DeltaNet: O(1) recurrent state, no growing cache.
        mid, r = self._load("Qwen__Qwen3-Coder-Next.json", "Qwen/Qwen3-Coder-Next")
        self.assertEqual(run_axiom_checks(mid, r), [])

    def test_linear_declared_per_token_fires(self):
        # The SSM overcount: claiming a linear layer grows KV per token.
        mid, r = self._load("Qwen__Qwen3-Coder-Next.json", "Qwen/Qwen3-Coder-Next")
        ax = r.spec.components["gated_delta_net_fused_attention"].axioms.checks[0]
        ax.storage = "per_token"
        try:
            self.assertIn("axiom_kv_growth_violation", _kinds(run_axiom_checks(mid, r)))
        finally:
            ax.storage = "constant_state"

    def test_swa_windowed_holds_and_per_token_fires(self):
        mid, r = self._load("deepseek-ai__DeepSeek-V4-Pro.json", "deepseek-ai/DeepSeek-V4-Pro")
        self.assertEqual(run_axiom_checks(mid, r), [])
        # The SWA overcount: claiming the windowed cache grows to all S.
        swa = r.spec.components["dsv4_swa_only_attention"].axioms.checks[0]
        swa.storage, swa.bound = "per_token", None
        try:
            self.assertIn("axiom_kv_growth_violation", _kinds(run_axiom_checks(mid, r)))
        finally:
            swa.storage, swa.bound = "windowed", "s_window"

    def test_constant_state_allows_constant_cache_but_not_lq_dependent(self):
        """New semantics: `constant_state` permits a cached recurrent-state
        stream as long as its storage is Lq-INDEPENDENT (the linear/SSM matrix
        state). A stream whose storage depends on Lq is still the overcount."""
        import sympy as sp
        from types import SimpleNamespace
        from backend.arch.eval.cache import CachedTensor
        from backend.arch.validate._axioms import _Call
        ax = SimpleNamespace(kind="kv_growth", storage="constant_state",
                             node=None, src="test", bound=None, expect=None)

        def judged(*tensors):
            return _Call(template="m", path="m", role="attention", label=None,
                         axioms=None, tensors=tensors, dims={})._kv_growth(ax)

        const_node = CachedTensor(
            node=0, name="recurrent_state", elements=sp.Integer(10),
            stored=sp.Integer(1), read=sp.Integer(1))
        self.assertEqual(judged(const_node), [])

        grow_node = CachedTensor(
            node=1, name="grows", elements=sp.Integer(10),
            stored=sp.Symbol("s_kv"), read=sp.Symbol("s_kv"))
        self.assertTrue(judged(grow_node),
                        "constant_state must fire on an Lq-dependent stream")


if __name__ == "__main__":
    unittest.main()
