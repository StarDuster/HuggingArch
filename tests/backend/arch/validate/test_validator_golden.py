"""Layer-3 ship-gate regression guard.

CLAUDE.md §4 Layer 3: a spec ships when `model_storage_bytes.diff == 0`
AND `unmatched_hf_tensors == []` AND `param_mismatches == []`.

Before this file, the ONLY test exercising the validator on real
models (`test_compound_op_params.TestSnapshotRegression`) checked
param TOTALS. Totals can be byte-equal while per-tensor NAME matching
is completely broken — a spec can emit `ffn.weight` where the
checkpoint has `ffn.gate.weight`, same byte count, zero total-diff,
but 60+ tensors unmatched. That blind spot let a DeepSeek-V4-Pro
state with 67 unmatched + 19 param_mismatches sit green; the matcher
fixes that closed it had no test holding them.

This file is that test. Two tiers:

  • STRICT — models whose spec is provably ship-gate-clean. Asserts
    the FULL gate (0 bytes diff, 0 unmatched, 0 param_mismatches).
    Any regression in any matcher path — the header's name selection, the
    declared checkpoint addresses names are composed from, `CheckpointJoin`
    — trips this immediately.

  • WATERMARK — models whose spec still has KNOWN matcher gaps (the
    validator's per-tensor matching is imperfect for several
    architectures; tracing every one is a separate effort). We record
    the CURRENT counts and assert they don't grow. This is a ratchet:
    a regression (count up) fails the test; a spec fix (count down)
    ALSO fails it — intentionally — forcing the baseline down so
    progress is locked in. The recorded numbers are a fence around a
    known-imperfect state, NOT an endorsement of it.

Runs offline against the git snapshots under
`backend/arch/snapshots/` — no network. A model whose snapshot
is missing is skipped, not failed.
"""
from __future__ import annotations

import functools
import pickle
import sys
import unittest

from tools.corpus_pins.checks.coverage_gate import gate_errors, validate_offline
from tests.snapshots import snapshot_config
import pytest
import unittest.mock as mock

from backend.arch.validate._finding import Finding
from pathlib import Path

pytestmark = pytest.mark.regression  # walks every snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import SNAPSHOT_DIR

# Provably ship-gate-clean: 0 byte diff, 0 unmatched HF tensors, 0
# param_mismatches. Went from 1 → 25 models when the per-tensor matcher's
# universal prefix bug (bare spec stamp vs prefixed HF key), the MoE
# `expert`/`experts` fold, the folded-primary router-gate stamp, and the
# `final_norm`→`norm` faithfulness fix all landed. DeepSeek-V4-Pro remains
# the canonical hardest entry (MTP `mtp.X.*` stack, model-root
# `parameter_tensor`/`parameter_vector` HC weights, four heterogeneous
# block templates sharing block-internal spec_ids, a bare-Parameter Gate).
# The rest lock in the coverage win across MoE (Qwen3/MiMo/DSv2-3/Ling/
# Kimi-K2/GLM-5), MTP, vision (Step-3.7 resblocks), and dense checkpoints.
_STRICT_MODELS = (
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-Coder-Next",
    "XiaomiMiMo/MiMo-V2-Flash",
    "XiaomiMiMo/MiMo-V2.5",
    "XiaomiMiMo/MiMo-V2.5-Pro",
    "moonshotai/Kimi-K2-Base",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Thinking",
    "inclusionAI/Ling-2.6-1T",
    "zai-org/GLM-5.1",
    "zai-org/GLM-5.2",
    "THUDM/glm-4-9b",
    "THUDM/glm-4-9b-chat",
    "google/gemma-4-12B-it",
    "meta-llama/Llama-2-7b-hf",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "openai/gpt-oss-20b",
    "stepfun-ai/Step-3.7-Flash",
    # Fully clean against the header's own tensor counts.
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-Nemo-Base-2407",
    # Router node renamed `block_sparse_moe`→`block_sparse_moe.gate` (checkpoint
    # name) + F32 quant override re-keyed to `.gate.weight`.
    "MiniMaxAI/MiniMax-M2.7",
    # Kimi-K3: SituAndMul gate modeled via the parameterized `gated_ffn`
    # (`act: situ`) family + a `situ` op; the MoonViT vision tower's nested
    # `repeat: Lv` block now flattens with its count so per-node param matching
    # is exact. Fully clean (0/0/0).
    "moonshotai/Kimi-K3",
    # Kimi-K2.5: its 6 residual param_mismatches were NOT INT4 packing (the old
    # watermark comment's guess) — they were the MoonViT vision tower's nested
    # `repeat` block undercounting in the flatten, the same bug Kimi-K3 exposed.
    # Fixed at the matcher (`_flatten_node` repeat multiplier); now clean.
    "moonshotai/Kimi-K2.5",
)

# Known-imperfect: (max_unmatched, max_param_mismatches). Ratchet DOWN when
# a spec is fixed (the test fails until you lower the number — the point).
_WATERMARK_MODELS = {
    # LongCat is fully clean: physical tensor matching reads materialized
    # checkpoint names and packing rather than aggregating display-node suffixes.
    "meituan-longcat/LongCat-Flash-Thinking-2601": (0, 0),
}


def _run_validate(model_id: str):
    # Offline, so the test never depends on network: snapshots carry
    # everything `validate` needs (config + the safetensors header).
    return validate_offline(model_id)


@functools.cache
def _report_bytes(model_id: str) -> bytes:
    return pickle.dumps(_run_validate(model_id))


def _validated(model_id: str) -> dict:
    """`validate(model_id)` on the unpatched tree, run once per process: the
    strict, watermark and coverage gates ask the same models. Each caller gets
    its own copy of the report. Tests that patch the orchestrator call
    `_run_validate` directly."""
    return pickle.loads(_report_bytes(model_id))


class TestValidatorShipGateStrict(unittest.TestCase):
    """Full Layer-3 gate on provably-clean specs."""

    def test_strict_models_pass_full_ship_gate(self):
        for model_id in _STRICT_MODELS:
            with self.subTest(model_id=model_id):
                snapshot_config(model_id)
                r = _validated(model_id)
                bytes_diff = r["model_storage_bytes"]["diff"]
                unmatched = r.get("unmatched_hf_tensors") or []
                mismatches = r.get("param_mismatches") or []
                self.assertEqual(
                    bytes_diff, 0,
                    f"{model_id}: model_storage_bytes.diff={bytes_diff:+,} "
                    f"(spec={r['model_storage_bytes']['spec']:,} "
                    f"actual={r['model_storage_bytes']['actual']:,})",
                )
                self.assertEqual(
                    unmatched, [],
                    f"{model_id}: {len(unmatched)} unmatched HF tensors — "
                    f"first few: {[u['pattern'] for u in unmatched[:5]]}",
                )
                self.assertEqual(
                    mismatches, [],
                    f"{model_id}: {len(mismatches)} param_mismatches — "
                    f"first few: "
                    f"{[(m['spec_node'], m['ratio']) for m in mismatches[:5]]}",
                )


class TestValidatorShipGateWatermark(unittest.TestCase):
    """Ratchet on known-imperfect specs — counts must not grow."""

    def test_watermark_models_do_not_regress(self):
        for model_id, (max_unmatched, max_mismatch) in _WATERMARK_MODELS.items():
            with self.subTest(model_id=model_id):
                snapshot_config(model_id)
                r = _validated(model_id)
                unmatched = len(r.get("unmatched_hf_tensors") or [])
                mismatches = len(r.get("param_mismatches") or [])
                self.assertLessEqual(
                    unmatched, max_unmatched,
                    f"{model_id}: unmatched_hf_tensors grew to {unmatched} "
                    f"(watermark {max_unmatched}) — a matcher path regressed.",
                )
                self.assertLessEqual(
                    mismatches, max_mismatch,
                    f"{model_id}: param_mismatches grew to {mismatches} "
                    f"(watermark {max_mismatch}) — a matcher path regressed.",
                )
                # Ratchet: if a spec fix dropped the count, the watermark
                # is now stale — fail loudly so it gets lowered, locking
                # the improvement in.
                self.assertEqual(
                    unmatched, max_unmatched,
                    f"{model_id}: unmatched dropped to {unmatched} "
                    f"(watermark {max_unmatched}) — lower the watermark in "
                    f"_WATERMARK_MODELS to lock this improvement in.",
                )
                self.assertEqual(
                    mismatches, max_mismatch,
                    f"{model_id}: param_mismatches dropped to {mismatches} "
                    f"(watermark {max_mismatch}) — lower the watermark in "
                    f"_WATERMARK_MODELS to lock this improvement in.",
                )


class TestCoverageGate(unittest.TestCase):
    """The coverage gate in one assertion across EVERY snapshot: a spec is
    `valid` iff every checkpoint tensor maps to a spec node and the byte
    diff is 0. This is the regression guard that makes
    `unmatched_hf_tensors` a real ship gate — the anchor a wrong MoE spec
    used to slip past on aggregate bytes."""

    def test_every_snapshot_is_valid(self):
        for snap in sorted(SNAPSHOT_DIR.glob("*.json")):
            model_id = snap.stem.replace("__", "/")
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    gate_errors(_validated(model_id)), [],
                    f"{model_id}: coverage/byte gate failed")

    def test_a_name_stands_for_one_address(self):
        """The gate joins names by equality, so a name stands for one place in
        the checkpoint: the tensors that share it (a heterogeneous stack's
        templates, an MTP layer appended to the backbone's index space) sit
        under one checkpoint prefix, never under two modules that merely end
        alike."""
        from tools.corpus_pins.corpus import CorpusModel
        from tools.corpus_pins.pins.gate_surfaces import _analysis

        for snap in sorted(SNAPSHOT_DIR.glob("*.json")):
            with self.subTest(snapshot=snap.stem):
                analysis = _analysis(CorpusModel(snap.stem))
                program = analysis.program
                prefixes: dict = {}
                for row in analysis.iter_checkpoint():
                    node = program.by_uid(row.uid)
                    prefixes.setdefault(row.name, set()).add(
                        program.checkpoint_prefix(node))
                shared = {name: sorted(p) for name, p in prefixes.items()
                          if len(p) > 1}
                self.assertEqual(shared, {})


class TestDiagnosticGates(unittest.TestCase):
    """Shape, cache and cross-layer findings are publish errors, not FYI."""

    _MODEL = "meta-llama/Llama-3.2-1B"

    def test_shape_mismatch_gates_valid(self):
        finding = Finding(
            kind="dim_map_mismatch", template="block",
            node="delta", op="gated_delta_rule",
            detail="head axis disagrees", shapes=[[1, 16], [1, 48]],
        )
        with mock.patch(
            "backend.arch.validate._orchestrator._run_shape_consistency_check",
            return_value=[finding],
        ):
            result = _run_validate(self._MODEL)
        self.assertFalse(result["valid"])
        self.assertTrue(any("dim_map_mismatch" in str(e) for e in result["errors"]))

    def test_type_contract_gates_valid(self):
        finding = Finding(
            kind="attention_missing_kv_cached", template="attention",
            detail="attention has no cache state declaration",
        )
        with mock.patch(
            "backend.arch.validate._orchestrator.run_type_contract_checks",
            return_value=[finding],
        ):
            result = _run_validate(self._MODEL)
        self.assertFalse(result["valid"])
        self.assertIn(finding.detail, result["errors"])

    def test_cross_layer_contract_gates_valid(self):
        finding = Finding(
            kind="cross_layer_stream_no_consumer",
            detail="stream 's' is produced but no component reads it",
        )
        with mock.patch(
            "backend.arch.validate._orchestrator.run_cross_layer_checks",
            return_value=[finding],
        ):
            result = _run_validate(self._MODEL)
        self.assertFalse(result["valid"])
        self.assertIn(finding.detail, result["errors"])

    def test_unchecked_stream_anchors_do_not_gate(self):
        """With no recorded forward source the stream anchors cannot be
        checked; the spec cannot supply the source, so it is reported, not
        gated."""
        finding = Finding(
            kind="cross_layer_src_unchecked",
            detail="stream anchors of s were not checked",
        )
        with mock.patch(
            "backend.arch.validate._orchestrator.cross_layer_anchor_notes",
            return_value=[finding],
        ):
            result = _run_validate(self._MODEL)
        self.assertTrue(result["valid"])
        self.assertNotIn(finding.detail, result["errors"])
        self.assertIn(finding.detail, [
            w["detail"] for w in result["forward_source_warnings"]])


if __name__ == "__main__":
    unittest.main()
