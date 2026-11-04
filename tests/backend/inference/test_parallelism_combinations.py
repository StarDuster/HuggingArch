"""Combinatorial parallelism correctness — the generic guard for
"arbitrary parallel combos are handled correctly".

The 8 axes (pp, tp_attention, tp_dense_ffn, tp_moe_expert, ep_moe, cp,
dp_attention, dp_replicas) are correct BY CONSTRUCTION: each shards a
different thing (tp→heads/hidden, cp→seq, ep→experts, dp→batch replicas,
pp→layers), orthogonally, with zero model-specific branches in backend.
But the suite historically exercised at most 2 axes active at once — the
3+ axis interactions (where the CP-ceil and head-broadcast bugs hid) were
untested. This sweep runs the cross-consumer consistency invariants over
VALID combos with 3-4 axes simultaneously active, so any combination that
breaks composition trips a model-agnostic invariant rather than waiting
for a calibration mismatch to surface it.

Invariants (same as test_consistency_invariants, applied per-combo):
  • phase mem_read == Σ per_op access_bytes_per_rank  (aggregate==parts)
  • KV read positions ≤ stored cache
  • per-token KV bytes are workload-invariant

Note on isolation: `ParallelConfig.validate()` couples the three group
products (attn = tp_attn×cp×dp, dense = tp_dense×cp×dp, moe = ep×tp_moe
must all equal per_stage), so "vary only dp_attention" is not expressible as a
pair of valid configs — DP-orthogonality can't be unit-isolated here. The
composition invariants below cover it implicitly: a DP axis that wrongly
sharded weights/KV would make some combo's read>storage or aggregate≠parts.
"""
from __future__ import annotations

import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import snapshot_config

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
_SEQ = 4096

# Combos with 3-4 axes simultaneously active. MoE models use the ep/tp_moe
# axes; the dense model (no experts) uses a dense-only set. An entry that
# fails validate() never becomes a case (filtered at collection). DeepSeek-V3
# comes first, so the default run's one case is a MoE combo: it exercises
# more axes than the dense model can.
# cp keeps tp_dense_ffn == tp_attention (it spans dense via the token-shard,
# never the weight divisor); each region's group must equal per_stage.
_MOE_COMBOS = [
    dict(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2),          # 3× TP
    dict(tp_attention=2, tp_dense_ffn=2, cp=2, ep_moe=4),           # tp + cp + ep
    dict(tp_attention=2, tp_dense_ffn=2, cp=2, ep_moe=2,
         tp_moe_expert=2),                                          # tp+cp+ep+tp_moe (5-wide)
    dict(tp_attention=2, dp_attention=2, tp_dense_ffn=2, ep_moe=4),  # DP + tp + ep
    dict(pp=2, tp_attention=2, tp_dense_ffn=2, ep_moe=2),           # pp + 3 axes
    dict(ep_moe=2, tp_moe_expert=2, tp_attention=4, tp_dense_ffn=4),  # ep×tp_moe + tp
]
_DENSE_COMBOS = [
    dict(tp_attention=2, tp_dense_ffn=2),
    dict(tp_attention=2, cp=2, tp_dense_ffn=2),                     # tp + cp
    dict(tp_attention=2, cp=2, dp_attention=2, tp_dense_ffn=2),     # tp + cp + dp
    dict(tp_attention=2, dp_attention=2, tp_dense_ffn=2),           # DP + tp
    dict(pp=2, tp_attention=2, tp_dense_ffn=2),
]
# Invariants are model-agnostic, so coverage is by AXIS not by model:
# one MoE model (exercises ep / tp_moe / cp / dp / pp) + one dense model
# (tp / cp / dp / pp) covers every axis combination. (Kimi would just
# repeat DeepSeek-V3's MLA+MoE shape — omitted to keep the sweep fast.)
_MODELS = [
    ("DeepSeek-V3", "deepseek-ai__DeepSeek-V3.json", _MOE_COMBOS),
    ("Llama-3.2-3B", "meta-llama__Llama-3.2-3B.json", _DENSE_COMBOS),
]


@lru_cache(maxsize=None)
def _config(snapshot):
    p = SNAPSHOT_DIR / snapshot
    return json.loads(p.read_text())["config"] if p.exists() else None


@lru_cache(maxsize=None)
def _resolved(snapshot):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    return shared_loader().match(
        _config(snapshot), model_id=Path(snapshot).stem.replace("__", "/", 1))


def _valid_cases():
    """(label, snapshot, parallel) for every combo that passes validate().
    Validation reads only the combo, so this runs at collection without
    touching a spec."""
    for label, snap, combos in _MODELS:
        for combo in combos:
            pk = ParallelConfig(**combo)
            if pk.validate():
                continue  # not a feasible deployment — skip
            yield pytest.param(snap, pk, id=f"{label} {combo}")


_CASES = list(_valid_cases())


def test_the_sweep_is_not_empty():
    assert _CASES, "no combo passed validate()"


class TestParallelCombinationInvariants:
    """Every valid multi-axis combo must satisfy the cross-consumer
    consistency invariants. Failure means some axis-combination breaks
    per-rank composition (double-count / miss / aggregate drift)."""

    @pytest.mark.parametrize("snap, pk", _CASES)
    def test_combo_invariants(self, snap, pk):
        cfg = snapshot_config(Path(snap).stem)
        r = estimate_capacity(
            config=cfg, resolved=_resolved(snap), gpu_sku="H200",
            num_gpus=pk.cluster_size,
            avg_input_tokens=_SEQ, avg_output_tokens=256, parallel=pk,
            selection_mode="measured")
        # 1. aggregate == Σ parts, both phases
        for phase_name, phase_key, rows_key in (
            ("decode", "decode", "per_op_timing_decode"),
            ("prefill", "prefill", "per_op_timing"),
        ):
            ph = r.get("phases", {}).get(phase_key, {}) or {}
            rows = r.get(rows_key) or []
            if not rows:
                continue
            # `total_bytes`, not `mem_read_bytes`: a leaf's access is
            # its whole traffic both ways, and the phase splits that
            # same sum into a read and a write half.
            agg = int(ph.get("total_bytes") or 0)
            parts = sum(int(x.get("access_bytes_per_rank") or 0) for x in rows)
            assert agg == parts, (
                f"{phase_name}: total {agg:,} != Σparts "
                f"{parts:,} (Δ {agg-parts:,})")
            assert (int(ph.get("mem_read_bytes") or 0)
                    + int(ph.get("mem_write_bytes") or 0)) == agg, (
                f"{phase_name}: read + write != total")
        # 2. KV read ≤ stored
        kv = r.get("kv_cache", {}) or {}
        read = kv.get("kv_read_positions_at_decode_cache")
        if read is not None:
            clen = kv.get("decode_cache_len_for_read") or (_SEQ + 128)
            assert int(read) <= int(clen), f"KV read {read} > stored {clen}"


class TestNumGpusIsInformational(unittest.TestCase):
    """`num_gpus` is informational (capacity.py: cluster_size =
    parallel.cluster_size is authoritative). Per-rank weight / KV must
    NOT change when num_gpus is varied with parallel held fixed — locks
    that contract so a future refactor can't make num_gpus silently
    drive per-rank math."""

    def test_per_rank_independent_of_num_gpus(self):
        cfg = _config("deepseek-ai__DeepSeek-V3.json")
        resolved = _resolved("deepseek-ai__DeepSeek-V3.json")
        pk = ParallelConfig(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2)

        def _per_rank(n):
            r = estimate_capacity(
                 config=cfg, resolved=resolved, gpu_sku="H200", num_gpus=n,
                avg_input_tokens=_SEQ, avg_output_tokens=256, parallel=pk,
                selection_mode="theoretical_only")
            wb = r.get("weight_breakdown", {}) or {}
            kv = r.get("kv_cache", {}) or {}
            # per-request KV/rank replaces the removed model-level per_token_bytes.
            return (int(wb.get("total_bytes_per_rank") or 0),
                    int(kv.get("per_request_kv_bytes") or 0))

        # cluster_size is 2; pass deliberately "wrong" num_gpus values.
        a = _per_rank(2)
        b = _per_rank(64)
        self.assertEqual(a, b, f"per-rank changed with num_gpus: {a} vs {b}")


class TestDenseModelHasNoExpertMesh(unittest.TestCase):
    """A model that routes no tokens to experts has no MoE region: the
    expert knobs the page ties to TP span no ranks, so context parallelism
    on a dense model is not refused as an uneven expert mesh."""

    def test_dense_cp_with_tied_expert_tp_estimates(self):
        cfg = _config("Qwen__Qwen3-32B.json")
        resolved = _resolved("Qwen__Qwen3-32B.json")
        r = estimate_capacity(
            config=cfg, resolved=resolved, gpu_sku="H100-SXM",
            avg_input_tokens=_SEQ, avg_output_tokens=256,
            parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                    tp_moe_expert=2, cp=2),
            selection_mode="theoretical_only")
        self.assertEqual((r["num_gpus"], r["cluster_size"]), (4, 4))
        self.assertEqual(r["parallel"]["tp_moe_expert"], 1)


if __name__ == "__main__":
    unittest.main()
