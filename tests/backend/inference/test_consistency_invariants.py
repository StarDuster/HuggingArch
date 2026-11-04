"""Cross-consumer consistency invariants — structural enforcement of the
"single source of truth" rule (CLAUDE.md §1).

Most bugs in this repo are NOT logic errors; they are two consumers
reading the same fact and disagreeing (e.g. KV storage counted the
latent once but KV read counted it ×N heads; an aggregate drifting from
the sum of its parts). Point goldens catch a specific cell moving;
these invariants catch the whole CLASS — they hold for EVERY model with
no magic numbers, so a new consumer that re-derives a fact inconsistently
trips them regardless of which model exposes it.

Kept model-agnostic on purpose: one representative per attention family
(MLA+MoE / GQA-dense / MoE-MLA) from committed snapshots, offline.
"""
from __future__ import annotations

import sys
import unittest

import pytest

from tests.snapshots import snapshot_config
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig


# (label, snapshot, parallel) — one per attention family. Modest TP so
# per-rank math exercises sharding without needing many ranks.
_CASES = [
    ("DeepSeek-V3 (MLA+MoE)", "deepseek-ai__DeepSeek-V3.json",
     dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8, ep_moe=1)),
    ("Llama-3.2-3B (GQA dense)", "meta-llama__Llama-3.2-3B.json", dict()),
    ("Kimi-K2.5 (MLA, EP+CP)", "moonshotai__Kimi-K2.5.json",
     dict(ep_moe=8, cp=8)),
]
_SEQ = 8192


@lru_cache(maxsize=None)
def _config(snapshot):
    return snapshot_config(snapshot)


@lru_cache(maxsize=None)
def _resolved(snapshot):
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    cfg = _config(snapshot)
    if cfg is None:
        return None
    return shared_loader().match(
        cfg, model_id=Path(snapshot).stem.replace("__", "/", 1))


@lru_cache(maxsize=None)
def _estimate(snapshot: str, parallel_key: tuple):
    cfg = _config(snapshot)
    if cfg is None:
        return None
    return estimate_capacity(
         config=cfg, resolved=_resolved(snapshot), gpu_sku="H200", num_gpus=8,
        avg_input_tokens=_SEQ, avg_output_tokens=256,
        parallel=ParallelConfig(**dict(parallel_key)),
        selection_mode="measured",
    )


def _cases():
    for label, snap, par in _CASES:
        r = _estimate(snap, tuple(sorted(par.items())))
        yield label, r


class TestAggregateEqualsSumOfParts(unittest.TestCase):
    """Phase `total_bytes` MUST equal Σ of its `per_op_timing` rows'
    `access_bytes_per_rank`, and its read/write split MUST be a partition of
    that same sum. The phase aggregate and the per-op breakdown are two
    consumers of the same leaf access bytes; if they disagree, one path
    silently re-derives differently (exactly the failure mode behind the
    head-broadcast decode-read bug — the per-op sum WAS the source of truth,
    so this would have flagged any future aggregate that stopped matching it).

    Held against `total_bytes`, not `mem_read_bytes`: a leaf's
    `access_bytes_per_rank` is `weight + metadata + act_in + act_out`, its
    whole traffic in both directions. The two were interchangeable only while
    `mem_write_bytes` was the literal zero."""

    def _check_phase(self, label, phase_name, phase, rows):
        if not rows:
            self.skipTest(f"{label}: no per_op rows ({phase_name})")
        agg = int(phase.get("total_bytes") or 0)
        parts = sum(int(r.get("access_bytes_per_rank") or 0) for r in rows)
        self.assertEqual(
            agg, parts,
            f"{label} {phase_name}: total_bytes {agg:,} != "
            f"Σ per_op access_bytes_per_rank {parts:,} "
            f"(Δ {agg - parts:,}) — aggregate drifted from its parts",
        )
        read = int(phase.get("mem_read_bytes") or 0)
        write = int(phase.get("mem_write_bytes") or 0)
        self.assertEqual(
            read + write, agg,
            f"{label} {phase_name}: read {read:,} + write {write:,} != "
            f"total {agg:,} — the split moved the sum it names halves of")
        self.assertEqual(
            write, sum(int(r.get("act_out_bytes_per_rank") or 0) for r in rows),
            f"{label} {phase_name}: the phase and the rows disagree about "
            f"what the step wrote")

    def test_decode_aggregate_matches_parts(self):
        for label, r in _cases():
            if r is None:
                continue
            self._check_phase(
                label, "decode", r.get("phases", {}).get("decode", {}),
                r.get("per_op_timing_decode"))

    def test_prefill_aggregate_matches_parts(self):
        for label, r in _cases():
            if r is None:
                continue
            self._check_phase(
                label, "prefill", r.get("phases", {}).get("prefill", {}),
                r.get("per_op_timing"))


class TestActiveParamsAreCountedNotInverted(unittest.TestCase):
    """`active_params_per_rank` MUST be the count the walk holds, not the
    rank's resident bytes divided by a weight width.

    The inversion agrees with the count on a dense model at a uniform dtype,
    which is why it stood: every parameter on the rank is active on every
    token. On an MoE it answers a different question — the rank holds every
    expert and a token routes to a few — and it read 3.05e10 on Qwen3-30B-A3B
    against the walk's 3.35e9, under a field name that says "active". Nothing
    pinned the field, so nothing could have caught it.

    The claim asserted here is the one that survives both: the field equals
    the per-rank walk's own total, and on a sparse model it is strictly below
    what that rank stores.
    """

    def test_a_sparse_rank_activates_less_than_it_stores(self):
        for label, snap, par in _CASES:
            r = _estimate(snap, tuple(sorted(par.items())))
            if r is None:
                continue
            wb = r.get("weight_breakdown") or {}
            resident_bytes = float(wb.get("total_bytes_per_rank") or 0)
            dtype_bytes = float(wb.get("weight_dtype_bytes") or 0)
            if not (resident_bytes and dtype_bytes):
                continue
            inverted = resident_bytes / dtype_bytes
            summary = r.get("summary") or {}
            total_p = float(summary.get("total_params") or 0)
            active_p = float(summary.get("total_active_params") or total_p)
            sparse = total_p > 0 and active_p < total_p * 0.9

            for phase_name in ("prefill", "decode"):
                declared = float((r.get("phases", {}).get(phase_name) or {})
                                 .get("active_params_per_rank") or 0)
                if declared <= 0:
                    continue
                self.assertLessEqual(
                    declared, inverted * 1.001,
                    f"{label} {phase_name}: active_params_per_rank "
                    f"{declared:,.0f} exceeds the {inverted:,.0f} parameters "
                    f"this rank even stores",
                )
                if sparse:
                    # The whole point. Equality here is the inversion coming
                    # back: the rank's resident count wearing the active name.
                    self.assertLess(
                        declared, inverted * 0.9,
                        f"{label} {phase_name}: the model activates "
                        f"{active_p / total_p:.1%} of its parameters, but "
                        f"active_params_per_rank {declared:,.0f} is the "
                        f"rank's full resident count {inverted:,.0f}",
                    )


class TestDecodeRowsCarryTheCacheRead(unittest.TestCase):
    """The per-op decode rows MUST account for the cache the step reads.

    Decode latency IS the cache read at any serving batch — the weights are
    read once and the cache is read once per request — so Σ per-op access
    bytes cannot be smaller than what the deployment says the cache costs.
    The two sides are derived by different routes on purpose: `kv_cache`
    reports the cache's own law per request, the rows come from the traffic
    derivation's leaves, and this asserts they meet.

    It is the invariant that went missing. `traffic.derive` clipped the
    cache-resident edges down to their fresh row and billed the bulk to the
    attention COMPONENT, while the per-op projection walks LEAVES only and
    skips every component — so the entire cache read landed on a row nothing
    downstream read, and decode `t_sol` reported a model that never touched
    its cache (Llama-3.2-3B at 4K: 12.7 ms, of which flash_attention was
    noise; the cache alone is 20 ms of H100 bandwidth). Every phase
    aggregate agreed with every part, because the parts agreed with each
    other about a number that was missing from all of them."""

    def test_decode_rows_cover_the_cache(self):
        for label, r in _cases():
            if r is None:
                continue
            rows = r.get("per_op_timing_decode")
            kv = r.get("kv_cache", {}) or {}
            stored = float(kv.get("effective_storage_tokens_per_request") or 0)
            read_positions = float(
                kv.get("kv_read_positions_at_decode_cache") or 0)
            batch = float((r.get("phases", {}).get("decode", {})
                           or {}).get("batch") or 0)
            if not (rows and stored and read_positions and batch):
                continue
            per_position = float(kv["per_request_kv_bytes"]) / stored
            cache_read = per_position * read_positions * batch
            rows_total = sum(int(x.get("access_bytes_per_rank") or 0)
                             for x in rows)
            self.assertGreaterEqual(
                rows_total, cache_read,
                f"{label}: Σ per-op decode access bytes {rows_total:,.0f} < "
                f"the cache read alone {cache_read:,.0f} — the rows are "
                f"missing the read that decode is made of",
            )


class TestKVReadNeverExceedsStorage(unittest.TestCase):
    """A query can never read more KV positions than are stored. DSA
    (read=Min(Lq,K), store=Lq) and SWA (read=store=Min(Lq,W)) both must
    satisfy read ≤ store; a regression that lets the read expr outgrow
    the storage expr (or vice-versa for an over-eager top-K) trips here
    without per-model thresholds."""

    def test_read_le_storage(self):
        for label, r in _cases():
            if r is None:
                continue
            kv = r.get("kv_cache", {}) or {}
            read = kv.get("kv_read_positions_at_decode_cache")
            if read is None:
                continue
            cache_len = kv.get("decode_cache_length") or (_SEQ + 128)
            self.assertLessEqual(
                int(read), int(cache_len),
                f"{label}: KV read positions {read} > stored cache "
                f"{cache_len} — a query read more than was cached",
            )


class TestPerTokenKVIsWorkloadInvariant(unittest.TestCase):
    """KV cost PER TOKEN is a property of the architecture + dtype, not
    of the workload: it must not change with how many tokens we plan to
    decode. (Per-REQUEST bytes correctly DO scale — cache len = input +
    output/2 — so the invariant lives at the per-token level.) Locks the
    boundary that the decode-read fix changed per-op read bytes but left
    the storage-per-token model untouched."""

    def test_per_token_bytes_same_across_output_len(self):
        for label, snap, par in _CASES:
            cfg = _config(snap)
            if cfg is None:
                continue
            pk = ParallelConfig(**par)

            def _per_request(out_len):
                r = estimate_capacity(
                     config=cfg, resolved=_resolved(snap), gpu_sku="H200",
                    num_gpus=8,
                    avg_input_tokens=_SEQ, avg_output_tokens=out_len,
                    parallel=pk, selection_mode="theoretical_only")
                return float((r.get("kv_cache") or {}).get(
                    "per_request_kv_bytes") or 0.0)

            # The served per-request numbers at two workloads must lie on
            # ONE storage law: their difference equals the reader's own
            # bytes_at difference at the same two lengths. A per-token
            # model that shifted with the workload would break the
            # equality; so would a payload computed off a different law
            # than the budget's.
            from tests.snapshots import shared_loader
            from backend.arch.eval.cache_view import cache_view_from_analysis
            from backend.arch.eval.program import Program
            from backend.arch.eval.transforms import Deployment
            resolved = shared_loader().match(
                cfg, model_id=Path(snap).stem.replace("__", "/", 1))
            analysis = Deployment(
                Program.of(resolved), config=pk,
            ).at(batch=1, input_tokens=128, logit_positions=1).analyze()
            reader = cache_view_from_analysis(analysis)
            len_a, len_b = _SEQ + 1, _SEQ + 512
            want = (reader.storage_bytes_at(len_b)
                    - reader.storage_bytes_at(len_a))
            got = _per_request(512) - _per_request(1)
            self.assertAlmostEqual(
                got, want, delta=max(1.0, abs(want) * 1e-9),
                msg=(f"{label}: per-request KV moved {got:,.0f} between "
                     f"out=1 and out=512 but the storage law says "
                     f"{want:,.0f} — two laws are in play"),
            )


def _weights_profile(roles, fmt):
    from backend.arch.eval.precision import PrecisionProfile
    return PrecisionProfile.canonicalize(
        {"weights": [{"select": {"roles": roles}, "format": fmt}]})


class TestOneWeightLedger(unittest.TestCase):
    """Tree per-rank, memory.weight_bytes_per_gpu, and the ledger are
    one number — as shipped, downcast, upcast, and per-role."""

    @pytest.mark.regression  # minutes-long
    def test_tree_matches_memory_across_profiles(self):
        profiles = (
            None,
            {"preset": "w_fp8_kv_fp8"},
            _weights_profile("all", {"dtype": "F32"}),
            _weights_profile(["attention"], {"scheme": "fp8"}),
        )
        for label, snap, par in _CASES:
            cfg = _config(snap)
            if cfg is None:
                continue
            for precision in profiles:
                for aux in (False, True):
                    r = estimate_capacity(
                         config=cfg, resolved=_resolved(snap), gpu_sku="H200", num_gpus=8,
                        avg_input_tokens=128, avg_output_tokens=32,
                        parallel=ParallelConfig(**dict(par)),
                        selection_mode="theoretical_only",
                        include_aux_weights=aux,
                        precision=precision,
                    )
                    tag = f"{label} {precision} aux={aux}"
                    tree = float((r.get("weight_breakdown") or {})
                                 .get("total_bytes_per_rank") or 0)
                    gpu = float((r.get("memory") or {})
                                .get("weight_bytes_per_gpu") or 0)
                    self.assertAlmostEqual(
                        tree, gpu, delta=1.0,
                        msg=f"{tag}: tree {tree:,.0f} vs "
                            f"memory.weight_bytes_per_gpu {gpu:,.0f}",
                    )
                    self.assertEqual(
                        (r.get("memory") or {}).get("weight_bytes_total"),
                        r.get("weight_bytes_total"),
                        msg=f"{tag}: wire has two cluster totals",
                    )


class TestReadsTrackResidentUnderCast(unittest.TestCase):
    """Reads and storage both come from the resident format, so a
    uniform upcast (every matrix BF16 → F32, ×2) moves billed and
    resident together and the ratio is stable. Both sides are uniform
    plain dtypes: on a mixed checkpoint (INT4 experts + BF16 attention)
    a cast from native moves the parts by different factors, and the
    ratio legitimately shifts with the read mix. (Absolute billed ≤
    resident is false when a tied LM head rereads the embedding table.)"""

    def test_upcast_does_not_open_a_read_storage_gap(self):
        # The decode batch is pinned: an upcast that no longer fits would
        # otherwise collapse the batch, and fewer routed experts read is a
        # workload change, not a read/storage gap.
        base = _weights_profile("all", {"dtype": "BF16"})
        upcast = _weights_profile("all", {"dtype": "F32"})
        for label, snap, par in _CASES:
            if int(par.get("pp") or 1) > 1:
                continue
            cfg = _config(snap)
            if cfg is None:
                continue
            pk = ParallelConfig(**dict(par))
            native = estimate_capacity(
                 config=cfg, resolved=_resolved(snap), gpu_sku="H200", num_gpus=8,
                avg_input_tokens=128, avg_output_tokens=32,
                parallel=pk, selection_mode="theoretical_only",
                prefill_batch_size=4, decode_batch_size=4,
                precision=base,
            )
            up = estimate_capacity(
                 config=cfg, resolved=_resolved(snap), gpu_sku="H200", num_gpus=8,
                avg_input_tokens=128, avg_output_tokens=32,
                parallel=pk, selection_mode="theoretical_only",
                prefill_batch_size=4, decode_batch_size=4,
                precision=upcast,
            )

            def _ratio(result):
                resident = float((result.get("memory") or {})
                                 .get("weight_bytes_per_gpu") or 0)
                rows = (result.get("per_op_timing_decode")
                        or result.get("per_op_timing") or [])
                billed = sum(
                    float(row.get("weight_read_bytes_per_rank") or 0)
                    + float(row.get("metadata_bytes_per_rank") or 0)
                    for row in rows
                )
                return billed / resident if resident else 0.0

            n_ratio, u_ratio = _ratio(native), _ratio(up)
            if n_ratio <= 0:
                continue
            self.assertAlmostEqual(
                u_ratio, n_ratio, delta=0.05,
                msg=f"{label}: billed/resident {n_ratio:.3f} BF16 vs "
                    f"{u_ratio:.3f} with F32 weights — reads and storage "
                    f"did not move together",
            )


class TestPeakActivationUsesTheEdgeDtype(unittest.TestCase):
    """Peak occupancy prices each produced tensor at
    `ResolvedPrecision.edge_bytes`, the same answer HBM traffic uses. A
    scalar on the peak path was the dual: a per-role activation rule
    moved the rows and left the OOM headroom where it was.

    Residual adds and norms often have no role, so covering every named
    role does not move the whole worst-block peak. The join is per
    produced tensor: a roled leaf follows its role's rule, a role-less
    leaf stays at the model's own dtype.
    """

    def test_roled_leaves_follow_per_role_width(self):
        from backend.arch.dtypes import dtype_bytes
        from backend.arch.eval.precision import PrecisionProfile, ResolvedPrecision
        from backend.arch.layout.physical import physical_kernel_shapes
        from tests.snapshots import shared_loader
        from backend.inference.activation_peak import _eval_leaf_records
        from backend.inference.step_eval import evaluate_step

        loader = shared_loader()
        for label, snap, par in _CASES:
            cfg = _config(snap)
            if cfg is None:
                continue
            resolved = loader.match(
                cfg, model_id=Path(snap).stem.replace("__", "/", 1))
            if resolved is None:
                continue
            analysis = evaluate_step(
                resolved, ParallelConfig(**dict(par)), phase="prefill",
                batch=1, input_tokens=128, output_tokens=0,
            ).analysis
            program = analysis.program
            base = analysis.resolved_precision
            roles = sorted({base.role_value(n.role)
                            for n in program.nodes() if n.role})
            self.assertTrue(roles, f"{label}: program has no roles")
            role_of = {
                n.aggregation_id: n.role
                for n in program.nodes() if not n.children
            }
            over = ResolvedPrecision(PrecisionProfile.canonicalize(
                {"activations": [{"select": {"roles": roles},
                                  "format": {"dtype": "F32"}}]}), program)
            widen = 4 / dtype_bytes(base.model_dtype)
            shapes = physical_kernel_shapes(analysis)
            rec_base, _ = _eval_leaf_records(program, shapes, base)
            rec_over, _ = _eval_leaf_records(program, shapes, over)
            saw_roled = False
            for key, rows in rec_base.items():
                by_id = {r["id"]: r for r in rec_over[key]}
                for r in rows:
                    o = by_id[r["id"]]
                    if role_of.get(r["id"]):
                        if r["out_bytes"]:
                            saw_roled = True
                            self.assertEqual(
                                o["out_bytes"], r["out_bytes"] * widen,
                                f"{label} {r['id']}: roled leaf did not "
                                f"follow its role's edge dtype",
                            )
                    else:
                        self.assertEqual(
                            o["out_bytes"], r["out_bytes"],
                            f"{label} {r['id']}: role-less leaf moved "
                            f"under a role-only rule",
                        )
            self.assertTrue(
                saw_roled, f"{label}: no roled leaf produced bytes",
            )

    def test_peak_asks_the_precision_profile(self):
        src = (PROJECT_ROOT / "backend/inference/activation_peak.py").read_text()
        self.assertIn("edge_bytes", src)
        self.assertNotIn(
            "_ACT_FUSIONS", src,
            "peak still grafts from an op-name allow-list",
        )
        self.assertNotIn(
            "_eval_fusion_matches", src,
            "peak still matches fusions besides rewrite_serving_graph",
        )


class TestRenderJoinIsUid(unittest.TestCase):
    """Shape join in rendering/ is on uid. Reconstructing a key from a
    namespaced id (`__block__<tmpl>/…` → `tmpl|path`) is the dual that
    cut 3 deleted."""

    def test_rendering_does_not_rebuild_join_keys_from_ids(self):
        root = PROJECT_ROOT / "backend" / "rendering"
        offenders = []
        for path in root.rglob("*.py"):
            src = path.read_text()
            rel = path.relative_to(root)
            if 'startswith("__block__")' in src:
                offenders.append(f"{rel}: strips a __block__ prefix to look up")
            if 'f"{template}|' in src or "f'{template}|'" in src:
                offenders.append(f"{rel}: rebuilds template|path")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
