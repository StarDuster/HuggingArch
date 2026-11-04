#!/usr/bin/env python3
"""Axiom-battery self-audit for HuggingArch inference accounting.

The deterministic half of `audit-curves` (see SKILL.md). For a sample of models
it rebuilds the SAME curves the compare page's 7 charts show — across seq and
batch — and checks each against first-principles infra **axioms** (see
`axioms.py`): a universal monotonicity tier that holds for every architecture,
and a spec-derived tier whose expected SHAPE is read from the model's own
declared attention read/storage cardinality. No magic thresholds; a violation
prints expected-vs-measured so it is self-explaining.

Usage:
    python sweep.py [--models N] [--include M1,M2] [--seqs ...] [--batches ...] [--json out]
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path


def _ensure_hashseed():
    """Pin PYTHONHASHSEED before any real work so the sweep is reproducible
    across processes. Dict/set iteration order (hence float-sum accumulation
    order AND argmax tie-breaks) is hash-seeded, and a borderline axiom result
    that sits on a rail can otherwise flip run-to-run purely from that — the
    exact "same-seed rerun disagrees" flake. Hash seed must be fixed BEFORE the
    interpreter starts, so re-exec ourselves once with it pinned. (The `--seed`
    flag only controls model sampling, not this.)"""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from axioms import (
    AxiomResult, derive_expected_shape, run_axioms, is_structural,
)
from reconcile import reconcile_model

# Span the FULL compare-page range, both ends: short (1K) through 1M — the long
# end is where S²/over-materialisation/OOM bugs surface, so it must be covered.
# (Enough points for monotonicity + tail/steepest log-log slope; --seqs to widen.)
_DEFAULT_SEQS = [1024, 8192, 65536, 262144, 1048576]


# The deployment the sweep audits: one Scaling-panel mode at one degree
# (`parallel_sweep.config_for_degree`), set from --mode / --degree.
_MODE, _DEGREE = "dp_ep", 8


def _par():
    from backend.inference.parallel_sweep import config_for_degree
    return config_for_degree(_MODE, _DEGREE)


def _est(cfg, model_id, seq, b):
    """One estimate at (seq, batch). b == 'max' → capacity-admitted batch."""
    from backend.inference.capacity import estimate_capacity
    # Precision is the as-shipped profile (the request's `precision` default),
    # the same deployment the compare page draws.
    kw = dict(config=cfg, model_id=model_id, gpu_sku="B200", num_gpus=_DEGREE, avg_input_tokens=seq,
              avg_output_tokens=256, gpu_memory_fraction=0.9, parallel=_par(),
              selection_mode="measured")
    if b == "max":
        kw["prefill_batch_size"] = -1     # capacity max
        # decode_batch_size left None → auto max
    else:
        kw["decode_batch_size"] = int(b)
        kw["prefill_batch_size"] = int(b)
    return estimate_capacity(**kw)


def _module_flops(phase_dict):
    """Module flops-by-role from the phase's `flops_by_module` tree. Only the
    COMPONENT-level rows carry a role (`self_attn`→attention, the FFN→ffn_moe/
    ffn_dense, router→router); leaf rows are role=None and the block_template
    row double-counts — both are skipped. The per_op_timing leaf rows have NO
    role, so this is the only sound module attribution."""
    acc: dict = {}
    for m in phase_dict.get("flops_by_module") or []:
        role = m.get("role")
        # skip wrapper rows that double-count their children (block_template /
        # block / aux_block) and the role-less leaves.
        if role in (None, "block_template", "block", "aux_block"):
            continue
        acc.setdefault(role, []).append(float(m.get("flops_per_rank") or 0))
    # fsum → the total is independent of accumulation order (belt-and-suspenders
    # with the hashseed pin: the attn_frac these feed then can't wobble on order).
    return {role: math.fsum(vals) for role, vals in acc.items()}


def _role_buckets(rows):
    """Σ access_bytes by role (HBM footprint per module), plus the single
    top-access op AND the top-access STRUCTURAL op (not a raw primitive).
    access_bytes_per_rank already folds the layer count, so it is the total HBM
    a module touches — the right 'where does the work go' measure, and it
    sidesteps the t_op×count MoE over-count (all experts vs top-K)."""
    by_role: dict = {}
    top = None
    struct = None
    for x in rows or []:
        role = x.get("role") or "other"
        op = x.get("op", "?")
        acc = float(x.get("access_bytes_per_rank") or 0)
        by_role[role] = by_role.get(role, 0.0) + acc
        rec = {"op": op, "role": role, "access": acc, "t": float(x.get("t_op_s") or 0)}
        if top is None or acc > top["access"]:
            top = rec
        if is_structural(op) and (struct is None or acc > struct["access"]):
            struct = rec
    return by_role, top, struct


def build_curves(label, cfg, model_id, seqs, batches):
    """Assemble the per-model curve bundle axioms.py consumes."""
    S = {
        "kv_per_req": {}, "max_decode_batch": {}, "max_prefill_batch": {},
        "kv_budget": {},
        "tpot_ms": {b: {} for b in batches}, "prefill_ms": {b: {} for b in batches},
        "prefill_tok_s": {b: {} for b in batches}, "decode_tok_s": {b: {} for b in batches},
        "attn_frac": {b: {} for b in batches},
        "role_flops": {}, "top_access_op": {},
        # reconciliation inputs: the roofline decomposition the estimate already
        # produces (t_sol = Σ_leaf max(compute, memory, comm, launch)), so each chart value can
        # be reconstructed from first principles and shown predicted-vs-measured.
        "roofline": {}, "peaks": {}, "mem": {},
        # The weights a B=1 decode step actually reads, per rank — the
        # engine's own number (`weight_read_bytes_per_rank`), not a
        # re-derivation from the per-token active set: a routed bank reads for
        # the rows it RECEIVES, which under DP attention are every rank's.
        "weight_read_b1": {},
    }
    # A point whose estimate raises is reported, not skipped: a swallowed
    # exception leaves the curve empty and every axiom passes vacuously.
    errors: list = []
    for seq in seqs:
        for b in batches:
            try:
                r = _est(cfg, model_id, seq, b)
            except Exception as e:
                errors.append({"seq": seq, "batch": str(b),
                               "error": f"{type(e).__name__}: {str(e)[:160]}"})
                continue
            ph = r.get("phases") or {}
            dec, pre = ph.get("decode") or {}, ph.get("prefill") or {}
            dt = float(dec.get("t_sol_s") or 0)
            pt = float(pre.get("t_sol_s") or 0)
            if dt > 0:
                S["tpot_ms"][b][seq] = dt * 1e3
                eff = float((dec.get("binding") or {}).get("effective_tokens_per_step") or (b if b != "max" else 0)) or 1
                S["decode_tok_s"][b][seq] = eff / dt
            if pt > 0:
                S["prefill_ms"][b][seq] = pt * 1e3
                effp = float((pre.get("binding") or {}).get("effective_tokens_per_step") or 0) or seq
                S["prefill_tok_s"][b][seq] = effp / pt
            # module flops-by-role (component-level; leaf per_op rows have no
            # role) — of a point the estimate priced. One that admits no
            # request has no prefill phase and no breakdown; read as a share
            # it would be an attention fraction of 0 the model never had.
            if pt > 0:
                roles = _module_flops(pre)
                S["role_flops"][(seq, b)] = roles
                tot = sum(roles.values()) or 1.0
                S["attn_frac"][b][seq] = roles.get("attention", 0.0) / tot
            for phase, key in (("prefill", "per_op_timing"), ("decode", "per_op_timing_decode")):
                _, top, struct = _role_buckets(r.get(key))
                if top:
                    S["top_access_op"][(seq, b, phase)] = {"top": top, "struct": struct}
            # roofline terms per phase for the reconciliation pass
            S["roofline"][(seq, b)] = {
                phase: {
                    "compute": float(d.get("t_compute_s") or 0),
                    "memory": float(d.get("t_memory_s") or 0),
                    "comm": float(d.get("t_comm_s") or 0),
                    "sol": float(d.get("t_sol_s") or 0),
                    "access": float(d.get("total_bytes") or d.get("mem_read_bytes") or 0),
                    "flops": float(d.get("flops_per_step") or 0),
                } for phase, d in (("decode", dec), ("prefill", pre))
            }
            if not S["peaks"]:
                S["peaks"] = {"bw": float(r.get("peak_hbm_bw_bytes_per_sec") or 0),
                              "flops": float(r.get("peak_flops_per_sec") or 0)}
            if b == "max":
                kv = r.get("kv_cache") or {}
                mem = r.get("memory") or {}
                ap = r.get("activation_peak") or {}
                S["kv_per_req"][seq] = float(kv.get("per_request_kv_bytes") or 0)
                S["max_decode_batch"][seq] = float(r.get("max_decode_batch_per_gpu") or 0)
                S["max_prefill_batch"][seq] = float(r.get("max_prefill_batch_per_gpu") or 0)
                S["kv_budget"][seq] = float(mem.get("kv_budget_bytes") or 0)
                S["mem"][seq] = {
                    "weight_per_gpu": float(mem.get("weight_bytes_per_gpu") or 0),
                    "usable": float(mem.get("usable_bytes") or 0),
                    "act_peak": float(ap.get("peak_activation_bytes_per_rank") or 0),
                }
            if b == 1:
                S["weight_read_b1"][seq] = sum(
                    float(x.get("weight_read_bytes_per_rank") or 0)
                    for x in (r.get("per_op_timing_decode")
                              or r.get("per_op_timing") or []))
    return {"model": label, "seqs": seqs, "batches": batches, "series": S,
            "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=4)
    ap.add_argument("--include", default="deepseek-ai__DeepSeek-V3,deepseek-ai__DeepSeek-V3.2,deepseek-ai__DeepSeek-V4-Pro")
    ap.add_argument("--seqs", default=",".join(str(s) for s in _DEFAULT_SEQS))
    ap.add_argument("--batches", default="1,16,64,max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--mode", default="dp_ep",
                    choices=("dp_ep", "cp_ep", "tp_ep", "tp_tp"))
    ap.add_argument("--degree", type=int, default=8)
    a = ap.parse_args()
    global _MODE, _DEGREE
    _MODE, _DEGREE = a.mode, a.degree

    snap_dir = ROOT / "backend/arch/snapshots"
    all_snaps = sorted(p.stem for p in snap_dir.glob("*.json"))
    include = [s for s in a.include.split(",") if s]
    pool = [s for s in all_snaps if s not in include]
    pick = []
    if pool and a.models > len(include):
        n = a.models - len(include)
        step = max(1, len(pool) // max(1, n))
        pick = [pool[(a.seed + i * step) % len(pool)] for i in range(n)]
    chosen = include + pick
    seqs = [int(x) for x in a.seqs.split(",")]
    batches = [(x if x == "max" else int(x)) for x in a.batches.split(",")]

    cohort: dict = {}
    expected: dict = {}
    errors: list = []
    for stem in chosen:
        snap = snap_dir / f"{stem}.json"
        if not snap.exists():
            continue
        label = stem.split("__")[-1]
        try:
            doc = json.loads(snap.read_text())
            cfg = doc["config"]
            cohort[label] = build_curves(label, cfg, doc["model_id"], seqs, batches)
            expected[label] = derive_expected_shape(doc["model_id"], cfg)
            errors.extend({"model": label, **e} for e in cohort[label]["errors"])
        except Exception as e:
            errors.append({"model": label, "error": str(e)[:160]})

    results: list = []
    reconciliation: dict = {}
    for label, curves in cohort.items():
        results.extend(run_axioms(curves, expected.get(label, {"known": False})))
        try:
            reconciliation[label] = reconcile_model(curves, expected.get(label, {}))
        except Exception as e:
            reconciliation[label] = [f"reconciliation failed: {type(e).__name__}: {e}"]

    fails = [r for r in results if not r.passed]
    by_model: dict = {}
    for r in results:
        by_model.setdefault(r.model, [0, 0])
        by_model[r.model][1] += 1
        if r.passed:
            by_model[r.model][0] += 1

    def _d(r: AxiomResult):
        return {"model": r.model, "axiom_id": r.axiom_id, "tier": r.tier,
                "chart": r.chart, "passed": r.passed, "expected": r.expected,
                "measured": r.measured, "detail": r.detail}

    out = {
        "sampled": chosen, "mode": _MODE, "degree": _DEGREE, "seqs": seqs, "batches": [str(b) for b in batches],
        "n_fail": len(fails),
        "pass_summary": {m: f"{p}/{t} pass" for m, (p, t) in sorted(by_model.items())},
        "expected": {m: {k: e.get(k) for k in ("known", "labels", "variants",
                     "kv_class", "decode_class", "prefill_band", "has_unbounded", "is_moe")}
                     for m, e in expected.items()},
        "failures": [_d(r) for r in fails],
        "reconciliation": reconciliation,
        "errors": errors,
    }
    # Human-readable reconciliation goes to STDERR so stdout stays pure JSON
    # (gate.py parses it). This is the "explain every chart" summary.
    for label in cohort:
        for line in reconciliation.get(label, []):
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _ensure_hashseed()   # re-exec once with PYTHONHASHSEED=0, then run for real
    main()
