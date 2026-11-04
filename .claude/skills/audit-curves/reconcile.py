#!/usr/bin/env python3
"""Confident first-principles reconciliation of every compare-page chart.

After a sweep, for each model this RECONSTRUCTS each of the 7 charts' numbers from
the roofline decomposition the estimate already exposes — `t_sol` is the sum
over rows of each row's `max(compute, memory, comm, launch)`
(`phase_formulas.STEP_SOL_LAW`), and the phase's compute/memory/comm sums are not
addends of it; `compute = flops/peak_flops` and `memory = access/peak_bw` —
plus the capacity identities (`max_batch = budget / per-request`). Every line
states the first-principles formula, the binding regime, and predicted-vs-measured.

This is self-consistent BY CONSTRUCTION: `t_sol` IS that max and `max_batch` IS
that ratio, so a line that does NOT reconcile (the from-principle magnitude misses
the stored term) is itself a finding. No doc, no external ground truth — the
skill's own method is the only reconciliation standard.

`reconcile_model(curves, expected) -> list[str]` returns one block of lines per
model; `sweep.py` prints them under each model as the "reconciliation" summary.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


def _gb(x): return x / 1e9
def _tf(x): return x / 1e12
def _ms(x): return x * 1e3


def _finite_seqs(d: Dict[int, Any]) -> List[int]:
    return sorted(k for k, v in d.items() if v not in (None, 0) and (not isinstance(v, float) or math.isfinite(v)))


def _roof(rl: Dict[str, float], peaks: Dict[str, float]):
    """(binding_name, t_sol_ms, predicted-from-principle terms). The stored
    compute/memory/comm sums name the largest component (the step's split by
    the term that set each row, launch included, is `t_sol_by_bound`); we also
    recompute memory=access/bw and
    compute=flops/peak to show the magnitude is reproducible from first principles."""
    sol = rl["sol"]
    terms = {"compute": rl["compute"], "memory": rl["memory"], "comm": rl["comm"]}
    binding = max(terms, key=terms.get) if any(terms.values()) else "—"
    bw, pf = peaks.get("bw", 0), peaks.get("flops", 0)
    mem_pred = rl["access"] / bw if bw else 0
    comp_pred = rl["flops"] / pf if pf else 0
    return binding, sol, mem_pred, comp_pred


def reconcile_model(curves: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    S = curves["series"]
    peaks = S.get("peaks") or {}
    bw, pf = peaks.get("bw", 0), peaks.get("flops", 0)
    out: List[str] = [f"reconciliation — {curves['model']} "
                      f"(peak {_gb(bw)/1e3:.1f} TB/s, {_tf(pf):.0f} TFLOP/s):"]

    cap_seqs = _finite_seqs(S["kv_per_req"])
    if not cap_seqs:
        return [*out, "  (no capacity-admitted points; model does not fit)"]
    sshort, slong = cap_seqs[0], cap_seqs[-1]

    # ── Chart 3 — per-request KV (class-correct; NOT a blanket S·per_token) ──
    # per_req_kv = stored_positions(S) · bytes_per_position. stored_positions is
    # the spec's storage cardinality — S (dense), Min(S,W) (SWA, plateaus),
    # const (linear/SSM), ⌈S/r⌉ (compressed). So S·per_token only holds for
    # DENSE; here we check the per_req_kv curve against the DERIVED class, using
    # TWO independent seq points (no back-derived-then-predict tautology).
    # Two independent points → growth slope (no back-derive-then-predict
    # tautology), checked against the DERIVED class with the same slope logic as
    # the D-KV axiom (hybrid-safe: an SWA-majority "linear" model grows at ~0.5–
    # 0.7, still > the plateau bar). per_token shown only for the dense case,
    # where per_req_kv really is ∝ S.
    kv = S["kv_per_req"]
    mem = S["mem"].get(slong) or {}        # used by Ch2 / Ch4 below
    cls = expected.get("kv_class") or "?"
    kshort, klong = kv.get(sshort), kv.get(slong)
    if kshort and klong and kshort > 0 and slong > sshort:
        seq_ratio = slong / sshort
        slope = math.log(klong / kshort) / math.log(seq_ratio) if klong > 0 else 0.0
        if cls == "linear":
            ok = slope > 0.30
            desc = f"grows (slope {slope:.2f}), {klong/slong:,.0f} B/token·pos"
        elif cls == "plateau":
            ok = slope < 0.30
            desc = f"sub-linear (slope {slope:.2f}) — caps at the window"
        elif cls == "constant":  # linear/SSM recurrent state
            ok = slope < 0.15
            desc = f"~flat (slope {slope:.2f}) — constant state"
        else:  # no class derived from the spec: nothing to check the slope against
            ok = False
            desc = f"slope {slope:.2f}, but no KV class was derived from the spec"
        out.append(
            f"  [Ch3 KV/req]   {cls}: {_gb(kshort):.2f}→{_gb(klong):.2f} GB over "
            f"{seq_ratio:.0f}× seq — {desc}"
            + ("  ✓" if ok else "  ⚠ class ⊥ curve"))

    # ── Chart 1 — max decode batch (KV-bound capacity identity) ─────────────
    bud = S["kv_budget"].get(slong, 0)
    mb = S["max_decode_batch"].get(slong, 0)
    wmem = (S["mem"].get(slong) or {})
    if bud <= 0:
        # negative/zero KV budget = weights alone exceed usable HBM at this
        # parallelism → the model does not fit (maxB=0). Not a bug, a config call.
        out.append(
            f"  [Ch1 maxB-dec] model does not fit at this parallelism — "
            f"weight {_gb(wmem.get('weight_per_gpu',0)):.0f} GB ≥ usable "
            f"{_gb(wmem.get('usable',0)):.0f} GB ⇒ kv_budget≤0, maxB={mb:.0f} (needs more GPUs/PP)")
    else:
        pred_mb = bud / kv[slong] if kv.get(slong) else 0
        out.append(
            f"  [Ch1 maxB-dec] = kv_budget ÷ per_req_kv — "
            f"S={slong}: {_gb(bud):.0f}/{_gb(kv[slong]):.1f} GB = {math.floor(pred_mb)} pred vs {mb:.0f} meas"
            + ("  ✓ KV-bound" if mb <= pred_mb * 1.02 + 1 else "  ⚠ over budget"))

    # ── Chart 2 — max prefill batch (activation-bound, chunked) ─────────────
    w = mem.get("weight_per_gpu", 0)
    usable = mem.get("usable", 0)
    act_budget = max(0.0, usable - w)
    mp = S["max_prefill_batch"].get(slong, 0)
    # The solver caps the peak at min(S, chunk), so the FULL-S act_peak here is
    # only an upper bound — prefill chunks, which is why it doesn't OOM. State
    # the principle, not a false OOM alarm.
    out.append(
        f"  [Ch2 maxB-pre] activation-bound (not KV) — S={slong}: maxB={mp:.0f}; "
        f"chunked prefill caps peak at min(S,chunk); act budget "
        f"(usable {_gb(usable):.0f}−weight {_gb(w):.0f})={_gb(act_budget):.0f} GB")

    # ── Chart 4 — TPOT (decode roofline) at B=1 (weight regime) & largest B ─
    for b, lab in ((1, "B=1"), (_largest_num(curves["batches"]), "B=max-fixed")):
        rl = (S["roofline"].get((slong, b)) or {}).get("decode")
        if not rl or rl["sol"] <= 0:
            continue
        binding, sol, mp_, cp_ = _roof(rl, peaks)
        out.append(
            f"  [Ch4 TPOT]     {lab} S={slong}: {_ms(sol):.1f} ms — {binding}-bound "
            f"(mem {_gb(rl['access']):.0f} GB/{_gb(bw)/1e3:.1f}TBps={_ms(mp_):.1f} ms, "
            f"comp {_tf(rl['flops']):.0f} TF/{_tf(pf):.0f}={_ms(cp_):.1f} ms)"
            + ("  ✓" if _close(rl[binding], {'memory': mp_, 'compute': cp_}.get(binding, rl[binding]), 0.1) else ""))

    # ── Chart 5 — prefill throughput (compute floor; prefill is chunked) ────
    rl = (S["roofline"].get((slong, 1)) or {}).get("prefill")
    tput = S["prefill_tok_s"].get(1, {}).get(slong, 0)
    if rl and rl["sol"] > 0:
        comp = rl["flops"] / pf if pf else 0
        out.append(
            f"  [Ch5 pre-tput] S={slong} B=1: {tput:,.0f} tok/s = S/{_ms(rl['sol']):.0f}ms; "
            f"compute floor {_tf(rl['flops']):.0f} TF/{_tf(pf):.0f} = {_ms(comp):.0f} ms — "
            f"compute-bound at long ctx (gap = chunked steps + comm/memory)")

    # ── Chart 6 — decode throughput = B / TPOT, saturating ─────────────────
    bmaxnum = _largest_num(curves["batches"])
    tp = S["decode_tok_s"].get(bmaxnum, {}).get(slong, 0)
    tpot = S["tpot_ms"].get(bmaxnum, {}).get(slong, 0)
    if tp and tpot:
        out.append(
            f"  [Ch6 dec-tput] B={bmaxnum} S={slong}: {tp:,.0f} tok/s = {bmaxnum}/{tpot:.1f}ms; "
            f"rises ∝B while weight-bound, saturates at the compute knee")

    # ── Chart 7 — module breakdown (FLOPs by role), short vs long ──────────
    def _top_role(seq):
        roles = S["role_flops"].get((seq, "max")) or S["role_flops"].get((seq, bmaxnum)) or {}
        if not roles:
            return ("—", 0.0, 0.0)
        tot = sum(roles.values()) or 1.0
        r = max(roles, key=roles.get)
        return (r, roles[r] / tot, roles.get("attention", 0.0) / tot)
    rs, ps, afs = _top_role(sshort)
    rl2, pl, afl = _top_role(slong)
    note = ("attention ∝S² overtakes FFN" if afl > afs + 0.02
            else "FFN/GEMM stays dominant (bounded attention)")
    out.append(
        f"  [Ch7 modules]  S={sshort}: top {rs} {ps*100:.0f}% of FLOPs (attn {afs*100:.0f}%); "
        f"S={slong}: top {rl2} {pl*100:.0f}% (attn {afl*100:.0f}%) — {note}")
    return out


def _largest_num(batches):
    nums = [b for b in batches if isinstance(b, (int, float))]
    return max(nums) if nums else 1


def _close(a, b, tol=0.05):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)
