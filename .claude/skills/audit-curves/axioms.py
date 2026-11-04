#!/usr/bin/env python3
"""First-principles infra axioms for the audit-curves self-audit.

Two tiers, plus role/regime/conservation laws. The design rule: **no tuned
thresholds**. The only constants are float-noise tolerance (ε) and the two
*physical* prefill-exponent bands that separate the only two regimes a
transformer prefill can be in — linear (~1) vs quadratic (~2).

  Tier-0 (universal) — monotonicity. Holds for EVERY architecture; needs no
    spec and no attention class. A violation is an unconditional accounting bug.

  Tier-1 (spec-derived) — the curve SHAPE (constant/plateau/linear KV, flat/linear
    decode, linear/quadratic prefill) is DERIVED from the model's own declared
    attention read/storage cardinality (`kv_cache_topology`), never asserted.
    Curve ⊥ declaration ⇒ a spec-declaration bug or an accounting bug.

  M / R / K — module-dominance, batch-regime, capacity-conservation. Read off
    `per_op_timing` roles + the capacity identity. Threshold-free where possible
    (relative ordering, sign of a second difference).

Each axiom is a pure function `(curves, expected) -> list[AxiomResult]`. The
caller (sweep.py) builds `curves` from `estimate_capacity` over the seq×batch
sweep — the same fields the compare page's 7 charts read — and `expected` from
`derive_expected_shape`.

`A5` cp_scheme-invariance lives in `backend/inference/invariants.py` (an exact
output-vs-output invariant, not a curve law) and is referenced, not duplicated.
`A6` dtype/roofline is a documented phase-2 extension, not built here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# `_EPS` is a genuine float-noise tolerance (monotonicity, dominance ties).
# `_P_*` are descriptive exponent bands used only to LABEL the derived shape in
# `derive_expected_shape` (for reconciliation display and the M-axiom guards) —
# they no longer gate any pass/fail. The former slope-rail shape "axioms"
# (D-KV/D-Dec/D-PreExp) were DELETED: KV runtime bytes have no external ground
# truth (unlike weights ↔ checkpoint), so a curve-slope check against the spec's
# own declared cardinality is vacuous, and its magic rails were never axioms.
# See SKILL.md "What was deleted, and where the intent really lives".
# ─────────────────────────────────────────────────────────────────────────────
_EPS = 1e-3                    # float-noise tolerance for monotonicity (real)
_P_LINEAR = (0.80, 1.30)       # descriptive exponent band: linear-regime label
_P_QUADRATIC = (1.60, 2.20)    # descriptive exponent band: quadratic-regime label

# RAW arithmetic / broadcast / reduce primitives. These are the building blocks
# a decomposed einsum/outer-product leaves behind; a real kernel fuses them, so
# one of these being the single LARGEST HBM op (out-reading every GEMM and
# attention op) means a decomposed intermediate is the biggest thing in memory —
# the HC `[B,S,Hc,Hc,H]` outer product, the pre-fix indexer `mul`/`sum_reduce`.
# NOT in this set (legitimately allowed to be #1): GEMMs (`*linear*`, matmul),
# attention, fused activations (`silu_and_mul`, `gelu_and_mul`), bare activations
# (`silu`,`gelu`,`relu`), norms, and view ops — all of which read their natural
# `[B,S,·]` tensors proportional to the GEMMs around them.
_RAW_PRIMITIVE = frozenset({
    "mul", "add", "sub", "div", "neg", "broadcast", "head_broadcast",
    "sum_reduce", "mean_reduce", "scale", "residual_add",
})


def is_raw_primitive(op: str) -> bool:
    o = (op or "").lower()
    return any(p == o or o.startswith(p) for p in _RAW_PRIMITIVE)


# Layout/view ops materialise a buffer BY DESIGN (legit to be large), but they
# are not the "real compute" baseline a raw primitive should be compared to.
_VIEW_OPS = frozenset({
    "concat", "transpose", "slice", "view", "view_split", "view_merge",
    "head_split", "head_merge", "reshape", "permute", "split", "stack", "cat",
})


def is_structural(op: str) -> bool:
    """A real compute op (GEMM / attention / activation / norm) — the thing that
    should top the HBM ranking. Excludes raw primitives AND view/layout ops."""
    o = (op or "").lower()
    if is_raw_primitive(o):
        return False
    return not any(v == o or o.startswith(v) for v in _VIEW_OPS)


@dataclass
class AxiomResult:
    axiom_id: str
    tier: str            # "0" | "1" | "M" | "R" | "K"
    chart: str           # compare-page chart number(s) this governs
    model: str
    passed: bool
    expected: str        # what first principles / the spec predict
    measured: str        # what the curve shows
    detail: str = ""     # one-line human explanation when failed


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pairs(d: Dict[int, float]):
    """Sorted (x0,y0,x1,y1) consecutive pairs over a {x: y} curve, finite only."""
    xs = sorted(k for k, v in d.items() if v is not None and math.isfinite(v))
    return [(xs[i], d[xs[i]], xs[i + 1], d[xs[i + 1]]) for i in range(len(xs) - 1)]


def _first_violation_nondecr(d: Dict[int, float]):
    """First point where the curve DROPS more than ε (non-decreasing law)."""
    for x0, y0, x1, y1 in _pairs(d):
        if y0 > 0 and y1 < y0 * (1.0 - _EPS):
            return (x0, y0, x1, y1)
    return None


def _first_violation_noninc(d: Dict[int, float]):
    """First point where the curve RISES more than ε (non-increasing law)."""
    for x0, y0, x1, y1 in _pairs(d):
        if y0 > 0 and y1 > y0 * (1.0 + _EPS):
            return (x0, y0, x1, y1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Spec-derived expected shape
# ─────────────────────────────────────────────────────────────────────────────
def derive_expected_shape(model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Read each attention component's declared read/storage cardinality and
    compose a model-level expectation. `model_id` is the HF id (`org/name`) the
    loader resolves the spec by. Returns a dict with `known: bool`, False only
    when no spec is registered for `model_id`; any other failure raises rather
    than reading as an unknown class.

    Classification uses only boundedness (the existing `_storage_is_bounded`,
    `limit(expr,Lq,∞).is_finite`):
      storage None        → 'constant'  (no cache: linear/SSM)
      storage bounded      → 'plateau'   (windowed / capped)
      storage unbounded    → 'linear'    (dense / compressed ⌈S/r⌉ both grow)
    Read side picks the prefill exponent: any unbounded read stream ⇒ an S²
    term exists ⇒ prefill band = quadratic; all-bounded/None ⇒ linear.
    """
    out: Dict[str, Any] = {
        "known": False, "labels": [], "variants": [],
        "kv_class": None, "decode_class": None, "prefill_band": None,
        "has_unbounded": False, "has_cache": False, "is_moe": False, "streams": [],
    }
    from backend.arch.loader import SpecLoader
    from backend.arch.structure import iter_reachable_components
    from backend.arch.block_synthesis import binding_of
    from backend.arch.eval.transforms import CallSite
    from backend.inference.kv_topology import kv_cache_topology
    from backend.arch.validate._axioms import _storage_is_bounded

    resolved = SpecLoader().match(config, model_id=model_id)
    if resolved is None:
        return out

    any_attn = False
    for name, bg in iter_reachable_components(resolved):
        role = getattr(bg, "role", None)
        if role in ("ffn_moe", "router"):
            out["is_moe"] = True
        if role != "attention":
            continue
        any_attn = True
        out["labels"].append(getattr(bg, "kv_label", None))
        out["variants"].append(getattr(bg, "attn_variant", None))
        # Evaluate the component at every call the model makes of it (DSv3
        # calls its MLA in the dense layers, the MoE layers and the MTP
        # block); an attention component nothing calls is a spec fact the
        # audit must not paper over, so it raises.
        sites = [site for site in CallSite.calls(resolved)
                 if binding_of(resolved.spec, site.component, name)]
        if not sites:
            raise ValueError(f"{model_id}: attention component {name!r} has no call")
        for site in sites:
            topo = kv_cache_topology(resolved, name, site=site)
            # Per-STREAM, not the headline: the headline storage expr for a DSA/
            # CSA component is the bounded main-attention path (Min(W,Lq)), but a
            # cached node like `compressor: ceiling(Lq/4)` grows. ANY cached node
            # that stores ∝ S means an unbounded stream exists — it is then read
            # densely in PREFILL (score matrix is Sq × stored_positions ⇒ an S²
            # term) and its cache grows ⇒ KV(S) is linear. So `has_unbounded`
            # (over every cached node) is the master signal for both prefill
            # quadratic-ness and KV growth.
            store = getattr(topo, "kv_storage_positions_per_query_expr", None)
            exprs = [store] + [getattr(n, "storage_positions_expr", None)
                               for n in (getattr(topo, "cached_nodes", None) or [])]
            present = [x for x in exprs if x is not None]
            if present:
                out["has_cache"] = True
            for x in present:
                if _storage_is_bounded(x) is False:
                    out["has_unbounded"] = True
            out["streams"].append({
                "component": name, "call": site.call, "label": getattr(bg, "kv_label", None),
                "variant": getattr(bg, "attn_variant", None),
                "store": str(store),
                "nodes": [str(getattr(n, "storage_positions_expr", None))
                          for n in (getattr(topo, "cached_nodes", None) or [])],
            })

    if not any_attn:
        return out

    # Compose model-level asymptotes from the master signal.
    if not out["has_cache"]:
        out["kv_class"] = "constant"          # linear / SSM: no growing cache
    elif out["has_unbounded"]:
        out["kv_class"] = "linear"            # some cached node stores ∝ S
    else:
        out["kv_class"] = "plateau"           # every cached node capped (SWA)

    # decode_class: 'flat' is asserted ONLY for a model that caches nothing
    # unbounded (constant-state / pure-SWA) — for those a rising TPOT is a
    # real bug. When something IS stored ∝ S, TPOT may rise OR sit flat
    # (weight-bound at the tested batch / model size), so we don't demand a
    # rise — U-DecT already forbids a DROP.
    out["decode_class"] = "grows" if out["has_unbounded"] else "flat"
    out["prefill_band"] = _P_QUADRATIC if out["has_unbounded"] else _P_LINEAR
    out["known"] = True
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tier-0 — universal monotonicity (no spec)
# ─────────────────────────────────────────────────────────────────────────────
def _mono_axiom(axiom_id, chart, model, law, series_by_b, kind: str) -> List[AxiomResult]:
    """`series_by_b`: {batch: {seq: value}}. `kind`: 'nondecr' | 'noninc'."""
    res: List[AxiomResult] = []
    finder = _first_violation_nondecr if kind == "nondecr" else _first_violation_noninc
    arrow = "↓" if kind == "nondecr" else "↑"
    for b, curve in series_by_b.items():
        v = finder(curve)
        if v is None:
            continue
        x0, y0, x1, y1 = v
        res.append(AxiomResult(
            axiom_id, "0", chart, model, passed=False,
            expected=f"{law} ({kind.replace('decr','-decreasing').replace('noninc','non-increasing')})",
            measured=f"B={b}: {y0:.4g}@{x0} {arrow} {y1:.4g}@{x1}",
            detail=f"{law} broke at seq {x0}→{x1}"))
    if not res:
        return [AxiomResult(axiom_id, "0", chart, model, True, law, "monotone", "")]
    return res


def ax_U_KV(c, e):
    return _mono_axiom("U-KV", "3", c["model"], "KV(S) non-decreasing",
                       {"_": c["series"]["kv_per_req"]}, "nondecr")


def ax_U_PreT(c, e):
    return _mono_axiom("U-PreT", "5", c["model"], "prefill t(S) non-decreasing",
                       _numeric_only(c["series"]["prefill_ms"]), "nondecr")


def ax_U_DecT(c, e):
    return _mono_axiom("U-DecT", "4", c["model"], "decode TPOT(S) non-decreasing",
                       _numeric_only(c["series"]["tpot_ms"]), "nondecr")


def ax_U_CapD(c, e):
    return _mono_axiom("U-CapD", "1", c["model"], "max_decode_batch(S) non-increasing",
                       {"_": c["series"]["max_decode_batch"]}, "noninc")


def ax_U_CapP(c, e):
    return _mono_axiom("U-CapP", "2", c["model"], "max_prefill_batch(S) non-increasing",
                       {"_": c["series"]["max_prefill_batch"]}, "noninc")


def ax_U_BatT(c, e):
    """TPOT(B) non-decreasing at fixed S — transpose the (B,S) grid (fixed
    batches only; 'max' is not necessarily the largest numeric batch)."""
    by_seq = _transpose(_numeric_only(c["series"]["tpot_ms"]))
    return _mono_axiom("U-BatT", "4", c["model"], "TPOT(B) non-decreasing @fixed S",
                       by_seq, "nondecr")


def ax_U_BatThru(c, e):
    by_seq = _transpose(_numeric_only(c["series"]["decode_tok_s"]))
    return _mono_axiom("U-BatThru", "6", c["model"], "throughput(B) non-decreasing @fixed S",
                       by_seq, "nondecr")


def _transpose(series_by_b: Dict[Any, Dict[int, float]]) -> Dict[int, Dict[Any, float]]:
    """{batch: {seq: v}} → {seq: {batch_index: v}} so monotonicity runs along B.
    Batch labels are mapped to a sortable rank (numeric batches, then 'max')."""
    bsort = sorted(series_by_b, key=lambda b: (b if isinstance(b, (int, float)) else 10 ** 9))
    rank = {b: i for i, b in enumerate(bsort)}
    seqs = set()
    for curve in series_by_b.values():
        seqs |= set(curve)
    out: Dict[int, Dict[Any, float]] = {}
    for s in seqs:
        out[s] = {rank[b]: series_by_b[b][s] for b in series_by_b if s in series_by_b[b]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 shape "axioms" (D-KV / D-Dec / D-PreExp) were DELETED. Their intent —
# "verify hybrid (windowed) attention does NOT scale ∝ S" — is real, but a
# curve-slope check cannot enforce it non-vacuously: the estimate DERIVES the KV
# curve from the spec's declared cardinality, so checking that curve's shape
# against the same declaration is tautological (measured/declared ≡ per_token,
# constant by construction; a ×N over-count is invisible). And there is no
# external ground truth for KV runtime bytes to anchor to (unlike weights ↔
# checkpoint). The honest home for the intent is a SPEC-vs-CONFIG check anchored
# to config.json's `sliding_window` / `layer_types` — see SKILL.md. The shape
# DERIVATION (`derive_expected_shape` → kv_class etc.) stays, used only to LABEL
# curves in reconciliation and to guard the M-axioms; it gates no pass/fail.
# ─────────────────────────────────────────────────────────────────────────────
def _smallest_b(series_by_b: Dict[Any, Dict[int, float]]) -> Dict[int, float]:
    numeric = [b for b in series_by_b if isinstance(b, (int, float))]
    key = min(numeric) if numeric else next(iter(series_by_b))
    return series_by_b.get(key, {})


def _numeric_only(series_by_b: Dict[Any, Dict[int, float]]) -> Dict[Any, Dict[int, float]]:
    """Drop the 'max' batch: its batch VALUE varies with seq (capacity), so a
    time/throughput curve along it is legitimately non-monotone — monotonicity
    laws only hold at a FIXED batch."""
    return {b: v for b, v in series_by_b.items() if isinstance(b, (int, float))}


# D-Variant is a CROSS-model axiom: it needs the whole cohort, so it is run
# separately by the harness (see `cross_model_axioms`).


# ─────────────────────────────────────────────────────────────────────────────
# M / R / K — module-dominance, batch-regime, capacity-conservation
# ─────────────────────────────────────────────────────────────────────────────
def ax_M_NoPwTop(c, e):
    """A raw arithmetic/broadcast/reduce primitive must never be the single
    largest HBM consumer. In a correctly-fused net the biggest HBM op is always
    a GEMM, an attention op, a fused activation, or a norm — they read their
    natural `[B,S,·]` tensors. A raw `mul`/`broadcast`/`sum_reduce` out-reading
    EVERY structural op means a decomposed intermediate (an outer product, an
    unfused score matrix) is the biggest thing in memory. The primitive must
    exceed the top structural op by more than float-noise: a bare `>` flips
    run-to-run when a primitive and a GEMM are within rounding of each other,
    so the margin makes 'is the single largest' a robust verdict — the same
    role _EPS plays for the monotonicity laws. It absorbs numerical ties, it is
    not a tuned cutoff (the axiom stays a dominance test, not a magnitude bar)."""
    bad = None
    for (seq, b, phase), pair in c["series"]["top_access_op"].items():
        top, struct = pair.get("top"), pair.get("struct")
        if not top or not is_raw_primitive(top.get("op", "")):
            continue
        if (top.get("access", 0) > (struct or {}).get("access", 0) * (1.0 + _EPS)
                and (bad is None or top["access"] > bad[1]["top"].get("access", 0))):
            bad = ((seq, b, phase), pair)
    if bad is None:
        return [AxiomResult("M-NoPwTop", "M", "7", c["model"], True,
                            "top HBM op is a GEMM/attention/activation", "ok", "")]
    (seq, b, phase), pair = bad
    top, struct = pair["top"], pair.get("struct") or {}
    return [AxiomResult(
        "M-NoPwTop", "M", "7", c["model"], False,
        "top HBM op is a GEMM/attention/activation",
        f"{phase} S={seq} B={b}: '{top['op']}' reads {top['access']/1e9:.0f}GB "
        f"> top compute op '{struct.get('op','-')}' {struct.get('access',0)/1e9:.0f}GB",
        f"raw '{top['op']}' out-reads every structural op ⇒ unfused/over-materialised intermediate")]


def ax_M_AttnGrows(c, e):
    """Dense-attention models: attention's time fraction must rise with S
    (the S² term overtakes the linear FFN). Skip when no unbounded read exists
    (linear/SWA attention legitimately stays a flat or shrinking fraction)."""
    if not e.get("known") or not e.get("has_unbounded"):
        return []
    frac = _smallest_b(c["series"]["attn_frac"])
    xs = sorted(k for k, v in frac.items() if v is not None)
    if len(xs) < 3:
        return []
    lo, hi = frac[xs[0]], frac[xs[-1]]
    ok = hi >= lo - _EPS  # non-decreasing fraction across the full range
    return [AxiomResult("M-AttnGrows", "M", "7", c["model"], ok,
                        "attention time-fraction ↑ with S (dense)",
                        f"frac {lo:.2f}@{xs[0]} → {hi:.2f}@{xs[-1]}",
                        "" if ok else "attention fraction falls with S despite an O(S²) read")]


def ax_M_RouterCheap(c, e):
    """MoE: the router/gate must cost less than the expert GEMMs it dispatches."""
    if not e.get("is_moe"):
        return []
    worst = None
    for (seq, b), roles in c["series"]["role_flops"].items():
        router = roles.get("router", 0.0)
        expert = roles.get("ffn_moe", 0.0)
        if (expert > 0 and router > expert
                and (worst is None or router / expert > worst[2])):
            worst = (seq, b, router / expert)
    if worst is None:
        return [AxiomResult("M-RouterCheap", "M", "7", c["model"], True,
                            "router ≪ expert GEMM", "ok", "")]
    seq, b, ratio = worst
    return [AxiomResult("M-RouterCheap", "M", "7", c["model"], False,
                        "router < expert GEMM time",
                        f"S={seq} B={b}: router={ratio:.1f}× expert time",
                        "router outweighs experts ⇒ gate over-materialised")]


# NB: there is deliberately NO "R-Concave" (throughput(B) is concave) axiom.
# It looks tempting — throughput should saturate once compute-bound — but the
# premise is physically FALSE: at small batch the per-step time is dominated by
# loading the (MoE or dense) weights, so as the batch fills, throughput rises
# SUPER-linearly while those weights amortise, then saturates. Throughput(B) is
# S-shaped (convex → concave), not concave. The sound, regime-agnostic content
# of the batch axiom (A2) is exactly the two monotonicity laws we already have:
# U-BatThru (throughput non-decreasing in B) and U-BatT (TPOT non-decreasing in
# B). Asserting concavity flagged Qwen3-235B's legitimate MoE expert-fill regime.


def ax_K_Conserve(c, e):
    """Capacity ceiling: the admitted KV (max_decode_batch · per_request_kv) can
    never EXCEED the budget — that would be over-committing HBM. Under-fitting is
    legitimate (another limit — max_tokens, activation, the parallel solver —
    binds first), so only an OVERSHOOT past the budget is a conservation bug."""
    bad = None
    for seq in c["series"]["max_decode_batch"]:
        mb = c["series"]["max_decode_batch"].get(seq, 0)
        kv = c["series"]["kv_per_req"].get(seq, 0)
        budget = c["series"]["kv_budget"].get(seq, 0)
        if mb <= 0 or kv <= 0 or budget <= 0:
            continue
        used = mb * kv
        over = (used - budget) / budget
        if over > _EPS and (bad is None or over > bad[1]):
            bad = (seq, over, used, budget)
    if bad is None:
        return [AxiomResult("K-Conserve", "K", "1,3", c["model"], True,
                            "max_batch·per_req_kv ≤ kv_budget", "ok", "")]
    seq, over, used, budget = bad
    return [AxiomResult("K-Conserve", "K", "1,3", c["model"], False,
                        "max_batch·per_req_kv ≤ kv_budget",
                        f"S={seq}: admits {used/1e9:.1f}GB > budget {budget/1e9:.1f}GB ({over*100:.0f}% over)",
                        "admitted KV exceeds the budget ⇒ a batch/KV factor is mis-counted")]


def _bw(c):
    return (c["series"].get("peaks") or {}).get("bw", 0)


def ax_W_ActiveFloor(c, e):
    """At B=1 decode you must read the weights that step reads, so
    TPOT(B=1) ≥ weight_read_per_rank / peak_bw. The read is the engine's own
    (`weight_read_bytes_per_rank`): a routed bank reads for the rows it
    RECEIVES, which under DP attention are every rank's, so recomposing it
    from the per-token active set would set the floor too low. At the shortest
    seq (KV negligible) decode is weight-bound, so this floor also fixes the
    bs=1 ordering. Below the floor ⇒ decode access is under-counted."""
    bw = _bw(c)
    by_b1 = c["series"]["tpot_ms"].get(1, {})
    seqs = sorted(s for s, v in by_b1.items() if v)
    if not bw or not seqs:
        return []
    s = seqs[0]
    tpot, aw = by_b1[s], (c["series"].get("weight_read_b1") or {}).get(s, 0)
    if not tpot or not aw:
        return []
    floor = aw / bw * 1e3
    ok = tpot >= floor * 0.95
    return [AxiomResult("W-ActiveFloor", "W", "4", c["model"], ok,
                        f"TPOT(B=1) ≥ weight-read floor {floor:.2f} ms",
                        f"TPOT(B=1,S={s})={tpot:.2f} ms, read {aw/1e9:.0f} GB ÷ "
                        f"{bw/1e12:.1f} TBps = {floor:.2f} ms",
                        "" if ok else "TPOT below the active-weight load time ⇒ decode access under-counted")]


def ax_W_TotalFloor(c, e):
    """At large batch ~every expert is hit, so a decode step must load ALL
    weights once: TPOT(largest fixed B) ≥ total_weight_per_rank / peak_bw. Below
    it (the compare page's linear MoE batch-interpolation does this) is
    physically impossible — you cannot decode faster than reading the weights."""
    bw = _bw(c)
    bnum = [b for b in c["series"]["tpot_ms"] if isinstance(b, (int, float))]
    if not bw or not bnum:
        return []
    blarge = max(bnum)
    by_bl = c["series"]["tpot_ms"].get(blarge, {})
    seqs = sorted(s for s, v in by_bl.items() if v)
    if not seqs:
        return []
    s = seqs[0]
    tpot, tw = by_bl[s], (c["series"]["mem"].get(s) or {}).get("weight_per_gpu", 0)
    if not tpot or not tw:
        return []
    floor = tw / bw * 1e3
    ok = tpot >= floor * 0.95
    return [AxiomResult("W-TotalFloor", "W", "4", c["model"], ok,
                        f"TPOT(B={blarge}) ≥ all-weight floor {floor:.2f} ms",
                        f"TPOT(B={blarge},S={s})={tpot:.2f} ms, total weight {tw/1e9:.0f} GB ÷ "
                        f"{bw/1e12:.1f} TBps = {floor:.2f} ms",
                        "" if ok else "TPOT below the all-weight load time ⇒ impossible (under-counted / bad interpolation)")]


_PER_MODEL_AXIOMS = [
    ax_U_KV, ax_U_PreT, ax_U_DecT, ax_U_CapD, ax_U_CapP, ax_U_BatT, ax_U_BatThru,
    ax_M_NoPwTop, ax_M_AttnGrows, ax_M_RouterCheap, ax_K_Conserve,
    ax_W_ActiveFloor, ax_W_TotalFloor,
]


def run_axioms(curves: Dict[str, Any], expected: Dict[str, Any]) -> List[AxiomResult]:
    """Run every per-model axiom, guarding each so one crash can't sink the rest."""
    results: List[AxiomResult] = []
    for ax in _PER_MODEL_AXIOMS:
        try:
            results.extend(ax(curves, expected))
        except Exception as exc:
            results.append(AxiomResult(ax.__name__, "?", "?", curves.get("model", "?"),
                                       False, "axiom evaluates", f"raised {type(exc).__name__}",
                                       str(exc)[:160]))
    return results

# Note on variant advantage (the user's A4): rather than a fragile CROSS-model
# magnitude comparison (a 671B model's prefill exceeds a 7B model's regardless of
# attention class — size, not class, dominates), each variant's advantage is
# checked PER-MODEL against its own declared cardinality: D-KV (storage class),
# D-Dec (constant-state ⇒ flat decode), D-PreExp (bounded reads ⇒ no spurious S²;
# unbounded reads ⇒ never super-quadratic). That is the sound, size-independent
# form of "the variant must behave as declared".
