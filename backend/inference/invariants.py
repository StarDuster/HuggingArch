"""Inference invariant battery — "is this round of accounting trustworthy?"
without hand-calculating every number.

The point (see `docs/validation.md` Part II, classes A–E): you can't verify every
inference number by hand across models × configs. Instead shrink the
human-trust surface to a tiny set of frozen per-op axioms (class A), and let a
battery of EXACT invariants guarantee everything built on them is leak-free.
Each invariant here is an output-vs-output equality that needs NO knowledge of
the correct constant — if it doesn't hold, there is a bug, full stop. (This is
how the V4-Pro ulysses cp-sharding bug was caught: `cp_scheme` invariance, not
hand-calc.)

Categories implemented here:
  • C (perturbation) — `cp_scheme` structural ratios (including the K/V
    producer ratio Ulysses and pcp owe); batch linearity (batch size doubling
    doubles forward FLOPs, leaves params invariant).
  • B (conservation/closure) — phase closure (Σ per-op time == the phase
    t_sol the UI shows); model-vs-engine FLOPs cross-check.

Run for any model:  `python -m backend.inference.invariants <model_id>`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional


# Default deployment for the battery. cp>1 is what makes ulysses≠pcp (and
# the cp-sharding bugs) observable; TP1/CP8/EP8 on 8×B200×cp is the recon
# config the invariants were validated against.
_DEFAULT_PARALLEL = dict(
    tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1,
    ep_moe=8, cp=8, pp=1, dp_attention=1, dp_replicas=1,
)
_REL_TOL = 0.02
_SEQ = 65536


@dataclass
class InvariantResult:
    name: str
    category: str          # "C-perturbation" | "B-closure"
    passed: bool
    detail: str


def _estimate(config: Dict[str, Any], scheme: str, *, gpu_sku: str,
              num_gpus: int, parallel: Dict[str, Any],
              avg_input_tokens: int, avg_output_tokens: int,
              resolved=None) -> dict:
    from backend.inference.capacity import estimate_capacity
    from backend.inference.parallel_config import ParallelConfig
    return estimate_capacity(
        config=config, resolved=resolved, gpu_sku=gpu_sku, num_gpus=num_gpus,
        avg_input_tokens=avg_input_tokens, avg_output_tokens=avg_output_tokens,
        # The battery's deployment, stated: block-128 FP8 weights on every
        # weight class and an FP8 KV cache, the rest as shipped.
        precision={"preset": "w_fp8_kv_fp8"},
        parallel=ParallelConfig(cp_scheme=scheme, **parallel),
        selection_mode="measured",

    )


def _approx(a: float, b: float, tol: float = _REL_TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


#: For two sides that are the SAME addition over the same list — a fold and
#: its parts. They agree bit-for-bit in practice; the tolerance exists only
#: so a future serialization round-trip cannot trip it. Deliberately not
#: `_approx`, whose 1.0 scale floor is meaningful for counts and blinding
#: for a quantity measured in seconds.
_FOLD_TOL = 1e-9


# ---------------------------------------------------------------------------
# C — cp_scheme invariance (the ratios each walked axis owes)
# ---------------------------------------------------------------------------


def _extent(shape: Any) -> float:
    """Elements in a declared shape, or 0 when it is not one."""
    if not isinstance(shape, (list, tuple)) or not shape:
        return 0.0
    total = 1.0
    for axis in shape:
        if not isinstance(axis, (int, float)):
            return 0.0
        total *= float(axis)
    return total


def _kv_producer_factor(row: dict, other: dict, cp: int,
                        producers: FrozenSet[str] = frozenset()) -> float:
    """`pcp / ulysses` for one row, from layout-walk provenance.

    A K/V producer under Ulysses runs on the resident sequence cut and then
    exchanges that cut for heads at the attention boundary; PCP computes the
    whole context. Every other producer keeps ratio one.  The Ulysses result
    carries the structural producer set from its layout walk; the rows' own
    shapes then remain what is checked, never what licenses the check.
    """
    row_id = other.get("aggregation_id")
    if isinstance(row_id, str) and row_id in producers:
        return float(cp)
    return 1.0


def _inv_cp_scheme(uly: dict, pcp: dict, cp: int) -> InvariantResult:
    """What the two context-parallel schemes owe each other.

    Per-rank FLOPs follows the structural output ratio: both schemes cut the
    same attention matrix `cp` ways, while a K/V producer under Ulysses runs
    on `1/cp` of the attended rows before the head exchange.

    Per-rank ACTIVATION is NOT invariant, and asserting it was is what kept
    this red. The schemes shard different axes, so each operand follows the
    axes it has:

      * ulysses cuts the HEAD axis and keeps every token, so a headed
        operand is `1/cp` and a head-less one is whole;
      * pcp cuts the QUERY-TOKEN axis and keeps every head, so the query
        and the head-less per-token operands are `1/cp` and the KV — which
        every rank must attend in full — is whole.

    So a query operand is the same size either way (`1`), a KV operand is
    `cp` times larger under pcp, and a head-less per-token operand (DSA's
    `[B, s_query, top_k]` index) is `1/cp` of ulysses'. Those three ratios
    are the schemes' definition; any OTHER ratio is a sharding factor that
    leaked, which is the bug this check was built for (the V4-Pro
    indexer-score one, where a head-shard was applied to a head-less op).
    """
    def _rows(r):
        return {x.get("aggregation_id") or x.get("op"): x
                for x in (r.get("per_op_timing") or [])}

    u, p = _rows(uly), _rows(pcp)
    allowed = (1.0, float(cp), 1.0 / cp)
    producers = frozenset(
        value for value in (uly.get("head_repartition_kv_producers") or ())
        if isinstance(value, str))
    for aid, row in u.items():
        other = p.get(aid)
        if other is None:
            continue
        op = row.get("op")
        factor = _kv_producer_factor(row, other, cp, producers)
        fu = float(row.get("flops_per_rank") or 0)
        fp = float(other.get("flops_per_rank") or 0)
        if not _approx(fp, fu * factor):
            return InvariantResult(
                "cp_scheme_invariance", "C-perturbation", False,
                f"op={op}: per-rank FLOPs ulysses={fu:,.0f} "
                f"pcp={fp:,.0f} (expected ratio {factor:.3g}) — the walked "
                "structure says these producers do not owe equal work")
        ou, op_out = _extent(row.get("shape_out")), _extent(other.get("shape_out"))
        if ou and op_out and not _approx(op_out, ou * factor):
            return InvariantResult(
                "cp_scheme_invariance", "C-perturbation", False,
                f"op={op}: per-rank output ulysses={ou:,.0f} "
                f"pcp={op_out:,.0f} (expected ratio {factor:.3g}) — "
                "both schemes hand the next op the shapes their walked axes "
                "imply")
        ins_u = row.get("shape_in") or []
        ins_p = other.get("shape_in") or []
        if len(ins_u) != len(ins_p):
            continue
        for index, (a, b) in enumerate(zip(ins_u, ins_p, strict=False)):
            ea, eb = _extent(a), _extent(b)
            if not ea or not eb:
                continue
            ratio = eb / ea
            if not any(_approx(ratio, ok) for ok in allowed):
                return InvariantResult(
                    "cp_scheme_invariance", "C-perturbation", False,
                    f"op={op} input {index}: pcp is {ratio:.3g}x ulysses "
                    f"({a} vs {b}) — cp={cp} admits only 1 (a query axis), "
                    f"{cp} (KV, whole under pcp) or 1/{cp} (head-less per "
                    "token, whole under ulysses); anything else is a leaked "
                    "sharding factor")
    return InvariantResult(
        "cp_scheme_invariance", "C-perturbation", True,
        "ulysses and pcp follow the structural FLOPs/output ratios, and "
        "every operand follows the axis its scheme cuts")


# ---------------------------------------------------------------------------
# B — phase closure (Σ per-op time / FLOPs == the phase total the UI displays)
# ---------------------------------------------------------------------------


def _draft_sum(res: dict, pm: dict, key: str) -> float:
    """A counted speculative draft's part of a decode step's per-op sum: one
    forward's rows (`per_op_timing_draft`) times the step's forwards. Zero
    for any phase that counts no draft."""
    draft = pm.get("speculative_draft")
    if not draft:
        return 0.0
    return int(draft["forwards"]) * sum(
        float(r.get(key) or 0) for r in res.get("per_op_timing_draft") or [])


def _inv_phase_closure(res: dict) -> List[InvariantResult]:
    out: List[InvariantResult] = []
    phases = res.get("phases") or {}
    for phase, rows_key in (("prefill", "per_op_timing"),
                            ("decode", "per_op_timing_decode")):
        rows = res.get(rows_key) or []
        if not rows:
            continue
        pm = phases.get(phase) or {}
        # A phase may decline to publish a step time (pp>1 without a
        # pipeline sum: these rows are one stage's, and a stage's wall time
        # is not the model's step). There is then no displayed total for
        # the rows to close against — the closure is not violated, it is
        # not posed. Reported rather than skipped silently, so a total that
        # vanishes for any OTHER reason still shows. At pp>1 WITH the
        # pipeline sum the closure IS posed — just not by these rows: the
        # bound stage's sum must equal its entry in `per_stage`, and the
        # per-stage sums plus the draft must equal the published total.
        if pm.get("pipeline_time") and pm.get("t_sol_s") is not None:
            pt = pm["pipeline_time"]
            per_stage = pt.get("per_stage") or []
            stage = pt.get("stage")
            checks = []
            ok = True
            if per_stage and isinstance(stage, int) and stage < len(per_stage):
                bound = sum(float(x.get("t_op_s") or 0) for x in rows)
                expect = float(per_stage[stage])
                scale = max(abs(bound), abs(expect))
                ok = (abs(bound - expect) <= _FOLD_TOL * scale
                      if scale else True)
                checks.append(
                    f"bound stage Σ t = {bound:.6f}s vs "
                    f"per_stage[{stage}] = {expect:.6f}s")
            if per_stage:
                total = (sum(float(s) for s in per_stage)
                         + _draft_sum(res, pm, "t_op_s"))
                phase_total = float(pm.get("t_sol_s") or 0)
                scale = max(abs(total), abs(phase_total))
                ok_total = (abs(total - phase_total) <= _FOLD_TOL * scale
                            if scale else True)
                checks.append(
                    f"Σ stages + draft = {total:.6f}s vs "
                    f"phase t_sol = {phase_total:.6f}s")
                ok = ok and ok_total
            out.append(InvariantResult(
                f"phase_closure[{phase}]", "B-closure", ok,
                "; ".join(checks) if checks else
                "pipeline step time (Σ stages + hops)"))
            continue
        withheld = str(pm.get("time_unpublished_reason") or "")
        if withheld and pm.get("t_sol_s") is None:
            out.append(InvariantResult(
                f"phase_closure[{phase}]", "B-closure", True,
                f"no displayed total to close against — {withheld}"))
            continue
        per_op = (sum(float(x.get("t_op_s") or 0) for x in rows)
                  + _draft_sum(res, pm, "t_op_s"))
        phase_total = float(pm.get("t_sol_s") or 0)
        # Held to a FOLD's tolerance, not `_approx`'s. `_approx` floors its
        # scale at 1.0, which is the smallest meaningful unit of a count and
        # a fifty-fold overstatement of a decode step: it turned into "agree
        # within 20 ms", and a millisecond-scale step could publish a total of
        # zero and pass. It did — DeepSeek-V3 decode on a config that does not
        # fit reported Σ 0.0084s against a published 0.0 and read as closed,
        # the slack admitting 2.4x the step. Nothing here needs slack: the
        # phase's `t_sol` IS `sum(row.t_op_s)` over this same list, so the two
        # sides are one addition and agree exactly (measured 0.00e+00 across
        # the corpus). The measured track is not mixed in — the overlay keeps
        # `t_op_s` at the theoretical rooftop and puts corpus times in
        # `t_measured_s` — so calibration gives this no slack to absorb.
        scale = max(abs(per_op), abs(phase_total))
        ok = abs(per_op - phase_total) <= _FOLD_TOL * scale if scale else True
        out.append(InvariantResult(
            f"phase_closure[{phase}]", "B-closure", ok,
            f"Σ per-op t = {per_op:.6f}s vs phase t_sol = {phase_total:.6f}s"
            + ("" if ok else " — displayed total ≠ Σ parts (an op was "
               "dropped or double-counted)")))
    return out


def _inv_flops_closure(res: dict) -> List[InvariantResult]:
    """Every fold of the step's FLOPs lands on the same integer.

    `phases._flops_rows` builds the table and `pipeline.analyze` builds the
    per-op rows the step's FLOPs are folded from. Those used to be two
    independent walks with nothing
    forcing them to describe the same step, and for a long time they did not —
    the table billed the LM head at every prefill position, kept aux (MTP)
    stacks the rows dropped, charged one layer of a template instantiated zero
    times, and omitted the final norm. Each was invisible in the totals because
    only one of the two moved.

    They are now two folds of one `StepEvaluation`, so this is close to a
    tautology and is kept anyway: it costs a sum over rows that already exist,
    and the thing it still catches is a fold that drops or double-counts rows —
    which is a real way to be wrong and the only one left here.

    Two legs, because the headline has two decompositions. The breakdown
    table is the grafted + fused tree read back — its leaves ARE the per-op
    rows merged by uid, so the two legs share them and the per-op leg now
    checks only the structural rows (template headers, wrappers) the walk
    adds on top. The table leg still catches a fold that drops or
    double-counts rows, and both legs are checked exactly: measured
    0.00e+00 across all 119 phase-cells of the corpus.
    """
    out: List[InvariantResult] = []
    phases = res.get("phases") or {}
    for phase, rows_key in (("prefill", "per_op_timing"),
                            ("decode", "per_op_timing_decode")):
        rows = res.get(rows_key) or []
        if not rows:
            continue
        pm = phases.get(phase) or {}
        table = pm.get("flops_by_module") or []
        if not table:
            continue
        # Depth 0 is the whole tree counted once: a block template charged its
        # layers, or a model-level node. `flops_per_rank` when a deployment was
        # described (the headline is then per-rank too), else the global column.
        key = ("flops_per_rank"
               if any(r.get("flops_per_rank") is not None for r in table)
               else "flops_full")
        shown = sum(float(r.get(key) or 0) for r in table if r.get("depth") == 0)
        per_op = (sum(float(r.get("flops_per_rank") or 0) for r in rows)
                  + _draft_sum(res, pm, "flops_per_rank"))
        headline = float(pm.get("flops_per_step") or 0)

        def _rel(side: float) -> float:
            scale = max(abs(side), abs(headline))
            return abs(side - headline) / scale if scale else 0.0

        rel_tab, rel_row = _rel(shown), _rel(per_op)
        ok = max(rel_tab, rel_row) <= _FOLD_TOL
        out.append(InvariantResult(
            f"flops_closure[{phase}]", "B-closure", ok,
            f"phase flops_per_step = {headline:.6e}; Σ breakdown = "
            f"{shown:.6e} (rel {rel_tab:.2e}); Σ per-op = {per_op:.6e} "
            f"(rel {rel_row:.2e})"
            + ("" if ok else " — the table and the per-op rows are describing "
               "different steps")))
    return out


def _inv_traffic_partition(res: dict) -> List[InvariantResult]:
    """A leaf's published traffic terms are a PARTITION of its access.

    `access_bytes_per_rank` is `weight_read + metadata + act_in + act_out +
    convert` (the last being the quantised copy of its own operand a kernel
    needs when the edge does not carry it) —
    the leaf's whole HBM traffic, both directions — and each of those four is
    now its own column. So this is an addition against itself, and it is here
    for the same reason `_inv_flops_closure` is: to fail the moment one term
    stops being folded and starts being recovered.

    That is not hypothetical. Every one of them was published except the
    weight read at THIS step, which is `min(rows x read_per_row, total)` —
    the saturation min over the rows the LEAF runs on (the busiest rank's,
    for a routed leaf). Consumers did not have those rows (only
    the step's token count), so they re-derived a min that omitted them
    and took activation as `access - weight - metadata`. A remainder cannot be
    wrong out loud: under balanced routing the re-derivation agrees and the
    error is zero, and under an imbalanced trace the difference is published as
    activation traffic. Both halves of the activation term are folds of engine
    columns now, and this says so on every estimate rather than in the one
    script that happened to check.
    """
    out: List[InvariantResult] = []
    for phase, rows_key in (("prefill", "per_op_timing"),
                            ("decode", "per_op_timing_decode")):
        rows = res.get(rows_key) or []
        if not rows:
            continue
        worst = 0
        worst_row = ""
        parts = 0
        access = 0
        for row in rows:
            terms = sum(int(row.get(k) or 0) for k in (
                "weight_read_bytes_per_rank", "metadata_bytes_per_rank",
                "act_in_bytes_per_rank", "act_out_bytes_per_rank",
                "convert_bytes_per_rank"))
            whole = int(row.get("access_bytes_per_rank") or 0)
            parts += terms
            access += whole
            if abs(terms - whole) > abs(worst):
                worst = terms - whole
                worst_row = str(row.get("aggregation_id") or row.get("op") or "?")
        # Integers on both sides of one addition: exact, no tolerance.
        ok = worst == 0
        out.append(InvariantResult(
            f"traffic_partition[{phase}]", "B-closure", ok,
            f"Σ terms = {parts:,} vs Σ access = {access:,}"
            + ("" if ok else f" — worst leaf {worst_row} off by {worst:+,}; a "
               f"term is being recovered by subtraction rather than folded")))
    return out


# ---------------------------------------------------------------------------
# C — batch linearity (doubling B doubles forward FLOPs; params batch-invariant)
# ---------------------------------------------------------------------------


def _inv_batch_linearity(resolved) -> InvariantResult:
    """Forward FLOPs are linear in batch and params are batch-independent —
    a precise invariant needing no correct constant. Walk-level (no
    deployment), so it isolates the spec's compute math. Run against `eval`:
    a gate guards the numbers production serves, and those are moving there —
    guarding the engine slated for deletion would verify the wrong producer.
    """
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    program = Program.of(resolved)
    r1 = Deployment(program).at(batch=1, input_tokens=128).analyze().total()
    r2 = Deployment(program).at(batch=2, input_tokens=128).analyze().total()
    f1, f2 = int(r1.flops), int(r2.flops)
    p1, p2 = int(r1.params), int(r2.params)
    if p1 != p2:
        return InvariantResult("batch_linearity", "C-perturbation", False,
                               f"total_params changed with batch ({p1:,} vs {p2:,})")
    if f1 > 0 and f2 != 2 * f1:
        return InvariantResult("batch_linearity", "C-perturbation", False,
                               f"total_flops not ×2 when B doubles ({f1:,} vs {f2:,})")
    return InvariantResult("batch_linearity", "C-perturbation", True,
                           f"FLOPs ×2 ({f1:,}→{f2:,}), params invariant")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_inference_invariants(
    config: Dict[str, Any],
    resolved=None,
    *,
    gpu_sku: str = "B200",
    parallel: Optional[Dict[str, Any]] = None,
    avg_input_tokens: int = _SEQ,
    avg_output_tokens: int = 256,
) -> List[InvariantResult]:
    """Run the exact-invariant battery for one model. Each invariant is
    wrapped so one failure (or exception) doesn't sink the rest — a raising
    invariant is itself a failure."""
    parallel = parallel or dict(_DEFAULT_PARALLEL)
    from backend.inference.parallel_config import ParallelConfig
    # The cluster is what the degrees multiply to, read off the config
    # rather than restated as `8 × cp × pp` with ep_moe frozen at 8.
    num_gpus = ParallelConfig(**parallel).cluster_size
    results: List[InvariantResult] = []

    def _guard(name: str, category: str, fn: Callable[[], Any]):
        try:
            r = fn()
            results.extend(r if isinstance(r, list) else [r])
        except Exception as exc:
            results.append(InvariantResult(
                name, category, False, f"raised {type(exc).__name__}: {exc}"))

    dep = dict(gpu_sku=gpu_sku, num_gpus=num_gpus, parallel=parallel,
               avg_input_tokens=avg_input_tokens, avg_output_tokens=avg_output_tokens,
               resolved=resolved)
    uly = {}
    pcp = {}

    def _run_pair():
        nonlocal uly, pcp
        uly = _estimate(config, "ulysses", **dep)
        pcp = _estimate(config, "pcp", **dep)
        return _inv_cp_scheme(uly, pcp, max(1, int(parallel.get("cp") or 1)))

    _guard("cp_scheme_invariance", "C-perturbation", _run_pair)
    _guard("phase_closure", "B-closure", lambda: _inv_phase_closure(uly) if uly else [])
    _guard("flops_closure", "B-closure", lambda: _inv_flops_closure(uly) if uly else [])
    _guard("traffic_partition", "B-closure",
           lambda: _inv_traffic_partition(uly) if uly else [])
    if resolved is not None:
        _guard("batch_linearity", "C-perturbation", lambda: _inv_batch_linearity(resolved))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_config_and_spec(model_id: str):
    """Config dict (for capacity) + ResolvedSpec (for the engine invariant),
    via the same config read as validate (`hf.metadata.fetch_config`)."""
    from backend.arch.loader import SpecLoader
    from backend.hf.metadata import fetch_config

    config = fetch_config(model_id)
    if config is None:
        raise ValueError(f"Cannot fetch config for {model_id}")
    try:
        resolved = SpecLoader().match(config, model_id=model_id)
    except Exception:
        resolved = None
    return config, resolved


def main():
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Run the exact-invariant battery for a model's inference accounting")
    p.add_argument("model_id")
    args = p.parse_args()

    config, resolved = _resolve_config_and_spec(args.model_id)
    results = run_inference_invariants(config, resolved)
    n_pass = sum(1 for r in results if r.passed)
    print(f"=== inference invariants: {args.model_id} ===")
    for r in results:
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} [{r.category}] {r.name}: {r.detail}")
    print(f"\n{n_pass}/{len(results)} invariants hold")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
