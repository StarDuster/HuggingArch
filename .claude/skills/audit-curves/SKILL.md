---
name: audit-curves
description: Proactively hunt inference-accounting bugs by checking every compare-page curve against first-principles infra AXIOMS — universal monotonicity laws, a capacity-conservation inequality, weight-roofline floors, and module-dominance checks. Anything that violates an axiom is a candidate bug. Use for a self-audit pass or as the body of a Ralph loop.
---

# audit-curves

The deterministic self-audit behind `agent-driven-spec-inspection`. For a sample
of models it rebuilds the **same curves the compare page's 7 charts show** —
across seq and batch — and checks each against infra **axioms** instead of magic
thresholds. Every axiom names its tier, the chart it governs, and (for the
spec-derived tier) the spec fact it derives its expectation from. A failure
prints expected-vs-measured, so it is self-explaining.

Why axioms, not thresholds: absolute cutoffs (`>200 GB`, `>4×`) produce false
positives (a small model's legit `B·S·H` traffic trips an absolute GB bar) and
blind spots (an over-materialised intermediate that happens to have FLOPs slips
a "compute-free" gate). The battery keys off **shape / monotonicity / regime /
dominance** instead.

**Honesty note.** Every surviving check is a knob-free law — a monotonicity,
an inequality, or a dominance ranking, whose only constant is float-noise ε.
The former Tier-1 *shape* "axioms" (`D-KV` / `D-Dec` / `D-PreExp`) were
**deleted**: they classified a continuous curve slope against a tuned rail, and
worse, the quantity they checked has no external ground truth — KV runtime bytes
are DERIVED from the spec's declared cardinality, so a curve-shape check against
that same declaration is vacuous (see "What was deleted"). The shape *derivation*
survives only as a LABEL for reconciliation and to guard the M-axioms.

## The laws

- **Tier-0 — universal monotonicity.** The only laws that hold for *every*
  architecture, no spec, no number: KV(S), prefill-t(S), decode-TPOT(S) are
  **non-decreasing** at fixed batch; max_batch(S) is **non-increasing**;
  TPOT(B)/throughput(B) are monotone at fixed seq. A violation is an
  unconditional accounting bug.

- **M / K / W — dominance, conservation, roofline floors.** Module dominance
  (a raw primitive never out-reads every compute op; router < experts; attention
  fraction rises for unbounded-read models), capacity conservation
  (admitted KV ≤ budget), and the weight-bandwidth floors (TPOT ≥ weight/bw).
  Each is an inequality or ranking, not a threshold.

The **shape derivation** (`derive_expected_shape` → `kv_class` /
`has_unbounded`, read from each attention component's declared cardinality via
`kv_cache_topology(...)` + `_storage_is_bounded`) is retained, but it no longer
drives any pass/fail — it only LABELS curves for the reconciliation display and
guards the M-axioms (e.g. M-AttnGrows runs only when `has_unbounded`).

## The battery (see `axioms.py` for the deliberated definitions)

| id | tier | chart | law |
|---|---|---|---|
| U-KV / U-PreT / U-DecT | 0 | 3/5/4 | KV, prefill-t, TPOT non-decreasing in S (fixed B) |
| U-CapD / U-CapP | 0 | 1/2 | max decode/prefill batch non-increasing in S |
| U-BatT / U-BatThru | 0 | 4/6 | TPOT(B), throughput(B) monotone at fixed S |
| M-NoPwTop | M | 7 | a raw `mul`/`broadcast`/`sum_reduce` never out-reads every compute op |
| M-AttnGrows | M | 7 | unbounded-read models: attention time-fraction ↑ with S |
| M-RouterCheap | M | 7 | MoE router < expert GEMM |
| K-Conserve | K | 1,3 | admitted KV (max_batch·per_req) ≤ kv_budget |
| W-ActiveFloor | W | 4 | TPOT(B=1) ≥ active-weight/peak_bw (bs=1 is weight-bound; fixes the ordering) |
| W-TotalFloor | W | 4 | TPOT(large B) ≥ all-weight/peak_bw (every expert hit ⇒ load all weights once) |

Tier-0 covers charts 1–6; M/K cover chart 7 (module breakdown) and the
capacity identity. The batch-regime axiom (A2) is the monotonicity pair U-BatT +
U-BatThru — there is deliberately **no concavity axiom**: MoE/dense weight
amortisation makes throughput(B) S-shaped (convex→concave), so "throughput is
concave" is physically false (it flagged Qwen3-235B's legitimate expert-fill
regime). **M-NoPwTop is the workhorse** — it caught the V4 HC
`[B,S,Hc,Hc,H]` outer product (`mul` out-reading every GEMM) and is the same
signal that flags the pre-fix indexer's raw `mul`/`sum_reduce`. Cp-scheme
invariance (A5) is an exact output-vs-output invariant living in
`backend/inference/invariants.py`, referenced here, not duplicated. Dtype/roofline
(A6: halve weight bytes ⇒ weight mem halves, compute-bound t halves; `compute_dtype`
peak on the right tier) is a phase-2 extension, not yet built.

## What is actually formalised (and what isn't)

Be precise about which checks are knob-free laws — the difference decides how
much a red is worth.

- **Genuine laws (an inequality / monotonicity / dominance ranking; only constant
  is float-noise ε).** Tier-0 `U-*` (7 monotonicity laws); `K-Conserve`
  (`admitted_KV ≤ budget`); `W-ActiveFloor` / `W-TotalFloor` (`TPOT ≥ weight/bw`,
  a roofline lower bound — the lone blemish is a 5% slack); `M-RouterCheap`
  (`router_flops < expert_flops`); `M-AttnGrows` (fraction non-decreasing);
  `M-NoPwTop` (a raw primitive never out-reads every compute op). A red here is a
  real accounting bug.

- **Remaining debt — `M-NoPwTop`'s op list.** The *predicate* is sound and
  threshold-free, but *what counts as a raw primitive* is the hand-maintained
  `_RAW_PRIMITIVE` / `_VIEW_OPS` frozensets + `startswith` matching, so a new or
  renamed op silently escapes classification. The tie is guarded by an ε margin;
  the enumeration itself is the debt. Fix: move `is_raw_primitive` onto a
  first-class op-declared flag (`elementwise`/`reduce`/`view`/`compute`) per
  `[[feedback_axiom_op_component_level]]`, so the audit reads a declared property.

## What was deleted, and where the intent really lives

`D-KV` / `D-Dec` / `D-PreExp` (the slope-rail "shape axioms") were **removed**.
They tried to verify the right thing — *hybrid (windowed) attention must NOT
scale ∝ S* — but a curve check can't enforce it:

- **No external ground truth.** Weights anchor to checkpoint bytes; KV runtime
  bytes do not. The estimate DERIVES the KV curve from the spec's declared
  cardinality, so checking that curve's shape against the same declaration is
  vacuous — `measured / declared ≡ per_token`, constant by construction, and a
  ×N over-count is invisible (verified empirically on gemma-4).
- **The rails were never axioms.** 0.15 / 0.30 / `_P_*` bucketed a continuous
  slope; for a mixed model the true slope is the byte-weighted blend of its
  streams, so no fixed rail is correct.

**The intent now lives where it belongs — a spec-vs-config check in the
validator**, not a curve. `validate/_axioms.py::_check_config_windowing` (runs
in every `run_axiom_checks`, warning-only): if `config.json` declares an active
sliding window (`sliding_window` > 0 and not `use_sliding_window: false`) but the
spec caches **no bounded KV stream** (`_storage_is_bounded` over the resolved
topology — DSA/CSA `Min(S,W)` count as bounded, dense `Lq` does not), it flags
`windowing_declared_but_unmodelled`. Anchored to config.json, FP-free across SWA
hybrids / DSA / window-disabled models; escape hatch = declare `sliding_window`
in `ignored_config_fields`. It closes the validator's known window/sparsity blind
spot, and on first run caught a real one: **Mistral-7B-v0.1 modelled as dense GQA**
(KV grows ∝ S where it should cap at 4096).

## Determinism

The sweep pins `PYTHONHASHSEED=0` (re-exec guard in `sweep.py`) and uses
`math.fsum` for its role sums. Dict/set iteration order is hash-seeded, and a
`M-NoPwTop` argmax over near-tied ops could otherwise flip between processes even
at the same `--seed` (which only controls model sampling); the ε margin plus the
pinned seed make it reproduce bit-for-bit. **Tell:** a red that reproduces across
reruns is real; one that flips was borderline noise — never a genuine law.

## Run

```
cd /path/to/HuggingArch
uv run python .claude/skills/audit-curves/sweep.py --models 6 --seed <n>
```

Always includes the DeepSeek V3/V3.2/V4 anchor trio plus a seeded sample of
others; sweeps seq × batch and prints, as JSON: `pass_summary` per model, the
derived `expected` shape per model (so you can see what the spec declared), and a
`failures` list. Each failure carries `axiom_id`, `tier`, `chart`, `expected`,
`measured`, `detail`. Flags: `--seqs` (default 5-point grid; pass the full
9-point grid for a deep run), `--batches` (default `1,16,64,max`), `--json out`,
`--mode` / `--degree` (the deployment: a Scaling-panel mode `dp_ep` / `cp_ep` /
`tp_ep` / `tp_tp` at a GPU count, default `dp_ep` × 8, built by
`parallel_sweep.config_for_degree`). A point whose estimate raises lands in
`errors`: a sweep with errors on every point audited nothing, whatever
`pass_summary` says.
In a Ralph loop, vary `--seed` to rotate the random models.

## Reconciliation — explain every chart, not just the violations

The battery flags what's WRONG; the **reconciliation** (`reconcile.py`) proves we
can explain what's RIGHT. After each sweep it prints (to **stderr**, so stdout
stays pure JSON for `gate.py`) a confident first-principles line per chart per
model, reconstructing each number from the roofline the estimate already exposes:

- `t_sol = Σ_leaf max(compute, memory, comm, launch)`, `memory = access/peak_bw`, `compute = flops/peak_flops`
- `max_decode_batch = kv_budget / per_request_kv`; `per_request_kv = S · kv_per_token`
- module split from `phases.*.flops_by_module` (component-level roles; leaf rows are role-less)

This is self-consistent BY CONSTRUCTION, so a line that does NOT reconcile (the
from-principle magnitude misses the stored term) is itself a finding. Example:

```
[Ch4 TPOT]   B=max S=131072: 89.9 ms — memory-bound (mem 696 GB/7.8TBps=89.8 ms, comp 18 TF/4500=4.1 ms)  ✓
[Ch7 modules] S=2048: top ffn_moe 74% (attn 22%); S=131072: top attention 79% — attention ∝S² overtakes FFN
```

It is also a coverage check on the battery itself: building the reconciliation is
how the role-attribution (per-op `role` is `None`; use `flops_by_module`) and the
chunked-prefill / weight-bound-decode regimes were pinned down, which in turn made
the `M-AttnGrows` / `M-RouterCheap` axioms non-vacuous. **No doc, no external
ground truth — the method reconciles against first principles only.**

## Per failure: explain or escalate

1. **Read `expected` vs `measured`** — the axiom already states the
   first-principles prediction and what the curve did.
2. **Locate the op.** For `M-NoPwTop`, re-run `estimate_capacity` at that
   (model, seq, batch) and read `per_op_timing[_decode]` to find the
   over-materialised op's `shape_in` (an extra axis = an unfused outer product).
   For a `K-Conserve` / `W-*` red, reconcile the flagged number against the
   roofline terms the estimate already exposes (see Reconciliation above).
3. **Decide:** a Tier-0 monotonicity violation is always a bug. `M-NoPwTop` is an
   unfused/over-materialised intermediate — fix it as a primitive op (if the math
   IS one op, e.g. HC mixing = a contract matmul) or a fusion pattern (if it maps
   to a real framework kernel, e.g. the indexer's `fp8_paged_mqa_logits`).

See `[[feedback_curve_anomaly_selfaudit]]` (the method) and
`[[feedback_axiom_op_component_level]]` (axioms belong at the op/component level).

## Fixing — step 2 (gated, via gate.py)

A fix lands only when **anchored by deterministic evidence** — else an autonomous
loop becomes the "confident error amplifier" the validation system exists to
prevent. `gate.py` enforces both anchors:

1. the targeted axiom goes **red → green** (sweep no longer flags it),
2. **no new violations** appear AND every golden/recon test stays **green**
   (capacity / validator / kv-hybrid / kv-read-cardinality / axioms / invariants).

```
# 1. snapshot the clean tree (records violations + that goldens are green):
uv run python .claude/skills/audit-curves/gate.py baseline --out .audit_baseline.json
# 2. apply the fix (ops.py / a component yaml / apply_fusion.py / fusion_plans/…)
# 3. verify it's anchored; --commit lands it only on PASS:
uv run python .claude/skills/audit-curves/gate.py verify \
    --baseline .audit_baseline.json \
    --target "DeepSeek-V4-Pro|M-NoPwTop" \
    --commit "fix(...): ..."
```

`--target` is the violated-axiom key `model|axiom_id` (verify prints the live
keys when the target isn't gone). **PASS** (exit 0) auto-commits on `--commit`;
**FAIL** (exit 1) = revert and file for human review. The gate is necessary, not
sufficient: green means "no regression in the golden surface", not "provably
correct" — sanity-check the fixed number against first principles too.
