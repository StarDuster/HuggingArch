# Inference package — design notes

What lives in `backend/inference/`, how the pipeline is ordered, and where
each number comes from. The deep dives are in `docs/` — this file is the
code-adjacent map. See
[`docs/inference_cost_model.md`](../../docs/inference_cost_model.md) for the
roofline, [`docs/parallelism.md`](../../docs/parallelism.md) for sharding and
communication, and
[`docs/evaluation_model.md`](../../docs/evaluation_model.md) for the
evaluation core these modules consume.

## Scope

This package turns a resolved spec plus a deployment
(`ParallelConfig`, GPU SKU, dtypes, workload knobs) into serving numbers:
TTFT / TPOT, per-op timing rows, capacity (max batch / max tokens), weight
and KV budgets, and the calibration overlay that puts measured numbers
beside the theory.

## Core separation — model capability vs user config

- The **spec** states what the model is. It carries no deployment facts:
  no degrees, no dtypes, no fusion. Which parallelism *can* divide a
  dimension is a property of the dimension itself, classified once in
  `arch/dims.yaml` (`shardable_by`); which world actually splits a stored
  weight's axis is declared on that weight (`weight_axes`), and
  activations declare nothing.
- The **user config** (`ParallelConfig`, defined next to the layout
  walk in `arch.layout`) states what the deployment does:
  TP / EP / CP / PP / DP degrees, replica counts. This package
  re-exports it. Consistency constraints
  (`tp_attention × cp × dp_attention = attention_world_per_replica`, the
  MoE analogue) are checked on the config, and a no-op knob (TP on an MLA latent)
  warns instead of silently pricing.

Per-rank numbers are physical folds of one evaluation. `LayoutPropagation`
carries each edge's placement during the evaluator walk; `physical.py` then
combines global extents with those layouts. The pipeline never evaluates a
second local model and never divides a completed model total afterward.

## The pipeline (`pipeline.py`)

Pass ordering, each pass consuming the previous:

1. **Evaluate once** — `StepEvaluation.analysis` runs the program with
   `LayoutPropagation`, yielding global semantic layers plus edge layouts.
2. **Select layer multiplicity** — pipeline-parallel ownership becomes one
   `Layer[int]`: the stage's view of placement (`layout/stages.py`,
   `StagePlacement.view(stage).rows`), which a stage index names across the
   step's analyses; FLOPs, storage, cache and communication all read it, and
   the stage pays the hops across its exit cut.
3. **Derive physical layers** (`layout/physical.py`, `persistent.py`) — rank
   shapes, FLOPs, traffic, cache views and resident artifacts are pure folds
   over that `Analysis`.
4. **Materialize communication** (`layout/materialize_comm.py`) — state
   mismatches become explicit collective events and graph nodes with local
   payload geometry and process-group identity.
5. **Rewrite fusion** (`eval/fusion.py`) — a framework binding produces one
   `Rewrite`; each layer rebases with its own combine law. No fused result
   object reconstructs all concerns.
6. **Per-op roofline** (`phases.py`, `row_roofline.py`) — `t_sol = Σ_op
   max(t_compute, t_memory, t_comm, t_launch)` per phase, prefill and
   decode, with the decode graph swapped to phase variants the spec
   declares (absorbed MLA). Four bounds (`row_roofline.BOUNDS`): a
   collective's `t_comm` is α-β — its bytes over the link plus one link
   latency per traversal of the latency-optimal small-message schedule
   (`comm_volume.latency_hops`), the same latency on every fabric — and a
   kernel that is not a collective is never faster than the launch floor
   (`t_launch = n × launch_floor`). The two floors are declared once in
   `defaults.py` and a request may set its own. Rows are serial (no
   compute/comm overlap); the per-bound sums a phase shows are components,
   not addends of `t_sol`. See
   [`docs/inference_cost_model.md`](../../docs/inference_cost_model.md).

`eval/projection/result.py` is outside this serving path. It produces
read-only `NodeReport`/`ModelReport` trees for validation, the inspector and
API presentation; those reports stay at the presentation boundary and are
never fed back into physical analysis.

## Phases (`phases.py`)

Prefill and decode are the same graph under different bindings —
`s_query = s_ctx` vs `s_query = 1` with `s_kv` the accumulated cache — plus
per-phase graph variants
declared at the spec level. KV read/storage laws come per cached tensor from
`arch/eval/cache.py`; physical cache geometry is derived from the same
analysis and restricted by the same stage (`CacheView.on(view)`).

## Capacity (`capacity.py`)

Decode is KV-bound (residual HBM ÷ per-request KV at the target length),
prefill is activation-bound, weights are the floor. Native weight bytes come
from `resident_weights` on the walk's artifacts (parameters + metadata +
buffers). A pipeline stage is the same join after `artifacts_for_layers`
over the stage's weight projection (`StageView.weights`: what runs there,
plus the parameters and tied tables it reads). Which stage binds is a memory
question: every stage is priced (`_StageProfile`, holding its `StageView`),
the workload-free choice is re-bound by the ceilings at the request length,
and the weight tree, the per-op rows and the phases are all built for the
bound stage's index.
The display tree is the module breakdown beside those numbers; serving-cast
and per-role dtype still price through the tree, because those rewrite
storage format. A serving-dtype override casts only when the requested
width is below the tensor's resident width (precision accessories above
the dominant width are exempt). The calibration overlay
(`calibration_overlay.py`) joins the measured corpus per op signature and
reports `t_measured` beside `t_sol` — beside, never instead.

## Weight breakdown (`weight_breakdown.py`)

The per-module storage view (weights / buffers / quant metadata per rank).
Still carries the legacy module-class summary for display grouping; byte
totals are anchored to the eval tensor join, and `source_logical_name`
keeps scheme renames (`.weight_packed`) joinable to the op's declared name.

## Invariants held by tests

- **Prefill CP schemes** (`invariants._inv_cp_scheme`): Ulysses and pcp owe
  equal per-rank FLOPs except a K/V decompression producer, where Ulysses
  computes `1/cp` of PCP's resident rows before the attention-entry A2A; the
  expected ratio is read from walked sequence-axis identities. Per-rank
  activation access differs only by the factors each scheme's cut implies
  (query alike, KV ×cp under pcp, a head-less per-token operand 1/cp) —
  `test_perturbation_invariants.py` on the DeepSeek family at `cp = 8`.
  A pcp rank's own work (`1/cp` of the pairs, the cache read through the CP
  collectives) and Ulysses's resident MLA decompression are pinned in
  `test_cp_cache_reads.py`.
- **Phase closure and batch linearity** across the class-diverse cohort.
- **Capacity golden** (`tests/golden/capacity_baseline.json`): exact
  max-batch / max-tokens, 1e-6 relative on times.
- **Curve axioms** (`.claude/skills/audit-curves`): Tier-0 monotonicity +
  weight floors + reconciliation per attention class.
