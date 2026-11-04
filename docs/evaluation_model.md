# The Evaluation Model: One Program, Many Bindings

Every number HuggingArch shows — a parameter count, a FLOP total, a KV byte
figure — is backed by the algebraic expression that produced it. Hover a value
and its formula appears; the formula and the number cannot disagree, because
they are two evaluations of one expression. That is not a display nicety. It is
what makes the analysis auditable, and it is what lets the validator check
geometry without executing a single numeric kernel.

This document explains the evaluation core in `backend/arch/eval/`: the
program/binding/evaluation trio, the layers an analysis is made of and the four
kinds they come in, why shape is a first-class propagated quantity, and how the
symbolic and numeric sides are one walk rather than two engines. It builds on
the spec vocabulary from the [technical design](design.md).

## Layers of the runtime

YAML is authoring. The runtime is one exit, one walk, then rewrites:

```
YAML / BlockGraph
        │  iter_tree
        ▼
   SpecIRTree     expanded graph (nodes, spine, stream table, this call's I/O)
        │  Program.of
        ▼
   Program        walk schedule (stacks, stages, pairings) — holds the tree
        │  Evaluator + LayoutPropagation (a walk Rule)
        ▼
   Analysis       shapes, cost, KV, per-rank
        │  materialize_comm then Framework.rewrite
        ▼
   Rewrite        same Analysis, rebased — not a second evaluation
```

- **`SpecIRTree`** (`spec_tree/`) is the expanded graph. Template I/O lives on
  this invocation's `BlockMeta`. The stream table and embed→layers→norm→head
  order live on the tree. Structure questions read here, not `spec.model`.
- **`Program`** (`eval/program.py`) is the walk schedule, not a second graph.
  Stacks (backbone plus aux — a vision tower, an MTP head) are sequences of
  stages: an op, or a block template evaluated once and multiplied by count.
  28 identical layers are one template and a count. Pairing and this run's
  remaps live here. `contributions` is the roots-times-counts rule every
  total follows.
- **`Binding`** (`eval/binding.py`) is the value domain: `d_model → 7168` or a
  `sympy` symbol. Extents stay global. What one rank holds is the layout's
  answer; a weight cut that does not divide is handled only as its declared
  `fit` says (`exact` refuses, `replica_cells` co-owns, `pad` pads).
- **`Evaluator`** (`eval/evaluator.py`) walks the program under a binding.
  Layout is a Rule on that walk (`eval/rule.py`'s `Rule` answers per-rank
  questions with the identity — local == global — and `LayoutPropagation`
  overrides them). Comm injection and fusion rewrite a copy of the Program
  afterwards; Inspector vs the serving graph are two drawings of that rewrite.
- **`Deployment`** (`eval/transforms.py`) is the composition root: it assembles
  the walk with the layout rule for one deployment (`Workload`, `Phase`) and is
  the only module in `eval/` allowed to import `arch.layout`.

The walk's input includes the precision profile; the evaluator resolves it into
a `ResolvedPrecision` on `Analysis.precision` (`eval/precision.py`), and
`Analysis.precision_effective()` reports each object's shipped and effective
format. A rewrite (`eval/layer.py`'s `Rewrite`: `origins`, `dropped`, composed
with `then`) maps new nodes back to the walked ones; layers rebase across it,
and the pre-fusion analysis stays reachable as `Analysis.walked`.

What comes out is an **`Analysis`** (`eval/analysis.py`): layers over the
program's nodes, plus the binding it was evaluated against.

## Layers, and the four kinds they come in

An analysis is not a tree of records. It is a set of **layers**
(`eval/layer.py`) — each a partial map from node uid to one kind of fact:
`Layer[Cost]`, `Layer[Shape]`, `Layer[StoredTensor]`, instance counts, roles,
cache laws. A consumer that wants cost receives cost, not a twenty-nine-field
record; attribution questions (`attention_params` vs `ffn_params`) are a `zip`
and a group-by over layers that already exist, never a second traversal.

Layers come in four kinds, and the distinction decides where each is computed:

1. **Propagated** — the value at a node depends on the *same layer's* value at
   its neighbors, so an ordered walk is unavoidable. There are exactly two:
   **shapes** (an output shape is a function of input shapes and the op's rule)
   and **sharding states** (a tensor's distributed state follows from its
   producer's, through `LayoutPropagation.transition` in
   `arch/layout/propagation.py`; `eval/transition.py` holds only the records the
   rule and the walk share).
2. **Walk-computed** — needs the *frame* (the dim environment of the enclosing
   template and component call), but not the neighbors: **cost** (flops /
   params), **storage declarations**, **node dims**, **cache positions and
   state sizes** (a window length is rebound per template; only the frame knows
   which). These are computed during the walk because the environment exists
   only there — not because anything flows.
3. **Sideband** — external facts joined in by identity, neither propagated nor
   derived: **formats** (a weight's representation follows from its role and
   the model's quant context, resolved by lookup). Measured kernel times are the
   same kind of fact, joined by the calibration overlay outside the analysis
   (`inference/calibration_overlay.py`); they sit beside the theory and never
   overwrite it.
4. **Derived** — pure functions of the finished analysis: **traffic** (HBM
   bytes; `eval/traffic.py`), the **cache footprint** (`eval/cache.py`
   assembles per-component `Cache` objects and their storage/read laws), the
   **capacity and checkpoint joins** (`eval/capacity.py`,
   `eval/checkpoint.py`), the `Analysis` join methods themselves
   (`tensors()`, `by_role`, `by_stack`, `total`), and the per-rank fold
   (`arch/layout/`). Collectives themselves are not a layer: they are grafted
   as graph nodes onto a copy of the Program (`materialize_comm_tree`). Derived layers can traverse the *graph* — a liveness
   analysis reads program order — but never depend on the walk's carried
   state, and never on their own value at another node.

The governing rule is one sentence: **the walk computes only what needs the
frame; everything else derives.** Every value that could be derived but is
materialized instead is a fork — a stored copy with nothing forcing it to agree
with its source. The system's byte and pricing bugs have historically been
exactly such forks, which is why the boundary between kinds 2 and 4 is treated
as architecture, not filing.

Why traffic in particular must be derived: flops is *extensive* — additive over
any partition of the nodes, so a fused node's flops is the sum of its members
and the value survives graph rewrites by summation. Traffic is a *boundary*
quantity — decided by which tensors cross the cut, not by the interior. Fusing
`attention_score → softmax → attention_apply` into one flash kernel removes the
score matrix from the boundary; no combination of the members' traffic values
can express that, so the only correct computation is to re-derive from the
rewritten graph's shapes — at which point the fusion saving is simply
`Σ pre − Σ post`, with no "subtract the eliminated buffer" logic to get wrong.

## Shape is first-class

`in:` on a spec node is dataflow, not a comment. The evaluator feeds each
node's op the already-propagated shapes of the nodes it names, and the op
answers through a declarative rule. Built-in primitives use `ShapeKind` in
`arch/ops_base.py`: `preserve`, `broadcast`, `axis_replace`, `axis_select`,
`axis_concat`, `contract`, `permute`, `source`, or `explicit`. YAML
`custom_ops` may declare the first seven; `source` / `explicit` need a
DimMap in `axis_propagation.py`.

One mechanism runs at three scales: between nodes inside a component, across
nested component invocations, and across model layers. Axes carry identity as
well as extent — `eval/dim.py`'s `Dim` pairs the symbolic expression with the
value, so a per-rank shape can show *which* axis the parallelism landed on, and
a formula popover renders `1*8*1*576` with the 8 identifiably the sharded head
count.

Where an op's `shape_in` template names dim symbols, the **binding is
authoritative** over whatever extent the upstream produced. That one rule is
what lets a decode attention read the whole cache (`s_kv` positions) while the
K/V leaves feeding it emit a single new token. The phase binds the step's query
length as `s_query` (prompt length in prefill, `n_verify` in decode); op
templates still write the residual axis as `s_ctx`, which is a lookup alias of
`s_query`, not a second dim. `s_past` is the keys before the step's first
query, and `s_max` the model's published context length.

## Symbolic evaluation

One `Evaluator` walks the program once, with the workload's counts as
symbols and the model's dims as numbers. Every extent is a `Dim` whose
identity half is the quantity in the model's names and whose law half is a
law in the counts, read at a point by `Analysis.at`. How a quantity goes from
that walk to the page, through cells, the where column and the declarations
derived names come from, is in [symbolic_evaluation.md](symbolic_evaluation.md).

Evaluation is strict about coverage: an op the walk cannot
model **raises**. There is no skip channel — a gap that returns a plausible
number is indistinguishable from a wrong implementation, and an oracle cannot
tell the two apart. A refusal says whose it is to fix (`eval/refusal.py`): a
`SpecRefusal` is the spec's, an `EngineGap` the engine's, a
`DeploymentRefusal` the deployment's: a combination of parallel knobs the
walk does not run whatever the model, stated once on `ParallelConfig`
(`refusals(phase)`) so the walk raises it and `validate()` reports it.

## Parallelism: global extents, layout for per-rank geometry

A per-rank analysis is the same program, same binding. Extents stay global.
Heads, expert copies and sequence cuts live on the layout `Block`: equal
replica cells that tile the rank group, or padding a weight declares
(`fit: pad`). A degree that does not divide its dim is handled only as declared
— never rewritten into a quieter divisor and audited after the fact.

Because per-rank shapes are the ground truth and traffic is derived from them,
two context-parallel schemes (Ulysses and pcp) must agree where their per-rank
work is the same: equal per-rank FLOPs, and per-rank activation traffic that
differs only by the factors the schemes' layouts imply ({1, cp, 1/cp}).
`inference/invariants.py` (`_inv_cp_scheme`) states it, and
`tests/backend/inference/test_perturbation_invariants.py::TestCpSchemeInvariantActivation`
checks it on DeepSeek-V3, V3.2 and V4-Pro. Communication itself
is grafted as first-class graph nodes at state-mismatch edges — see
[parallelism](parallelism.md).

## Two trees: structure versus evaluation

The structural IR, **`SpecIRNode`** (`arch/spec_tree/nodes.py`), is *"the model
has this structure"* — resolved once per model and queried by consumers that
need no numbers: the validator, the renderer, the capacity planner. It is the
canonical identity carrier (`display_path`, `aggregation_id`, `template`,
`op`, `role`). The evaluation side keys everything by node uid and reaches
identity through the program's nodes; structure is never duplicated into the
result.

`eval/projection/` is the presentation layer: the one place analyses
serialize into the shapes the pipeline, the API and the dashboard consume —
the tree-shaped model result, the per-leaf access rows, the formula payload.
Projections are read-only assemblies of analysis layers; nothing there
recomputes a cost or a byte. Its read surface is pinned by a ratchet test
(`tests/meta/test_eval_boundary.py`).

## Where this lives in the code

| Concern | Location |
|---|---|
| Canonical dimension registry | `arch/dims.yaml` + `arch/dims.py` |
| Primitive `flops_fn` / `params_fn` / `shape_kind` | `arch/ops_base.py` (`OpSpec`), `arch/ops_registry.py` (registrations) |
| Shape DimMap (axis correspondence) | `arch/axis_propagation.py` |
| The compiled model (stacks, templates, aliases, contributions) | `arch/eval/program.py` |
| Value domains | `arch/eval/binding.py` |
| Layout (global extents + per-edge state) | `arch/layout/` |
| The walk | `arch/eval/evaluator.py` |
| The result container and its joins | `arch/eval/analysis.py` |
| Layers (the column algebra) | `arch/eval/layer.py` |
| Axis identity beside extent | `arch/eval/dim.py` |
| Derived: HBM traffic | `arch/eval/traffic.py` |
| Derived: cache footprint laws | `arch/eval/cache.py` |
| Derived: weight/storage bytes, checkpoint join | `arch/eval/capacity.py`, `arch/eval/checkpoint.py` |
| The rule contract (identity per-rank answers) | `arch/eval/rule.py` |
| Sharding transition rule + state algebra | `arch/layout/propagation.py` (records shared with the walk: `arch/eval/transition.py`) |
| Composition root (walk + layout rule per deployment) | `arch/eval/transforms.py` (`Deployment`) |
| Precision profile resolved against a program | `arch/eval/precision.py` |
| Cache bound to a deployment | `arch/eval/cache_view.py` |
| Refusal attribution | `arch/eval/refusal.py` |
| Physical fold / collectives / comm bytes | `arch/layout/` |
| Weight formats / quant representation | `arch/eval/formats.py`, `arch/eval/storage.py` |
| Fusion (`Framework.rewrite`) | `arch/eval/fusion.py` |
| Presentation projections | `arch/eval/projection/` |
| Structure tree (shared IR substrate) | `arch/spec_tree/` |
