# HuggingArch Technical Design

This is the deep-dive technical document for HuggingArch. It explains how the
spec system, the evaluation core, the validation system, the quantization
model, and the agentic spec-generation pipeline work, and what design
trade-offs they make.

If you are new to the project, start with the [project README](../README.md)
for the pitch, quickstart, and feature overview, then come back here for
the deep dive.

## Contents

1. [Spec system: Primitive / Component / Model](#1-spec-system-primitive--component--model)
2. [Evaluation core](#2-evaluation-core)
3. [Information sources](#3-information-sources)
4. [Validation: gates and diagnostics](#4-validation-gates-and-diagnostics)
5. [Quantization](#5-quantization)
6. [Agentic spec-generation pipeline](#6-agentic-spec-generation-pipeline)
7. [User-facing surfaces](#7-user-facing-surfaces)
8. [Spec authoring contract: naming convention & dtype matching](#8-spec-authoring-contract-naming-convention--dtype-matching)
9. [The compiler lens](#9-the-compiler-lens)

---

## 1. Spec system: Primitive / Component / Model

HuggingArch organizes every model into a **three-layer op system** —
primitive op, component (compound op), and model — with each layer carrying
its own grammar and responsibility boundary.

### 1.1 Primitive ops

`OpSpec` and `OP_REGISTRY` live in `backend/arch/ops_base.py`; the built-in
ops are registered with `register_op(OpSpec(...))` in
`backend/arch/ops_registry.py`, and `backend/arch/ops.py` is the lookup facade
(`get_op`, `is_primitive`, …). Each primitive carries a parameter-count
formula (`params_fn`), a FLOPs formula (`flops_fn`), a shape specification
(`shape_kind`, plus a hand-written DimMap for an irregular case; see §2.3),
and the weight tensors it stores (`tensors=(TensorDecl(".weight", …),
TensorDecl(".bias", …, when="bias"))` — the one declaration every weight
name and byte count is derived from). The formulas are not strings — they are
sympy `Symbol`-based lambdas, so `linear` returns
`2·B·s_ctx·in·out + B·s_ctx·out·bias` directly. An `OpSpec` also declares how
it couples across ranks (`couples_on` / `axis_use`) and, for attention ops,
per-query KV-accounting hooks.

There are currently about 100 primitives, covering: linear-algebra and
attention math (`linear`, `embedding`, `rmsnorm` / `layernorm`, `rope`,
`attention_score`, `softmax`, `attention_apply`); an activation family
registered from a single generator (`silu`, `gelu`, `relu`, `relu_squared`,
`sigmoid`, `tanh`, `softplus`, …); zero-FLOP view ops (`view_split`,
`view_merge`, `transpose`, `permute`, `slice`, `concat`); attention-variant
and state-space kernels (`swa_*`, `dsa_*`, `mamba2_scan`, `gated_delta_rule`,
`conv1d`); positional and multimodal ops (`mrope`, `rope_2d`, `xdrope`,
`patchify`, `spatial_merge`, `conv2d`); routing and selection (`topk`,
`hash_route`); Hyper-Connection mixing (`hc_mix`, `hc_split_sinkhorn`); and the collective-communication ops the parallelism model injects
(`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, and the
pipeline's point-to-point `send_recv`).

The global registry is maintained in code because its FLOPs / params formulas
and Python-only shape or cache laws are shared contracts. A model spec may add
an atomic, schema-limited `custom_ops` declaration; composite structure still
belongs in components. Human-readable head reshapes such as `head_split` /
`head_merge` are **not** primitives — they are library components (§1.2) that
wrap a `view_split` / `view_merge` with a `transpose`.

### 1.2 Components (compound ops)

This is the layer that absorbs model-structure diversity. Each component is a
YAML file living in one of **three tiers**, where the directory *is* the tier
declaration (no extra field or tag):

- `backend/arch/components/library/` — general-purpose topologies reused
  across models (GQA, MLA, SwiGLU, MoE, sliding-window attention, …). Grows
  continually as new models are added.
- `backend/arch/components/adapters/` — thin binding / `module` /
  `inner_modules` shims tailored to one model family, handling
  checkpoint-naming differences (`wq` vs `q_proj`, stacked-Parameter experts
  vs `ModuleList` experts) without introducing new graph nodes.
- `backend/arch/components/drafts/` — the sandbox tier for components an
  agent generates during spec generation. Drafts are parsed at load time but
  do **not** enter the global namespace; a draft is visible only to a spec
  that lists it in a top-level `drafts:` field (see §6.7), so a broken draft
  cannot pollute other specs.

Each component describes a **DAG**:

- A component with one formal input does not declare `inputs`; that formal
  is named `x` inside the component. Author `inputs` only for a different
  interface. Repeating the default as `inputs: [x]` is a load error.
  `outputs` names the tensors exposed to the parent.
- `graph` — DAG nodes. Each node's `op` field can name a primitive **or**
  recursively name another component, supporting unlimited nesting.
- `in:` per node — declares which upstream nodes feed the node. The engine
  follows these edges to propagate shape (see §2.3).
- `dims` per node — local dim bindings (e.g. `out: "n_heads * d_head"`) that, together
  with the propagated upstream shape, drive sympy evaluation.
- `chain` / `repeat` / `active` (on a graph node) — copies of one node.
  `chain: N` runs the copies in sequence: copy *i* reads copy *i−1*, and a
  reader of the node sees only the last (a vision tower's block list, a fold
  such as MiMo's speech-bank accumulator). `repeat: N` places the copies side
  by side: every copy reads the node's own `in:`, and a reader sees all of
  them (a shared-expert bank); `active` is its FLOPs multiplier where only
  some copies run per token. The two arrangements cost the same FLOPs,
  parameters and storage and differ only in the bytes on the edge leaving the
  node, which is why the arrangement is declared rather than inferred; a node
  may not declare both (`schema/graph.py`). Routed expert banks are written as
  a leading expert axis instead ([MoE expert axis](moe_expert_axis.md)), and
  `repeat:` remains only on shared-expert banks.
- `params` — parameterizable slots that let an attention type / FFN type /
  norm type be injected by the caller (component as a role).
- `cache.family` / `cache.entries` (attention components) — declare the
  KV-cache topology: which graph values persist, their optional dtype override,
  the cache-family label, and — for each recurrent state — the node that
  carries it and its `shape` (the state's axes, outermost first; its size is their product).
  The per-node storage *law* is not a component field; it is
  resolved from graph op laws (the node, subcomponent output, consumer, or
  attention anchor) — see
  [KV Cache Storage](kv_cache_storage.md) for the full model and its
  cross-check.
- `attn_variant` — optional compute-flavor label, written only when it
  differs from `cache.family` (for example MLA cache + DSA compute).
- Component claims live only in
  `backend/arch/claims/components/<identity>.yaml`, outside the executable
  component YAML; the loader attaches them by identity to the standalone
  component or model-inline block of that name, and an `axioms:` key in a
  component YAML, draft, or inline block is a load error. The validator judges
  the claims against the cache each call of the component grows in the model's
  own analysis; `src` citations are human-reviewed, not fetched or verified
  mechanically.

Example: `gqa_attention.yaml` (abridged — `k_proj` / `v_proj` mirror `q_proj` with `n_kv_heads`)

```yaml
role: attention
description: "Plain Grouped-Query Attention with no QK-norm. Llama, Qwen2, Mistral."
cache:
  family: GQA
  entries:
    - { node: k_rope, kind: kv }
    - { node: v_split_heads, kind: kv }
graph:
  q_proj:
    op: linear
    in: x
    dims: { in: d_model, out: "n_heads * d_head", bias: Qb }
    weight_axes:
      - placement_by_symbol:
          n_heads: { world: attention, parts: n_heads, fit: replica_cells }
          n_audio_heads: { replicated: true }
      - { replicated: true }
  # k_proj, v_proj: the same, counted in n_kv_heads
  q_split_heads:   { op: head_split, in: q_proj, dims: { n_heads: n_heads, d_head: d_head } }
  k_split_heads:   { op: head_split, in: k_proj, dims: { n_heads: n_kv_heads, d_head: d_head } }
  v_split_heads:   { op: head_split, in: v_proj, dims: { n_heads: n_kv_heads, d_head: d_head } }
  q_rope:          { op: rope, in: q_split_heads, dim: d_head }
  k_rope:          { op: rope, in: k_split_heads, dim: d_head }
  qk_score:        { op: attention_score, in: [q_rope, k_rope], dims: { n_heads: n_heads, d_head: d_head } }
  softmax:         { op: softmax, in: qk_score, dims: { n_heads: n_heads } }
  attn_apply:      { op: attention_apply, in: [softmax, v_split_heads], dims: { n_heads: n_heads, d_head: d_head } }
  o_merge_heads:   { op: head_merge, in: attn_apply, dims: { n_heads: n_heads, d_head: d_head } }
  o_proj:
    op: linear
    in: o_merge_heads
    dims: { in: "n_heads * d_head", out: d_model }
    weight_axes:
      - { replicated: true }
      - placement_by_symbol:
          n_heads: { world: attention, parts: n_heads, fit: replica_cells }
          n_audio_heads: { replicated: true }
outputs: [o_proj]
```

Notable properties:

- `role: attention` declares how the component is attributed and which slot it normally fills in a parent
  block template. Every role or slot name — a component's `role:`
  (`BlockGraph.role`), a model's `components:` slot, a call-site override
  (`GraphNode.role`) — is classified exactly once in
  `backend/arch/roles.yaml` (about twenty names in six kinds: attention, ffn,
  memory, norm, residual, wrapper), and the schema and `SpecLoader` reject a
  non-empty name the registry does not classify. A role may `refine:` a
  broader one (`attention_swa` refines `attention`), so a rule that selects
  `attention` — a `quant_context.per_role` entry, a precision rule — reaches
  it too. The schema permits an omitted
  role for utility sub-components (compressor, indexer, `head_split`, ...);
  missing attribution on a weight-bearing component is a structural warning,
  not a load error.
- `graph` is a dict, not a list — each key is a node name; values reference
  one another by `in:` to form the DAG. `op: head_split` / `head_merge` here
  name **library components**, not primitives (see §1.1).
- `dims` values are sympy expression strings: `out: "n_heads * d_head"` is parsed into
  a symbolic product, not a string concatenation.
- Sharding is declared in two places. Which parallelism *may* divide a
  dimension is a property of the dimension (`kind` / `shardable_by` in
  `backend/arch/dims.yaml`); which one actually splits each stored weight axis
  is declared on the node (`weight_axes`): `q_proj` splits its output axis by
  head (column-parallel), `o_proj` its input axis (row-parallel), and `fit`
  says what happens when the head count does not divide the degree. Activations
  declare nothing — their per-rank shape follows from the dims. See
  [parallelism](parallelism.md).
- `cache.entries` / `cache.family` tag the KV-cache topology. The K entry is
  `k_rope`, the post-RoPE value that is stored, even though attention consumes
  the same node as an input. `attn_variant` is only declared when compute flavor differs from
  that topology. Adding a new linear-attention family is one cache declaration that
  all downstream tools then recognize.
- The claims file `claims/components/gqa_attention.yaml` pins a growth-law
  expectation: at each call of the component the validator checks that the
  engine's per-position KV total equals `2·n_kv_heads·d_head` in that call's
  dims (see §4).
- `bias: Qb` makes "QKV bias on/off" a parameterizable dim — the model spec
  passes `Qb=0` or `Qb=1` to reuse the same component.

The whole graph is a standard `q_proj → head_split → rope → attention_score →
softmax → attention_apply → head_merge → o_proj` GQA flow. The component
system is open: MQA is `n_kv_heads=1`, MHA is `n_kv_heads=n_heads`, partial-RoPE is a variant that
swaps the `rope` node. New models compose like this.

### 1.3 Model spec

Defined in `SpecDocument` (`backend/arch/document.py`). Describes: how
many block templates this model has, which template each layer instantiates,
which component fills each slot inside the block. e.g. `qwen2.yaml` uses one
`block` template plus `components: {attention: gqa_attention, ffn_dense: swiglu_ffn}`,
and `repeat: n_layers` expands the backbone; a hybrid model (some layers
MLA, some SWA) selects a template per layer with `per_layer: {assign, map}`
on the repeated stack.

`SpecDocument` is the authoring schema and only that: every field in it is one a
person types, `extra="forbid"` so a misspelled key is an error naming the key,
and `per_layer: {assign, map}` has a type of its own rather than an ad-hoc
read inside the loader. `SpecLoader.lower()` takes it one way to `ArchSpec`
(`backend/arch/schema/`), the assembly product the engine reads: the
block graphs that came out of binding components into slots, the transitive
component closure, the role map. Nothing downstream re-reads the document to
reach its own conclusion.

Minimal example:

```yaml
# models/qwen2.yaml — coding agent only needs to write this
match:
  model_type: [qwen2]

components:
  attention: gqa_attention
  ffn_dense: swiglu_ffn
  norm: rmsnorm
  block: pre_norm             # block pattern: norm→attn→residual→norm→ffn→residual

dims:
  d_model: config.hidden_size
  n_heads: config.num_attention_heads
  n_kv_heads: config.num_key_value_heads ?? n_heads
  d_head: config.head_dim ?? "d_model // n_heads"
  Qb: "1"
  d_ffn: config.intermediate_size
  n_vocab: config.vocab_size
  n_layers: config.num_hidden_layers
  s_max: config.max_position_embeddings   # the published context length

model_inputs:
  input_ids: { kind: tokens }

checkpoint: { root: model }               # tensor names live under `model.`

model:
  - { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding, dims: { n_vocab: n_vocab, d_model: d_model }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }] }
  - { name: layers, repeat: n_layers, template: block, inputs: [embed_tokens], streams: { hidden: { carries: true } } }
  - { name: norm, op: rmsnorm, dim: d_model, weight_axes: [{ replicated: true }], inputs: [layers] }
  - { name: lm_head, op: linear, checkpoint: { path: lm_head }, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], tied_weight: { to: embed_tokens, when: config.tie_word_embeddings }, inputs: [norm] }
```

### 1.4 Model-layer expressiveness boundary

The currently supported scope is **decoder-only (multimodal) LLMs**. The
model layer is **an ordered list whose dataflow is authored** — a
deliberate scope contraction. Understanding the boundary is useful for spec
authors and for anyone extending the system.

**Edges are `inputs:`.** `model:` is an ordered array, and every entry
that reads something names it: `embed_tokens` reads `input_ids`, `layers`
reads `embed_tokens`, `norm` reads `layers`, `lm_head` reads `norm`. Only a
zero-input op (a parameter, a buffer) names nothing. The names bind to the
reader's formals by position, one to one:

```yaml
- name: language_model.model.norm
  op: rmsnorm
  dim: d_model
  inputs: [language_model.model.layers]
- name: language_model.model.output_attn_res_norm
  op: rmsnorm
  dim: d_model
  inputs: [language_model.model.norm]
```

**What the model receives is declared.** Token ids, images and audio clips
come from outside the graph, so no node produces them; each is a
`model_inputs:` entry, read by name like a layer. Token ids are
`{ kind: tokens }`, one id per prompt position, as many as the step's
`s_ctx`. An image declares its own axes — `pixel_values: { kind: image,
axes: [n_img, img_h, img_w] }`, images per request and the size the processor
hands the tower — and an audio clip `{ kind: audio, axes: [n_aud, s_audio] }`;
the workload gives them (`Workload.media`). An image is read by a `patchify`
layer citing the processor's patch size and resize factor, so its rows are
its patch grid, one attention segment per image. Each encoder runs in its
own phase, `encode`, whose binding holds the media axes and none of the
language model's step; the language model's prefill and decode do not walk
the encoders. The rows an encoder hands the language model are declared
where they land: every edge into the language model states `prompt_slots:
{ domain: <media input>, defines: true | false }`; the positions a request's
media fill are each defining edge's rows, reported on the encoder's side. The
language model's step is its own text workload: its handoffs carry no rows,
no prefill or decode number moves with the media, and an encoder's weights and
activations are the encoder's accounting, none of the language model's
capacity. The estimate prices each encoder's step over the request's media on
its own (`encode`), beside prefill and decode and never inside them; the
Inspector shows an encoder's FLOPs per media item as an expression in the
item's axes. A raw image's processed size comes from the model's own
processor configuration (kept in its snapshot), resized by the rule of the
processor class it declares. A video is `[n_vid, vid_frames, vid_h, vid_w]`,
cut into temporal patches (the last frame repeated to a whole patch), by the
same `patchify` that cuts an image, a video of one frame. A module the forward
runs again on another input (a model's videos through its image tower,
Step-3.7's crops through its vision model) is a `call:` of the layer that
declares it: the loader writes the call out as that layer under the call's
name, and every weight it reads is the called layer's, stored once. One walk
binds one size per media input; a request whose items differ in size is
refused. Where each axis lives follows from its
place in the tensor: the batch is the request owner's, text rows take the
sequence cut, an encoder's rows arrive whole, a width is whole. An input that
nothing produced and nothing declares is refused, not read off an op's
signature.

A repeated stack hands its value from one iteration to the next through
the stream it carries (`streams: { hidden: { carries: true } }`);
every other value that crosses layers is a stream too (see
`backend/arch/DESIGN.md`, "Cross-layer wiring").

Constraints:

- `inputs:` only allows references to **earlier entries** — earlier
  top-level layers, earlier siblings inside a group, an appended stream
  an earlier stack declares, or a model input. The `ArchSpec` validator in `schema/` rejects
  forward jumps, self-references, and cycles, keeping model-level dataflow
  acyclic.
- This field is model-level dataflow and selects the inbound tensor the
  evaluator hands the layer. It does not change a layer's declared
  parameter formula by itself; downstream shape-derived values follow the
  selected input's shape.

**Visual collapsing of periodic interleaving.** Hybrid models often follow
"every N layers, X layers of template A and Y layers of template B" cycles
— e.g. Qwen3-Next's `(linear×3 + full)`, GPT-OSS's `(SWA + GQA)` two-cycle
alternation. `find_periodic_span()` in `backend/rendering/_periodic.py` and the
cycle emitter inside `layout_assignments_in_order()` in
`backend/rendering/_layout.py` detect the longest periodic
pattern and fold it into a single "× N cycles" container card. Detection
requires period ≥ 2, ≥ 2 full cycles, and ≥ 2 distinct templates. This is
a **rendering-layer** transformation only — schema semantics are unchanged
and authored `per_layer: {assign, map}` has already lowered to the internal
layer-index-to-template assignments used by this renderer.

**Why not arbitrary DAG?** Not a technical limitation; the cost is
elsewhere. Arbitrary DAG drags in encoder-decoder (T5 / BART), cross-stack
attention, multi-stack speculative decoding — each requires not just spec
schema changes but **proportional complexity in the inference analyzer, KV
cache modeling, and parallelism semantics** (e.g. encoder KV is produced
once and broadcast globally; decoder KV is per-step — different
mathematical objects).

To keep the system convergent, HuggingArch currently scopes "analyzable
models" to **decoder-only (multimodal) LLMs**. This covers the vast
majority of recent and near-future releases. VLMs follow vision tower +
projector + decoder-only LLM and fit into the existing model-layer with
`group:` nesting. True encoder-decoder and Mixture-of-Depth are out of
scope and require independent schema / analyzer extensions; not on the
roadmap.

This boundary lets the four subsystems (op schema, symbolic eval,
verification, inference analyzer) preserve simple invariants — "every layer
produces per-token KV", "layer index uniquely maps to one block template" —
which is what enables HuggingArch to converge as the model landscape keeps
moving.

---

## 2. Evaluation core

HuggingArch's value isn't just numbers — it's **explainable formulas**.
Hovering any value reveals the symbolic expression behind it. This requires a
unified algebra running through all three op layers, and one evaluation core
producing every number.

> This section summarizes the core. For the full treatment — SpecIRTree /
> Program / Analysis, the four kinds of layer, derived traffic, and per-rank
> evaluation — see the
> [Evaluation model deep dive](evaluation_model.md).

### 2.1 Canonical dim symbols

Centrally registered and documented in `backend/arch/dims.yaml`: model
symbols include `d_model`, `n_heads`, `n_kv_heads`, `d_head`, `d_ffn`,
`n_vocab`, `n_layers`, and the architecture-specific descriptive names in that
file; a spec may declare a symbol of its own under `dim_symbols:`. Some
dims are supplied by the engine from the request rather than bound by the
model: `B` (the batch per rank), `s_query` (the step's query length —
`input_tokens − prefix` in prefill, `n_verify` in decode: 1, or the draft
length under speculative decoding), `s_past` and `s_kv`. Op templates write
the residual sequence axis as `s_ctx`; it is a lookup alias of `s_query`,
not a second dim. `s_max` is the model's published context length
(`config.max_position_embeddings`), bound by every spec whose config states
one. `resolve_dimensions` rejects an authored `dims:` key that neither the
registry nor the spec's `dim_symbols:` registers. Every primitive's
`flops_fn` / `params_fn` returns a sympy `Expr`, so a primitive's output is
already an algebraic expression rather than a string. The formula printer
(`_DotPrinter` in `ops_base.py`) writes `*` and `^` so a formula copies back
into sympy, and pins operand ordering (`B, s_ctx, s_query, n_verify, s_kv`
first) so a formula never shuffles.

### 2.2 One YAML exit, one walk, then rewrites

YAML is authoring. Structure leaves it once, through `iter_tree`
(`spec_tree/build.py`), as a **`SpecIRTree`**: nodes, edges, roles, zones,
the model spine (embed → layers → norm → head), the stream table, and this
invocation's template I/O on **`BlockMeta`**. Consumers that need structure
read that tree, not `spec.model` / `get_block` a second time.

**`Program`** (`eval/program.py`) is the walk schedule, not a second copy of
the graph. It holds the tree and arranges it into stacks and stages (an op,
or a block template × count). Pairing and this run's dim remaps live here.
**`Binding`** is the value domain — numeric sizes or sympy symbols. Extents
stay global; what one rank holds is the layout's answer.

**`Evaluator`** walks the program under a binding. **LayoutPropagation** is a
Rule on that walk (sharding does not build another graph). The result is an
**`Analysis`**: layers over the program's nodes (cost, shapes, storage,
formats, instances, cache laws). Attribution is a zip/group-by over those
layers, not a second traversal.

The walk's input includes the deployment: the parallel layout, the batch and
context, and the precision profile (`PrecisionProfile`, request field
`precision`). The evaluator resolves the profile against the program into a
`ResolvedPrecision` on `Analysis.precision` — each weight, edge, matmul and
cached tensor takes the most specific rule that selects it, or the precision
the model ships with — and `Analysis.precision_effective()` reports both
values per object. `eval/transforms.py` (`Deployment.analyze`) is the one
place that assembles the walk with the layout rule.

After the walk, serving rewrites a copy of the Program (the analysis is not
re-run):

1. Collective materialization (`materialize_comm_tree`, `layout/`) grafts
   collectives where placement states disagree.
2. `Framework.rewrite` fuses kernels (after comm, because a fused op has no
   formula of its own — cost is the absorbed ops'). Layers rebase off the
   `Rewrite`. Inspector draws the pre-fusion Program; the inference compute
   graph draws the post-fusion Program. Compute-graph L1 is a view over L2
   nodes (`backend/views/abstraction.py`), not another derivation.

Do not re-parse YAML, re-read `dim_values`, or guess from op names to answer
a question the tree or the Analysis already holds.

Model totals preserve heterogeneous structure: a hybrid backbone sums to
`n_mla_layers·MLA_FLOPs + (n_layers - n_mla_layers)·MHA_FLOPs`, not a flattened scalar, so a formula
still shows *which* layer types contribute.

### 2.3 Shape inference is first-class

The walk propagates shape along dataflow edges. `in:` is not a comment — each
node's op receives the already-propagated upstream shapes, and answers through
a **declarative** rule: built-in primitives declare a `shape_kind` — including
`preserve`, `broadcast`, `axis_replace`, `axis_select`, `axis_concat`,
`contract`, `permute`, `source`, and `explicit` (`ShapeKind` in `ops_base.py`).
YAML `custom_ops` may declare the first seven; `source` / `explicit` need a
DimMap in `axis_propagation.py`. Adding a built-in op means declaring its
kind (or a DimMap for an irregular case), never writing a bespoke shape
solver. One
mechanism runs at three scales: inside a component, across nested component
calls, and across model layers. Axes carry identity beside extent, so a
per-rank formula can show which axis the parallelism landed on.

Making shape first-class is what lets the validator run a **geometric
consistency check without executing any numerics** (§4): supported op rules
check rank agreement, matching non-axis dims, and one extent on every axis an
op's signature contracts (the Q/K head width, the key axis of the
probabilities and V).
Their findings (`shape_mismatches`) are publish-gate errors: a graph whose
tensors do not compose cannot ship merely because its checkpoint bytes add up.
They prove the tensors compose, not that the graph computes what `forward()`
computes.

### 2.4 One walk, two halves

There is no symbolic engine beside a numeric one: the evaluator walks a
program once, with the workload's counts (batch, query and cache lengths,
media counts) as symbols and the model's dims as numbers. Every extent it
emits is a `Dim` with two halves. The identity half is the quantity in the
model's names (`d_model`, `n_heads`, `s_kv`), and it is the formula a tooltip
prints. The law half is the same quantity as a law in the counts, and a point
(`Analysis.at`) reads the number beside the formula from it. The formula and
the number are two readings of one walk, so they cannot drift from each other
or from the rest of the dashboard, and KV bytes fold the same walk.

### 2.5 Bytes are derived, not carried

HBM traffic, KV footprints, weight bytes, and the checkpoint join are
**derived** (weight bytes in one place, `Analysis.tensors()`, which joins
storage, formats and instance counts; `Analysis.iter_checkpoint()` adds the
scale tensors a format implies) — pure functions over the finished analysis — rather than values
carried by the walk. The rule (see the
[deep dive](evaluation_model.md)): the walk computes only what needs the
evaluation frame; everything else derives, because every materialized copy of
a derivable fact is a fork that nothing forces to agree with its source. One
consequence: a kernel fusion's saving needs no arithmetic — the fused node's
traffic is computed from its own boundary, where the eliminated intermediate
simply does not appear.

## 3. Information sources

To make spec generation cheap and accurate, HuggingArch pulls facts from
multiple sources on the HF model — **without ever downloading weights**:

- **`config.json`** — fetched by `fetch_config()` in `backend/hf/metadata.py`,
  trying the custom-model directory → bundled snapshot
  (`backend/arch/snapshots/`) → live HF Hub, the same order `fetch_metadata()`
  uses. This is the source of
  scalar dims (`d_model`, `n_layers`, `n_heads`, `d_ffn`, `n_vocab`, ...).
- **Model skeleton (`print(model)`)** — `_fetch_model_str()` builds an
  empty model on the meta device via `init_empty_weights()`, then calls
  `print(model)` to capture the nested module hierarchy. No weight
  download, no forward execution. Result is cached at
  `~/.cache/huggingarch/model_info/`.
- **Forward source code** — `backend/hf/forward_source.py`
  (`describe_forward_source()`) tries, in order, the snapshot, custom-model
  files, the HF repo's modeling files, local transformers via
  `inspect.getsource()`, and the model-info cache. The available text is fed to the spec-generation agent for
  connectivity review and to a best-effort, non-gating AST diagnostic.
- **Safetensors index** — `backend/hf/metadata.py` (`fetch_metadata()`)
  issues 2–3 HTTP range requests to fetch `model.safetensors.index.json`
  plus per-shard metadata blocks, returning per-tensor name / shape / dtype
  and the aggregate on-disk sizes (`model_storage_bytes`,
  `quantization_metadata_bytes`, `checkpoint_storage_bytes`). No weight bytes
  are downloaded. This is the ground truth for the storage-bytes gate and the
  per-tensor packing / dtype anchor (§4).

For models that are not in the transformers main branch and only published
via `trust_remote_code` in their model card (Kimi, MiMo, GLM-MoE, etc.),
HuggingArch fetches `modeling_*.py` / `configuration_*.py` from the model
repo and feeds the forward source to the agent as text — never importing
`flash_attn` and friends, so the analyzer doesn't need a CUDA environment to
read those models.

### Example: model skeleton

```
DeepseekV2ForCausalLM(
  (model): DeepseekV2Model(
    (embed_tokens): Embedding(102400, 2048)
    (layers): ModuleList(
      (0): DeepseekV2DecoderLayer(
        (self_attn): DeepseekV2Attention(...)
        (mlp): DeepseekV2MLP(...)              # dense
      )
      (1-26): 26 x DeepseekV2DecoderLayer(
        (self_attn): DeepseekV2Attention(...)
        (mlp): DeepseekV2Moe(...)              # MoE
      )
    )
  )
)
```

`print(model)` shows what modules exist; the **forward** sources show how
they connect (block pattern, residual placement, MoE routing logic, the
selection logic for heterogeneous layers).

---

## 4. Validation: gates and diagnostics

Letting an agent write YAML isn't enough — LLMs routinely produce
"plausible-looking but wrong-by-counts" specs. Validation lives in the
package `backend/arch/validate/`, composed by `_orchestrator.py`; the
publication entry point adds the analysis replay (CLI:
`python -m backend.inference.validate <model_id>`). It returns a result dict
whose `errors[]` list is the publish gate: `valid = len(errors) == 0`.
Checkpoint-backed errors use external data. The axiom subsystem analyses the
whole model once and, at every component call, judges the component's claims
(from `claims/components/`) against the cache that call grows and checks that
an SWA-family call keeps a capped cached tensor; when the model calls an
attention-kind component that persists a cache (`roles.kind_of(role) ==
"attention"`), it also checks active `config.sliding_window` against the
presence of a bounded KV stream.
Everything not on the gate list is a
**non-gating diagnostic** — fed back into the agent's refinement loop, but
not itself a reason to block a spec from shipping.

> This section summarizes the validation layer. For the full treatment — the
> deterministic-anchor principle, the byte anchor's blind spot and how axioms
> cover it, and the guard-ablation evidence and its current-tree caveat — see the
> [Validation deep dive](validation.md).

Validation returns early with `No spec matched for ...` — a resolution
prerequisite, not a guard result; a config or checkpoint that cannot be fetched
raises (infrastructure, not an error for the spec writer). Once a config and spec resolve, the gate (`_gating_errors` in
`_orchestrator.py`) collects errors under eight levers — the same levers the
guard-ablation study switches off one at a time:

| Lever | Ground truth | Gates on | Modules |
|---|---|---|---|
| `weight` | safetensors header: names, shapes, dtypes, bytes | an HF tensor no spec leaf claims, or a spec-stored name the header lacks; per-tensor element mismatch; weights + declared buffers ≠ `model_storage_bytes`; the spec's checkpoint total ≠ the shard headers' own byte spans; the per-rank weight ledger and tied copies | `_matching`, `_checkpoint_names`, `_weight_ledger`, `_tied_copy`, `eval/checkpoint` |
| `packing` | header per-tensor shape + dtype; `quant_context` | stored element count and dtype that each contradict the scheme (a compensating packing error the byte sum cannot see); a quant override that selects no tensor; a role holding two stored precisions in one stack; prefill and decode forms disagreeing on a tensor's format; an unknown `quant_method` not described by `quant_context` | `_packing_consistency` |
| `shape` | the spec's dims and op rules | tensors that do not compose (§2.3), and per-op DimMap inconsistencies | `_shapes`, `axis_propagation` |
| `analysis` | the walk itself | an op the walk reaches with no layout rule, or a per-rank join that does not build | `_analysable` |
| `schedule` | `config.json` per-layer lists and named param values | a per-layer schedule that leaves a listed value unhandled; a config-named param not bound to its op | `_schedule_exhaustiveness`, `_activation` |
| `structural` | config rope sections; declared masks; the stored weights | a text rotation that contradicts `mrope_section` / `xdrope_section`; a mask declaration that is inconsistent; a parameter that is stored but never read | `_rotary_positions`, `_attention_masks`, `_unread_parameters` |
| `axiom` (+ `swa_window`) | component claims; `config.sliding_window`; cache declarations | a KV-growth claim the analysis contradicts (also in each `decode_variant`); DSA ops or `kda_layers` with no `attn_variant`; an SWA-family cache that grows without a cap; a windowed config modeled entirely as dense attention; inconsistent KV / attention type contracts; a selection read whose cache declares no bounded read law | `_axioms`, `_type_contracts`, `_selection_reads` |
| `source` | forward source | a cross-layer stream whose `src` anchor is absent from the source, or that has no producer or no reader | `_cross_layer_channels` |

The publication entry point (`backend/inference/validate.py`) adds an analysis
replay to these. Everything else validation computes is a non-gating
diagnostic fed back to the agent: the forward-source AST diff (call order,
residual count), module execution at small configs, the FFN activation versus
`hidden_act`, parameter counts versus the model card, reuse between drafts,
and whether forward source was available at all.

The rest of this section covers the schema-load checks that run before the
gate and the checkpoint join.

### 4.1 Schema-load hard checks

In `backend/arch/document.py`, `backend/arch/loader.py`, and
`backend/arch/schema/`. The
document is parsed first, and rejects what only an author can get wrong:

- **A key the document schema does not name is an error naming the key.**
  `extra="forbid"` on every model, so `dim:` for `dims:` fails at the file
  rather than loading a spec with no dimension bindings, and a selector
  spelled `strides:` fails rather than matching every layer.
- **Every `per_layer.assign` type must have a `map` entry that binds it.**
  A generator with nothing to build from selects layers and produces
  nothing.

Lowering then runs, and the `ArchSpec` root validator `raise ValueError` for:

- **`template:` references must resolve to an existing block template.**
  "Existing" means any of: components from the library / adapters tiers;
  drafts activated by this spec's top-level `drafts:` field; or inline blocks
  in this spec's `blocks:` section. Only errors when none resolves.
- **`inputs:` references must name an earlier entry in the same `model:`
  list.** The validator walks the model layer list top-down, accumulates
  seen `name`s into a set, and on every `inputs:` entry checks set
  membership. This rejects forward-jumping references, self-references, and
  any A↔B mutual reference — so model-level dataflow is a forward DAG.
- **Further rejects:** duplicate `inputs:` entries; a `tied_weight.to` that does
  not name an earlier layer; conflicting quant-override selectors; quant
  sibling tensor names (`*_scale`, `qweight`, …) anywhere in component or
  inline-block YAML; and role names `roles.yaml` does not classify.

An important property: **the validator does not use a topology whitelist.** When
the agent encounters a structure the library can't express, it defines the
new component (as a draft it activates, or inline in the spec) and the
reference can resolve in the loader's local namespace. This layer hard-rejects
bad references and unsupported schema constructs, but novelty by itself is not
an error.

### 4.2 Storage bytes and the per-tensor join

The unit of comparison is **storage bytes**, not element count. The spec side
is produced by `Analysis.iter_checkpoint()`: the walk's storage/formats join,
plus format-implied scale siblings, each carrying on-disk dtype, packing, and
byte count. `_orchestrator.py` sums those
bytes and compares against the safetensors header aggregate; a non-zero
`model_storage_bytes.diff` appends an error. Element count (`Σ numel`) is retained only as
a secondary diagnostic — reliable for single-precision checkpoints, but not
gating, because bytes are **the only unit that needs no conversion** in a
mixed-precision checkpoint (FP4 routed experts + FP8 attention + BF16 norms
in one file).

The per-tensor join comes first and gates on its own: an HF tensor pattern no
spec leaf claims (`unmatched_hf_tensors`) or a pattern whose physical element
count disagrees with its leaf (`param_mismatches`) is an error. The byte total
is a checksum on that join — it catches a wrong width but not a swapped
submodule with the same total. The reverse direction — a name the spec stores
that the header lacks — gates too (`_checkpoint_names`).

The gate rests on the fact that **every safetensors tensor name carries
structured information** — both topology and quantization packing — which the
matcher (`backend/arch/validate/_matching.py`) parses.

#### What tensor names tell us about topology

HuggingFace models follow a fairly stable naming convention:

- `model.layers.X.self_attn.q_proj.weight` — attention Q projection in some
  layer; `.layers.X.` is the block-internal boundary, prefix is model-level
  path, suffix is the block-internal module path.
- `model.layers.X.mlp.experts.X.gate_proj.weight` — a single expert in an
  MoE expert bank; `.experts.X.` denotes one of N homogeneous experts.
- `model.vision_tower.encoder.blocks.X.attn.q_proj.weight` — vision tower
  uses `.blocks.X.` instead of `.layers.X.`.
- `model.language_model.layers.X.self_attn.q_proj.weight` — multimodal
  nested stack (Gemma 4, Kimi K2.5), with an extra `language_model.`
  prefix.

The spec declares each of those parts — the model's `checkpoint.root`
(`model`, `language_model`, or none), each stack's address (its name under the
root, or `checkpoint.prefix`), and where an index sits (a stack's own,
`weight_copies.checkpoint: unrolled` for an expert bank, `checkpoint.index:
local` on a repeated node) — and `spec_tree.compose_module` composes every
stored tensor's full name from them, so the spec names the four tensors above
exactly as the checkpoint does.

#### What tensor names tell us about quantization

Each quant scheme is a YAML under `backend/arch/quant_schemes/` (awq,
compressed-tensors, fp4, fp8, fp8_ue8m0, gptq, modelopt, mxfp4,
mxfp4-pack-quantized, nvfp4), parsed by the
dataclasses in `backend/arch/quant_schemes/__init__.py` (§5). A scheme declares its
packed-weight storage suffix (GPTQ/AWQ `qweight`, MXFP4 `_blocks`, FP8 the
bare `.weight` at fp8 dtype) and its metadata siblings (`scales` / `qzeros` /
`scale_inv` / …), which are accounted separately and do not count as logical
  weight.

In other words: the quantization layout is reflected in tensor names and
dtypes. Seeing `qweight` indicates GPTQ/AWQ-style packing (the bit width comes
from the resolved scheme/config);
seeing a `_scales` sibling at UE8M0 means MXFP4 per-block exponents.
HuggingArch doesn't guess: when a spec is generated, the agent translates
`config.quantization_config` once into the spec's structured `quant_context`
(per role, overrides, default dtype); from then on the scheme comes from
`quant_context`, and the rest is name- and dtype-driven self-identification.

#### Matching procedure

1. **Name both sides in full.** Every stored tensor
   (`Analysis.iter_checkpoint()`) carries the name composed from the spec's
   declared addresses; the header carries the snapshot's normalized names.
2. **Join by exact equality** (`CheckpointJoin`). Nothing is stripped or
   folded, so a vision tower's `q_proj` cannot stand in for the text
   backbone's, and an MTP layer cannot hide behind its backbone namesake.
3. **Aggregate, don't pair 1:1.** In a heterogeneous stack, the same
   module path can be claimed by multiple spec nodes (e.g.
   `input_layernorm` exists in both dense and MoE templates); the matcher sums
   every matching spec node and compares the total against the HF group.
4. **Unmatched tensors.** HF-side tensors with no spec claimant land on an
   unmatched list — usually a sign that the spec missed a module (forgot
   the router gate, expert bias, etc.).

#### The publish gate

Final criterion: `validate()` returns `valid: true`, equivalently an empty
`errors` list. Within the checkpoint gate, spec-derived storage bytes must equal
the safetensors byte total and the gated unmatched-HF list must be empty. Gate
errors and the available per-node logical-parameter diagnostics are written
back to the agent for the next refinement round. The aggregate storage error
itself does not claim a per-module byte localization.

This layer also indirectly catches shape errors: if a `linear`'s
`out_features` is misdeclared, its param count immediately diverges, the
diff localizes it to a specific module, and the agent's feedback reads
"`self_attn.o_proj` claimed X params, actual is 1.5X" — more direct than
any graph-level shape solver could give.

Graph-level `shape_mismatches` gate as well, under the `shape` lever — see §2.3.

---

## 5. Quantization

Quantization is a sweet spot for the spec system because HuggingArch
**decouples parameter structure from weight storage format** by design.

> This section summarizes quantization. For the full model — per-role quant
> context, schemes as data, per-tensor materialization, and why mixed precision
> makes bytes the validation unit — see the [Quantization deep dive](quantization.md).

**Architecture (params / FLOPs) is dtype-agnostic.** The `linear` op in a
spec cares how many parameters exist and how many FLOPs it produces, not
whether they live as BF16, FP8, or INT4. The model structure doesn't
change with the quantization scheme.

**Storage and bandwidth (memory traffic) is dtype-aware.**
`eval.formats` applies tensor overrides, per-role schemes, and defaults
during the walk. Packed data bytes and metadata siblings come off
`Analysis.iter_checkpoint()`; `backend/inference/weight_breakdown.py`
aggregates that join rather than maintaining an independent overhead formula.

**Quant schemes** live as YAMLs under `backend/arch/quant_schemes/`
(awq, compressed-tensors, fp4, fp8, fp8_ue8m0, gptq, modelopt, mxfp4,
mxfp4-pack-quantized, nvfp4). Each declares
storage suffixes and dtype, packing factors, metadata siblings (scale / zero /
etc.), and bit-width resolution — no Python. The spec's `quant_context` names
the scheme per role; the serving path never re-reads `config.quantization_config`.

This decoupling lets one spec model different deployment precisions. The one
entry point is the **precision profile** (`backend/arch/precision_profile.py`,
request field `precision`): rules `selector → format` in four tables —
weights, activations (the edges between kernels), matmul, and cache —
selected by declared vocabulary (roles, stacks, cache slot kinds), never by
node or tensor names. The evaluator resolves it into `ResolvedPrecision`
(`backend/arch/eval/precision.py`): each object takes the most specific rule
that selects it, or what the model ships with, and
`Analysis.precision_effective()` reports shipped and effective side by side
(`/precision/effective`). So the same spec can simulate FP8 deployment from a
BF16 checkpoint, or answer "with NVFP4 weights and an FP8 KV cache, how big a
model fits on one card?" — a different binding of the same walk, with the
spec untouched.

---

## 6. Agentic spec-generation pipeline

The original goal: "input an HF model id, output a verified spec." This is
an agentic pipeline implemented by the `backend/spec_worker/` package.

### 6.1 Vast majority: compose existing components

Most newly released models are familiar topologies — hybrid GQA + MLA,
SwiGLU FFN + 256-expert MoE, sliding window, a different RoPE. The
component library already has a wide selection (and grows with each new
model). New models only need to fill the attention / ffn / norm slots in
the model spec with existing component names. The adapter layer
(`components/adapters/`) is a thinner shim: it handles tensor-name
differences between same-topology checkpoints (e.g. `wq` vs `q_proj`)
without polluting the library baseline.

### 6.2 Prompt construction

`backend/spec_worker/_prompts/system.py::build_system_prompt()` assembles the
system prompt. Its helpers in `_prompts/components.py` expand it with:

- **Op registry table** — every primitive op's name + formula signature.
- **Component manifest** — each library + adapter component's name, role,
  and one-line description. Note: the full YAML graphs are NOT inlined;
  the agent reads any component file on demand to avoid context blowup.
- **Schema docs + curated examples** — a few verified specs for the agent
  to learn from.
- **Ground truth** — this model's forward source, `config.json`, and
  safetensors tensor list.

### 6.3 Prompt architecture diagram

```
┌──────────────────────────────────────────────────┐
│  System Context (auto-expanding)                 │
│  ├── Primitive Ops table   ← OP_REGISTRY         │
│  ├── Component Library     ← components/*.yaml   │
│  ├── Schema docs           ← static              │
│  └── Reference examples    ← curated models      │
├──────────────────────────────────────────────────┤
│  Ground Truth 1: Model Architecture              │
│  └── print(model) from transformers              │
│      via init_empty_weights (no weights needed)  │
├──────────────────────────────────────────────────┤
│  Source Input 2: Forward Logic                   │
│  └── describe_forward_source() strategies        │
│      available DecoderLayer / Attention / MLP    │
├──────────────────────────────────────────────────┤
│  Ground Truth 3: Weight Tensors                  │
│  └── safetensors index.json + shard headers      │
│      per-pattern shape, dtype, count             │
├──────────────────────────────────────────────────┤
│  Task + Instructions + Rules                     │
└──────────────────────────────────────────────────┘
```

The coding agent writes YAML and invokes
`python -m backend.inference.validate`; it may read the YAML component library
listed by the manifest, but the prompt forbids reading implementation Python.

### 6.4 Agent backends

Abstracted in `backend/agents/base.py`. Currently implemented: `claude.py`,
`codex.py`, `kimi.py` (CLI subprocess wrappers). `spec_worker.runtime` acquires
a per-model lock and the worker spawns the agent as a subprocess. The agent
writes YAML into a per-run overlay (`spec_worker/overlay.py`), never the
shared tree, calls the validator, and uses errors plus diagnostics to refine
the spec. When the agent stops, the worker runs `validate()` once more in a
fresh subprocess; a non-empty `errors` list sends the agent back for a repair
round (bounded by `spec_worker/limits.py`), and only an empty one is
published. A run that fails on one backend can fail over to the next
(`spec_worker/failover.py`), and a stopped run resumes from its draft
(`spec_worker/resume.py`).

### 6.5 Pipeline flow

```
Trigger: spec-generation or refinement request (starts a background job)
         │
         ▼
   ┌─ build prompt ────────────────────────────┐
   │  1. build_system_prompt()                  │
   │     - primitive ops table          ← auto  │
   │     - component manifest           ← auto  │
   │     - formal bases (`base:` stems) ← auto  │
   │     - schema doc                   ← static│
   │     - example specs                ← curated│
   │  2. get_ground_truth(model_id)             │
   │     - model skeleton + forward source      │
   │     - safetensors tensor patterns          │
   └───────────────────────────────────────────┘
         │
         ▼
   subprocess: the configured coding-agent CLI
         │
         ▼
   coding agent, in its per-run overlay:
     1. Write YAML under the overlay's models/
     2. python -m backend.inference.validate <model_id>
     3. Read errors and diagnostics → fix YAML → goto 2
         │
         ▼
   worker re-validates (fresh subprocess)
     errors → repair round · no errors → publish
         │
         ▼
   frontend polls the job status (/api/spec/status)
```

### 6.6 Auto-expansion guarantees

| Prompt section          | Source                       | Auto-expands when…           |
|-------------------------|------------------------------|------------------------------|
| Primitive ops table     | `OP_REGISTRY` at runtime     | New `register_op()` call     |
| Component library       | Scans `components/library/`, `adapters/` and `drafts/` | New YAML file added |
| Reference examples      | Hardcoded filenames          | Manual update (curated)      |
| Model architecture      | `_metadata.get_model_info()` subprocess | Automatic per model          |
| Forward methods         | `describe_forward_source()`  | Best-effort per model        |
| Tensor patterns         | `spec_worker/_metadata.py::get_tensor_section` | Automatic per model |
| Validation result       | `inference.validate.validate()` | Automatic per model  |

### 6.7 The tail case: genuinely new topology

When a model introduces a truly novel topology (DSA's sparse attention
indexer, KDA's kernel decomposition, etc.) that the existing library can't
express, the agent is steered toward adding a new component YAML under
`components/drafts/`, listing its stem in top-level `drafts:`, and referencing
it locally. Inline `blocks:` cover model-specific block graphs. New components
go through the same loader and validator checks — so
**extending the system uses the same machinery as using it**. This is what
keeps HuggingArch from becoming a "supports N whitelisted architectures"
tool as the model landscape keeps moving.

### 6.8 After publish: review, evidence check, Agent Fix, PR

Validation answers "does the spec agree with the checkpoint"; it cannot
answer "does the spec compute what the source computes". Three agentic steps
follow a published spec:

- **Semantic review** (`spec_worker/reviewer.py`) — a reviewing agent reads the
  spec, the staged forward source, `config.json` and the tensor list, and
  writes a step-by-step table of forward ↔ spec node ↔ verdict, a defect
  list, and PASS / NO_PASS. A failing validate forces NO_PASS.
- **Evidence check** (`spec_worker/review_evidence.py`) — each claim the review
  makes is checked against the run's snapshot and marked verified,
  unverified or contradicted, so a hallucinated finding does not become a
  work order.
- **Agent Fix** (`spec_worker/fixer.py`) — a fixing agent works only from
  verified defects, the validate errors re-run now, and repairable load
  rejections; its result is re-loaded and re-validated before it is stored.

After a PASS, the author can open a pull request: an admission dry run, then
a spec-bundle PR opened by a GitHub App (`backend/service/spec_pr.py`). An
agent that needs an op the registry lacks is chained to an op writer and
reviewer (`spec_worker/_op_chain.py`, `backend/op_workers/`).

### 6.9 Key files

| File                                       | Role                                                        |
|--------------------------------------------|-------------------------------------------------------------|
| `backend/spec_worker/worker.py`            | Job execution, re-validation, repair rounds, publish        |
| `backend/spec_worker/_prompts/`            | System / schema / task prompts                              |
| `backend/spec_worker/_ground_truth.py`, `_metadata.py` | Skeleton, forward source, config, tensor section |
| `backend/spec_worker/overlay.py`           | The per-run directory the agent writes into                 |
| `backend/spec_worker/reviewer.py`, `review_evidence.py`, `fixer.py` | Review, evidence check, Agent Fix  |
| `backend/spec_worker/failover.py`, `resume.py` | Backend failover and resuming a stopped run             |
| `backend/inference/validate.py`            | The publish entry point (validate + analysis replay)        |
| `backend/arch/validate/`                   | Publish gates and structured diagnostics                    |
| `backend/arch/ops_registry.py`             | Primitive op registry (auto-feeds the prompt)               |
| `backend/arch/components/{library,adapters,drafts}/*.yaml` | Component manifest                          |
| `backend/arch/models/*.yaml`               | Model specs (what the coding agent writes)                  |

### 6.10 Agent selection / concurrency

Per deployment:

- Env: `SPEC_AGENT_CLI=codex|claude|kimi`
- Flag: `python -m backend.api --spec-agent-cli codex`
- Env: `SPEC_AGENT_MAX_CONCURRENT=10`
- Flag: `python -m backend.api --spec-agent-max-concurrent 10`

In multi-user mode a signed-in user's runs use that user's own agent
credentials and per-task routing (account page), not the host's.

---

## 7. User-facing surfaces

The web app (`web/`, a React + TypeScript single-page app built with Vite and
served by FastAPI) has six pages. Every number on them is read from the same
evaluation (§2); no page computes its own.

| Page | Route | Answers |
|---|---|---|
| Inspector | `/` (also `/inspector.html`) | What one model is: dashboard, architecture diagram, per-block composition; compare mode for 2–5 models |
| Gallery | `/gallery.html` | How the whole library compares: release timeline, size, KV cache, and each model's decode Pareto frontier (tok/s per user against tok/s per GPU, over every layout of one node) |
| Inference | `/inference.html` | Will it fit and how fast, under a parallel deployment; sweeps |
| Playground | `/playground.html` | Module comparison, and a model editor that swaps components in a loaded model |
| Fusion | `/fusion.html` | Which vLLM / sglang fusions fire for a model |
| Kernel Bench | `/kernel-corpus.html` | Measured kernel timings, by op, shape, GPU, library and version |

### 7.1 Inspector

**Dashboard.** Model type, layer count, hidden size, total and active
parameters, checkpoint size and precision mix, published context length;
attention dims grouped by layer type, each beside its spec symbol; FFN / MoE
configuration; and the weight bytes beside the checkpoint's own bytes — the
result of the storage gate (§4).

**Architecture rendering.** `backend/rendering/builders.py` and its helper
modules expand the spec into a hierarchical node tree, every node carrying
shape, params, FLOPs. The Inspector page (`web/src/pages/Inspector.tsx`) lays
it out with ELK and renders it (`web/src/pages/BlockDag.tsx`): blocks open into
their attention / FFN / norm / residual graph, and an attention opens into a
structure view (its paths — Q, shared K/V latent, K, V, attention core,
output) or a per-op view, with each cached tensor's per-token write and read.
It is a live, expandable graph of the same evaluation, not a static image.

**Hover dual-render.** Behind every numeric value sits a sympy expression.
One walk writes both: the formula is its identity half, the number beside
it the law half read at the request's point (§2.4). Hovering shows both, e.g. a KV
cache size beside `2 * n_kv_heads * d_head * dtype_bytes * s_ctx` with the
model's dims substituted.

**Compare mode.** Two to five models side by side (`/compare` endpoint), diffing
attention type, FFN configuration, parameter distribution, per-token KV
cache and per-layer FLOPs. The old `/compare.html` link redirects here.

### 7.2 Inference

Where Inspector answers "what does this model look like", the inference page
answers "can / how do I deploy it on this card". This is the day-to-day
question for MaaS providers and chip vendors.

> This section summarizes the analyzer. For the full cost model — the
> prefill/decode roofline, why SOL is a per-op sum, fusion, and how TTFT /
> TPOT / throughput / max-batch are derived — see the
> [Inference cost model deep dive](inference_cost_model.md).

**Inputs.** A `ParallelConfig` (`backend/arch/layout/parallel_config.py`:
`tp_attention`, `tp_dense_ffn`, `tp_moe_expert`, `ep_moe`, `cp`, `dcp`, `pp`,
`dp_attention`, `dp_replicas`, …; see [parallelism](parallelism.md)); a GPU from
`backend/inference/gpu_specs.csv` (HBM capacity and bandwidth, peak BF16 / FP8
TFLOPs); batch, input and output tokens; a precision profile; and the serving
framework, which decides the fusions.

**Memory.** Per-rank weights come from the resident-weight join on the walk's
artifacts (a pipeline stage filters those artifacts, then joins); KV and
activations come from layout on the same analysis — extents stay global.
Each cached tensor carries its own storage law (`arch/eval/cache.py`): GQA K·V
grow per token, an MLA latent stores `(d_kv_lora + d_qk_rope)` per position
with no per-head fanout, a window caps at `s_window`, a recurrent state is
constant; hybrid models sum per layer type. See the
[KV deep dive](kv_cache_storage.md). A request past the published context
length (`s_max`) is flagged.

**Time.** Each leaf op is bounded by four terms — compute, memory, comm (a
collective's bytes plus a link latency per hop) and the kernel launch floor —
and a step sums its leaves with no overlap between kernels (the law as the
popovers write it: `phase_formulas.STEP_SOL_LAW`; derivation and assumptions
in the [cost model](inference_cost_model.md#the-roofline-four-bounds-per-op)).
The page shows the step split by the term that bounds each op, and the per-op
table beside it.

**Measured overlay.** `backend/inference/calibration_overlay.py` joins eligible
per-op shapes with the measured corpus. Measured time is kept beside the SOL
track rather than overwriting it; a signed-in user's private rows overlay the
shared corpus. See [measured calibration](measured_calibration.md).

**Sweeps.** The Scaling panel sweeps one estimate across GPUs and parallel
degree, and across batch × context (`/inference/estimate-parallel-sweep`,
`/inference/estimate-sweep`).

### 7.3 Playground

**Module comparison.** Pick several components that fill the same role (say
MLA, DSA, CSA, HCA and KDA), each bound as a shipped model binds it, and
compare them under one fixed contract — the same hidden size, batch, context
and GPU: per-layer params, weight storage, KV per request across context
lengths, prefill / decode cost, and a compute-graph diff whose nodes are
paired by declaration. It prices a drop-in replacement, not an architecture
ranking (`backend/api/routes/playground.py`).

**Model editor.** Load a model, swap the component in any slot of any layer
group, and adjust dims; the editor names any dim the new component needs that
nothing binds, and reports old → new for every group and the whole model.

### 7.4 Fusion and Kernel Bench

The Fusion page lists the fusion plans and framework bindings and shows which
fire for a model (`/fusion/registry`, `/fusion/fired`). The Kernel Bench page
browses the measured corpus — every implementation of an op side by side,
and the same kernel across library versions (`backend/service/routes/bench.py`,
`bench_browse.py`; see [measured calibration](measured_calibration.md),
§5.4).

---

## 8. Spec authoring contract: naming convention & dtype matching

> This is the contract every spec author must follow. Violating it produces
> unmatched coverage or matching diagnostics, breaks cross-family component
> reuse, and prevents the refine loop from converging automatically.

### 8.1 Core principle: node names mirror checkpoint tensor paths

For every **leaf node with params** in a spec, its structural module path must
align with the checkpoint module path. The walk's format layer applies
the selected quant scheme's physical suffix (`.weight`, `.weight_packed`,
`qweight`, ...), and the full name is joined to the checkpoint header by
equality.

Good:

```yaml
input_layernorm:          { op: rmsnorm, ... }   # tensor: model.layers.X.input_layernorm.weight
post_attention_layernorm: { op: rmsnorm, ... }   # tensor: model.layers.X.post_attention_layernorm.weight
self_attn:                { op: mla_attention }  # internal q_a_proj/q_b_proj/...
mlp:                      { op: moe_ffn }
shared_head.head:         { op: linear, ... }    # tensor: model.layers.X.shared_head.head.weight
```

Bad — never do this:

```yaml
pre_attn_norm:    { op: rmsnorm }    # ✗ no model.layers.X.pre_attn_norm.weight exists
pre_ffn_norm:    { op: rmsnorm }     # ✗ should be post_attention_layernorm
router:          { op: linear }      # ✗ in transformers, MoE router is mlp.gate
embed:           { op: embedding }   # ✗ should be embed_tokens
```

### 8.2 Why this constraint pays off

Mirroring checkpoint names creates stable claimant groups between spec leaves
and checkpoint modules; heterogeneous leaves may aggregate to one canonical
module key. From that you get:

1. **Automatic verifiability.** Empty gated `unmatched_hf_tensors` means every
   checkpoint tensor has a spec claimant; the overall stop condition is the
   `valid` field returned by `validate()`.
2. **Per-node dtype precision.** Pull each tensor's actual dtype from the
   safetensors header (BF16 / FP8_E4M3 / INT4-packed); aggregate by leaf
   to compute mixed-precision weight bytes exactly.
3. **No translation tables.** If specs were allowed to use semantic
   aliases (`pre_attn_norm`), every spec would need a `name_map:
   spec_node → tensor_pattern` mapping — error-prone. Using tensor names
   directly removes the need entirely.
4. **Higher LLM spec-gen accuracy.** The refine prompt can directly say
   "`mlp.router` didn't match; closest tensor is `mlp.gate.weight`" —
   trivial to fix at a glance.

### 8.3 Why the matrix-of-abstractions isn't broken

Even though node names are tensor-shaped, the spec still does four things
tensor names can't:

| Aspect           | Spec provides                                          | Safetensors provides              |
|------------------|--------------------------------------------------------|-----------------------------------|
| Structure (DAG)  | ✓ parallel branches / residuals / concat / MoE routing | ✗ only a flat list                |
| Formulas         | ✓ FLOPs / params dim expressions                       | ✗ shape only, no FLOPs            |
| Component reuse  | ✓ `moe_ffn` shared across 10+ specs                    | ✗ no family-level abstraction     |
| Dim parameterization | ✓ one spec covers V3 / V3.1 / R1 at different sizes | ✗ each checkpoint stands alone   |

### 8.4 Exact matching

`backend/arch/validate/_matching.py::CheckpointJoin` joins the spec's stored
tensor names to the checkpoint header's on equality. The spec's names are
composed from declarations only: the model's `checkpoint: { root }`, each
stack's name or `checkpoint: { prefix, index }`, an op's
`checkpoint: { path }`, `checkpoint: { index: local }` on a repeated node, and
`tensor_suffix` where a tensor's stored suffix is not the op's. Nothing folds
singular/plural expert names, drops wrapper segments or guesses which tower a
name belongs to; the authored path must use the checkpoint's spelling.
Rendering dtype attachment uses the materialized tensor index in
`backend/rendering/_quant_match.py`, not a second matcher.

### 8.5 Author checklist (run through before writing a new spec)

1. List the checkpoint header's tensor names (the spec-gen prompt prints
   them grouped, each numeric segment's index set in braces).
2. Does each leaf node in your spec (excluding component-typed `op`s)
   store under one of those names, with `X` exactly where a declaration
   (`checkpoint:` prefix / `index`, unrolled `weight_copies`) indexes it?
3. If you used wrapper nodes (`block:` wrapping a moe_block), run
   `build_block_graph_data` and verify `unmatched_nodes` is empty.
4. If you used `group:` for an aux module (MTP / vision), do **not**
   flatten its contents — use `template:` to reference an inline block so
   the DAG can express parallel branches / concat / fanout.

---

## 9. The compiler lens

HuggingArch's core concepts map onto compiler concepts with unusual
precision, and the vocabulary is worth having even though the project is
deliberately not a compiler. The spec system is an IR — a closed primitive
set as the instruction vocabulary, components as expandable intrinsics,
models as modules. The spec-generation agent is an untrusted frontend whose
output is published only through the verifier (the validation gates); the
evaluation core is an interpreter that runs over symbolic workload counts,
which is textbook abstract interpretation — the same semantics over an
expression domain.
Fusion is instruction selection against a target triple (framework x
hardware); sharding is an SPMD distribution pass that borrows DTensor's own
state vocabulary; the absorbed-decode graph swap is context-driven
specialization; kernel_bench is target-machine characterization feeding a
profile-guided cost model, with the physical row key as the ABI between
the two.

Two places the analogy earns its keep. First, the classic IR disciplines
apply directly, and the project's worst bugs have been classic IR bugs: a
field dropped in lowering, an implicit fallthrough consuming the wrong
edge, an instruction modeled with the wrong semantics — all caught, and
now prevented, by verifier-style machinery (load-time lints, unknown-op
raises, coverage gates). Second, "how does a compiler solve this?" is a
legitimate first question in any design discussion here, because it
usually has a tested answer.

And one place it deliberately stops: there is no execution backend, by
design. HuggingArch lowers to *accounts* — expressions, bytes, times —
anchored to external ground truth (checkpoint bytes, measured kernels)
rather than to generated code. The goals this serves are accuracy of the
account itself, model-design assistance (the account being algebra means
design questions become equation-solving rather than parameter sweeps),
and driving kernel work (a true speed-of-light bound makes the
measured-vs-SOL gap attributable). None of those need codegen, a pass
manager, or any of the machinery a compiler carries for its own sake; the
lens is kept for its vocabulary, not its obligations.
