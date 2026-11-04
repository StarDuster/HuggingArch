# `components/` — three-tier compound op library

YAML files here are reusable compound ops referenced by model specs in
`backend/arch/models/`. They are organised into three physical
subdirectories whose **directory location is the tier declaration** —
no schema field, no tag.

## `library/` — distinct topology building blocks

Anything whose graph encodes a **new computation structure**: a new
node sequence, a new connectivity, or a structural variant of an
existing topology that cannot be expressed by binding + naming alone.

Examples:
- `pre_norm`, `post_norm`, `mtp_block_shared_head`, `mtp_block_with_head`
- All `*_attention` (mla, gqa, swa, mha, vit, gated_delta_net, …)
- All `*_ffn` and `moe_ffn_*`
- `router_noaux_tc`
- Model-prefixed but structurally distinct: `gemma4_4norm_scaled_block`,
  `step3p5_mtp_block`, `longcat_block`, `qwen35_vision_block`,
  `kimi_k25_vision_block`, `minimax_m2_decoder_block`, …

A model-family prefix in the name (`gemma4_*`, `qwen35_*`, …) is a
**historical "first observed in" label**, not a scope marker. Any model
whose forward matches the topology may compose against it. When a
second model family adopts the same structure, `git mv` to a generic
name and update references — the rename in git history captures the
promotion.

### Admission rule
If you remove all `params:` bindings and naming clauses and what
remains is structurally novel (vs. the rest of `library/`), it
belongs here.

### A per-layer variant is a BINDING, not a copy

When a model's layers differ in one part of an otherwise identical
graph — which compressor an attention runs, whether it owns its
indexer or reads a shared one — the shared part is written once with
a `params:` slot and each layer group binds the piece it runs
(`GraphNode.params`, nested as
`params: { attn: { template: X, params: { … } } }` at the call site).
The CSA2 family is the worked example: `csa2_attention` holds the
q-LoRA, the window KV, the one sparse score and the grouped output
LoRA, and its `compressor` / `indexer` / `compressed` slots bind the
form of `Attention._compress_kv` that layer runs — its own compressor
and indexer and a compressed-KV store, or the shared latent, the
shared selection and a plain read. Four nearly-identical components,
one per layer kind, is the anti-pattern this replaces: the copies
drift, and the drift is invisible because each one is internally
consistent.

Two things to know before composing this way. A component's
`cache:`, `reads:` and `stream_out:` are the TEMPLATE's, so a fact
that differs per layer kind — one layer stores the compressed KV,
another borrows it — has to live in the bound piece, not in the
shared body. And a KV cardinality is recorded per NODE, so one slot
can hand a key stream to a score and nothing else: a second tensor
out of the same slot would have to cross as a secondary port, which
arrives carrying the main output's cardinality (two cardinalities at
one score is refused) and which the graph view cannot resolve when
the slot declares `module: null`. Give each returned tensor its own
slot instead — that is why `_compress_kv` is three (see the note in
`csa2_attention`).

### A function constant is a binding too

An op whose math takes a real number beside its operands — `clamp` its
bound, `situ` its betas — reads it from its node's frame under a
dims.yaml symbol of kind `constant` (`clamp_limit`, `situ_beta`,
`situ_linear_beta`; `OpSpec.consts`). A component never names the config
field: `swiglu_ffn_clamped`'s clamps read `clamp_limit`, which the model's
`dims:` binds where its config keeps it (`config.swiglu_limit`,
`config.text_config.swiglu_limit`). Where one part of a model reads another
config scope, the node or the call site rebinds the symbol with `dims:` —
GLM-5.3's vision clamps write `dims: { dim: d_vis_ffn, clamp_limit:
vis_clamp_limit }` and the model binds `vis_clamp_limit` to
`config.vision_config.swiglu_limit`. A config that publishes none leaves the
constant unbound: absent, never a default.

### Encapsulation parity rule (hard invariant)

> **Spec components must be at least as encapsulated as the source
> code's modules / functions / methods. Spec MAY roll up multiple
> source-level operations into one component (more abstract);
> spec MUST NOT split a single source-level function into separate
> top-level nodes (less abstract).**

The spec's component boundaries express the architecture's semantic
units. When the source author wrote `def hc_pre(...)` as a method,
they drew a semantic line that says *"this pipeline is one
conceptually atomic operation, callable from multiple sites,
testable in isolation"*. Spec inlining below that line silently
deletes the line. Downstream consumers (the agent, validate,
rendering, future analysis tools) lose any way to recognize "these
seven nodes are the same `hc_pre` invoked at four call sites" —
the architecture looks accidental instead of designed.

Direction is asymmetric on purpose:

- **Source has a function, spec has a component**: ✓ required.
- **Source has a function, spec inlines its body**: ✗ violates parity.
- **Spec has a component, source inlines that body across multiple sites**:
  ✓ allowed (spec abstracts a recurring pattern the source author
  hand-wrote repeatedly). Document the rollup in the component
  description.

Practical checklist when authoring a model spec:

1. Read the upstream `forward()` source for each block first.
2. Every Python `def` / `class` / `nn.Module` boundary in that source
   becomes a candidate `library/` component (if reusable across
   models) or `adapters/` shim (if checkpoint-naming-only divergence).
3. If the schema makes the encapsulation hard to express
   (multi-output ports, sibling-weight flattening, etc.), prefer
   *fixing the schema* over inlining. Inlining is technical debt
   that compounds: every model that copies the inlined body adds
   another multiplicative restructure cost when the schema gap eventually
   gets closed.

Known violations to clean up: see CHANGELOG / `drafts/` directory.

### `role:` — what slot can this fill, NOT what compute it does

`role` answers a single question: *"can a model spec drop this in as
its `attn:` / `ffn:` / `block:` / etc.?"*. It is **not** a compute
classifier. A KV-compressor sub-module performs attention compute, but
it cannot be plugged in as a model attention slot — so it is roleless
(`role: None`).

Allowed values (enforced by `Literal` on `BlockGraph.role`):

| role         | what fills the slot                                                                          |
|--------------|----------------------------------------------------------------------------------------------|
| `attention`  | complete attention impl: Q/K/V projections + score + output. Plug-in for spec `attn:`.       |
| `ffn_dense`  | complete dense FFN (gate/up/down + activation).                                              |
| `ffn_moe`    | complete MoE FFN (router + dispatch + experts + combine + optional shared experts).          |
| `router`     | MoE routing pipeline (gate / topk / scoring); rendered separately in the slot view.          |
| `memory`     | lookup memory a block reads beside attention and FFN: a token-id-indexed table gated into the residual (`dsv41_engram`, `qwen_ple`). |
| `hyper_connection` | Hyper-Connections residual mixing (`hc_pre`, `hc_post`, `hc_head`, `dsv41_hc_mixes`, …); a block's `hc_*_{fn,base,scale}` parameter nodes carry it too. |
| `block`      | full transformer block (`pre_norm`, `gemma4_4norm_scaled_block`, `dsv4_block`, …).           |
| `aux_block`  | auxiliary stack wrapper (MTP head, vision tower wrapper).                                    |
| (omit)       | sub-component / utility (compressor, indexer, `head_split`, `head_merge`, `head_broadcast`). |

A node inside a component may carry a finer role that refines the
component's (`roles.yaml` `refines:`) where the checkpoint stores it at a
different precision from the rest of the component: `attention_output`
(`o_proj`), `attention_kv_up`, `attention_indexer`, `attention_compressor`,
`attention_position_embedding`, `attention_short_conv`, `ffn_moe_shared`,
`ffn_moe_latent`, `router_hash_table`. Inside one role and one stack the checkpoint stores every
weight matrix at one precision (the validator's quant-role gate).

Sub-components must be roleless. Examples currently in tree:
- `dsv4_compressor_overlap`, `dsv4_compressor_no_overlap`, `dsv4_indexer`
  — used INSIDE `dsv4_*_attention`, but not pluggable on their own.
- `csa2_compressor_gated` / `csa2_compressor_plain` /
  `csa2_shared_latent` (the `compressor` slot), `csa2_indexer` /
  `csa2_indexer_candidate_source` / `csa2_indexer_shared_keys` /
  `csa2_shared_selection` (the `indexer` slot) over `csa2_index_keys` +
  `csa2_index_score`, and `csa2_compress_store` /
  `csa2_compress_borrowed` (the `compressed` slot) — the pieces
  `csa2_attention`'s three slots bind.
- `head_split`, `head_merge`, `head_broadcast` — tensor-shape utilities.

#### Two-level role
- **Component-level** (this field): explicit, declared by the YAML
  author. Validated against the `Literal` whitelist by Pydantic.
- **Leaf-level** (`spec_tree.build_ir_node`): every leaf in the spec
  IR inherits the role of its nearest non-None ancestor component.
  This is what FLOPs/params/quant routing
  (`quant_context.per_role.<role>`) keys off — never the
  component-level field directly. So a `linear` leaf inside a
  `dsv4_compressor_overlap` (roleless) inside a `dsv4_csa_attention`
  (`role: attention`) gets `role=attention` at the leaf.

Adding a new role is a schema change: register it in
`backend/arch/roles.yaml` (`kind` + `meaning` — the loader rejects names
the registry does not classify), extend the `Literal` in
`schema.py`, decide leaf-inheritance semantics, then update
`quant_schemes` consumers.

## `adapters/` — pure binding + checkpoint-naming shims

A single-node wrapper around a `library/` component that supplies
bindings and/or states a specific model's checkpoint naming via
`module:` / `inner_modules:`. Adapters contribute **no new graph structure** of
their own.

Today: `qwen35_full_moe_block` / `qwen35_linear_moe_block` (bind
`pre_norm` with this family's attention + the packed fused MoE FFN),
`qwen35_full_dense_block` / `qwen35_linear_dense_block` (the same with the
dense swiglu FFN), `qwen35_mtp_packed_inner_block` (the MoE MTP layer whose
checkpoint stores the packed expert bank, rotated by one position stream as
vLLM runs it), `hunyuan_vit_mlp`
(`clip_vision_mlp` stored as dense_h_to_4h / dense_4h_to_h),
`qwen3_next_linear_moe_block` (the same for Next's fused-QKV linear
attention), `gemma_geglu_ffn` (`gated_ffn` with the Gemma family's
tanh-approximate GELU), `chatglm_block` and siblings (checkpoint naming).

### Admission rule
The body must be `op: <library_component>` plus `params:` plus
optional `module:` / `inner_modules:`. If you find yourself wanting to add a second
graph node, the adapter is doing too much — promote the new structure
to `library/` instead.

### Reuse policy
Adapters are scoped to one model family. A second family wanting the
same binding pattern means the binding itself is a structural
variant — promote the inner topology to `library/` (with a generic
name) and have both families' adapters bind against it.

## `drafts/` — agent-generated, spec-scoped staging

Agent-authored components from `/generate-spec` runs land here first,
NOT in `library/`. Drafts are parsed at startup but DO NOT enter the
global namespace: a draft is only visible to a model spec that
explicitly opts in by listing its name under a top-level `drafts:`
field, and only for the duration of that spec's load. So a buggy
draft yaml from one V4-Pro run does not affect any other spec, does
not pollute the manifest shown to the next agent run, and stays out
of the cross-model reuse pool.

```yaml
# in backend/arch/models/some-model.yaml
drafts:
  - dsv4_hc_block          # only this spec sees these names
  - dsv4_hca_attention
```

Lifecycle:
1. Agent writes new component yamls to `components/drafts/<name>.yaml`
   and adds their names under the spec's `drafts:` list. Before it wires
   one in, it asks which existing components already hold that graph:
   `python -m backend.arch.component_reuse components/drafts/<name>.yaml`
   ranks every component of the same `role:` by the share of the draft's
   edges (by op) it holds, with the file to read. That is the same
   comparison the validator reports as `reuse_warnings` once the spec is
   built (`arch/component_reuse.py` is the one implementation), so a draft
   restating a library component is cheapest to catch here. A maintainer
   can run it on any component file, including a `library/` one.
2. Maintainer reviews — fixes naming, validates the topology actually
   maps to forward source, deletes throwaways.
3. After review, promote to `library/`:
   ```
   python -m backend.arch.promote <name>
   ```
   This `git mv`s the yaml to `library/<name>.yaml`. Once promoted,
   the spec's `drafts:` entry becomes a no-op (library wins) and may
   be removed at the next edit.

Drafts that fail to parse are silent until referenced — only specs
that list a broken draft fail to load. This means an aborted agent
run leaving a half-finished yaml in `drafts/` does not spam every
server boot with warnings.

### Admission rule
Anything an agent writes during spec-gen. There is no admission
review at write time — that gate moves to the promotion step.

## Cross-tier references

`library/` and `adapters/` feed the same flat name namespace, so YAML
refs like `op: pre_norm` resolve correctly regardless of subdirectory.
`drafts/` is per-spec scoped (see above) and does not contribute to
that shared namespace. Adapters reference library components freely;
library components should not reference adapters (would invert the
dependency). Drafts may reference any tier; library/adapters must not
reference drafts (drafts are by definition unreviewed).

### Calling a component from the model root

A model-level entry may call a library component by `op:` — a pure
function at the root, like DeepSeek-V4's `hc_head` or Gemma's
`logit_softcap`. Name what it reads with `inputs:`; model-level
dataflow has no default wiring, and the names bind to the component's
formals by position. A post-processing step on the logits names the head
(`inputs: [lm_head]`), so it reads the sampled positions the head
produces rather than every position.

## Why two tiers — checkpoint ≠ source

A spec describes two things that are *almost* the same but not quite:

- a **forward computation graph** (which ops, in what order, with what
  dependencies and FLOPs)
- a set of **stored tensor names** that the analyzer matches against
  the checkpoint's `safetensors.index.json`

These often diverge in real models because a checkpoint is the output
of *some* training framework (Megatron-Core, internal vendor stack,
DeepSpeed, …) and that framework's storage layout doesn't have to
match HF transformers' `modeling_*.py`. Common patterns:

- **Stacked-expert weights**: an MoE layer in the checkpoint may be a
  single `nn.Parameter` of shape
  `[n_expert_copies, 2*d_ffn_expert, d_model]` (grouped-GEMM friendly,
  no `.weight` suffix because it's not an `nn.Linear`), while the HF
  `modeling_*.py` reads as `nn.ModuleList([SwiGLU(...) for _ in
  range(n_expert_copies)])` and *internally* hooks a weight remap on load.
- **Bare `nn.Parameter` instead of `nn.Linear`**: DSv3's routing gate
  is `self.weight = nn.Parameter(...)`, so the state-dict key is
  `mlp.gate.weight` (no `.linear.` indirection). The forward graph
  still has a Linear-like matmul, but the tensor lives one level up.
- **Asymmetric subgraphs**: same model can mix conventions — Qwen3.5's
  main backbone uses stacked-Parameter experts (perf path), while its
  MTP head uses a per-expert ModuleList (standard path). One
  `modeling_*.py` class, two checkpoint layouts.
- **Quantization wrappers**: GPTQ / AWQ / compressed-tensors append
  suffixes (`*_qweight`, `*_scale_inv`, `*_packed`) that don't appear
  anywhere in the original `nn.Module` hierarchy.

This is what library/adapters splits along:

- **`library/`** encodes the **forward computation graph** (semantic
  structure: what depends on what, what the FLOPs are). Read HF source
  for this.
- **`adapters/`** encodes the **checkpoint naming** (which library
  component to bind, which node names to rewrite to land on real
  tensors). Read `safetensors.index.json` for this.

Spec authoring discipline:

> **Source is ground truth for compute. Checkpoint is ground truth for
> naming.** Don't infer tensor paths from `modeling_*.py` — the
> training framework can have moved them. Always cross-check against
> the actual safetensors index before committing a spec.
