# KV Cache Storage: Modeling and Validation

KV cache storage is the one inference quantity that the weights do not anchor.
A layer's cache footprint is decided by its attention *mechanism* — how many
past positions each query keeps, and how large each stored position is — not by
any tensor on disk. It is therefore exactly the class of fact the
bytes‑vs‑checkpoint validator is blind to, and where the largest errors hide: a
sliding‑window layer that forgets its bound caches the entire sequence instead
of the last `s_window` tokens, and its per‑request KV blows up by orders of magnitude
at long context — with nothing in the parameter count to catch it.

This document explains how HuggingArch represents KV storage per cached node,
where the number actually comes from, and how a separately authored declaration
cross-checks the engine's computation. It assumes the spec-system vocabulary from the
[technical design](design.md) (primitive op, component, model).

## The unit of accounting: a cached node

Each attention component declares a `cache:` block — a family tag and the list
of entries written to the KV cache. For a plain attention the entries are the
key and value projections; for a compound mechanism there can be several with
*different* growth behavior. DeepSeek‑V4's CSA layer, for instance, caches three:

```yaml
cache:
  family: MLA
  entries:
    - { node: kv_concat, kind: kv }
    - { node: compressor, kind: compressed_kv }
    - { node: indexer, kind: index_key }
```

a sliding‑window latent (`kv_concat`), an overlap KV compressor pool
(`compressor`), and the lightning indexer's FP8 key cache (`indexer`). Storage
is accounted **per node** — each is sized by its own law and its own dtype — and
the layer's per‑request footprint is their sum. There is no single "KV bytes per
token" for such a layer; the per‑node view is the only one that composes.

Each entry also declares its **kind** — what the slot holds. The vocabulary is
`slot_kinds:` in `backend/arch/cache_families.yaml`:

| Kind | Entry | Holds |
|---|---|---|
| `kv` | `node:` | attention keys or values |
| `latent` | `node:` | MLA's compressed KV latent |
| `rope_key` | `node:` | the rotary key MLA caches beside its latent |
| `index_key` | `node:` | a sparse-attention indexer's key |
| `compressed_kv` | `node:` | a compressor's pooled KV, one row per window |
| `compressor_state` | `state:` | a compressor's buffers for the window it has not finished pooling |
| `recurrent_state` | `state:` | the matrix a recurrence kernel rewrites every step |
| `conv_state` | `state:` | a causal conv's window of its last `kernel_size` inputs |
| `token_window` | `state:` | the last token ids a hashed n-gram lookup reads |

A family cannot stand in for the kind: one MLA layer keeps a latent, a rope key
and an indexer key, and a deployment chooses each one's precision separately.
An entry that names a sub-component with its own cache (`compressor`, `indexer`
above) gives the kind of that sub-cache's per-token tensors — the indexer's
compressor pools an `index_key` — while the sub-cache's states keep their own.
A `state:` carried by a kernel that declares `carries_recurrent_state` is a
`recurrent_state`, and only such a state is.

An entry's `dtype:` is the precision the model stores it at: a spelling the
forward source shows (DSv3.2's FP8 indexer key, Kimi-K3's F32 KDA state), or the
config field the checkpoint states it in, tried in order and ending on the dtype
the HF forward stores the slot at (or on `model`, the model's own precision).
Nemotron-3's SSM state is `dtype: "config.mamba_ssm_cache_dtype ?? F32"`: its
config sets the field to `float32`, the transformers config class defaults it
to `float32`, and the HF forward caches the F32 state its kernels return. The
shipped precision is the HF forward's, so a checkpoint whose config field
disagrees with the chain's last term is refused rather than either source
winning silently.

What a cache slot is actually stored at in a deployment is resolved in order:
the precision profile's rule for that slot kind (for slot kinds that take a
format), then the spec's `quant_context.cache_formats[kind]`, then the entry's own
declaration, then the model's dtype with no scale. A `token_window` slot declares
`storage_format: false` and must carry its own dtype. Named cache formats live
in `arch/cache_formats/` (FP8 per-tensor, per-group-128, per-token-head, and
power-of-two scales with a 128-wide Hadamard rotation; INT4 per-token-head;
NVFP4); a format's scales are billed beside the data — per-position scales grow
with the cache, static ones are paid once.

## Three storage laws

Every cache entry grows in one of three ways as the cache length `s_kv`
increases:

| Law | Cardinality | Used by |
|---|---|---|
| `per_token` | grows without a fixed bound as `s_kv` grows | dense MLA latent, GQA/MHA K·V, compressed streams |
| `windowed` | capped at a window: `Min(s_kv, s_window)` | sliding‑window attention (SWA) |
| `constant_state` | independent of `s_kv` (a fixed recurrent state) | linear attention / SSM — GLA, GatedDeltaNet, Mamba2 |

These three cover stored entries: a cache either grows with the sequence,
saturates at a window, or has fixed recurrent state. A `kv_growth` claim may
also use `storage: none` for an attention mechanism with no cache entry.

## Where the number comes from: the graph's op laws

The storage cardinality is **computed, not declared**. Ops that specialize cache
behavior carry a `kv_storage_positions_per_query_fn` returning a sympy expression
in `s_kv`: a sliding-window score op returns `Min(s_kv, s_window)`, and an overlap
compressor returns `ceil(s_kv/compression_ratio)`. A declared recurrent state is
handled as one fixed position. Its elements are the product of its declared
`shape` (axes outermost first, as the model's HF forward allocates it: the
delta-rule matrix `[heads, d_v, d_k]`, the conv window `[conv_dim, kernel_size]`),
its last axis is the one quantization scales group over, and each axis
shards with the carrier kernel's axis of the same identity. `arch/eval/cache.py` resolves each cached
tensor's law in five steps: the node's own op, a subcomponent's output op, an op
that consumes the node (through views), a downstream block-anchor scoring kernel,
and the component's attention anchor. The **storage** law and the **read** law
fall back independently — a DSA index key stores every position but is read
through a top-k — and readers that state contradictory laws are refused. The
consumer path is what lets a score op describe eviction of a tensor whose
producer emitted all positions. `inference.kv_topology` is a view of that layer
(`cache_of_component`); it does not walk the component graph again.
Per‑request bytes then follow directly:

```
per_request_bytes = Σ_node  elements_per_token_node × dtype_bytes_node × positions_node(s_kv) / positions_over_node
                  + the format's scale bytes (per-position scales grow, static ones are paid once)
```

where `positions_over` is the degree the position axis is split over (decode
context parallelism, a layer-split CP); tensor parallelism splits KV heads, and
an MLA latent with no head axis stays replicated. `arch/eval/cache_view.py`
binds these laws to one deployment — per-token growth plus a fixed floor,
bytes at a given length, the `growth_formula` the UI shows, and a per-stage view
— and capacity divides the KV budget by it. A model whose caches are all
windowed or recurrent has no token ceiling (`max_tokens` is `None`). Whether a
model is hybrid is read off the leaf terms of `growth_formula`, not off the
cache family: three dense-FFN layers and 58 MoE layers that all cache an MLA
latent are one law over `n_layers`.

The window, the compression ratio, the constant state — all of it falls out of
the ops the author wired into the graph. No component‑level scalar participates.
This is why removing a spec's declared window string changes no number: the
sizing never read it.

## The cross‑check: `kv_growth` axioms

Because the storage law follows from *which op* produces a node, an author can
wire the wrong one — a dense op on a windowed layer, a per‑token op where the
source pools and evicts — and the engine will faithfully size the wrong
footprint. Nothing in the graph is self‑evidently wrong. So each cached node's
growth law can also be declared independently as a `kv_growth` claim. Claims
live only in `backend/arch/claims/components/<identity>.yaml`, whose `subject` is
the component identity; the loader attaches the file to the standalone component
or model-inline block of that name, and an `axioms:` key inside a component
YAML, draft, or inline block is a load error:

```yaml
schema: huggingarch.claims/v1
subject: dsv4_csa_attention
axioms:
  checks:
  - kind: kv_growth
    node: kv_concat
    storage: windowed
    bound: s_window
    src: 'model.py::Attention.forward — the SWA latent KV is sliding-window: cached positions = min(s_ctx, s_window), evicted past the window.'
  - kind: kv_growth
    node: compressor.pooled_out
    storage: per_token
    src: 'model.py::Compressor — the overlap KV compressor stores ⌈s_ctx/4⌉ positions and keeps ALL of them (no eviction).'
```

A claim has `kind: kv_growth` and the fields `node`, `expect`, `storage`,
`bound`, and `src`. The `storage` field is **not read by the sizing math**. It is
a claim — optionally carrying a `src` citation for human review — that
`validate/_axioms.py` checks against the cache the engine *independently
computes*. The validator analyses the whole model once
(`Deployment(Program.of(resolved)).at(batch=1, input_tokens=1).analyze()`),
walks every component call node in the program, and reads the cache that call
grows with `cache.by_call` — the entries `cache.derive` counts from. In the
parent's cache a nested sub-component's tensors are prefixed by its name
(`indexer.compressor.pooled_out` in a CSA call); the sub-component's own call
names them from itself (`compressor.pooled_out` in the indexer call), which is
how `dsv4_indexer`'s claims name them. `expect` and `bound` resolve in that call's
dims (`analysis.node_dims`); stored lengths stay symbolic in the cache length.
A `windowed` claim requires the node's stored length to stay bounded as
`s_kv → ∞`; `constant_state` requires it to be free of `s_kv` entirely;
`per_token` requires it to grow. A mismatch is reported as
`axiom_kv_growth_violation`. Per-layer dim variants are judged at each call; a
template-bound copy (e.g. `gqa_qknorm_attention__rope-mrope`) reports under the
component it was bound from; a block a stack runs directly as its `template:` is
judged too and keeps no cache of its own; a component the model never calls is
not judged; `decode_variant` forms are judged on the decode phase's own analysis
(`Deployment.during("decode")`) and report under their own name. If
the model analysis fails, one gating `axiom_unjudged` finding carries the
analysis error.

The validator does not fetch or verify the `src` citation. The mechanical gate
is declaration versus engine computation; source fidelity remains a review
responsibility.

This is a declaration-versus-computation check: the ops compute the footprint,
the claims file states the expected class, and the validator asserts that they
agree. The node name appearing in both `cache.entries` and the axiom is not
redundancy; it identifies the computation being checked.
Collapsing the two — inlining the law onto the cache entry, say — would give the
declaration and the computation a shared source, and a wiring mistake would
become invisible: the engine would simply compute whatever the single
declaration asserted. The apparent duplication *is* the check.

## The sliding‑window contract

Two of the three laws are self‑evident from a correct graph (a dense cache grows;
a recurrent state declares its own `shape`). The one law that is easy to lose
is the window — a sliding‑window layer wired with a dense score op looks
structurally identical to a global one, and simply caches everything. This is the
128 TB family, and it is guarded by a contract in `validate/_axioms.py`, judged
on the same whole-model analysis as the claims:

> In a component call whose `cache.family` is in the windowed family, every
> cached tensor that grows with the sequence must cap, and at least one must grow.
> Otherwise `swa_cache_unbounded` gates the spec **invalid** (controlled by the
> `swa_window` ablation lever).

The window is expressed in the graph — a `sliding_window_kv` over the stream's
`cache.entries` names, with the model's window as `s_window` — so spec writers
author no claim for it. The check reads the **stored cache length** each call
grows, not the computed FLOPs expression. A windowed layer and a global one emit
structurally identical attention FLOPs (the window bounds *cached tokens*, not
the score arithmetic), so a `flops_expr.has(Min)` test would false‑positive on
every windowed component and mishandle sparse attention. Boundedness of the
stored length is unambiguous, so the contract stays free of false positives while
still catching a layer that labels itself SWA but caches the full sequence.

There is also a separate **model-level config gate** in
`validate/_axioms.py`. If `config.sliding_window` is a positive integer and
`use_sliding_window` is not false, and the model calls at least one
attention-kind component that persists a cache (`roles.kind_of(role) ==
"attention"`), the validator requires at least one such call
to keep a growing cached tensor whose stored length caps, read off the same
whole-model analysis. If none does, `windowing_declared_but_unmodelled` enters
`axiom_warnings` and the orchestrator promotes it to `errors`, making the spec
invalid. A model that truly ignores the config field declares `sliding_window`
with a reason under `validation_exemptions`.

That gate is intentionally model-level: it catches a windowed config modeled
entirely as dense attention, but the existence of one bounded stream does not
prove that every scheduled windowed layer is correct. The per-call
`swa_cache_unbounded` contract and each `kv_growth` declaration-versus-computation
check supply the more local checks. The implementation returns no config-window
warning when the model calls no attention-kind component that persists a cache.

## Worked example: DeepSeek‑V4 CSA

At `s_kv = 256 K`, one CSA layer's per‑request KV decomposes by cached node:

| Node | Law | Positions | Per position | Bytes |
|---|---|---|---|---|
| `kv_concat` | `windowed` | `Min(s_kv, 128)` = 128 | 30.7 KB | 3.9 MB |
| `compressor.pooled_out` | `per_token` | `⌈s_kv/4⌉` = 64 K | 30.7 KB | 1.97 GB |
| `indexer.compressor.pooled_out` | `per_token` (FP8) | `⌈s_kv/4⌉` = 64 K | 3.8 KB | 246 MB |

(The FP8 index key is what vLLM stores; the reference implementation keeps it
in BF16. The overlap compressor also holds two small FP32 `compressor_state`
entries per request — a constant term, independent of `s_kv`.)

The growing nodes sum to the layer's footprint; each line is traceable to the node and
storage law it comes from. A single "positions × bytes/token" summary cannot
represent this — the positions differ per node (128 vs 64 K) and so do the dtypes
(BF16 latent vs FP8 index) — which is why the per‑node model is the primitive and
the per‑layer number is derived from it, never the reverse.

## What component‑level `kv_window` was, and why it is gone

Earlier specs carried a component-level `kv_window: "min(s_ctx, s_window)"`
string. The key itself is now rejected by the loader. It was a
*third* representation of the window that neither computed the footprint (the op
does) nor supplied the checked expectation (the axiom does), surviving only as the
sliding‑window contract's proxy — and a weak one, since a string can be present
while the layer caches the full sequence. The window lives in the op that
computes it; the sliding-window contract reads the stored length that op
produces, and a `kv_growth` claim in `claims/components/` states the expected
law where a reviewed claims file exists. Nothing declares a window field.

## Where this lives in the code

| Concern | Location |
|---|---|
| Which nodes are cached | component `cache.entries` (spec YAML) |
| Storage cardinality (the number) | graph op `kv_storage_positions_per_query_fn` laws resolved by `arch/eval/cache.py`; `inference/kv_topology.py` is a view of that layer |
| Per‑tensor storage / read laws, per‑request bytes | `arch/eval/cache.py` (assembled per cached tensor; one stored size per position, data and dynamic scales, on `CachedTensor.position_bytes`) |
| Laws bound to a deployment (growth, floor, `growth_formula`) | `arch/eval/cache_view.py` (`CacheView`) |
| Cache families and slot kinds | `arch/cache_families.yaml` |
| Named cache formats | `arch/cache_formats/` |
| Growth-law declaration + human source citation | `arch/claims/components/<identity>.yaml` (`kind: kv_growth`), attached by identity in `arch/spec_parsing.py` (`_attach_claims`) |
| The cache one component call grows | `arch/eval/cache.py` (`by_call`) |
| Declaration ⟺ computation cross‑check + active-window config gate | `arch/validate/_axioms.py` |
| Sliding‑window contract | `arch/validate/_axioms.py` (`swa_cache_unbounded`) |
