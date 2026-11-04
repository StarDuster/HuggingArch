# Attention Families: Mechanisms and Their Cost Signatures

HuggingArch models roughly a dozen attention variants — from vanilla multi-head
attention to compressed-latent MLA, sparse DSA, and constant-state linear/SSM
layers. Each is an ordinary component (see the [technical design](design.md) §1),
but they differ along three cost axes that drive the entire inference model. This
document walks the families, the mechanism behind each, and the cost signature it
produces. It is the *mechanism* companion to the [KV cache storage](kv_cache_storage.md)
deep dive, which covers the storage side in depth, and it feeds the
[inference cost model](inference_cost_model.md), which turns these signatures into
TTFT / TPOT / capacity.

## The three cost axes

An attention variant is not characterized by a single "attention FLOPs" number.
Three quantities matter, and they move independently:

1. **Compute per query** — how much math one query does against the cache.
2. **KV storage law** — how the cache grows with sequence length: `per_token`
   (grows), `windowed` (caps at a window), or `constant_state` (fixed). This is
   the subject of the [KV storage deep dive](kv_cache_storage.md).
3. **KV read cardinality** — how many cached positions a query actually attends
   to, which can be far smaller than what is stored.

The last two are distinct, and the gap between them is exactly what sparse and
windowed attention exploit: a layer can *store* the whole sequence yet *read* only
a sparse subset, or store only a window. The cost model tracks both — a
per-request storage figure and a per-query read figure — never conflating them.

## Dense: MHA / GQA / MQA

The baseline. Every query attends to every cached position; K and V are stored
per head. Per-token-per-layer KV is `2 · n_kv_heads · d_head · bpe` — the
factor of two for K and V. Multi-head attention (MHA) has
`n_kv_heads = n_heads`; grouped-query attention (GQA) fans `n_heads` query
heads onto fewer `n_kv_heads`; multi-query (MQA) is the `n_kv_heads = 1`
extreme. Storage is `per_token`, read is the full `s_kv`, and prefill compute is quadratic in sequence. Everything else is a
departure from this shape that trades compute or memory for scale.

## MLA: a compressed latent, and the decode absorb

Multi-head latent attention (DeepSeek) stores the KV as a single small latent of
size `d_kv_lora` plus RoPE-only keys of size `d_qk_rope`, shared across all
heads rather than fanned out per head: `(d_kv_lora + d_qk_rope) · bpe` per
token, a large KV reduction versus GQA.

MLA has **two graph forms**, and the phase decides which:

- **Decompressed** (prefill) — `kv_b_proj` expands the latent into per-head K and
  V, then standard attention runs. More compute, used while processing the prompt.
- **Absorbed** (decode) — `kv_b_proj` is folded into the Q-side and O-side per-head
  matmuls so the KV never leaves the `d_kv_lora + d_qk_rope` space. Less compute and a
  smaller read. The component declares it (`decode_variant:
  mla_attention_absorbed`), and the decode walk swaps it in
  (`rebind_decode_variants`, `arch/eval/transforms.py`). The latent and the
  RoPE key are joined once per position — one KV head, as the kernel reads
  it — and shown to every query head as a zero-byte broadcast.

A sparse (DSA) MLA prefills absorbed too: each query reads its selected rows
of the cached latent, which cannot be expanded per head ahead of the
selection, so vLLM and SGLang (beyond 2048 keys) run prefill through the same
sparse MLA kernel as decode. `phase_form` (`arch/eval/transforms.py`) is the
one place that says which phase runs which form; a component whose graph
runs DSA ops (`OpSpec.dsa`) runs its declared form in both phases.

A parallelism note falls out of the shape: the latent has no head dimension to
shard, so tensor parallelism *replicates* it across ranks rather than splitting
it — raising `tp_attention` does not shrink MLA KV. Data-parallel replicas can
raise aggregate throughput, but do not shrink the per-request latent on a rank.

## SWA: the window bounds cached tokens, not per-token bytes

Sliding-window attention caches only the last `s_window` positions and evicts the rest.
Its per-token bytes follow the *underlying* dense (GQA) shape — the window bounds
the **number** of cached tokens, not the cost of each. So storage is `windowed`
(`Min(s_kv, s_window)`) and read is likewise `Min(s_kv, s_window)`, while the per-token formula is
unchanged. This storage-vs-per-token distinction is why capacity math evaluates
per-token bytes on the dense shape but caps the position count at the window.

## DSA: sparse attention over a compressed index

DeepSeek Sparse Attention (V3.2) and GLM's variant add a **lightning indexer**: a
compact key cache that spans the full sequence and is scored to select a
sparse top-k of `n_topk_keys` positions per query. V3.2 / GLM cache one
uncompressed index key per token; Qwen3.8's QSA caches raw keys and pools them
into blocks each step; only DeepSeek-V4's CSA indexer stores a compressed
`⌈s_kv/compression_ratio⌉` pool. The cost signature is the clearest case of
storage diverging from read:

- **Storage grows** — the index-key cache is `per_token` and spans the whole
  sequence, because DSA attends *beyond* any local window. Its precision is a
  per-model fact: the component caches at the model dtype, and V3.2 declares
  FP8 for it in its own `quant_context.cache_formats`.
- **Read is capped** — a query attends to `Min(s_kv, n_topk_keys)` positions,
  not `s_kv`.

So decode cost is regime-dependent versus dense: at long context the read cap
dominates, at short context the sparsity does not yet bite. DSA composes onto a
latent backbone (V3.2: MLA + indexer) or a dense one (GQA + indexer). GLM-5.2
adds *shared* layers that run no indexer of their own: they read the top-k a
full layer published on the `dsa_topk` stream and store no index key.

## CSA / HCA: compressor plus windowed latent

DeepSeek-V4 chooses an attention per layer by `compress_ratio`: HCA (ratio 128)
and CSA (ratio 4, which adds an overlap compressor and a sparse indexer). These
layers cache *several* heterogeneous nodes at once — a windowed SWA latent, a
`per_token` compressor pool at `⌈s_kv/compression_ratio⌉`, and (CSA) an FP8 index cache — which is
why they are the motivating example for the per-node storage model. The
[KV storage deep dive](kv_cache_storage.md) works CSA through node by node.
DeepSeek-V4.1's CSA2 keeps a window plus one compressed KV that later layers
can borrow from an earlier one (`csa2_compress_borrowed`), and its compressors
declare decode variants of their own.

## Linear / SSM: a constant recurrent state

GatedDeltaNet, Mamba2, and gated linear attention (GLA) do not keep a per-token KV
cache at all. Each holds a **fixed-size recurrent state** per request — a matrix
like `[heads, d_v, d_k]` (GLA / GatedDeltaNet, and Kimi-K3's KDA, a GatedDeltaNet
variant with per-channel decay) or `[heads, d_head, d_state]`
(Mamba2) — that is updated token by token, and GatedDeltaNet / Mamba2 also keep
the window of `kernel_size` inputs their short causal conv reads. Each of these
tensors is its own `cache:` state entry, carried by the kernel that reads and
rewrites it (the scan, the conv). Their storage is a per-request state
(`recurrent_state`, `conv_state` slots): independent of `s_kv`, so the cache does
not grow with context, and compute is linear in sequence (a scan / recurrence)
rather than quadratic. It is a first-class cache class — a `recurrent` family
in `cache_families.yaml` — and it is why hybrid backbones interleave linear
layers (cheap context carry) with periodic full-attention layers (precise recall):
the two storage laws sum naturally in the per-layer KV total.

## The two cross-cutting distinctions

Everything above reduces to two ideas the cost model keeps separate where naive
accounting would merge them:

- **Storage is not read.** A DSA layer stores the full compressed sequence but
  reads a top-K; a windowed layer stores and reads a window; a dense layer stores
  and reads everything. Each cached tensor carries a storage law and a read
  law (`arch/eval/cache.py`), evaluated on different cardinalities of the same
  cache.
- **Prefill is not decode.** The same layer can present different graphs per
  phase — MLA's and DSA's decompressed-vs-absorbed swap, DeepSeek-V4's
  compressors — declared as `decode_variant` and run in its phase rather than
  one graph scaled by a factor.

Two vocabularies describe a cache and stay separate. The **cache family**
(`backend/arch/cache_families.yaml`: MHA, GQA, MLA, SWA, ViT, GLA-Linear,
GatedDeltaNet, Mamba2, each with a `kind` — full, windowed, recurrent, vision)
is how the cache is shaped; `attn_variant` (DSA, CSA, HCA, KDA, …) names the
compute flavor when it differs — CSA, HCA and CSA2 are all MLA-family caches.
Each cache entry also has a **slot kind** (`kv`, `latent`, `rope_key`,
`index_key`, `compressed_kv`, `compressor_state`, `recurrent_state`,
`conv_state`, `token_window`), and deployment precision picks a cache format
by slot kind: the precision profile first, then the spec's declaration, then
the model dtype.

## Where this lives in the code

| Family | Component YAMLs |
|---|---|
| MHA / GQA / MQA | `components/library/gqa_*.yaml`, `mha_attention.yaml` |
| MLA (decompressed / absorbed) | `mla_attention*.yaml`, `mla_no_qcomp_attention*.yaml`, `mla_output_gate_attention*.yaml` |
| DSA / QSA (sparse) | `mla_dsa_attention*.yaml`, `gqa_qknorm_partial_rope_dsa_attention.yaml`, `qwen_qsa_attention.yaml` |
| CSA / HCA / CSA2 (DSv4, DSv4.1) | `dsv4_csa_attention.yaml`, `dsv4_hca_attention.yaml`, `csa2_*.yaml` |
| SWA (windowed) | `swa_*.yaml`, `dsv4_swa_only_attention.yaml` |
| Linear / SSM | `gated_delta_net_*.yaml`, `kimi_k3_kda_attention.yaml`, `gla_*.yaml`, `nemotron_h_mamba_mixer.yaml` |

Per-tensor KV storage and read laws live in `arch/eval/cache.py` (bound to a
deployment in `cache_view.py`); the decode graph swap is
`rebind_decode_variants` in `arch/eval/transforms.py`.
