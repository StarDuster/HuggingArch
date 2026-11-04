# Quantization: Structure Is Dtype-Agnostic, Storage Is Not

A checkpoint's structure — how many parameters, how many FLOPs, what shapes — does
not change when it is quantized. What changes is how many bytes those parameters
occupy on disk and move across the bus. And modern checkpoints quantize different
parts differently: FP4 routed experts, FP8 attention, BF16 norms, all in one file.
HuggingArch models this by keeping the two concerns separate — structure is
dtype-agnostic, storage is dtype- and role-aware — which is what lets the byte gate
([validation](validation.md#4-the-checkpoint-anchor)) verify a mixed-precision checkpoint, and what lets a
single spec answer "how big a model fits if I serve it at FP8?".

## The decoupling

A `linear` in a spec declares its matrix geometry (`In × Out`) and the FLOPs that
follow from it. Neither depends on whether the weights live as BF16, FP8, or INT4 —
the architecture is the same model regardless of quantization. So parameter counts
and FLOP totals flow through the [evaluation model](evaluation_model.md) with no
dtype in sight.

Storage and bandwidth are where dtype enters. Weight bytes have two parts: the data
(`params × dtype_bytes`, adjusted for packing) and the quantization metadata (scale,
zero-point). A hard rule keeps the two layers from bleeding into each other — the
**logical dimension is sacred**: inside the spec a weight tensor's dim is the
matmul truth `In × Out`, always; packing ratios live in the quant scheme and the
engine derives on-disk bytes at materialize time. A 4-bit weight is still `In × Out`
logical parameters; it just occupies a quarter of the bytes.

## Quantization is per role, not per model

The crux is that a checkpoint has no single quantization. It is described per role,
in the model spec's `quant_context`, and every tensor is explained by exactly one
rule:

```yaml
quant_context:
  per_role:                                          # broad: role → method + config
    attention: { method: fp8, block_h: 128, block_w: 128 }
    ffn_dense: { method: fp8, block_h: 128, block_w: 128 }
    ffn_moe:   { method: fp8, block_h: 128, block_w: 128 }
    router:    { method: none }
    default:   { method: none }
  overrides:                                         # narrow: tensor name → dtype
    - tensors_re: 'mlp\.gate\.e_score_correction_bias$'
      dtype: F32
  default_dtype: BF16
```

Two layers cover everything. `per_role` assigns a quant *method* to the weights of
each role (attention, dense FFN, routed experts, router, …); `overrides` pin
specific tensors by name or regex to an explicit dtype for the long tail —
F32 routing biases and gains, a stack-specific layout. The default
is **quant-on-linear**: only linear weights of a quantized role are packed; norm
gains, biases, RoPE inverse-frequency buffers, and attention sinks stay at their
authored dtype. This is exactly where vendor mixed-precision lives: each tensor
materializes under one and only
one of these rules.

A role is also the unit a deployment chooses precision by, so inside one role
and one stack the checkpoint stores every weight matrix at one precision — the
validator refuses a role that the checkpoint splits. Where a checkpoint keeps
part of a role apart (MiMo-V2's BF16 `o_proj` beside FP8 `qkv_proj`, Kimi-K2.5's
BF16 shared experts beside INT4 routed ones), that part declares a finer role
from `roles.yaml` that refines the broader one (`attention_output`,
`ffn_moe_shared`, …) with `role:` on its node, and the spec gives it its own
`per_role` entry. A `per_role` entry reaches every role refining it unless that
role is named itself, so a finer role changes nothing for a model that does not
write one.

## Schemes as reusable data

A quant *method* — how a single linear is laid out on disk — is a YAML under
`quant_schemes/` (fp8, fp4, mxfp4, mxfp4-pack-quantized, awq, gptq,
compressed-tensors, modelopt, fp8_ue8m0, and nvfp4 — the last a deployment
format no shipped checkpoint selects). Each declares its storage dtype, its
**packing factor** (a tight enum `∈ {1, 2, 4, 8, 16}` enforced at registry import
when `_load_quant_schemes()` calls `_parse_format_variants()`),
and the **metadata siblings** that come along (scales, zero-points, `scale_inv`)
together with how many of them there are as a function of `In × Out` and the scale
strategy (per-group, per-channel, per-tensor, per-tile).

The scheme is also **self-identifying** in the tensor names and dtypes on disk: a
`qweight` tensor means GPTQ/AWQ 4-bit packing; a `_scales` sibling at UE8M0 means
MXFP4 per-block exponents. `config.quantization_config` is read when a spec is
generated (and when a snapshot classifies its header tensors); the agent
translates it once into `quant_context`, and from then on the scheme comes from
`quant_context` — the serving path never re-reads `quantization_config`. The
rest follows from names and dtypes, the same structured-name reading the
[validation byte gate](validation.md#4-the-checkpoint-anchor) relies on. A scheme may also declare that
the same checkpoint carries another scheme's sidecar (`also_stores`: DeepSeek-V4's
`scale_fmt: ue8m0` brings `fp8_ue8m0`'s `.scale`).

## Materializing bytes

`FormatPolicy` (`arch/eval/formats.py`) resolves each stored tensor's format in a
fixed order — **override, then per-role scheme (walking from the finer role up
the roles it refines), then default** — tagging every tensor with the rule that
claimed it (`override:<name or pattern>` / `role:<role>` / `default`). A spec
with no `default_dtype` is refused rather than assumed BF16. Bytes are joined
in one place, `Analysis.tensors()`; `Analysis.iter_checkpoint()` adds the scale
siblings a format implies:

```
storage_bytes = (In × Out) / packing_factor × storage_dtype_bytes
              + metadata siblings (accounted separately, not logical weight)
```

Metadata overhead scales with the scale strategy: per-group adds
`scale_bytes / group_size` (typically 128) per element, per-tile adds
`scale_bytes / tile_size`, per-tensor is negligible. The result is one
materialized tensor per leaf, each with its on-disk dtype, packing, and byte count
— the input to the byte gate.

## Why bytes, not elements, is the gate

A mixed-precision checkpoint has no single dtype to convert through. FP4 experts,
FP8 attention, and BF16 norms in one file mean the element count (dtype-blind) and
the on-disk header (dtype-specific) live in different units and cannot be compared.
Bytes is the only unit that needs no conversion — which is why the strongest
[validation gate](validation.md#4-the-checkpoint-anchor) is a byte diff, and why it *rests* on quant being
modeled per-tensor. For the same reason there is no global "serve at dtype X"
switch: deployment precision is chosen per object, by role and stack, below.

## Deployment precision: one spec, any serving format

Because structure is dtype-agnostic, deployment precision is an input to the
walk, not a rewrite of the spec. The one entry point is the **precision
profile** (`arch/precision_profile.py`, request field `precision`): a set of
rules `selector → format` in four tables —

- **weights** — the stored format of a weight (`arch/quant_schemes/`, with each
  scheme's `deployment_fields`);
- **activations** — the dtype of the edges between kernels
  (`arch/activation_formats/`; a scheme's `activation_options` offer the W4A4 /
  W8A8 choices);
- **compute** — the matmul precision; a pin never converts a weight, and a pin
  narrower than its weight is refused;
- **cache** — the format of each cache slot kind (`arch/cache_formats/`).

Selectors combine declared vocabulary only — roles (including `weight_class` and
`outside_roles`) × stacks (`backbone`, `aux:<kind>`), or cache slot kinds ×
stacks — never node or tensor names. The rule set is canonicalized so every
object has one most specific rule; ambiguity is refused. An object no rule
selects keeps what the model ships with (`quant_context` for weights, the
checkpoint's own dtype for edges, the cache entry's own declaration or the
spec's `quant_context.cache_formats` for caches). Presets live in
`arch/precision_presets.yaml` (`as_shipped`, `kv_fp8`, `kv_nvfp4`,
`w_fp8_kv_fp8`, `w_nvfp4_w4a4_kv_fp8`). The gallery's KV leaderboard offers
every preset that sets only the cache.

**A framework, not operand width, chooses a hardware MMA.** A deployed
binding's `precision_launches` lists audited candidates for one quant scheme and
requested activation format in the framework's own preference order. The
resolver first applies the launch's explicit family semantics:

- **Default launch** — every `mma_dtypes` family is needed simultaneously by
  the instruction. The resolver requires a peak column for all of them
  (`gpu_specs.peak_families`); missing one skips that candidate and records a
  `missing_card_mma` gap. This is the audited semantics of vLLM's TRT-LLM and
  DeepGEMM MXFP4 paths: their FP4 instruction is not a decorative operand
  width, so an H100 without FP4 cannot claim either launch.
- **Promotion-ladder launch** — the launch itself declares
  `promotion_ladder: true` (with its source anchor). The audited framework path
  can run on a card-supported subset of the declared families, typically by
  software-promoting one operand to another family (for example an MXFP4 weight
  dequantized beside a block-FP8 activation). The selector requires a
  non-empty intersection; only when all rungs are absent is it skipped with a
  `missing_card_mma` gap. The ladder is never inferred from a launch name.

An optional `when:` can still add a deployment fact (for example,
`moe_backend=deep_gemm` or the audited Hopper window of the Humming ladder) or
corroborate a scheme/request coordinate, but it is not the card-family gate.
If no candidate is viable, the row says so instead of borrowing a dtype. This
is why an MXFP4 weight does not imply an FP4 MMA: Humming's Hopper WGMMA path
is an explicitly declared ladder, while Blackwell TRT-LLM and DeepGEMM state
different simultaneous mixed families.

Pricing follows the same ownership rule. A selected default launch already has
all of its columns and prices directly. A selected promotion ladder asks its
declared families and takes the widest card-supported one, retaining the
activation conversion the audited path performs; for example H100's declared
`(F8, MXFP4)` Humming ladder prices at F8 even though it has no FP4 column.
Only when every ladder family is unavailable does pricing fall to the edge
dtype and remove that conversion. The same edge fallback remains for a walk
with no selected launch; a deployment-chosen dtype is refused instead of
borrowing a column. A merely partial default launch never receives ladder
promotion—that would silently change one audited instruction into another.

**As shipped means as served, and serving has a ceiling.** The width a model
ships running at is not the width its file stores: **nothing ships running wider
than 16 bits, anything already 16 bits is left exactly as stored, and nothing is
ever widened** (`dtypes.served_dtype`). One monotone rule, no framework in it —
which is what keeps it consistent with the companion rule that a BF16 checkpoint
stays BF16 even though a loader materialises fp32 parameters for part of it: a
cast a loader performs is not a serving width, and this rule cannot widen. To
RUN a width rather than ship one — fp32 inference, say — is a CHOSEN precision,
and the one entrance for choosing is the profile itself.

So `ResolvedPrecision` carries both `shipped_model_dtype` (what the file holds)
and `model_dtype` (what the deployment runs), derived from the dtype table's own
columns with no model names anywhere. The narrowing reaches a weight as an
`intent` against an unchanged `saved`, which is why the on-disk byte gate still
compares a gemma-2 tensor against F32 while its per-rank bytes, its KV law and
its matmul's GPU peak column are all 16-bit. `precision_effective` names
`framework`, not `profile`, as the source of such a row: nothing was chosen.

Both loaders were read at the pinned releases and both do narrow to 16 bits —
corroboration, not the premise; the `file:line` anchors are in
`dtypes.served_dtype`. They pick different 16-bit floats for a non-gemma float32
checkpoint (vLLM bfloat16, sglang float16), which under this rule is two names
for one width: same bytes, and `gpu_specs.csv` prices the two columns equally on
every row (held by `test_gpu_specs`), so no published number separates them. The
open end is the LABEL, not the number — a panel does not render per framework,
so one label is wrong for one of the two, and rendering per framework would mean
a framework dimension on the walk rather than a condition.

**One width, four sources.** "Width" is a single concept and the only question
is who decided it:

| Source | Decides |
|---|---|
| the file | what the checkpoint stores (`Representation.saved`) |
| a precision profile | what the deployment CHOSE to run (`WeightIntent`, the four rule tables) |
| the serving ceiling | what the model ships running at, narrowed from the file (`dtypes.served_dtype` → `ResolvedPrecision.model_dtype`) |
| a fused epilogue | what one fused launch writes, when its own epilogue decides it |

Two nearby mechanisms are NOT further sources and must not be folded in with
these, because neither answers "how wide":

- `ResolvedPrecision.conversion` + an `ActivationFormat` answer **did this
  launch actually perform a conversion, and do we bill bytes for it** — they
  graft a `quantise_activation` node (`inference.materialize_casts`) and put
  `convert_bytes` on the launch.
- A fusion plan's "output follows an operand" answers **where this launch
  writes and how wide**, with no new node.

The serving ceiling produces neither a node nor a byte, and gemma-2 shows it:
measured on both models, both deployments and both phases, the per-op node set
is identical before and after the serving width and `convert_bytes_per_rank` is
0. The fp32 tensor never exists on the device — the loader creates the parameter
at the model width — so there is nothing per step to convert and nothing to
bill. That is also why `conversion_to` returning None for plain→plain is correct
here and a gap elsewhere: a kernel that really does narrow an activation at run
time (sglang's post-mix `.to(dtype)`) needs a format to name and a node to
graft, which is the first bullet's business, not this row's.

The evaluator resolves the profile against the program into `ResolvedPrecision`
(`arch/eval/precision.py`) on `Analysis.precision`, and
`Analysis.precision_effective()` reports each object's shipped and effective
format side by side (`/precision/effective`). So the same spec simulates FP8
serving from a BF16 checkpoint, or answers "with NVFP4 weights and an FP8 KV
cache, how big a model fits on one card?"; the
[inference cost model](inference_cost_model.md) picks up the changed bytes,
while the parameter and FLOP structure never moves.

## Activation casts in the serving graph

A quantised GEMM reads a narrow copy of its activation operand. The serving
graph represents that copy as a `quantise_activation` node
(`inference.materialize_cast`), grafted after collective communication and
before fusion (comm graft → **cast graft → canonicalise** → fusion,
`serving.rewrite_serving_graph`). The governing rule is:

> **One quantised copy per (tensor, format), placed after its writer, moved
> upstream past pure copies.**

The graph models the efficient form. Where a framework quantises the same
tensor twice (for example vLLM's shared-expert linear beside its MoE prepare
step), that duplication is treated as an inefficiency of the framework, not
modelled.

### Graft

Every conversion the pre-graft walk reports (`ResolvedPrecision.conversion`)
receives one pending cast on the consumer's input edge.

### Canonicalise

The pending casts are rewritten to a fixed point of two rules.

**Common-subexpression elimination.** Casts with an equal key — source tensor
(`(writer uid, port)`, or the name of an input no node writes), target format,
and the readers' block template — merge into one. This merges, for example,
the gate and up projections of one MLP, MLA's `q_a` and `kv_a`, the shared and
routed experts of a MoE layer, and an attention's `wq_b` with its indexer's.
The template is part of the key because the IR holds one tree per block
template for all of its layers, so readers in two templates cannot share a
node.

**Commutation.** `cast(X(y)) → X(cast(y))` when X is the cast's source writer,
the cast is the only reader of X's output (`traffic.TensorReaders`), nothing
reads that output outside the graph's edges (it is not a published component
or block output, a model-level node, a cached tensor or context source, or a
stream read by a later layer), and X copies elements without touching the
innermost (scale-group) axis. Whether X qualifies is derived from its
declaration alone:

| Declaration of X | Element copy | Innermost axis preserved | Commutes |
|---|---|---|---|
| view op, or view-only component | yes | last output axis is the input's last, same extent, same element count | if preserved |
| `gather_source` (index gather; `permute` is the MoE dispatch) | yes, of that input | last output axis is the source's last (whole rows) | if preserved (`permute` yes, `gather` no) |
| non-reducing collective (`all_to_all`, `all_gather`, `send_recv`) | yes | per-rank innermost extent unchanged | if preserved |
| reducing collective (`all_reduce`, `reduce_scatter`) | no | — | no |
| `axis_concat` | yes, of every input | — | no (one launch would become one per input) |
| any other compute or source op | no | — | no |

As a consequence, the routed experts' cast moves past the expert-parallel
exchange and `permute` onto the dispatch's token input without any rule that
names the dispatch, and a non-reducing collective moves the narrow record
rather than the wide one.

### Position and launch count

A cast stands immediately after its source writer, in the writer's scope, so
that the writer's launch can absorb it during fusion (for example
`fused_add_rmsnorm` absorbs DeepSeek-V3's dense-FFN cast). When the writer is
in another block template (the residual stream from the previous layer) or
does not exist (a model input, stream, or auxiliary input), the cast stands in
the lowest scope that holds every reader, before the first node there that
holds one (`CastGraft.rewrite`). Such a copy is still a launch of its own,
never a conversion folded into its GEMM.

A copy runs once per tensor written — as often as the writer it follows,
regardless of how many readers take it. When it stands in its readers' scope
instead, it runs as often as its readers, which must agree
(`CastGraft.runs_with`); otherwise the graft is refused.

### Widths after the rewrite

Edge widths come from one walk rule (`traffic.declared_output`). A cast writes
its declared format; a fused plan writes its port's format; an op writes its
`output_dtype` or promoted dtype; a cached tensor takes its writer's width. An
element copy writes what it reads: the format is passed on and its record
width recomputed over the copy's own output (`_format_width`), so a partial
scale group is never inherited. A copy whose inputs hold different records,
or a format passed through a copy that touches the innermost axis, is refused.
An edge that already holds the wanted format needs no conversion. Collective
payloads are restated from this walk after the graft
(`MaterializedCommTree.with_payload_widths`), so an exchange moves the narrow
copy together with its scales.

## Where this lives in the code

| Concern | Location |
|---|---|
| Per-role assignment + tensor overrides | top-level `quant_context` (spec YAML) |
| Method layouts (packing, metadata siblings) | `arch/quant_schemes/*.yaml` + `arch/quant_schemes/__init__.py` |
| `quant_context` schema (per role, overrides, cache formats) | `arch/schema/quant.py` |
| Per-tensor format resolution | `arch/eval/formats.py` (`FormatPolicy`) |
| Bytes: the one join, and the on-disk list | `Analysis.tensors()`, `Analysis.iter_checkpoint()` |
| Deployment precision profile + presets | `arch/precision_profile.py`, `arch/precision_presets.yaml` |
| Profile resolved against a program | `arch/eval/precision.py` (`ResolvedPrecision`); `/precision/effective` |
| Framework launch and hardware MMA selection | `inference/framework_bindings/*.yaml` + `ResolvedPrecision._launch_for` |
| Activation and cache formats | `arch/activation_formats/`, `arch/cache_formats/` |
| Activation cast graft and canonicalisation | `inference/materialize_cast.py`, `inference/traffic.py` (`declared_output`) |
| Weight-byte + overhead breakdown | `inference/weight_breakdown.py` |
| Byte-diff + tensor-coverage gates | `arch/validate/_orchestrator.py` + `_matching.py` (see [validation](validation.md#4-the-checkpoint-anchor)) |
