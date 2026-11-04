Weight / param / byte exactness is NOT evidence of forward fidelity. A spec can
map every checkpoint tensor with the right shape and still compute a different
function: parameter-free ops (score functions, norms without weight, inverse
RoPE, clamps, Hadamard rotations), boolean switches, dataflow edges, reused
shared weights, layer patterns, and cache layouts all leave the tensor list
unchanged. Only a walk of the real `forward` catches them.

**Forward step.** A step is a computation that changes values at inference
time under THIS config: every projection, norm, activation, score function,
scale, rotation, clamp, gather / top-k, add / concat that feeds the output,
including `Model.forward` (embedding, final norm, head) and submodules below
attention / MoE (router, experts, compressor, indexer). NOT steps (list them
as `n/a (reason)` if you mention them; they never affect the verdict):
training-only paths (dropout, aux losses, QAT simulation), branches the config
turns off (cite the condition, e.g. `use_sliding_window=false`), pure
views / reshapes / transposes / contiguous (the spec's shape system absorbs
them), up-casts that change no stored or cached dtype, mask / position
bookkeeping (incl. MTP position-0 embedding zeroing), and cache writes (the
spec's `cache:` declarations cover them).

**Granularity.** One mapping row per distinct component — the unit
`backend.arch.structure` prints — not per layer. Layer patterns get one row
per assignment rule (a `per_layer` generator vs the config list or rule it
reproduces, e.g. `first_k_dense_replace`, `layer_types`, `compress_ratios`),
with the check done for every layer index inside that row. Auxiliary stacks
(MTP, towers) get their own rows. An `aux_kind: mtp` stack is the
speculative-decoding draft and sits outside the backbone totals. It follows
the repo's own MTP forward where the repo runs one (DeepSeek-V4
`inference/model.py` `MTPBlock`). When the HF modeling file only loads or
drops these weights, map the stack's rows to the vLLM / SGLang `*_mtp.py`
forward and cite it beside the dropping line; the same family's framework MTP
is the reference when this model has none of its own.

**Citations.** Code inside the listed full source files: cite `file:line`.
Code outside them (e.g. a backbone class imported from another package): cite
`Class.method` plus the quoted source line. Config defaults the config.json
omits: cite the configuration class field (e.g. `norm_topk_prob` absent →
class default). An item that does not apply is cited by the forward code or
config field that shows its absence. A framework file is cited only for a part
the repo forward does not run, beside the HF line that shows the gap.

**Checklist.** Tags say what a mistake changes: [cost] = FLOPs / weight or
activation bytes, [KV] = cache bytes, [sem] = forward semantics only (numbers
unchanged, still a defect).

Attention
- softmax scale: head_dim^-0.5 vs qk_head_dim / custom scale; YaRN / mscale
  applied to logits or folded into cos/sin [sem]
- attention sinks, logit softcap [sem]
- partial rotary: fraction and which dims rotate (first vs last slice) [sem;
  cost if the slice feeds a matmul]; pairing (interleaved vs rotate_half) and
  per-layer rope parameters — see the constant rule below [sem]
- qk-norm: present? per-head or full-width? before or after RoPE? weighted or
  weight-free (`q *= rsqrt(mean(q²)+eps)` has no tensor) [sem; bytes if unfused]
- SWA / NoPE / global layer pattern and window size (one row per assignment
  rule, checked for every layer index) [KV][cost]
- output-side ops: inverse RoPE on the attention output, output gates, grouped
  output projections [sem][cost]
- MLA / LoRA extra scales (e.g. sqrt(d_model / q_lora)) [sem]
- linear attention (Gated DeltaNet, KDA): `use_qk_l2norm_in_kernel=True`
  normalizes q and k before the recurrence even though the call is a kernel
  flag; each is an `l2norm` node feeding `gated_delta_rule` [sem; bytes]
MoE / FFN
- router score function: softmax vs sigmoid vs softplus vs sqrt(softplus); the
  gate tensor is identical for all of them [sem, changes top-k choice]
- top-k weight normalization on/off, including the config-class default [sem]
- routed_scaling_factor; where the correction bias applies (selection only vs
  weights) [sem]
- grouped routing (n_group / topk_group) — present only when the source does it [sem]
- shared experts, zero / identity experts, hash-routed layers [cost]
- dense-first layers / MoE layer frequency (one row per assignment rule,
  checked for every layer index) [cost]
- activation clamps (swiglu_limit) and activation type [sem]
- Hadamard / other fixed rotations (indexers, quantized paths) [sem][cost]
Model level
- embedding scaling (× sqrt(d_model)), residual scaling, tied vs untied head [sem][cost]
- MTP / auxiliary stacks: which hidden feeds them (pre- vs post-final-norm,
  before or after hyper-connection head), concat order, and their logits head —
  an MTP step that reuses the main lm_head still pays that projection (FLOPs +
  weight read) inside the MTP stack [cost]
- cache layout and dtype: latent vs expanded KV, K=V sharing, indexer / state
  cache dtype (fp8 / bf16 / fp32) [KV]

**One ground truth.** The HF repo forward decides every step it runs. For a
part it does not run, the spec follows vLLM or SGLang and its comment names the
HF gap and the file followed; when the two frameworks differ on that part, the
comment names the one followed and why.

**Constant not expressible.** A difference qualifies only if BOTH hold:
(a) no op in the spec vocabulary has a field or attribute that can declare it
(name the op you checked), and (b) it changes neither which ops run nor any
dataflow edge. Examples that qualify: a scale / theta / eps value, rotary
pairing (the `rope` op has no pairing attribute, so interleaved vs rotate_half
is recorded as "constant not expressible: rope pairing", never MISMATCH),
the section sizes of a multi-axis rope and 2-D axial vision rope, xdrope
sections, where YaRN mscale is applied, per-layer rope parameters. Never constants:
switches that decide whether a node exists (norm_topk_prob, qk-norm on/off,
a clamp), score functions, and any missing or extra elementwise transform.
A multiply by a constant is itself a node (`scale_const`); only its value is
exempt. The number of position streams a text rotation reads is an op, not a
constant: when the text rope parameters carry `mrope_section` (Qwen3-VL,
Qwen3.5, Qwen3.8), the rotation is `op: mrope` — a library attention takes it
as `params: { rope: mrope }`; when the rope type is `xdrope` with
`xdrope_section` (HunYuan-VL), it is `op: xdrope` — and validate holds the
spec to the config. Without a section list in the config, the HF rotary class
decides and the spec cites it: a class that sections by its own default
(`Qwen3_5MoeTextRotaryEmbedding`, `mrope_section` default [11, 11, 10]) is
`op: mrope`. A
vision tower that rotates each patch by its (h, w) grid position is
`op: rope_2d`; an encoder that rotates by one position per token is
`op: rope`. The
section sizes, θ and the SWA layers' θ are shown through `view:`
(`rope_scaling`, `rope_theta`, `swa.rope_theta`). If the HF forward folds that multiply into an adjacent weight, map it to
that node and cite the line. A missing operation needs a node — use a `custom_ops:`
entry if no primitive fits.

**Faithful idioms (never a defect).**
- Attention confined to fixed groups of G consecutive positions, both
  directions inside a group (a `[T/G, G, D]` reshape run with
  `is_causal=False` — MiMo audio's local transformer, `group_size` 4): the
  `swa_*` ops with `s_window: <G>`, `is_causal: "0"` and `is_chunked: "1"`.
  Each query sees exactly its group's G keys, the work of the block-diagonal
  form.
- An attention whose source runs it without a causal mask (`is_causal=False`,
  `causal=False` in the tower's modeling code) declares `is_causal: "0"` on its score, softmax and
  apply nodes. A bidirectional sliding window (`sliding_window_q =
  sliding_window_k = w`) is the `swa_*` ops at `s_window: <2w + 1>` with
  `is_causal: "0"`.
- A vision tower whose block stack mixes block types is an aux group holding
  a `repeat:` + `per_layer:` stack between its stem and head ops (MiMo
  vision, `fullatt_block_indexes`); a single-type tower is a singleton
  wrapper component.
- Per-row / per-column tokens of an image grid (one learned separator per
  merged row, HunyuanOCR `image_newline`) are counted from the image's own
  axes: `img_h // (patch * merge)` merged rows per image, over `B * n_img`
  images.
- The d_k^-0.5 query scale of a delta-rule kernel (`scale`, default
  `k.shape[-1] ** -0.5`) belongs to `gated_delta_rule`, as 1/sqrt(d_head)
  belongs to `attention_score`: no separate scale node. The q/k L2 norm
  does not; it is an `l2norm` node (checklist above).
- A patch embedding (Conv2d / Conv3d) whose stride equals its kernel is a
  `linear` over the flattened patch, `in = channels × kernel volume`; the
  library writes it that way (`kimi_k25_vision_tower.yaml`).
- A choice made inside a shared library component (a cache dtype, a
  constant, a missing node) is a defect of that component, tagged
  `[library: <component file>]`; the model spec cannot override it. A
  choice the component takes as a `params:` slot is the model spec's to
  bind, and a wrong binding is a defect of the model spec.
- `output_gate_type` sets only the Gated DeltaNet norm-gate activation
  (RMSNormGated; `swish` is silu, absent is silu), bound on the library
  GDN components as `params: { gate_act: … }`. The full-attention output
  gate is `sigmoid(gate)` in transformers, vLLM and sglang whatever the
  field says, so a sigmoid attention gate beside `output_gate_type: swish`
  is faithful.
- A cross-layer value written in the form the system context's
  "`forward()` pattern → stream form" table gives for its pattern: the layer
  loop as the stack's carried `hidden`; a threaded
  per-layer argument (`prev_topk_indices`) or a reused earlier layer's KV as
  an inner stream (`stream_out` / `reads`), whose readers declare no cache;
  hidden states collected at chosen layers as an appended stream, taken
  at the layer's exit (`after:`, DeepStack, Llava, EAGLE-3) or at its entry
  (`before:`, Kimi-K3 AttnRes) and read `each`, `all` or `concat`. The spec has no node for the list append, for
  the `torch.cat` of a `read: concat` join or for its per-element crop: the
  engine lays those down, so their absence is not a defect.
