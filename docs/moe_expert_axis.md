# How a MoE module is modelled: the experts are the outermost axis

Inside a MoE module, every tensor's outermost axis is the expert axis.

* every routed expert weight is `[n_expert_copies, out, in]`, declared by the
  node's `shape:`;
* every activation between the dispatch and the combine is
  `[n_expert_copies, M, d]`, where `M` is the MEAN rows one expert receives.

Nothing about a bank is a count: there is no `repeat:` over the experts, no
per-expert evaluation, no unit placeholder axis. A bank is one node whose
tensors are wider by one axis, and one launch of it runs every expert the rank
holds.

`E` comes only from declarations — a weight's `shape:`, `permute`'s declared
output, `dims.yaml`'s `n_expert_copies` — and never from a node's position, its
zone membership, a tensor's rank or a name's spelling.

---

## 1. How a bank is written

Every MoE component calls one of the expert templates in
`components/library` — `gated_experts`, `fused_gated_experts`,
`clamped_gated_experts`, `relu2_experts`, one per HF `*Experts` shape the
corpus has. Their weight leaves declare `shape: [n_expert_copies, out, in]`
and a 3-entry `weight_axes` whose axis 0 is
`{ world: moe_copies, parts: n_expert_copies, fit: exact }`.

The call site says how the checkpoint stores the expert axis:

* **unrolled** — one tensor per expert on disk, indexed after `experts`
  (Qwen3-MoE):

  ```yaml
  experts:
    op: gated_experts
    in: permute
    dims: { d_model: d_model, d_ffn: d_ffn_expert }
    expert_weight: unrolled
  ```

  with `inner_modules` where the checkpoint's names differ (Mixtral's
  `w1` / `w3` / `w2`);

* **leading_axis** — the checkpoint already stores the stacked tensor
  (Step-3.7, gpt-oss): `expert_weight: leading_axis`. gpt-oss's biases are
  `parameter_vector`s of `shape: [n_expert_copies, payload]`.

Qwen3-VL-MoE and Hy4 write their bank inline because their checkpoints store
each expert transposed (`stored_axes: [0, 2, 1]`).

The ops inside a template are the declared ones — `gate_proj` / `up_proj` /
`down_proj`, the packed `gate_up_proj` with its two slices, the activation and
the multiply — each the plain op over the leading axis, which is
`torch.matmul`'s batch rule (`ops.weight_leading_axes`). A grouped kernel that
runs the rank's experts in one launch is the MEASUREMENT layer's
(`resolve_op` → the `grouped_gemm` signature), and gate/up fusion is a
framework rewrite pattern (`inference/fusion_plans/`). Neither is ever written
into a spec or into the walk: a grouped op in the spec is how a bank gets
counted twice.

The expert axis is placed by its `weight_axes` entry like any other axis:
expert parallelism divides a SHAPE, the way every mesh does (`dims.MESHES`).
A rank holds a slice of the bank's leading axis; nothing holds fewer whole
copies of anything.

## 2. Where each consumer reads `E`

Always by declaration:

| consumer | reads `E` from |
|---|---|
| weight storage, params (`storage.from_declaration`, `ops.weight_leading_axes`) | the node's `shape:` |
| weight placement under EP (`weight_endpoints`) | the weight's `weight_axes` axis-0 entry and `dims.yaml` `n_expert_copies: shardable_by [moe_copies]` |
| checkpoint ledger (`copy_checkpoint.expert_artifacts`) | `storage.leads_with_expert_axis` (axis 0's identity is a `kind: expert_copies` symbol) and the call's `expert_weight` |
| the activation `[E, M, d]` | `permute`'s declared `shape_out`, carried by shape propagation; `combine`'s declared `shape_in` |
| active parameters (`Evaluator._routed_share`) | `leads_with_expert_axis` on the leaf's stored tensors, inside the zone's routed scope |
| pool facts (`Analysis._moe_pool_stacks`) | the routed node's expert-led tensor, its leading extent |
| the routed workload (`layout/moe.py`'s `permute`) | the node's binding, `n_expert_copies` — the only consumer that reads the number itself, because it is the one BUILDING the evidence every other reads |
| the routed-tensor check (`propagation._validate_routed_state`) | the axis whose identity is an expert axis (`storage.is_expert_axis`), and its placement against the routed-load evidence |
| a row gather before column parallelism (`layout/linear.py` `gather_rows_for_output`) | the axis after the expert axis, the expert axis found by identity (`is_expert_axis`) |
| per-rank extents (`layout/physical._local_dims`) | the routed-load evidence's mean for the rows axis, times the busiest rank's load factor; the expert axis cut like any other, located by identity |
| the route exchange's buffer (`materialize_flow.routed_flow_shape`) | `local_experts`, handed in by the caller, times the evidence's mean and that factor, rounded up to whole rows |
| `linear`'s batch axes (`layout/linear.py` `weight_batch_axes`) | the weight's leading axes |
| the measurement key and the cadence (`benchmark_identity`) | the signature's declared engine layout over the arriving `[E, M, d]` shape as one launch lays it out (`RoutedLaunch`, on the row as `routed`: over the experts the rows reach) — `grouped_gemm` and `moe_combine` read it as operand 0, `moe_permute` as its `engine_outputs`; the key names the pair `n_experts` / `s_expert_rows` (M's measurement name — not `s_query`, which is the dispatch's TOKEN side, `B × s_query`, and the two boundary cells carry both) |
| the roofline's kernel rows (`row_roofline.kernel_rows`) | the node's own declaration — `OpSpec.shape_in` / `CustomOpSpec.shape_in` beside `couples_on` — with the expert axis excluded from the rows by identity (`storage.is_expert_axis`). It reads no signature: a measurement cell and a saturation law are two questions |
| the `repeat:` refusal (`Evaluator._copies`) | `is_expert_axis` on the resolved count, and the routed scope |

A `repeat:` inside a dispatch zone, or over a symbol `dims.yaml` classifies
`kind: expert_copies`, is refused by the walk.

## 3. `M` is a mean, and the model of the spread is two numbers

`M = B·s_ctx·n_experts_active·dp_attention / n_route_targets` — the rows one
expert receives ON AVERAGE. It is kept exact (a `Rational`), because the
bank's total is `E · M` and any rounding here is multiplied by the expert
count.

A real routing trace gives each expert its own count, and a grouped GEMM is
called with those counts (cuBLAS 12.5+ `cublasGemmGroupedBatchedEx` takes
`group_size[]` and a per-group `m`). The model carries exactly two readings of
the distribution and no per-expert histogram:

* the mean `M`, which the walk binds and every zone tensor carries;
* the busiest rank, from the declared skew `ep_load_factor`
  (`ExpectedRoutedLoad.busiest_rank_load_factor`,
  `routed_row_capacity_per_rank`), which a routed tensor's per-rank shape
  carries (`layout.physical`): the rank holding the hottest expert is the
  one every per-rank row prices.

Both are published on one object (`ExpectedRoutedLoad`), and
`tests/backend/inference/test_routing_imbalance_pricing.py` holds them
together: a consumer that read the mean and not the skew would silently assume
balance. A measured grouped-GEMM cell is keyed on the uniform routed operand
over the experts its rows reach, so it answers for a balanced trace; the busiest-rank reading is what
the latency model adds on top. A display shows `M` as a mean.

`M` is never read as the batch: the zone rows' shape laws keep the expert axis
constant and `M` linear in the batch, and the FLOPs formula shows `B` only
inside `M`.

**At the kernel boundary the mean is padded up, and nowhere else.** `M` stays
exact everywhere the walk reads it — the storage, the FLOPs, the bytes, the
SOL time are all the mean's, because the work is conserved however the rows
fall. A kernel is handed a buffer, though, and no buffer is a fraction of a
row wide: a measurement pads every expert that RECEIVES rows up alike, and
an expert that receives none does nothing (`eval.routed.RoutedLaunch.
touched_experts`) — the convention `ExpectedRoutedLoad` points at by leaving the
rounding to the boundary that needs it. From a mean of one row up every
expert receives some and the launch is `[E_local, ceil(M), d]`; below it the
step's `ceil(E_local × M)` routed rows land on that many experts at one row
each (DeepSeek-V3's decode at B=1: 8 rows over 256 experts, 8 rows run, not
256) — the same saturation the weight read already takes (`Min(received,
E)`), and a measured cell's `n_experts` is those experts. Two consequences,
both carried rather than assumed: the measured cell's row total `n_tokens` is
those padded rows
(`benchmark_identity.routed_launch_rows`), and a job
states the law's own exact row count beside the padded one
(`launch_rows_exact` beside `launch_rows`) so a reader can see how much of
the measured launch was padding. The numerators a job carries are that
padded geometry's, since the latency it divides is the padded launch's.

That last one is **term by term and not an aggregate ratio**
(`calibration_overlay._at_geometry`): FLOPs follow the launch's rows, the activation read
and a kernel's own quantised copy of its operand follow the READ side's
elements, the activation write the WRITE side's, and the resident weight read
saturates with its scales. The dispatch boundaries are what tells those
apart from one ratio — `permute` reads the step's tokens and the routing that
picked their experts, neither of which widens because the routed rows were
padded, and writes the padded routed rows; `combine` is the reverse. So each is
`fixed + p x routed`, and `p x total` overcharges DeepSeek-V3's decode
boundary by 5.9%.

## 4. The activation peak holds the whole bank

A bank's `[E, M, ·]` buffers are all live at once, because that is what the
default forward does: transformers picks `grouped_mm` when no implementation
is requested (`modeling_utils.py`), and `grouped_mm_experts_forward`
(`integrations/moe.py`) materialises `tokens·k` rows and keeps them alive
through both projections to the weighting. vLLM's `fused_moe` allocates
`cache13` / `intermediate_cache2` over `M·top_k` rows; sglang allocates
`intermediate_cache1/2/3` over `num_tokens·topk`. HF's `eager` experts
implementation — a loop over the hit experts — peaks lower and is not the
default.

Open limitations of this model are listed in [known limitations §2](known_limitations.md).
