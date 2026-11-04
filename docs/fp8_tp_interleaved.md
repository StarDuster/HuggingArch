# TP-interleaved FP8 checkpoint grids

A fused projection's logical Q/K/V segments do not necessarily own separate
FP8 scale grids. MiMo-V2.5 stores four interleaved `[Q|K|V]` chunks; V2.5-Pro
stores eight. Each complete chunk owns its own ceil-padded grid. For V2.5 full
attention, the weight is `[13568, 4096]` and its scales are
`[4 * ceil((13568 / 4) / 128), 32] = [108, 32]`. The K segment's 192 local
rows sit inside the fused chunk's grid.

The quant entry declares `block_scale_layout`, including the checkpoint TP,
parallel world and the loaders known to handle this layout. This fact stays
with its `TensorFormat`, then travels with the walk's `WeightShard` and the
published `MatrixShard`. It does not change the logical head placement.
The planner applies the selected loader's rules and retains ordinary block
alignment checks for weights without this contract and for the reduction axis.

| Loader | Matching checkpoint TP | Other supported TP |
|---|---|---|
| SGLang v0.5.23, `whole_chunks` | Load one complete chunk and its scales | TP must divide checkpoint TP. Merge complete chunks, dequantize, deinterleave Q/K/V, and requantize one rank-local fused matrix. |
| vLLM v0.32.0, `head_gather` | Load one complete chunk and its scales | Gather each rank's Q heads and K/V heads from the checkpoint chunks, replicate K/V where necessary, then requantize one rank-local fused matrix. |

For V2.5 full attention, SGLang TP1/2/4 reads resident grids with 106/53/27
rows per rank; TP8 is refused. vLLM TP8 has local fused N=1856, including
replicated K/V, and a 15-row scale grid. V2.5-Pro's matching degree is TP8;
SGLang also supports TP1/2/4, while vLLM can gather heads at TP16.

The serving reader materializes a resident format on the existing analysis
when a loader requantizes. Checkpoint formats retain the original grids;
`iter_resident`, whole-model residency and rank projection share the resident
format. Unsupported SGLang degrees keep conservative analytical geometry and
receive a loader refusal in the planner. At degrees that cannot own whole
checkpoint chunks, the physical ledger omits the source scale grid's rank
surface: head gathering does not define a contiguous shard of that grid.
The global checkpoint tensors remain available through `iter_checkpoint`.
Load-time CPU buffers and traffic are not modelled.

Sources:

- [SGLang fused loader and deferred scale resolution](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/mimo_v2.py#L126-L321).
- [SGLang checkpoint TP selection](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/configs/model_config.py#L124-L145).
- [vLLM fused weight/scale sharding](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/mimo_v2.py#L495-L632).
