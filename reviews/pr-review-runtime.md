# Runtime / DCP / calibration review

Reviewed current runtime_laws.py, flashmla_schedule.py, sparse_dcp_contract.py, sparse_dcp_comm.py, materialized local traffic and row-roofline communication consumption, paged indexer / absorbed-BMM providers, and calibration selection. No model service, GPU job, large test suite, or production-code edit was performed by this review. Small CPU function probes only.

## Confirmed findings

### P1 — Calibration CLI selected a strategy that could not consume the new rows — FIXED by root

`scripts/estimate_kimi_k31_sglang.py:39` originally used `selection_mode='latest'` whenever `--calibration-rows` was supplied. However the SGLang indexer entry has a source citation without an explicit provider dispatch identity, and the newly measured absorbed BMM providers likewise have no declared identity in that binding. `op_calibration.latest_time_for` returns None without either a declared dispatch or an explicit provider pin. Physical-key equality alone is insufficient for latest selection.

Small CPU probe loading the actual evidence rows through `merge_validated_rows` reproduced indexer and V-BMM `latest -> ambiguous`, while `envelope -> measured`. Thus the evidence scripts' prior `physical_time_for` checks were correct but did not establish consumption by the public CLI. Root changed the CLI to explicit `envelope`, records the selector in summary, and is verifying actual public partial coverage. I confirmed the edited selector at lines39/57. No claim that a particular SGLang dispatch has been pinned is added.

### P2 — Windowed scheduler extrapolated beyond SGLang's admission boundary — FIXED by dcp_support

`SparseMlaLocalStages.work` originally selected windowed work for every known SM count and arbitrary flattened row count. The real SGLang `_maybe_precompute_flashmla_sched_meta` checks `4*(5*b+1+num_sm_parts*8) <= 48*1024` before precomputing; otherwise FlashMLA schedules itself. For the audited B300 head96 geometry, parts148 allows2220rows but rejects2221. A prediction still naming the windowed schedule after that transition would have wrong split counts and producer/combine traffic.

`runtime_laws.windowed_counts:29-33` now enforces the same boundary and explicitly rejects the unmodeled native fallback. This is reached after symbolic point substitution as well as direct numeric evaluation. I checked2220 succeeds and2221 raises the new explanatory ValueError. The primitive `windowed_schedule` remains a pure oracle of that algorithm; the runtime law owns dispatch admission.

## Other reviewed contracts

- 300 deterministic bounded CPU cases compared periodic `windowed_counts` totals/combining-row/combining-split counts with the independent source-style `windowed_schedule` oracle: all agreed. Patterns include zero, partial-block and heterogeneous lengths across multiple batches and hardware partition counts. This is CPU algorithm verification, not live ownership evidence.
- Verified source zero-length handling uses one block; metadata partition count is distinct from per-query split count. Flattened verification rows are used rather than a hardcoded74splits.
- Default candidate merge retains full K buffers and a separate localize pass. Its multicast recipient volume, fallback hotspot sender volume and extra dependency round remain distinct. Direct-exchange ring factors are applied to full collective payloads once; they produce seven remote candidate blocks and7/8 of projected head partials for world8.
- Local combine contracts separate producer FP32 split-partial writes, combine reads/final writes, and remote communication. Unknown arrival polling and integer/radix throughput are not presented as measured FLOPs. No additional arithmetic correctness issue was confirmed in the audited B300/head96/reference-profile domain.
- Paged indexer cache declaration derives exact byte extents from CacheFormat plus page geometry; it does not globally permit uint8 for FP8. Packing is page-SoA with FP32 scale tails; normal numerical check remains mandatory. BMM providers retain the actual sliced/transposed Q strides and contiguous per-head weights.
- The nine stored rows use exact shape/dtype/hardware/phase keys. Training rows only enter overlays; heldout rows carry separately generated actual seeds. Holdout demonstrates same-shape input/run repeatability, not unseen-shape/full-model prediction. CUDA-event L2-drain timing is not relabeled CUPTI warm-cache timing.

## Remaining limitations (not newly confirmed regressions)

- The runtime plan is an explicitly audited TP8/DCP8/B300 reference profile. Only B300 currently supplies an SM count in the built-in GPU table. If arbitrary non-Blackwell deployment SM counts become public inputs, fast-scheduler hardware/native-head eligibility must be guarded too; a positive SM count alone is not source proof of dispatch eligibility.
- MoE MXFP8×MXFP4 and cross-component Kimi fused-front remain uncalibrated. Assigning similar BF16/DeepGEMM measurements to those differently fused FlashInfer/TGV operations would be invalid.
- Parent's final public estimate/partial-coverage run remains the integration evidence; the small registry probe here does not replace it.
