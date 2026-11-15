# B300 K3.1 runtime calibration repair

Final scope: target-only Kimi-K3.1-Pro, B300 TP8/DCP8/EP2, KV64, batch1, output512. No speculation or AgentX result is implied. The original nine rows on kimi-k31-reports are unchanged. No PR.

## Final comparison

| Input | Predicted ms/token | Historical measured ms/token | Error | Predicted time using measurements |
|---:|---:|---:|---:|---:|
| 2048 | 9.340986 | 9.944850 | -6.07% | 78.81% |
| 16384 | 9.360186 | 10.081021 | -7.15% | 78.86% |

These are partial predictions compared against the existing unprofiled serving baseline, not new end-to-end runs. No aggregate multiplier, serving-latency offset, or latency-threshold row selection was fitted. The uncalibrated remainder is still theoretical and fully measured latency remains null.

## What changed

- Propagated the actual TP degree into both planner and estimator dispatch. Previously TP-conditioned bindings were silently disabled.
- Added source-anchored latent-path and complete-MoE execution spans. The complete span includes merged dense front, prepared routing and MXFP8 quantization, TRTLLM SiTU expert GEMMs, shared-expert side stream, join, deferred finalize/AR/norm, and GEMV/all-gather/add. It is a multi-kernel span, not a claim of one CUDA kernel.
- Allowed a span that owns the full dispatch/expert/combine region to consume original tokens. Partial boundary crossings still refuse. Whole spans do not inherit internal expert-row padding or dispatch membership.
- Preserved mathematical operands when a computation span contains communication; primitive wire shapes no longer replace its input signature.
- Retained both same-group collective payloads and their independent fixed floors and dispatch rungs. A payload aggregate is used for byte totals, never as a fabricated single launch.
- Replaced the entire measured MoE block once per layer. Its routing, quantization, shared activation, Norm and collective measurements are not charged again.

## Direct concurrency measurement

| Batch | Source two-stream execution µs | Serial control µs | Time saved |
|---:|---:|---:|---:|
| 1 | 44.856 | 57.346 | 21.78% |
| 8 | 83.600 | 97.282 | 14.06% |

Both controls use the same kernels, inputs and weights; the serial control removes the shared side stream. The source fork/join and low-SM pull collective are reproduced, followed by deferred finalize/push/norm and GEMV/all-gather/add. The logical MoE boundary uses prefix_sum=None; the outer decoder residual remains outside this measurement.

Every timing uses all eight ranks, maximum rank median, thirty graph samples, sixteen distinct routes and four dense-weight sets. Rotation avoids repeating one resident expert or dense matrix set in L2. The complete expert bank is present. No L2-drain subtraction is inserted between serving kernels. Cache/weight residency is a documented regime, not a universal device constant.

## Numerical and regression validation

Complete MoE numerical validation: batch1/8 × seeds17/43 × all8 ranks =32 successful checks. The independent reference uses FP32 dense GEMMs, explicit E2M1 decode, MXFP8 group32 rounding, SiTU(beta4, linear_beta25), weighted expert outputs, CPU/Gloo sums and RMSNorm. The existing FP8 error band was unchanged. Maximum normalized RMS error is 0.004382, maximum normalized element error 0.044099.

The narrower distributed path also passed32 checks and the actual standalone TRTLLM API passed four checks. Final logs: final-runtime-tests.log (87 passed,140 subtests), final-integration-tests.log (6 passed), roofline-comm-regression-tests.log (6 selected passed). Earlier broader structural checks passed163 tests plus47 subtests. Do not sum these overlapping suites as distinct tests.

## Intermediate result and cancellation

The latent-only correction predicted10.330ms at16K (+2.47%) and10.311ms at2K (+3.68%). It still added the shared branch serially. The measured complete block is44.856µs, whereas that intermediate model billed55.401µs per MoE block. Its smaller aggregate error therefore included cancellation. The final report uses the complete two-stream measurement even though the final absolute error is larger. Both intermediate artifacts remain for audit.

## Reproduction and artifacts

Use whole-moe-overlay.json with the code on codex/b300-calibration. The native planner generates the keys; no operator key is invented. Run prepare_runtime.py --whole, the eight-rank whole_moe_tp8.py with the pinned SGLang PYTHONPATH, export_runtime.py --whole, then estimate_runtime.py --whole. The narrower diagnostic uses --span-scope latent in the estimator and an equivalent narrowed planner binding.

Raw per-rank samples, numerical verdicts, serial controls, scripts, CSVs, exact native job keys and source snapshots are included. Critical MoE source files compared between historical SGLang1b2870d3 and recipe5792cf6e are byte-identical; hashes are in SUMMARY.json. The GPU stack is PyTorch2.14.1+cu130 / FlashInfer0.7.4, matching the recorded earlier benchmark.

All raw prediction outputs, including per-op symbolic laws, are archived in the [private results release](https://github.com/StarDuster/k3-performance-results/releases/tag/b300-runtime-calibration-20261115). The source branch keeps their compact summaries and hardware measurements; large expanded prediction JSONs are omitted from the source checkout.

Remaining limits: only two historical B1 contexts have end-to-end comparison; B8 is a numerical/timing span check. No claim is made for AgentX, DFLASH, other EP/TP layouts or arbitrary routing skew. About21% of final predicted time remains theoretical.
