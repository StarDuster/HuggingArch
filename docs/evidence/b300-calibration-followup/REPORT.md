# B300 calibration: validated measurements and remaining error

The calibration dataset and fusion fixes are saved on `codex/b300-calibration` in the user fork. No PR is opened. The original nine rows on `kimi-k31-reports` are unchanged. No AgentX rerun was started after cancellation.

This pass adds four TP8 multicast shapes, sixteen learned-weight AttnRes-plus-RMSNorm shapes measured in two explicit timing regimes, and four public CUTLASS group-32 MXFP8×MXFP4 shapes. All forty new measurements passed numerical checks. These are forty rows, not forty distinct physical keys: the sixteen AttnRes keys have separate cold and warm measurements.

| Measurement | Rows | Timing / usable scope |
|---|---:|---|
| SGLang TP8 multicast | 4 | Warm graph128; maximum rank median; residual=None, no fused norm |
| SGLang weighted AttnRes + RMSNorm | 16 | Native isolated L2-drain |
| Same AttnRes kernel | 16 | Warm graph64; separate implementation suffix |
| FlashInfer CUTLASS MXFP8×MXFP4, group32 | 4 | Cross-library alternative; one logical row per expert padded to four |

TP8 measured medians are 3.214 / 3.326 / 3.551 / 4.163 µs for 3584 / 7168 / 28672 / 57344 BF16 elements. The distributed checks use all eight ranks and an independent Gloo FP32 oracle, with seeds 17 and 43.

## Residual reference correction

The old unfused AttnRes signature and reference omitted the learned score gain/projection tensors already present in the model graph. Its sixteen old Torch measurements validated the wrong function and are retired into `legacy-reference-quarantine/`; they are excluded from the live corpus and both controlled overlays. The corrected unfused reference reads both weights. A separate SGLang fusion contract preserves both score inputs and includes the following RMSNorm. The provider implements the source BF16 cached score-weight product and matches an independent FP32 expression.

For batch1, the new kernel measures 8.736–10.752 µs with isolated L2-drain and 2.235–3.738 µs in graph64 steady state. Historical serving CUPTI medians at context16385 are 5.264–7.024 µs across history depths1–8. All fourteen sampled decode steps have186 matching kernel calls. History depth is inferred from forward order, and CUPTI event durations differ from standalone CUDA-event timing; this diagnostic is not imported as calibration. The historical and current attn_res.py files have identical SHA256, although the overall SGLang revisions differ.

## Controlled 16K comparison

Each overlay keeps the original41 library-backed rows, adds four TP8 rows, then adds exactly one set of sixteen AttnRes rows. No latency threshold chooses rows. The four CUTLASS alternative GEMMs are excluded because the serving recipe selects TRTLLM fused MoE. Other kernels retain their original measurement regimes, so these are partial mixed-clock diagnostics.

| AttnRes timing | Partial prediction ms | Historical TPOT ms | Error | Measured time share |
|---|---:|---:|---:|---:|
| cold | 12.7513 | 10.0810 | +26.49% | 64.28% |
| graph-warm | 11.5811 | 10.0810 | +14.88% | 60.67% |

The 5–8% error target has not been validated. Adding standalone timings does not by itself reconstruct serving latency. Remaining gaps include the selected full TRTLLM MoE implementation, two unfused AttnRes boundary calls, and serving PDL/multi-stream overlap and cache conditions. The model still combines theoretical and measured costs; fully measured latency remains null. No aggregate fitted multiplier is applied.

## Artifacts and checks

`calibration_rows.json` contains146 validated rows including eager references and alternate timing regimes. It is an audit corpus, not a recommended universal overlay. The four distributed TP8 rows remain in `tp8-communication/calibration_rows.json`. Use the explicitly named `controlled-*-calibration_rows.json` files only with the scope above. `alternative_group32_calibration_rows.json` stores the alternate grouped GEMM rows separately.

The native GPU cross-checks passed for every new measurement. Regression output is in `weighted-reference-tests.log` and `fusion-followup-tests.log`. Raw CSVs, graph samples, rank-local communication samples, source provenance, scripts and a SHA256 manifest are included. The historical serving comparison is target-only B1, TP8/DCP8/EP2, KV64, speculation disabled; it is not an AgentX measurement.

Earlier broad-overlay reports are retained for history and are superseded by this report. Their retired AttnRes rows are invalid and must not be reused. Existing library-only41-row estimates preceded the new fusion contract and are not a controlled comparison with this pass.
