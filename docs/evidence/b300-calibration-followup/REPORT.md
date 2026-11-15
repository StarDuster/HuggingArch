# B300 kernel calibration follow-up

Reused 9 existing measurements; added 99 validated positive-latency measurements. Physical-key deduplication is recorded in deduplication.json. Native failures/refusals are retained in results/ and coverage.log.

| Input | Calibration | Predicted ms | Historical TPOT ms | Relative error | Measured share |
|---:|---|---:|---:|---:|---:|
| 2048 | theory | 6.3817 | 9.9448 | -35.83% | 0.00% |
| 2048 | existing | 7.1790 | 9.9448 | -27.81% | 14.87% |
| 2048 | expanded | 74.3190 | 9.9448 | +647.31% | 98.26% |
| 16384 | theory | 6.3817 | 10.0810 | -36.70% | 0.00% |
| 16384 | existing | 7.1982 | 10.0810 | -28.60% | 15.10% |
| 16384 | expanded | 74.3382 | 10.0810 | +637.41% | 98.26% |

The historical target-only serving measurements used three repeats, disabled speculation and excluded profiled requests. They are not AgentX or DFLASH measurements. See benchmark/dcp-validation in the results archive for full runtime provenance. This comparison reuses the recorded baseline; it does not claim a fresh end-to-end remeasurement.

All three estimates above use the corrected branch (DCP formula reader, K3.1 fusion rules and explicit KV block size 64). Differences between existing and expanded isolate the overlay under those corrected model rules; differences from older report totals also include the rule/lookup corrections.

The measured-envelope selector can choose a faster provider than SGLang dispatches. Cold-cache L2-drain timings differ from a warm CUDA graph; serial summation does not model full PDL/multi-stream overlap. Unsupported operator/collective gaps remain theoretical. No global fitted latency multiplier is applied.

The previously suggested 5–8% error was a hypothesis. The table above, and only validated coverage in the artifacts, state what this run achieved. Partial coverage must not be described as a fully measured model.
