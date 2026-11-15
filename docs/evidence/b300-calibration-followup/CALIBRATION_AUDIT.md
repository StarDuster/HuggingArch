# Calibration audit and usable scope

Retained 41 original/library-backed rows; excluded 85 new eager reference rows from this supplementary overlay. The full raw corpus remains unchanged. Selection is based on implementation provenance, never closeness to the serving result.

| Input | Library-kernel partial estimate ms | Historical TPOT ms | Error |
|---:|---:|---:|---:|
| 2048 | 12.1186 | 9.9448 | +21.86% |
| 16384 | 12.1378 | 10.0810 | +20.40% |

The first broad overlay predicted 74.34 ms. Correcting the source-version eligibility and measuring actual SGLang kernels reduced it to 47.32 ms. The remaining broad-overlay leaf costs include linear 12.876 ms, eager AttnRes 9.390 ms and eager router 7.834 ms; these are real timings of those providers, not evidence that SGLang pays them.

The supplementary library overlay is partial, not an exact dispatch reconstruction. Unsupported MXFP4 expert kernels and TP8 collectives still use theory, AttnRes needs a weighted combine-plus-norm fusion contract, and cold isolated timings do not include serving overlap. A low aggregate error can result from cancellation and does not validate missing components. No 5–8% accuracy claim is supported by the available evidence.

Use library_kernel_calibration_rows.json when inspecting the library-only partial overlay; calibration_rows.json retains the complete cross-library corpus for audit and should not be represented as production-specific SGLang calibration.
