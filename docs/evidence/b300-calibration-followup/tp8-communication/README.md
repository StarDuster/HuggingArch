# TP8 multicast communication calibration

Actual pinned SGLang K3 all_reduce_push_res, residual=None, eight B300 GPUs. Every rank/shape passed exact comparison against Gloo FP32 sums for two independent random integer seeds (17,43).

| BF16 elements | Conservative rank-max p50 (us) |
|---:|---:|
| 3584 | 3.2139 |
| 7168 | 3.3262 |
| 28672 | 3.5512 |
| 57344 | 4.1627 |

Timing uses CUDA Graph steady state (128 calls/replay, 30 samples/rank), zero timed inputs to avoid repeated in-place overflow, and warm L2. No data-dependent shortcut is present. All original per-rank samples are retained. These are unfused reductions; do not identify them with fused all-reduce/residual/RMSNorm kernels.

`calibration_rows.json` and `measurements.csv` contain four validated exact PhysicalKeys. `../library_plus_tp8_calibration_rows.json` combines the prior 41-row library subset and these four rows; no existing physical key was remeasured.

The 16K partial serial estimate with these 45 rows is 12.4715 ms versus historical target-only 10.0810 ms (+23.71%). This adds real primitive cost but does not validate fused scheduling or overlap. Graph-warm communication and cold-drain standalone kernels have different timing regimes; the mixed overlay is explicitly a diagnostic and not a homogeneous performance model.

Two initial launch attempts failed before measurement: default lock device selection, then an incompatible one-device visibility workaround for the source P2P cache. The successful run retained the normal eight-device namespace and used GpuLock(device=local_rank). No measurement lock or P2P validation was bypassed.
