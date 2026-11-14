# Cache timing fixture on B300

Only the warm-cache regression fixture changed; `kernel_bench/timing.py` and the required cold/hot ratio1.1 are unchanged. The fixture now uses exactly3/4 of the runtime-reported L2 capacity, leaving room for the reduction scratch/output. Missing or nonpositive capacity skips with an explicit explanation; no SKU table or timing-selected size is used.

GPU7 was exclusive. Environment: Torch2.11.0+cu130, CUDA13.0, NVIDIA B300 SXM6 AC. The CUDA property reports132644864bytes (126.5MiB). The two preselected inputs were16MiB and99483648bytes (94.875MiB=3/4L2), fp32 `torch.amax` with scalar output. Each independent process used a different seed, standard15warmup/60samples and both ordinary timing modes. The input-size order was reversed in repeat2; no samples or repeats were dropped.

| Process |16MiB hot/cold us|Cold/hot|3/4L2 hot/cold us|Cold/hot|
|---|---:|---:|---:|---:|
|1|14.336/15.232|1.0625|20.352/27.360|1.3443|
|2|14.368/14.976|1.0423|20.544/28.832|1.4034|
|3|14.368/16.224|1.1292|20.384/27.552|1.3516|

A16MiB reduction occupies only12.65% of this L2 and has a small, unstable cold/hot difference. The fixed larger cache-resident working set exposes a repeatable memory-hierarchy difference with margin above1.1. This supports changing the fixture; it does not prove an exact decomposition of reduction/launch overhead, and the third16MiB repeat did pass the old threshold.

The updated whole `tests/kernel_bench/test_timing.py` passed3/3 in3.18s on GPU7. Raw per-run JSON/logs, reproduction script `probe.py`, combined `results.json`, and `pytest.log` are in this directory. GPU7 was released with4MiB used and0% utilization; `gpu7-release.txt` records the final query.

```bash
source /tmp/scratch-space/k3-performance/env.sh
cd /tmp/scratch-space/k3-performance
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/HuggingArch HuggingArch/.venv313/bin/python benchmark/timing-fixture/probe.py 1
# Repeat as separate processes with argument2 and3.
```
