"""Product defaults for the inference pages, declared once.

Every value here used to live in a frontend file, where it could only
drift or go stale without anyone noticing. None of these are derivable
facts (a derivable one belongs in the engine); they are product
conventions, and each says where the convention comes from. The
`/inference/defaults` endpoint ships this module's declarations.
"""
from __future__ import annotations

from typing import Dict

from backend.inference.gpu_specs import get_gpu


def _per_sku(seeds: Dict[str, int]) -> Dict[str, int]:
    """`seeds`, each key resolved against `gpu_specs.csv`."""
    for sku in seeds:
        if get_gpu(sku) is None:
            raise ValueError(
                f"a per-SKU default names {sku!r}, which is not a row of "
                f"gpu_specs.csv")
    return seeds


# vLLM's `gpu_memory_utilization` default. The fraction of HBM the
# deployment may budget for weights + KV; the rest is CUDA context,
# allocator pools and activation headroom.
GPU_MEMORY_FRACTION_DEFAULT = 0.9

# DP-EP degree seeded for MoE deployments, per GPU SKU. The degree is one
# compute TRAY of the high-speed fabric: a GB200/GB300 tray is 4 GPUs
# (NVL72 = 18 trays × 4 — `gpu_specs.csv:gpus_per_node=72` is the whole
# NVL72 domain, not the tray); an SXM node is 8. SKUs not listed use 8.
# H100-SXM seeds 16 — two SXM nodes — so the Inference page's landing
# deployment (DeepSeek-V3) opens at the DP16×EP16 wide-EP shape.
# The degree is the convention; that each key names a card is the sheet's
# to say, so the seeds are resolved against `gpu_specs.csv` at import — a
# SKU renamed there would otherwise fall back to 8 with nothing said.
MOE_DP_EP_DEGREE: Dict[str, int] = _per_sku({"GB200": 4, "GB300": 4,
                                             "H100-SXM": 16})
MOE_DP_EP_DEGREE_DEFAULT = 8


def moe_dp_ep_degree(gpu_sku: str) -> int:
    return MOE_DP_EP_DEGREE.get(gpu_sku, MOE_DP_EP_DEGREE_DEFAULT)

# One link traversal's latency, in µs, the same on every link — NVLink,
# PCIe, InfiniBand or RoCE — at this granularity. A collective pays it once
# per traversal its algorithm waits on in sequence (`comm_volume.ALGORITHMS`:
# a one-shot all-reduce 1, a two-shot or NVLS one 2 — what a framework runs
# inside an NVLink domain, `comm_cost.collective_ladder` — and NCCL's
# schedule elsewhere, an all-reduce of N ranks 2 log2 N, a reduce-scatter,
# all-gather or all-to-all log2 N, per phase of a hierarchical schedule;
# `comm_cost.collective_latency`). An estimate until Kernel Bench measures
# collectives; the request may set its own (`comm_latency_us`).
COMM_LATENCY_US_DEFAULT = 2.0

# The shortest one kernel launch takes, however little it does, in µs: the
# fourth term of the row roofline (`row_roofline`, `t_launch`). An estimate
# until Kernel Bench measures it; the request may set its own
# (`launch_floor_us`).
LAUNCH_FLOOR_US_DEFAULT = 1.0
