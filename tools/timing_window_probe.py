#!/usr/bin/env python3
"""Does this harness's timing window contain all of a launch's DRAM traffic?

An over-peak row is either a numerator that describes more movement than
the timed launch made, or a timing window that does not contain all of the
movement. `kernel_bench.timing.time_callable` sweeps L2 before every
sample, so READS are cold. Writes are the half that can escape: L2 is
write-back, so a launch's stores sit dirty in L2 when it ends and reach
DRAM only when something later evicts them, after an end event recorded
right behind the launch. The harness therefore sweeps L2 again inside the
window and subtracts that sweep's own cost.

So: time a pure read (fp32 amax), a pure write (fill) and a copy of the
same bytes through the harness, at sizes straddling L2, against the card's
spec peak from `inference/gpu_specs.csv`. A read and a write of n bytes move
the same DRAM traffic in opposite directions; a window that charges one of
them for traffic it does not move shows as a read/write ratio far from 1
inside L2, converging outside it.

Measured, RTX-4090 (L2 72 MiB, spec 1008 GB/s), 2026-10-25, idle GPU 0.
Old flush (memset, L2 left dirty): a 16 MiB read took 1.88x a 16 MiB write
(the read paid the flush's write-back, the write paid none of its own).
This method: read/write 0.83 at 16 MiB, 0.76 at 32 MiB, 0.98 at 512 MiB;
copy 0.86 of peak at 32 MiB, 0.91 at 512 MiB, nothing above it. One real
cell, vLLM's DeepSeek-V4 un-rotation at 208 tokens (27.3 MB): 0.60 of peak
under the old flush here, 1.47 from a clean L2 with no drain, 1.10 in the
corpus (old flush, bench card, L2 partly clean for a reason not
identified), 0.78 under this method. ~0.91 is what a memory-bound kernel
shows on this card, so an MBU above ~0.95 is already suspect.

Reads the engine's spec sheet for the peak rather than restating one:
the number a row is judged against is `peak_bw_at_test`, and that comes
from `gpu_specs.csv` (`kernel_bench/env.py`).

Run on an idle GPU — `CUDA_VISIBLE_DEVICES=0` when a sweep holds the other.
Usage: timing_window_probe.py [--mib 1,4,16,64,256,1024] [--iters 60]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_bench.timing import time_callable

def peak_for(name: str):
    """(sku, HBM GB/s) for the device torch reports, from the engine's sheet.

    The engine's own resolver disambiguates variants: an "NVIDIA H100 80GB
    HBM3" names its memory generation but not SXM, and matching only the
    family prefix read it as H100-PCIe (2000 GB/s) on an H100-SXM box.
    """
    from kernel_bench.env import _resolve_peaks
    sku, peak = _resolve_peaks(name)
    return sku, peak


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", default="1,4,16,32,64,128,256,1024")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        print("no CUDA visible", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    l2 = (getattr(props, "L2_cache_size", 0)
          or getattr(props, "l2_cache_size", 0) or 0)
    sku, peak = peak_for(props.name)
    print(f"{props.name} -> {sku}  L2 {l2 / 2**20:.0f} MiB  "
          f"spec HBM {peak and f'{peak:.0f}'} GB/s")
    print(f"{'buffer':>9} {'read':>9} {'of peak':>8} {'write':>9} "
          f"{'of peak':>8} {'copy':>9} {'of peak':>8} {'read/write':>10}"
          f"   buffer vs L2")
    sink = torch.empty((), device="cuda")
    for mib in [float(x) for x in args.mib.split(",")]:
        n = int(mib * 2**20) // 4
        src = torch.randn(n, device="cuda")
        dst = torch.empty_like(src)
        torch.cuda.synchronize()
        nbytes = 4 * n
        fits = "fits" if l2 and nbytes <= l2 else "spills"
        # The same bytes one way, the other way, and both (a copy). fp32
        # amax, not an int8 reduction: torch's int8 reduction runs at ~5%
        # of the card and would bound nothing.
        read = time_callable(lambda: torch.amax(src, dim=0, out=sink),
                             iters=args.iters).p50_us
        write = time_callable(lambda: dst.fill_(1.0), iters=args.iters).p50_us
        copy = time_callable(lambda: dst.copy_(src), iters=args.iters).p50_us

        def share(moved, us):
            if not peak:
                return "       -"
            return f"{moved / (us * 1e-6) / 1e9 / peak:8.3f}"
        print(f"{mib:8.0f}M {read:8.1f}u {share(nbytes, read)} "
              f"{write:8.1f}u {share(nbytes, write)} "
              f"{copy:8.1f}u {share(2 * nbytes, copy)} {read / write:10.2f}"
              f"   {fits} ({nbytes / max(l2, 1):.2f}x L2)")
    print("\nThese bytes are not in question, so a share above 1.000, or a "
          "read/write ratio far from 1 inside L2, is the timing window, not "
          "the numerator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
