"""What the device ACTUALLY ran — layer 4's evidence.

The first three guarantee layers prove a declaration is well formed, that its
identity is unique, and that its numbers agree with a reference. None of them
can tell `declared A, ran A` from `declared A, ran B` when B computes the same
thing: FlashKDA takes over `chunk_kda` on SM90+ unless `FLA_FLASH_KDA=0`, and
vLLM's CustomOp runs its native torch path unless `custom_ops=all`. Both
produce correct numbers under the wrong kernel's name, which is the one failure
a numeric check cannot see.

MEASURED FIRST, THEN DESIGNED. `ARCHITECTURE.md` described this layer as
comparing the symbol against a declared one. That is not possible: one
`kernel_impl`, three shapes, three different kernels:

    64x256x256      void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s16…>
    4096x4096x14336 ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
    512x2048x2048   ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn

cuBLAS picks per shape, and not merely a different tile — a different kernel
family, so even the `cutlass::`/`ampere_` provenance token moves. A declared
pattern would be either loose enough to catch nothing or wrong on every other
cell. The observation is present for every runnable provider and stable across
repeats.

So this records rather than declares, and the comparisons it supports are the
ones that need no expectation:

  * two kernels of one signature that launch the SAME sequence on the same cell
    are one kernel under two names, not two measurements (asserted by smoke);
  * the same (kernel_impl, cell, gpu, version) changing its `kernel_symbol`
    between runs is a substitution, visible by comparing rows.

A stitched torch reference launches a whole program — `torch_rope` runs 18
kernels — so the unit is a SEQUENCE. The column carries a digest of the whole
sequence (any change anywhere moves it) prefixed onto the name of the kernel
that consumed the most device time (so a human reading a row can see whose
kernel it was). Both halves, one column, because they are one fact.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Tuple

# The longest-running kernel's name is truncated to keep a CSV row readable.
# Wide enough for the informative part of the mangled forms seen so far —
# `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_
# 128x1_tn_align8>` is 91 characters.
_NAME_BUDGET = 96


class SymbolsUnavailable(RuntimeError):
    """Raised when the profiler produced no CUDA kernel at all.

    An empty observation is reported, never recorded as one: a blank
    `kernel_symbol` on a row that did run would read as "nothing to see"
    exactly where the evidence was supposed to be.
    """


def observe(call: Callable[[], Any]) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    """The CUDA kernels one `call` launches, in order, and their device time.

    `torch.profiler` rather than a stack trace, because a stack trace is the
    thing that lies here: an async CUDA error surfaces at the NEXT API call, so
    the stack names whatever the host was doing rather than what ran. A launch
    record is the observation; the stack is a guess.

    Costs ~3ms, so callers run it ONCE per case and outside the timed region —
    profiling serializes the CUDA stream and would inflate every latency it
    touched.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    call()                      # warm: first-call JIT/autotune is not the kernel
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()

    seq: List[str] = []
    spent: Dict[str, float] = {}
    for ev in prof.events():
        if not str(getattr(ev, "device_type", "")).endswith("CUDA"):
            continue
        if getattr(ev, "device_index", -1) < 0:
            continue
        name = str(ev.key or "")
        if not name:
            continue
        seq.append(name)
        spent[name] = spent.get(name, 0.0) + float(
            getattr(ev, "self_device_time_total", 0.0) or 0.0)
    if not seq:
        raise SymbolsUnavailable(
            "the profiler recorded no CUDA kernel for this call — either it ran "
            "entirely on the host or the build has no CUPTI")
    return tuple(seq), spent


def describe(call: Callable[[], Any]) -> str:
    """The `kernel_symbol` column value for one measured case.

    `<digest>:<name>` — the digest covers the whole launch sequence in order,
    so a substitution anywhere moves it; the name is the kernel that consumed
    the most device time, so the row says whose kernel it was without a lookup.
    """
    seq, spent = observe(call)
    top = max(spent, key=lambda k: (spent[k], k)) if any(spent.values()) else seq[0]
    return f"{digest(seq)}:{top[:_NAME_BUDGET]}"


def digest(symbols) -> str:
    """A short stable id for a launch sequence.

    Over the sequence, not the set: a kernel that runs its inner loop a
    different number of times is a different program, and collapsing the
    repeats would hide it.
    """
    h = hashlib.sha256()
    for s in symbols:
        h.update(s.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]
