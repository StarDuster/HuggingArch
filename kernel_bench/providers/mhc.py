"""The hyper-connection sublayer boundary — the "mHC seam", one provider set
per launch.

A manifold-hyper-connection model keeps `n_hidden_copies` parallel residual
copies and mixes them at every sublayer boundary. Both frameworks run that
boundary as two or three TileLang launches through one wrapper, and the cost
model prices each launch as its own span (`fusion_plans/mhc_*.yaml`). These
providers call each launch the way serving does, at the audited releases
(vLLM v0.32.0, sglang v0.5.22):

  mhc_post            the post-mix of a finished sublayer
    vLLM    model_executor/kernels/mhc/tilelang.py:791 `mhc_post_tilelang`
            (kernel `_MHC_POST_TILELANG_KERNEL`, tilelang_kernels.py:1584) —
            the standalone launch and the large-token first launch of
            `mhc_fused_post_pre_tilelang` (:944).
    sglang  kernels/ops/layernorm/mhc.py:1360 `mhc_post`, the thin wrapper
            around `mhc_post_tilelang` (:1311), as :1759 calls it.

  mhc_pre_gemm        the pre-mix projection
    vLLM    tilelang.py:22 `_hc_prenorm_gemm_outputs`, which takes DeepGEMM
            `tf32_hc_prenorm_gemm` where DeepGEMM is available and
            `_HC_PRENORM_GEMM_TILELANG_KERNEL` (tilelang_kernels.py:1117)
            otherwise. Called at :953.
    sglang  `_mhc_pre_gemm_sqrsum_dispatch()` (mhc.py:708 → the TileLang
            kernel `mhc_pre_gemm_sqrsum_tilelang`, :641), the launch
            `mhc_fused_post_pre` makes at :1788.
    deepgemm `tf32_hc_prenorm_gemm` itself — the kernel BOTH frameworks
            prefer on the default serving path, and its own measurement the
            way a marlin row is its own beside a vllm-native one.

  mhc_pre_big_fuse    the pre-mix epilogue + the sublayer norm
    vLLM    `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` (tilelang_kernels.py:1264),
            as tilelang.py:959 calls it.
    sglang  `mhc_pre_big_fuse_with_norm_tilelang` (mhc.py:936), as :1836
            calls it.

  mhc_fused_post_pre  a post-mix and the next projection in ONE launch
    vLLM    `_MHC_FUSED_TILELANG_KERNEL` (tilelang_kernels.py:1660), as
            tilelang.py:915 calls it on the small-token path.
    sglang  `mhc_fused_post_pre_fma_tilelang` (mhc.py:1391), as :1739 calls
            it below its `fma_token_threshold`.

  mhc_mega_post_pre   DeepInfer's shifted V4.1 seam in ONE DeepGEMM launch
            (`deep_gemm.mega_mhc`, HCLayerMixin._mega_mhc): the previous
            post, the current projection, and the carried-pre reduce +
            sublayer norm. The graph leaves the coefficient split billed;
            the provider exposes the projection operand that split reads.

DeepSeek-V4.1 runs the same seam on a shifted schedule — the coefficients a
sublayer computes are consumed by the NEXT one, so its epilogue collapses
the copies with a CARRIED pre-mix. Three of its launches are V4 kernels
(`_MHC_POST_TILELANG_KERNEL`, `_MHC_FUSED_TILELANG_KERNEL`, the prenorm
GEMM) and are measured by the arms above; two are its own and get their own
arms UNDER THE SAME SPANS, because the span's operands and bytes are the
V4 ones and only the kernel differs:

  vllm_mhc_pre_norm     the delayed epilogue, `MHC_PRE_NORM_KERNEL`
            (vLLM kernels/mhc/warmup.py:101 —
            `mhc_pre_big_fuse_with_norm_tilelang`,
            tilelang_kernels.py:299, with `use_pre_mix_in` /
            `save_pre_mix`, :322-323), as tilelang.py:445 calls it inside
            `mhc_fused_post_pre_delayed_tilelang`. It reads the carried
            `[tokens, hc]` pre-mix and writes the next one beside the
            span's three declared outputs; the carry is derived here from
            the same operands the oracle splits, so the cross-check stays
            one function.
  sglang_hc_combine_norm  sglang's collapse + sublayer norm,
            `hc_combine_norm` (kernels/ops/layernorm/hc_combine_norm.py:55),
            from `_combine` (srt/models/deepseek_v4_mhc.py:394). The kernel
            is compiled for V4.1's exact geometry (4 copies x 5120), so the
            arm declares it. sglang's stats launch
            (`hc_mix_stats_sinkhorn*`) computes the coefficients the V4
            epilogue derives in-launch — the plan leaves that split a billed
            node, and this arm canonicalises the launch's one write beside
            the coefficients derived from the same operands.

Both frameworks write the pre-mix projection as split-K partials plus a
per-token sum of squares, and how many splits is each kernel's own dispatch
(`compute_num_split` / `_compute_num_split_for_mhc_pre`). `to_canonical`
therefore reduces the partials and applies the rsqrt, which is the span's
declared output (`refs.hc_normalised_mixes`) — one form every provider's
output lands in, whatever it split the reduction into.

Import-guarded: library imports are lazy inside `build`.
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register
from .attention import _require_cuda


def _seam(inputs: Dict[str, Any]):
    """The seam's geometry off the residual every one of these launches
    carries: tokens, copies, model width."""
    t, hc, d = inputs["residual"].shape
    return int(t), int(hc), int(d)


def _epsilons(inputs: Dict[str, Any]):
    """The five scalars the pre-mix launches take beside their operands, in
    the order both frameworks' wrappers list them (`refs._hc_scalars`)."""
    return (float(inputs["rms_eps"]), float(inputs["hc_pre_eps"]),
            float(inputs["hc_sinkhorn_eps"]),
            float(inputs["hc_post_mult_value"]),
            int(inputs["sinkhorn_repeat"]))


def _normalised(mixes, sqrsum, reduce_width: int, rms_eps: float):
    """Split-K partials reduced and scaled by the residual's RMS — the
    pre-mix projection span's declared output."""
    import torch
    m = mixes.float()
    s = sqrsum.float()
    if m.dim() == 3:                     # [n_splits, tokens, mix]
        m, s = m.sum(0), s.sum(0)
    return m * torch.rsqrt(s.unsqueeze(-1) / reduce_width + rms_eps)


def _big_fuse_canonical(layer_input, post, comb):
    """The epilogue's three writes side by side, as the oracle states them."""
    import torch
    return torch.cat([layer_input.float(), post.float(),
                      comb.float().flatten(1)], dim=-1)


def _fused_canonical(mixes, sqrsum, residual_out, reduce_width, rms_eps):
    """The fused span's two ports side by side, as the oracle states them."""
    import torch
    return torch.cat([_normalised(mixes, sqrsum, reduce_width, rms_eps),
                      residual_out.float().flatten(1)], dim=-1)


def _torch_floor_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                       ) -> BenchCase:
    """The span's naive chain, eager and AT THE CELL'S PRECISION — what the
    fused launch is measured against.

    Not the oracle's routine: that one computes in fp32 on purpose, to be a
    trusted reference, and a floor timed in fp32 reports another precision's
    latency (the driver refuses it — "the launch's largest operand is
    float32 where the cell declares bfloat16"). The arithmetic is the same
    chain either way; what differs is the width it runs at, so the floor
    casts its operands the way every other torch floor here does.

    THE COEFFICIENTS ARE CAST TOO, and that is a real difference from the
    kernels rather than a tidy invariant. Both frameworks hold `post`, `comb`
    and the mix projection in fp32 at runtime and assert it, so a faithful
    naive chain would keep them fp32 — but in eager torch an fp32 coefficient
    against a bf16 residual PROMOTES the residual, and a full-width fp32
    operand is exactly what the driver refuses ("the launch's largest operand
    is float32 where the cell declares bfloat16"). There is no eager spelling
    that keeps the coefficients fp32 and the big operand bf16 in one op, so
    the floor casts them down (`:comb`/`post` and `fn` below).

    What that costs, stated so a ratio read off this floor can be discounted:
    the projection's floor runs a bf16×bf16 tensor-core GEMM where both real
    kernels compute at fp32/TF32 width (the TileLang fallback converts in
    registers; DeepGEMM's op IS `tf32_hc_prenorm_gemm`). The coefficients are
    the small operands — `fn` is 24×hc·d ≈ 0.69M elements against a 117M
    residual — so the BYTE effect is under a percent, but the floor is a
    slightly cheaper chain than the naive one the span prices. ONE DIRECTION
    follows from that, not two: a saving is read as floor / kernel, the floor
    is the numerator, and a numerator that is too small makes the ratio too
    small. So every speedup measured against this floor is CONSERVATIVE —
    the real naive chain would be slower and the span's saving is at least
    what is reported. The norm reductions do stay fp32
    (`mean(..., dtype=torch.float32)`), which costs no full-width temporary.
    """
    import torch
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype)
    sig = inv.signature

    def post_mix():
        """`hc_mix` then the sublayer spread — the two naive members."""
        comb = inputs["comb"].to(dt)
        residual, sublayer = inputs["residual"].to(dt), inputs["sublayer"]
        mixed = torch.einsum("tij,tih->tjh", comb, residual)
        return mixed + inputs["post"].to(dt).unsqueeze(-1) * \
            sublayer.to(dt).unsqueeze(-2)

    def projection(residual):
        """`rms_norm` then the mix GEMM."""
        flat = residual.reshape(residual.shape[0], -1)
        # fp32 accumulation, no fp32 copy of the whole row (see rope.py).
        scale = torch.rsqrt(
            flat.square().mean(-1, keepdim=True, dtype=torch.float32)
            + float(inputs["rms_eps"]))
        return (flat * scale.to(dt)) @ inputs["fn"].to(dt).t()

    if sig == "mhc_post":
        return BenchCase(run=post_mix, to_canonical=lambda o: o.float())
    if sig == "mhc_pre_gemm":
        return BenchCase(run=lambda: projection(inputs["residual"].to(dt)),
                         to_canonical=lambda o: o.float())
    if sig == "mhc_fused_post_pre":
        def fused():
            y = post_mix()
            return projection(y), y
        return BenchCase(run=fused, to_canonical=lambda o: torch.cat(
            [o[0].float(), o[1].float().flatten(1)], dim=-1))

    if sig == "mhc_mega_post_pre":
        def mega_mhc():
            """The Mega-MHC span's naive chain at the cell's width."""
            y = post_mix()
            mixes = projection(y)
            reduced = (inputs["pre"].to(dt).unsqueeze(-1) * y).sum(dim=1)
            normed = reduced * torch.rsqrt(
                reduced.square().mean(-1, keepdim=True, dtype=torch.float32)
                + float(inputs["norm_eps"])).to(dt)
            return (normed * inputs["norm_weight"].to(dt),
                    mixes, y)
        return BenchCase(run=mega_mhc, to_canonical=lambda o: torch.cat(
            [o[0].float(), o[1].float(), o[2].float().flatten(1)], dim=-1))

    # mhc_pre_big_fuse: the gates are fp32 by construction (sigmoids and a
    # softmax over a 24-wide vector); the reduce over the copies and the
    # sublayer norm are the chain's big operands and run at the cell's width.
    from ..refs import hc_mix_split
    hc, d = int(inputs["residual"].shape[1]), int(inputs["residual"].shape[2])

    def big_fuse():
        pre, post, comb = hc_mix_split(
            inputs["mixes"], inputs["sqrsum"], inputs["scale"],
            inputs["base"], hc, hc * d, inputs)
        residual = inputs["residual"].to(dt)
        reduced = (pre.to(dt).unsqueeze(-1) * residual).sum(dim=1)
        normed = reduced * torch.rsqrt(
            reduced.square().mean(-1, keepdim=True, dtype=torch.float32)
            + float(inputs["norm_eps"])).to(dt)
        return normed * inputs["norm_weight"].to(dt), post, comb

    return BenchCase(run=big_fuse, to_canonical=lambda o: torch.cat(
        [o[0].float(), o[1].float(), o[2].float().flatten(1)], dim=-1))


# --- mhc_post --------------------------------------------------------------

def _vllm_post_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                     ) -> BenchCase:
    _require_cuda()
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    # The wrapper takes `post` with a trailing singleton and squeezes it for
    # the kernel; the canonical tensor is [tokens, copies].
    post = inputs["post"].unsqueeze(-1)
    x, residual, comb = inputs["sublayer"], inputs["residual"], inputs["comb"]
    return BenchCase(
        run=lambda: mhc_post_tilelang(x, residual, post, comb),
        to_canonical=lambda o: o.float())


def _sgl_serving_context():
    """sglang's runtime config for the deployment the bindings describe —
    one rank, no context parallelism — as a context the call runs inside.

    sglang reads its parallel layout through process-wide config that a
    server publishes at start-up (`srt/runtime_context.py`), and a kernel
    wrapper may consult it: `mhc_post` (kernels/ops/layernorm/mhc.py:1366
    at v0.5.22) asks `is_dsa_prefill_cp_interleave()`, which reads
    `get_parallel().attn_cp_size` and raises "config namespace 'parallel'
    not published" in a process that never served — the error every
    `sglang_mhc_post` canary died of on the H100. `create_context` builds
    the config from default `ServerArgs` WITHOUT replacing the process's
    own, and `use_context` scopes it to the call, the way sglang's own test
    helper (`published_topology`, test/test_utils.py) does. Built once per
    process: it is the same default every call."""
    global _SGL_CONTEXT
    if _SGL_CONTEXT is None:
        from sglang.srt.runtime_context import SpawnRanks, create_context
        from sglang.srt.server_args import ServerArgs
        _SGL_CONTEXT = create_context(
            ServerArgs(model_path="dummy"), role="scheduler",
            ranks=SpawnRanks(world_rank=0, dp_rank=None))
    return _SGL_CONTEXT


_SGL_CONTEXT = None


def _sgl_post_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                    ) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.layernorm.mhc import mhc_post
    from sglang.srt.runtime_context import use_context
    context = _sgl_serving_context()
    post = inputs["post"].unsqueeze(-1)
    x, residual, comb = inputs["sublayer"], inputs["residual"], inputs["comb"]

    def run():
        with use_context(context):
            return mhc_post(x, residual, post, comb)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


# --- mhc_pre_gemm ----------------------------------------------------------

# The projection has TWO physical kernels behind one wrapper, and which one
# runs is `_hc_prenorm_gemm_outputs`'s own choice by package availability
# (`use_deep_gemm = is_deep_gemm_supported() or not use_tilelang_fallback`,
# vLLM tilelang.py:35-57). One provider through that wrapper would record one
# `kernel_impl` for either kernel, so rows from a box with DeepGEMM and a box
# without would be the same identity and silently incomparable. They are two
# providers instead, each naming the kernel it runs.
#
# SPLIT-K IS PART OF THE LAUNCH, not a reduction detail. The wrapper derives
# it from the card's SM count and the shape (`compute_num_split`,
# tilelang_kernels.py:45-54): on a 132-SM H100 that is 2 splits at 4096
# prefill tokens and ~112 at decode token counts. Measuring a 1-token decode
# cell at one split serialises a K = hc·d ≈ 28672 reduction that serving
# spreads over ~100 blocks — a different, under-parallel launch. So the
# DeepGEMM arm goes THROUGH the wrapper, which applies that function itself;
# the count is never computed here, because a copy of a framework's dispatch
# is the next thing to drift from it.

def _vllm_pre_gemm_deepgemm_build(inv: Invocation, env: Env,
                                  inputs: Dict[str, Any]) -> BenchCase:
    """The DeepGEMM projection as serving reaches it: through vLLM's wrapper,
    at the split count `compute_num_split` chooses for this card and shape.

    `use_tilelang_fallback=False` forces the DeepGEMM branch
    (`use_deep_gemm = is_deep_gemm_supported() or not use_tilelang_fallback`),
    so this measures that kernel even where the probe would have taken the
    other one — and still at the served schedule, because the wrapper sizes
    the split buffers before dispatching.

    It needs vLLM importable, not just `deep_gemm`: for this span DeepGEMM is
    not a library a deployment selects, it is what vLLM picks inside this
    wrapper, and calling it bare means inventing the schedule the wrapper
    would have chosen.
    """
    _require_cuda()
    from vllm.model_executor.kernels.mhc.tilelang import (
        _hc_prenorm_gemm_outputs,
    )
    t, hc, d = _seam(inputs)
    x = inputs["residual"].view(t, hc * d)
    fn, rms_eps = inputs["fn"], float(inputs["rms_eps"])
    return BenchCase(
        run=lambda: _hc_prenorm_gemm_outputs(x, fn, hidden_size=d, hc_mult=hc,
                                             use_tilelang_fallback=False),
        to_canonical=lambda o: _normalised(o[0], o[1], hc * d, rms_eps))


def _vllm_pre_gemm_tilelang_build(inv: Invocation, env: Env,
                                  inputs: Dict[str, Any]) -> BenchCase:
    """The TileLang projection, the kernel the wrapper takes where DeepGEMM is
    absent — at ONE split, which is what the wrapper itself allocates on that
    branch (`n_splits = compute_num_split(...) if use_deep_gemm else 1`,
    tilelang.py:38-42). So one split here is the served schedule rather than
    an invented one, and the buffers are shaped `[1, tokens, mix]` as the
    wrapper shapes them.
    """
    _require_cuda()
    import torch
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_PRENORM_GEMM_TILELANG_KERNEL,
    )
    t, hc, d = _seam(inputs)
    x = inputs["residual"].view(t, hc * d)
    fn, rms_eps = inputs["fn"], float(inputs["rms_eps"])
    mix = int(fn.shape[0])
    mixes = torch.empty(1, t, mix, dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(1, t, dtype=torch.float32, device=x.device)
    return BenchCase(
        run=lambda: _HC_PRENORM_GEMM_TILELANG_KERNEL(x, fn, mixes, sqrsum,
                                                     d, hc),
        to_canonical=lambda _o: _normalised(mixes, sqrsum, hc * d, rms_eps))


def _sgl_pre_gemm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                        ) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.layernorm.mhc import _mhc_pre_gemm_sqrsum_dispatch
    t, hc, d = _seam(inputs)
    x = inputs["residual"].view(t, hc * d)
    fn, rms_eps = inputs["fn"], float(inputs["rms_eps"])
    mix = int(fn.shape[0])
    # The kernel writes into buffers the caller allocates (mhc.py:1788).
    mixes = torch.empty(t, mix, dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(t, dtype=torch.float32, device=x.device)
    kernel = _mhc_pre_gemm_sqrsum_dispatch()
    return BenchCase(
        run=lambda: kernel(x, fn, mixes, sqrsum, mix, hc * d),
        to_canonical=lambda _o: _normalised(mixes, sqrsum, hc * d, rms_eps))


# There is deliberately NO bare `deep_gemm` arm calling
# `tf32_hc_prenorm_gemm` directly. One existed and it passed `num_split=1`,
# which is not the schedule any framework launches — 2 at 4096 prefill
# tokens, ~112 at decode counts — so its rows timed an under-parallel
# reduction and the speedup read off them was a ratio between two different
# launches. DeepGEMM for this span is reachable only as vLLM's own choice
# inside `_hc_prenorm_gemm_outputs`, so it is measured there
# (`_vllm_pre_gemm_deepgemm_build`) and nowhere else. Measuring it bare would
# mean this harness choosing the split count, i.e. holding a second copy of
# the framework's dispatch.


# --- mhc_pre_big_fuse ------------------------------------------------------

def _big_fuse_buffers(inputs: Dict[str, Any]):
    """The three outputs the epilogue writes, allocated as both frameworks'
    wrappers allocate them."""
    import torch
    t, hc, d = _seam(inputs)
    dev = inputs["residual"].device
    return (torch.empty(t, hc, dtype=torch.float32, device=dev),
            torch.empty(t, hc * hc, dtype=torch.float32, device=dev),
            torch.empty(t, d, dtype=inputs["residual"].dtype, device=dev))


def _vllm_big_fuse_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                         ) -> BenchCase:
    _require_cuda()
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
    )
    t, hc, _d = _seam(inputs)
    post, comb, layer_input = _big_fuse_buffers(inputs)
    # The kernel reads the projection's split-K partials; one split is what
    # the TileLang pre-GEMM path writes (tilelang.py:40).
    mixes = inputs["mixes"].unsqueeze(0)
    sqrsum = inputs["sqrsum"].unsqueeze(0)
    args = (mixes, sqrsum, inputs["scale"], inputs["base"],
            inputs["residual"], post, comb, layer_input, *_epsilons(inputs))
    # bf16 + contiguous, as the WRAPPER this call skips does
    # (tilelang.py:880-885) and as the sglang arm below does explicitly. The
    # raw kernel's buffer is bf16-typed, so handing it the weight slot's own
    # dtype would make the gate wider than the build: an fp32-gain cell would
    # pass `serves` and then hand fp32 to a bf16 buffer.
    norm_weight = inputs["norm_weight"].bfloat16().contiguous()
    norm_eps = float(inputs["norm_eps"])
    return BenchCase(
        run=lambda: _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
            *args, norm_weight=norm_weight, norm_eps=norm_eps),
        to_canonical=lambda _o: _big_fuse_canonical(
            layer_input, post, comb.view(t, hc, hc)))


def _sgl_big_fuse_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                        ) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.layernorm.mhc import (
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    t, hc, d = _seam(inputs)
    post, comb, layer_input = _big_fuse_buffers(inputs)
    mixes = inputs["mixes"].unsqueeze(0)
    sqrsum = inputs["sqrsum"].unsqueeze(0)
    mix = int(inputs["base"].shape[0])
    # Positionally, as mhc.py:1836 calls it: operands, the three outputs, the
    # norm gain in bf16, then the widths, the epsilons and the split counts.
    args = (mixes, sqrsum, inputs["scale"], inputs["base"],
            inputs["residual"], post, comb, layer_input,
            inputs["norm_weight"].bfloat16().contiguous(), d,
            *_epsilons(inputs), float(inputs["norm_eps"]), 1, hc, mix)
    return BenchCase(
        run=lambda: mhc_pre_big_fuse_with_norm_tilelang(*args),
        to_canonical=lambda _o: _big_fuse_canonical(
            layer_input, post, comb.view(t, hc, hc)))


# --- the V4.1 realizations of the epilogue span -----------------------------

def _carried_coefficients(inputs: Dict[str, Any]):
    """The `(pre, post, comb)` the span's oracle derives from the projection's
    writes — the form V4.1's delayed schedule CARRIES the `pre` in, computed
    here from the same operands so provider and oracle read one function."""
    from ..refs import hc_mix_split
    hc, d = int(inputs["residual"].shape[1]), int(inputs["residual"].shape[2])
    return hc_mix_split(inputs["mixes"], inputs["sqrsum"], inputs["scale"],
                        inputs["base"], hc, hc * d, inputs)


def _vllm_pre_norm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                         ) -> BenchCase:
    """DeepSeek-V4.1's delayed epilogue, called as `mhc_pre_delayed_tilelang`
    calls it (tilelang.py:186-210): one split, the carried pre-mix in and the
    next one out beside the span's three writes. The carried pre is the
    activated gate the previous seam stored; a one-cell measurement has no
    previous seam, so it is the gate these same operands derive
    (`_carried_coefficients`) — which is also what makes the oracle's
    collapse the same value.
    """
    _require_cuda()
    import torch
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL
    t, hc, d = _seam(inputs)
    post, comb, layer_input = _big_fuse_buffers(inputs)
    pre_mix, _p, _c = _carried_coefficients(inputs)
    pre_mix = pre_mix.contiguous()
    next_pre = torch.empty(t, hc, dtype=torch.float32,
                           device=inputs["residual"].device)
    mixes = inputs["mixes"].unsqueeze(0)
    sqrsum = inputs["sqrsum"].unsqueeze(0)
    norm_weight = inputs["norm_weight"].bfloat16().contiguous()
    args = (mixes, sqrsum, inputs["scale"], inputs["base"],
            inputs["residual"], post, comb, layer_input, norm_weight,
            pre_mix, next_pre, layer_input)
    return BenchCase(
        run=lambda: MHC_PRE_NORM_KERNEL(
            *args, hidden_size=d, rms_eps=float(inputs["rms_eps"]),
            hc_pre_eps=float(inputs["hc_pre_eps"]),
            hc_sinkhorn_eps=float(inputs["hc_sinkhorn_eps"]),
            hc_post_mult_value=float(inputs["hc_post_mult_value"]),
            sinkhorn_repeat=int(inputs["sinkhorn_repeat"]),
            norm_eps=float(inputs["norm_eps"]), hc_mult=hc,
            use_pre_mix_in=True, save_pre_mix=True, rms_numel=hc * d),
        to_canonical=lambda _o: _big_fuse_canonical(
            layer_input, post, comb.view(t, hc, hc)))


def _sgl_combine_norm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                            ) -> BenchCase:
    """sglang's collapse + sublayer norm for the shifted schedule. The launch
    writes only the normed input; the span's two coefficient ports are
    derived from the same operands beside it, so the canonical form is the
    span's.
    """
    _require_cuda()
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    t, hc, d = _seam(inputs)
    pre, post, comb = _carried_coefficients(inputs)
    x = inputs["residual"].view(t, hc * d)
    weight = inputs["norm_weight"].bfloat16().contiguous()
    eps = float(inputs["norm_eps"])
    return BenchCase(
        run=lambda: hc_combine_norm(x, pre.contiguous(), weight, eps),
        to_canonical=lambda y: _big_fuse_canonical(y, post, comb))


# --- mhc_fused_post_pre ----------------------------------------------------

def _vllm_fused_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                      ) -> BenchCase:
    _require_cuda()
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
    )
    _tokens, hc, d = _seam(inputs)
    fn, rms_eps = inputs["fn"], float(inputs["rms_eps"])
    mix = int(fn.shape[0])
    args = (inputs["comb"], inputs["residual"], inputs["post"],
            inputs["sublayer"], fn.view(mix, hc, d), hc, d, mix)
    return BenchCase(
        run=lambda: _MHC_FUSED_TILELANG_KERNEL(*args),
        to_canonical=lambda o: _fused_canonical(o[0], o[1], o[2],
                                                hc * d, rms_eps))


def _sgl_fused_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                     ) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.layernorm.mhc import (
        mhc_fused_post_pre_fma_tilelang,
    )
    t, hc, d = _seam(inputs)
    fn, rms_eps = inputs["fn"], float(inputs["rms_eps"])
    mix = int(fn.shape[0])
    dev = fn.device
    # Its own split-K and mix tiling, as mhc.py:1718 sets them for a
    # below-threshold step.
    tile_n = 2 if t < 8 else 3
    n_splits = 8 if (t < 8 and d <= 4096) else 4
    mixes = torch.empty(n_splits, t, mix, dtype=torch.float32, device=dev)
    sqrsum = torch.empty(n_splits, t, dtype=torch.float32, device=dev)
    residual_out = torch.empty_like(inputs["residual"])
    args = (inputs["comb"], inputs["residual"], inputs["post"],
            inputs["sublayer"], fn.view(mix, hc, d), mixes, sqrsum,
            residual_out, hc, d, mix)
    return BenchCase(
        run=lambda: mhc_fused_post_pre_fma_tilelang(
            *args, tile_mix_outputs=tile_n, split_k=n_splits),
        to_canonical=lambda _o: _fused_canonical(
            mixes, sqrsum, residual_out, hc * d, rms_eps))


# --- registration ----------------------------------------------------------
#
# Each span gets the framework launches that realize it and a torch floor
# running the span's own naive chain; the projection has two vLLM arms,
# because that wrapper has two kernels behind it (see above).
#
# `activation=("bf16",)` on the framework arms is a CAPABILITY, not a
# dispatch condition: the audited wrappers require a bf16 residual and assert
# it (vLLM tilelang.py:854-858), so an fp16 cell is one these kernels cannot
# serve. Declared, it declines; undeclared, it was accepted and then failed
# inside the wrapper — the gate wider than the build. The torch floor carries
# no such bound: it runs the chain at whatever width the cell declares, which
# is the whole point of a floor.

for _sig, _vllm, _sgl in (
    ("mhc_post", _vllm_post_build, _sgl_post_build),
    ("mhc_pre_big_fuse", _vllm_big_fuse_build, _sgl_big_fuse_build),
    ("mhc_fused_post_pre", _vllm_fused_build, _sgl_fused_build),
):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_{_sig}",
        import_name="vllm.model_executor.kernels.mhc.tilelang",
        serves=KernelSignature(signature=_sig, activation=("bf16",)),
        build=_vllm, fused_op=_sig,
    ))
    register(Provider(
        library="sglang", signature=_sig, kernel_impl=f"sglang_{_sig}",
        import_name="sglang.kernels.ops.layernorm.mhc",
        serves=KernelSignature(signature=_sig, activation=("bf16",)),
        build=_sgl, fused_op=_sig,
    ))

# The projection: one sglang arm, and two vLLM arms naming their kernel.
register(Provider(
    library="vllm", signature="mhc_pre_gemm",
    kernel_impl="vllm_mhc_pre_gemm_deepgemm",
    import_name="vllm.model_executor.kernels.mhc.tilelang",
    serves=KernelSignature(signature="mhc_pre_gemm", activation=("bf16",)),
    build=_vllm_pre_gemm_deepgemm_build, fused_op="mhc_pre_gemm",
))
register(Provider(
    library="vllm", signature="mhc_pre_gemm",
    kernel_impl="vllm_mhc_pre_gemm_tilelang",
    import_name="vllm.model_executor.kernels.mhc.tilelang_kernels",
    serves=KernelSignature(signature="mhc_pre_gemm", activation=("bf16",)),
    build=_vllm_pre_gemm_tilelang_build, fused_op="mhc_pre_gemm",
))
register(Provider(
    library="sglang", signature="mhc_pre_gemm",
    kernel_impl="sglang_mhc_pre_gemm",
    import_name="sglang.kernels.ops.layernorm.mhc",
    serves=KernelSignature(signature="mhc_pre_gemm", activation=("bf16",)),
    build=_sgl_pre_gemm_build, fused_op="mhc_pre_gemm",
))

# DeepSeek-V4.1's realizations of the epilogue span — the same span, its own
# kernels, so each arm names what it runs (see the module docstring). The
# sglang kernel is compiled for V4.1's geometry alone (4 copies x 5120,
# hc_combine_norm.py:62-65), which the arm declares rather than asserts
# inside the build.
register(Provider(
    library="vllm", signature="mhc_pre_big_fuse",
    kernel_impl="vllm_mhc_pre_norm",
    import_name="vllm.model_executor.kernels.mhc.warmup",
    serves=KernelSignature(signature="mhc_pre_big_fuse",
                           activation=("bf16",)),
    build=_vllm_pre_norm_build, fused_op="mhc_pre_big_fuse",
))
register(Provider(
    library="sglang", signature="mhc_pre_big_fuse",
    kernel_impl="sglang_hc_combine_norm",
    import_name="sglang.kernels.ops.layernorm.hc_combine_norm",
    serves=KernelSignature(signature="mhc_pre_big_fuse",
                           activation=("bf16",),
                           enum=(("d_model", (5120,)),
                                 ("n_hidden_copies", (4,)))),
    build=_sgl_combine_norm_build, fused_op="mhc_pre_big_fuse",
))

for _sig in ("mhc_post", "mhc_pre_gemm", "mhc_pre_big_fuse",
             "mhc_fused_post_pre", "mhc_mega_post_pre"):
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch",
        # The floor casts every operand to the cell's dtype and runs plain
        # torch ops (`_torch_floor_build`), so it runs at any float width.
        serves=KernelSignature(signature=_sig, activation=TORCH_FLOAT),
        build=_torch_floor_build, fused_op=_sig,
    ))
