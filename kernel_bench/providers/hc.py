"""The hyper-connection boundaries that are not the mHC seam — one provider
set per launch.

A hyper-connection model keeps `n_hidden_copies` parallel residual copies and
has to read one hidden state out of them before each sublayer and write the
sublayer's output back into all of them after. `mhc.py` covers the variant
that cross-mixes the copies with a Sinkhorn-normalised matrix; these are the
rest, and the frameworks ship each as its own launch:

  ihc_post          the sublayer spread with NO cross-mix (HY V4)
    vLLM    models/hy_v4/nvidia/triton_ihc.py:277 `triton_ihc_post`
            (kernel `_ihc_post_kernel`, :155), from hy_v4/nvidia/hc.py:195
            behind `triton_ihc_supported` (:19-27: CUDA, fp16/bf16, not
            batch-invariant).
    sglang  kernels/ops/layernorm/hy4_ihc.py:303 `fused_hy4_ihc_post`
            (kernel `_hy4_ihc_post_kernel`, :169).

  fused_hc_head     the exit-side reduce of the copies to one hidden state
    vLLM    model_executor/kernels/mhc/tilelang.py:1041
            `hc_head_fused_kernel_tilelang` → TileLang
            `hc_head_fuse_tilelang` (kernels/mhc/tilelang_kernels.py:1021),
            from models/deepseek_v4/nvidia/model.py:1516 and mtp.py:290.
    sglang  kernels/ops/layernorm/mhc_head.py:90 `fused_hc_head` (kernel
            `_hc_head_kernel`, :31), from
            srt/models/deepseek_v4.py:4053 `DeepseekV4Model.hc_head`, the
            CUDA call at :4072.

  hc_silu           the mix activation between the read gate's two projections
    vLLM    models/qwen4_exp/nvidia/ops/hc.py:461 `hc_silu` (kernel
            `_hc_silu_kernel`, :82), from nvidia/hyperconnection.py:142
            and :159.
    sglang  NO LAUNCH — the activation is a GEMM epilogue (below).

  hc_gate_mix       the READ: gated mean over the copies
    vLLM    .../ops/hc.py:465 `hc_gate_mix` (kernel `_hc_gate_mix_kernel`,
            :126), from hyperconnection.py:173 and :201.
    sglang  NO LAUNCH — fused into the mix GEMMs (below).

  hc_combine_norm   the WRITE, fused with the next read's grouped norm
    vLLM    .../ops/hc.py:480 `hc_combine_norm` (kernel
            `_hc_combine_norm_kernel`, :273), from
            hyperconnection.py:190 `combine_and_mix`.
    sglang  kernels/ops/elementwise/hc_combine_decode.py:41
            `hc_combine_apply_norm` (kernel
            `hc_combine_norm_decode_kernel`, jit/csrc/elementwise/
            hc_combine_decode.cuh:60), from
            `GatedResidual.combine_and_normalize`
            (srt/layers/hyperconnection.py:318, the call at :344) — the
            one-token decode path only.

Audited at vLLM v0.32.0 and sglang v0.5.22, the releases the bindings name.

TWO FRAMEWORKS OR ONE, said here rather than implied by what is missing.
`ihc_post` and `fused_hc_head` are bound in both, and the two kernels compute
the same function operand for operand — so both arms are registered and the
cross-check compares them.

SGLANG'S QWEN4EXP BOUNDARIES, settled by the v0.5.22 audit that bound
`hc_combine_norm` there. sglang's Qwen4Exp hyper-connection is
`GatedResidual` (srt/layers/hyperconnection.py:121, built by the model at
qwen4_exp.py:1530 / :1535), and it does not run vLLM's three launches:

* the READ is one fused chain — down projection, `silu(·/hc)`, up
  projection, sigmoid gate, gated mean over the copies. `mix()` tries
  two SM100 split-K GEMMs with the silu in the down projection's
  epilogue and the sigmoid + gated mean in the up projection's first
  (`hc_mix`, kernels/ops/elementwise/hc_mix.py:72, bf16/fp16 at 24
  rows or fewer, from hyperconnection.py:258-279), falls back to the
  persistent Triton `fused_hc_mix` (kernels/ops/gemm/hc_mix.py:182,
  kernel `_hc_mix_persistent_kernel` :27), else a torch.compile chain
  (`_mix_compute`, hyperconnection.py:198). `hc_silu` and `hc_gate_mix`
  are functions INSIDE those launches; sglang has no launch at either
  boundary, so neither is bound. That is this audit's decision, recorded
  here — not the unasked question it was before it.
* the grouped norm ahead of the read is its own launch,
  `grouped_gemma_rmsnorm` (kernels/ops/layernorm/
  grouped_gemma_rmsnorm.py:40, from hyperconnection.py:61), except where
  the previous combine fused it.
* the WRITE splits by phase and batch. Prefill, and any decode past one
  token: `hc_combine` / `hc_combine_split`
  (kernels/ops/elementwise/hc_combine.py:48 / :115, from
  hyperconnection.py:386 / :396) — the combine alone, the next read's
  norm unfused. Decode at exactly one token, bf16, SM100 (the side
  stream exists only at 4 copies of 2560 on a major-10 device,
  hyperconnection.py:186-196; the partials only for a one-token bf16
  batch, :301-315): the gate dot runs as `hc_combine_gate`
  (hc_combine_decode.py:25, prepared during the read at :312) and the
  combine + next norm as `hc_combine_apply_norm` — vLLM's
  `hc_combine_norm` boundary, with ONE operand encoded differently. The
  injection logits arrive as 8 fp32 split partials (`[rows, 8, 4]`),
  summed inside the kernel before the sigmoid
  (hc_combine_decode.cuh:80-81), where vLLM's takes one `[rows, 4]` row.
  The provider packs the span's logits into the first split: the sum the
  kernel gates on is the same value, and the launch measured — its
  reads, its grid — is the one sglang runs. The gate dot itself is a
  second launch in both frameworks (vLLM's inject linear, sglang's
  `hc_combine_gate`) and outside the span in both.

A SAME-NAMED KERNEL THAT IS NOT THIS SPAN. sglang ships
`kernels/ops/layernorm/hc_combine_norm.py:55 hc_combine_norm`, and it is
DeepSeek-V4's (`srt/models/deepseek_v4_mhc.py:441-443`), not Qwen4Exp's. It
is close to the OPPOSITE function: it REDUCES four streams to one
(`[m, 4·5120] → [m, 5120]`) under a `pre` gate and norms the result, where
this span scatters into every copy and keeps the multi-copy width
(`[T, hc·d]` in, two `[T, hc·d]` out). It also hard-asserts a 20480-wide
input. Binding it here on the strength of the name would measure a real
kernel computing a different function — the same trap as an anchor that
resolves against the wrong source tree.

The three vLLM Qwen4Exp entries go through the module-level wrappers
(`hc_silu` / `hc_gate_mix` / `hc_combine_norm`) rather than the private
`_hc_*` functions beneath them: the wrappers are `torch.ops.vllm` custom ops
(registered at hc.py:427-449), and that dispatch is part of the call the
model makes.

OPERAND ORDER IS DECLARED, NOT DERIVED — and nothing here reads it at run
time. Each span's `engine_inputs` in `op_signatures_catalog.py` states the
engine's emission of its operands as `(position, layout)` pairs, which is
what that field is for: a permute or a pack is a declaration, and
registration rejects a malformed one at import. The frameworks' argument
order is a separate fact, true of the call rather than of the graph, and it
lives here — at the call.

Three of the five permute between the two, which is why every provider below
takes its operands BY NAME out of `inputs`:

  ihc_post          contract: post, sublayer, residual
                    vLLM/sglang call: sublayer, residual, post
  fused_hc_head     contract: x, fn, scale, base       (both calls agree)
  hc_silu           contract: x                        (one operand)
  hc_gate_mix       contract: gate, xn
                    vLLM call: xn, gate
  hc_combine_norm   contract: block, injection, residual, norm_weight
                    vLLM call: residual, block, injection, norm_weight
                    sglang call: block, residual, injection, norm_weight
                    (the injection split into 8 fp32 partials — one value,
                    two encodings, not two operands)

A positional splat would read correct and compute a different function, and
the only thing between that and a published row is the numerical cross-check.
Do not reorder the contract to match a framework, or a framework's call to
match the contract: they answer different questions and each is written where
it is true. The plan's `io_mapping` answers a third question — which nodes of
the naive subgraph get absorbed — and is not an operand order at all, so it
is not comparable to either.

Import-guarded: library imports are lazy inside `build`.
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register
from .attention import _require_cuda


# --- ihc_post --------------------------------------------------------------
#
# Both frameworks take the same three tensors positionally and in the same
# order — the sublayer output, the residual, the per-copy gate — and assert
# the same thing about the gate (fp32). The only difference is the name.

def _ihc_post_args(inputs: Dict[str, Any]):
    """`(sublayer, residual, post)` — the frameworks' argument order, which
    is a PERMUTATION of the contract's (`post, sublayer, residual`, declared
    in the span's `engine_inputs`). Taken by name for that reason."""
    return inputs["sublayer"], inputs["residual"], inputs["post"]


def _vllm_ihc_post_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                         ) -> BenchCase:
    _require_cuda()
    from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_post
    args = _ihc_post_args(inputs)
    return BenchCase(run=lambda: triton_ihc_post(*args),
                     to_canonical=lambda o: o.float())


def _sgl_ihc_post_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                        ) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_post
    args = _ihc_post_args(inputs)
    return BenchCase(run=lambda: fused_hy4_ihc_post(*args),
                     to_canonical=lambda o: o.float())


# --- fused_hc_head ---------------------------------------------------------
#
# Again one function in two frameworks, including the `+ hc_eps` after the
# sigmoid (vLLM tilelang_kernels.py:1085-1090, sglang mhc_head.py:72). The
# two epsilons are passed, not defaulted: sglang's wrapper has no default for
# either, and vLLM's kernel reads them as compile-time constants, so a
# provider that omitted them would be measuring another function.

def _hc_head_args(inputs: Dict[str, Any]):
    """`(x, fn, scale, base)` — here the frameworks' order and the
    contract's agree, which is luck rather than a rule, so it is still read
    by name."""
    return (inputs["x"], inputs["fn"], inputs["scale"], inputs["base"])


def _vllm_hc_head_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                        ) -> BenchCase:
    """vLLM's exit reduce. Its wrapper writes a BF16 output unconditionally
    (`torch.empty(..., dtype=torch.bfloat16)`, tilelang.py:1050) and the
    TileLang kernel declares both the residual and the output `T.bfloat16`
    (tilelang_kernels.py:1046-1051) — which is why this arm declares bf16 and
    only bf16. An fp16 cell measured here would name fp16 and write bf16.
    """
    _require_cuda()
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
    )
    from ..refs import _HC_EPS, _HC_RMS_EPS
    args = _hc_head_args(inputs)
    return BenchCase(
        run=lambda: hc_head_fused_kernel_tilelang(*args, _HC_RMS_EPS,
                                                  _HC_EPS),
        to_canonical=lambda o: o.float())


def _sgl_hc_head_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                       ) -> BenchCase:
    """sglang's exit reduce. Takes the epsilons by keyword, as its own call
    site does (deepseek_v4.py:4072-4078), and returns `x.dtype`."""
    _require_cuda()
    from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head
    from ..refs import _HC_EPS, _HC_RMS_EPS
    args = _hc_head_args(inputs)
    return BenchCase(
        run=lambda: fused_hc_head(*args, norm_eps=_HC_RMS_EPS,
                                  hc_eps=_HC_EPS),
        to_canonical=lambda o: o.float())


# --- the Qwen4Exp read / write gates --------------------------------------

def _vllm_hc_silu_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                        ) -> BenchCase:
    """`silu(x / hc_count)` on the low-rank mix vector.

    The divisor comes from `refs`, which is where this family's function
    constants are stated — this is the one launch whose cell cannot supply
    it (its operand is the low-rank vector, which has no copy axis), so the
    oracle and this provider read the same single declaration.
    """
    _require_cuda()
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_silu
    from ..refs import _HC_SILU_COPIES
    x = inputs["x"]
    return BenchCase(run=lambda: hc_silu(x, _HC_SILU_COPIES),
                     to_canonical=lambda o: o.float())


def _vllm_hc_gate_mix_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                            ) -> BenchCase:
    """The gated mean over the copies.

    `hc_gate_mix(x, gate, hc_count)` — the normed stream FIRST, the gate
    second, which is the opposite of the contract's declared engine order
    (`gate, xn`). Mixing the two up computes `sigmoid(xn)·gate`: a
    plausible-looking tensor of the right shape and the wrong function.

    The copy count rides on the inputs because no operand of this launch
    carries it (see `refs._hc_gate_mix_inputs`).
    """
    _require_cuda()
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
    xn, gate = inputs["xn"], inputs["gate"]
    hc = int(inputs["n_hidden_copies"])
    return BenchCase(run=lambda: hc_gate_mix(xn, gate, hc),
                     to_canonical=lambda o: o.float())


def _vllm_hc_combine_norm_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """The write fused with the next read's grouped norm.

    `hc_combine_norm(residual, block_output, injection_logits, norm_weight,
    eps, hc_count)` — vLLM's order, which is a PERMUTATION of the contract's
    declared engine order (`block, injection, residual, norm_weight`).
    Passing them in the contract's order would hand the sublayer output
    where the stream belongs; both are activation tensors, so the call would
    succeed and write a correctly shaped answer.

    Two outputs, and the span declares both: the combined stream (the next
    write reads it) and the normalised one. `to_canonical` lays them side by
    side as the oracle does, so the cross-check covers both rather than
    whichever one came first.

    `hc_count` is read off the injection logits' own width — the launch's own
    operand says how many copies there are, so nothing here states it.
    """
    _require_cuda()
    import torch
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm
    from ..refs import _HC_RMS_EPS
    residual, block = inputs["residual"], inputs["block"]
    injection, weight = inputs["injection"], inputs["norm_weight"]
    hc = int(injection.shape[-1])
    return BenchCase(
        run=lambda: hc_combine_norm(residual, block, injection, weight,
                                    _HC_RMS_EPS, hc),
        to_canonical=lambda o: torch.cat([o[0].float(), o[1].float()],
                                         dim=-1))


def _sgl_hc_combine_norm_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """sglang's decode write fused with the next read's grouped norm.

    `hc_combine_apply_norm(block, residual, partials, weight, eps)` — the
    block FIRST, which is neither the contract's order nor vLLM's; taken
    by name for that reason. And the injection arrives SPLIT: the gate
    dot's 8 fp32 partials (`[rows, 8, hc]`, written by the companion
    `hc_combine_gate` launch), which the kernel sums before the sigmoid
    (`hc_combine_decode.cuh:80-81`). The span's `injection` is that same
    dot already reduced, so the partials are built with the logits in the
    first split and zeros elsewhere — the sum the kernel gates on is
    unchanged, and the launch that runs is the one sglang times, reads
    and grid included. The packing is at build time, not in the timed
    call.

    The kernel is hard-coded to 4 copies of 2560 in bf16 and refuses more
    than 4 rows (the TensorMatcher and CHECK_HOST in the .cuh); the
    `serves` gate states those. The dispatch that brings the MODEL here —
    one token in the whole forward, on a major-10 device — is the
    binding's `when:`, not this gate's: a gate states what the kernel
    runs, and the kernel runs 2-4 rows as faithfully as one.
    """
    _require_cuda()
    import torch
    from sglang.kernels.ops.elementwise.hc_combine_decode import (
        hc_combine_apply_norm,
    )
    from ..refs import _HC_RMS_EPS
    block, residual = inputs["block"], inputs["residual"]
    injection, weight = inputs["injection"], inputs["norm_weight"]
    hc = int(injection.shape[-1])
    partials = torch.zeros(injection.shape[0], 8, hc, dtype=torch.float32,
                           device=injection.device)
    partials[:, 0, :] = injection.float()
    return BenchCase(
        run=lambda: hc_combine_apply_norm(block, residual, partials,
                                          weight, _HC_RMS_EPS),
        to_canonical=lambda o: torch.cat([o[0].float(), o[1].float()],
                                         dim=-1))


# --- the torch floors ------------------------------------------------------

def _torch_floor_build(inv: Invocation, env: Env, inputs: Dict[str, Any]
                       ) -> BenchCase:
    """Each span's naive chain, eager and AT THE CELL'S PRECISION — what the
    fused launch's saving is measured against.

    Not the oracle's routine. That one computes in fp32 on purpose, to be a
    trusted reference, and a floor timed in fp32 reports another precision's
    latency — which the driver refuses outright ("the launch's largest
    operand is float32 where the cell declares bfloat16"), so the span would
    silently lose its floor and keep its kernel. The arithmetic is the same
    chain either way; what differs is the width it runs at.

    Where a coefficient is fp32 in the framework's runtime and the big
    operand is not (`ihc_post`'s gate, `fused_hc_head`'s projection), the
    floor casts the coefficient down, exactly as `mhc.py`'s floor does and
    for the same reason: in eager torch an fp32 operand against a bf16 one
    PROMOTES the bf16, and a full-width fp32 operand is what the driver
    refuses. The consequence runs ONE WAY — a saving is read as
    floor / kernel, so a floor that is slightly cheaper than the true naive
    chain makes the ratio too SMALL. Every speedup read off these floors is
    therefore conservative.

    The norm reductions stay fp32 (`mean(..., dtype=torch.float32)`), which
    costs no full-width temporary of the row.
    """
    import torch
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype)
    sig = inv.signature

    if sig == "ihc_post":
        def spread():
            post = inputs["post"].to(dt).unsqueeze(-1)
            sublayer = inputs["sublayer"].to(dt).unsqueeze(-2)
            return post * sublayer + inputs["residual"].to(dt)
        return BenchCase(run=spread, to_canonical=lambda o: o.float())

    if sig == "fused_hc_head":
        from ..refs import _HC_EPS, _HC_RMS_EPS

        def reduce_copies():
            x = inputs["x"].to(dt)
            flat = x.reshape(x.shape[0], -1)
            scale = torch.rsqrt(
                flat.square().mean(-1, keepdim=True, dtype=torch.float32)
                + _HC_RMS_EPS)
            mixes = (flat @ inputs["fn"].to(dt).t()) * scale.to(dt)
            pre = torch.sigmoid(mixes * inputs["scale"].to(dt)
                                + inputs["base"].to(dt)) + _HC_EPS
            return (pre.unsqueeze(-1) * x).sum(dim=1)
        return BenchCase(run=reduce_copies, to_canonical=lambda o: o.float())

    if sig == "hc_silu":
        from ..refs import _HC_SILU_COPIES

        def activate():
            x = inputs["x"].to(dt) / _HC_SILU_COPIES
            return x * torch.sigmoid(x)
        return BenchCase(run=activate, to_canonical=lambda o: o.float())

    if sig == "hc_gate_mix":
        hc = int(inputs["n_hidden_copies"])

        def gated_mean():
            g = inputs["gate"].to(dt).view(inputs["gate"].shape[0], hc, -1)
            x = inputs["xn"].to(dt).view(inputs["xn"].shape[0], hc, -1)
            return (torch.sigmoid(g) * x).sum(dim=1) / hc
        return BenchCase(run=gated_mean, to_canonical=lambda o: o.float())

    # hc_combine_norm: the two naive members, the combine and the grouped
    # norm, with the activation-dtype round trip between them that the
    # unfused graph really makes (and that the kernel reproduces on purpose).
    from ..refs import _HC_RMS_EPS
    hc = int(inputs["injection"].shape[-1])
    d = int(inputs["block"].shape[-1])

    def combine_and_norm():
        gate = 2.0 * torch.sigmoid(inputs["injection"].to(dt) / hc)
        write = inputs["block"].to(dt).unsqueeze(-2) * gate.unsqueeze(-1)
        residual = inputs["residual"].to(dt)
        out = residual.view(residual.shape[0], hc, d) + write
        normed = out * torch.rsqrt(
            out.square().mean(-1, keepdim=True, dtype=torch.float32)
            + _HC_RMS_EPS).to(dt)
        flat = out.reshape(out.shape[0], hc * d)
        y = normed.reshape(out.shape[0], hc * d) * (
            1 + inputs["norm_weight"].to(dt))
        return flat, y

    return BenchCase(run=combine_and_norm,
                     to_canonical=lambda o: torch.cat(
                         [o[0].float(), o[1].float()], dim=-1))


# --- registration ----------------------------------------------------------
#
# `activation=(...)` on a framework arm is a CAPABILITY, not a dispatch
# condition: a cell outside it is one the kernel cannot serve, and declaring
# it declines the cell instead of accepting it and failing inside the
# wrapper (the gate wider than the build). The torch floors declare none,
# because a floor runs the chain at whatever width the cell states — which
# is the whole point of a floor.
#
# Which widths, per arm, from the kernels rather than from each other:
#   * `ihc_post` — vLLM's own support predicate is
#     `x.dtype in (torch.float16, torch.bfloat16)` (triton_ihc.py:25), and
#     sglang's kernel is the same Triton body reached from the same model.
#   * `fused_hc_head` — vLLM's TileLang kernel types the residual AND the
#     output `T.bfloat16` (tilelang_kernels.py:1046, :1051) and the wrapper
#     allocates bf16 whatever it was given, so bf16 is the only cell it
#     measures faithfully. sglang's Triton kernel writes `x.dtype`, so it
#     serves both. The asymmetry is the kernels', not a modelling choice.
#   * the three Qwen4Exp launches — vLLM's are `x.new_empty(...)`, so the
#     output follows the input, and the model runs bf16/fp16. sglang's
#     `hc_combine_apply_norm` is one specialization, compiled for bf16 and
#     4×2560 with a 4-row ceiling (the TensorMatcher / CHECK_HOST in
#     hc_combine_decode.cuh), which its gate states verbatim; the norm
#     write follows the stream it reads, so no `output` slot narrows it.

for _sig, _vllm, _vllm_act, _sgl in (
    ("ihc_post", _vllm_ihc_post_build, ("bf16", "fp16"), _sgl_ihc_post_build),
    ("fused_hc_head", _vllm_hc_head_build, ("bf16",), _sgl_hc_head_build),
):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_{_sig}",
        import_name=("vllm.models.hy_v4.nvidia.triton_ihc"
                     if _sig == "ihc_post"
                     else "vllm.model_executor.kernels.mhc.tilelang"),
        serves=KernelSignature(signature=_sig, activation=_vllm_act),
        build=_vllm, fused_op=_sig,
    ))
    register(Provider(
        library="sglang", signature=_sig, kernel_impl=f"sglang_{_sig}",
        import_name=("sglang.kernels.ops.layernorm.hy4_ihc"
                     if _sig == "ihc_post"
                     else "sglang.kernels.ops.layernorm.mhc_head"),
        serves=KernelSignature(signature=_sig, activation=("bf16", "fp16")),
        build=_sgl, fused_op=_sig,
    ))

for _sig, _build in (("hc_silu", _vllm_hc_silu_build),
                     ("hc_gate_mix", _vllm_hc_gate_mix_build),
                     ("hc_combine_norm", _vllm_hc_combine_norm_build)):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_{_sig}",
        import_name="vllm.models.qwen4_exp.nvidia.ops.hc",
        serves=KernelSignature(signature=_sig, activation=("bf16", "fp16")),
        build=_build, fused_op=_sig,
    ))

# The one Qwen4Exp launch sglang has at vLLM's boundary — the decode write
# + norm, hard-compiled for bf16 and 4 copies of 2560, at most 4 rows (the
# .cuh's own limits). `hc_silu` and `hc_gate_mix` get no sglang arm because
# sglang has no launch at their boundaries (the header's audit). Blackwell
# only, as the binding says: the model creates the gate stream that leads
# here only on capability major 10 (srt/layers/hyperconnection.py:186-196),
# and the kernel's 32-byte `AlignedVector<bf16x2, 8>` does not compile
# below it (`sgl_kernel/vec.cuh:79`'s static_assert — the JIT build error
# that disabled this path on the H100).
register(Provider(
    library="sglang", signature="hc_combine_norm",
    kernel_impl="sglang_hc_combine_norm",
    import_name="sglang.kernels.ops.elementwise.hc_combine_decode",
    serves=KernelSignature(
        signature="hc_combine_norm", activation=("bf16",),
        phases=("decode",),
        enum=(("n_hidden_copies", (4,)), ("d_model", (2560,))),
        max_product=((("B", "s_query"), 4),)),
    build=_sgl_hc_combine_norm_build, fused_op="hc_combine_norm",
    min_sm=100,
))

for _sig in ("ihc_post", "fused_hc_head", "hc_silu", "hc_gate_mix",
             "hc_combine_norm"):
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch",
        # The floor casts every operand to the cell's dtype and runs plain
        # torch ops (`_torch_floor_build`), so it runs at any float width.
        serves=KernelSignature(signature=_sig, activation=TORCH_FLOAT),
        build=_torch_floor_build, fused_op=_sig,
    ))
