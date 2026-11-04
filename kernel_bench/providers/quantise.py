"""Activation-cast providers — the quantise a quantised matmul runs on its
input operand, and the dequantise back.

WHY THIS IS A CELL AND NOT A PREPARATION STEP

The frameworks launch it on its own. vLLM quantises the activation in
`vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:119`
and runs the GEMM at `:132`; sglang in
`python/sglang/srt/layers/quantization/fp8_utils.py:1355` with the GEMM
after it. Both read the whole wide activation and write the narrow one plus
its scales — zero FLOPs and non-zero bytes — and a row with no provider
bound to it is UNPLACED, which under the all-or-nothing rule
(`tests/backend/inference/test_measured_coverage.py`) takes the measured
time off every model with an FP8 GEMM. Until now the same arithmetic lived
inside this harness as an untimed preparation step in the GEMM builds
(`gemm._per_token_cast_to_fp8`), which is exactly the launch a server pays
for and nothing was pricing.

ONE SIGNATURE, TWO DIRECTIONS

`quantise_activation` carries both: the cell says which dtype the launch
reads (`activation`) and which it writes (`output`), and the oracle is the
same either way — a dequantise kernel is right when it returns the
activation as the narrow format holds it, which is what a quantise kernel's
output dequantises to. See `quant.ActivationCast`.

THE GRANULARITY IS PART OF THE CELL

Per-token, per-128-group and per-32-group are three kernels between one pair
of dtypes, and the frameworks ship them as three entry points. The cell
states which (`activation_granularity`, from
`ActivationFormat.granularity`), and so does every provider here; nothing
reads a group size off a model name or a shape.

WHAT IS NOT COVERED, AND WHY NOT

  * The DEQUANTISE direction has no framework provider. Neither vLLM nor
    sglang exposes a standalone activation dequantise launch at the audited
    releases: the widening happens in a GEMM's epilogue (`cutlass_scaled_mm`
    and `fp8_gemm_nt` both write bf16), and the weight-only fallbacks
    dequantise the WEIGHT inside the kernel rather than the activation. So
    the direction is torch-floor only here, and a framework entry point for
    it would be a new provider rather than a new signature.
  * FP4 activations (`nvfp4_dynamic`, `mxfp4_dynamic`) have real framework
    kernels and are registered DISABLED with the reason, so their cells
    report as the gap they are rather than as cells nobody wanted. What is
    missing is ours: an fp4 element is half a byte, torch has no dtype for
    it, and the scales come back swizzled into a layout nothing here can
    bring to the canonical one — so a row would be measured with no way to
    check it computed the cell.
  * DeepGEMM exposes no activation quantiser at all: it consumes already
    quantised `(elements, scales)` pairs and publishes only layout
    transforms (`get_mn_major_tma_aligned_tensor`,
    `transform_sf_into_required_layout`). Those move the SCALES and not the
    activation, so they are a different launch from this one; vLLM and
    sglang both reach them through their own wrappers
    (`vllm/utils/deep_gemm.py:433`, `sglang/srt/layers/
    deep_gemm_wrapper/entrypoint.py:24`).
  * A static per-TENSOR activation scale (vLLM
    `_custom_ops.py:1889 static_scaled_fp8_quant`, sglang
    `fp8_kernel.py:903 static_quant_fp8`) has no cell to serve: no declared
    activation format states a per-tensor granularity, and inventing one
    here would be a format no checkpoint runs under.

Every file:line above and below was read at the release each framework's
binding is audited at — vLLM v0.32.0, sglang v0.5.22
(`backend/inference/framework_bindings/{vllm,sglang}.yaml`). Paths are
relative to the repository root of each; sglang's are under `python/`.

Import-guarded; the framework bodies run only in the matching provider env.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.arch.activation_formats import (ACTIVATION_FORMATS,
                                             ACTIVATION_GRANULARITIES)
from backend.arch.dtypes import DTYPES, corpus_token

from ..env import Env
from ..invocation import Invocation
from ..quant import (ACTIVATION_EPS, FORMATS, ActivationCast, quantise_groups,
                     scale_group)
from . import BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda

#: The narrow end of a cast the torch floors can hold: a format some declared
#: activation format quantises to, AND one torch has a dtype for. The
#: intersection of two declarations rather than a list, so neither half can be
#: wider than the other — a format torch cannot hold has no tensor to be the
#: output of (fp4 is half a byte), and one no activation format states is a
#: cell no checkpoint produces.
_NARROW = tuple(sorted(
    {corpus_token(fmt.dtype) for fmt in ACTIVATION_FORMATS.values()}
    & {name for name, grid in FORMATS.items() if grid.torch_cast}))

#: The wide end: the dtypes that hold values as they are and that torch has a
#: dtype for — `arch.dtypes`' own `quantized is False` rows with a `torch`
#: spelling and a corpus token (bf16, fp16, fp32).
_WIDE = tuple(sorted(
    row.token for row in DTYPES.values()
    if row.token and row.quantized is False and row.torch))

#: The scale dtypes (`metadata`) those narrow formats are declared with —
#: fp32 and ue8m0 — every one an encoding the torch floors quantise to
#: (`quant.activation_scale_encoding`). Derived, like `_NARROW`, so a format
#: added beside them widens both together.
_NARROW_SCALES = tuple(sorted(
    {corpus_token(fmt.scale_dtype) for fmt in ACTIVATION_FORMATS.values()
     if corpus_token(fmt.dtype) in _NARROW}))

#: Every granularity a declared activation format states. The torch floors
#: serve all of them because they group by reshaping the row at whatever the
#: cell says — which is also why they decline a row that does not divide
#: (`declines_an_untileable_row`) rather than naming widths
#: here.
_EVERY_GRANULARITY = tuple(sorted(ACTIVATION_GRANULARITIES))


# ---------------------------------------------------------------------------
# torch floors — the arithmetic itself, at the cell's own grid
# ---------------------------------------------------------------------------

def torch_quantise_activation(x, narrow: str, group: int, scale_encoding: str,
                              wide_dtype, eps: float):
    """`x` → `(elements, scales)` at the declared grid, in torch ops.

    The one implementation of the grid (`quant.quantise_groups`), which the
    oracle also runs — so what this provider measures and what the
    cross-check compares against cannot drift apart. Its arguments are
    spelled out rather than bundled so the row's `launch_signature` shows
    the group and the scale convention the cell was measured at.
    """
    return quantise_groups(x, FORMATS[narrow], group, scale_encoding,
                           wide_dtype, eps=eps)


def torch_dequantise_activation(elements, scales, wide_dtype):
    """`(elements, scales)` → the wide activation, one multiply per group.

    Two passes rather than one, because torch has no arithmetic on an fp8
    tensor: the widening cast is its own kernel and the multiply is the
    next. So this is an upper bound on the direction rather than a tight
    floor, and a framework kernel that fuses the two will beat it — which
    is what the envelope is for.
    """
    return elements.to(wide_dtype) * scales.unsqueeze(2)


def declines_an_untileable_row(axis: str = "d_model"):
    """`impl_params` for a launch holding a quantised row `axis` wide:
    decline a cell whose row is not a whole number of the cell's own scale
    groups, before any input is allocated.

    Not expressible as a `divisible` on the signature: the modulus is the
    CELL's (its own `activation_granularity`), not the kernel's, so there is
    no fixed number to declare. Every kernel that holds such a row groups it
    by reshaping, and the frameworks assert the same thing — vLLM
    `fp8_utils.py:665` (`assert x.shape[-1] % group_size == 0`), sglang
    `fp8_kernel.py:625`.

    The width axis is the launch's own: `d_model` for a cast or for the
    dispatch that moves the record, `d_ffn` for a gated activation's half.
    One refusal for all of them — written twice, the two copies diverged on
    which axis they read.
    """
    def refuse(inv: Invocation) -> str:
        width = int(inv.dims[axis])
        group = scale_group(inv, width=width)
        if group <= 0 or width % group:
            raise NotImplementedError(
                f"a row of {width} is not a whole number of "
                f"{group}-element scale groups")
        return ""
    return refuse


def _torch_quantise_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    cast: ActivationCast = inputs["cast"]
    return BenchCase(
        run=Launch(torch_quantise_activation, inputs["x"], cast.narrow,
                   cast.group, cast.scale_encoding, cast._torch_wide(),
                   ACTIVATION_EPS),
        to_canonical=cast.dequantise,
    )


def _torch_dequantise_build(inv: Invocation, env: Env,
                            inputs: Dict[str, Any]) -> BenchCase:
    cast: ActivationCast = inputs["cast"]
    # Quantising to get the launch's operands is setup, not the measurement:
    # a dequantise cell prices reading the narrow elements and writing the
    # wide ones, and a server reads elements some earlier launch wrote.
    elements, scales = cast.quantise(inputs["x"])
    return BenchCase(
        run=Launch(torch_dequantise_activation, elements, scales,
                   cast._torch_wide()),
        to_canonical=lambda out: out.float().reshape(out.shape[0], -1),
    )


register(Provider(
    library="torch", signature="quantise_activation",
    kernel_impl="torch_activation_quantise", import_name="torch",
    serves=KernelSignature(signature="quantise_activation",
                           activation=_WIDE, output=_NARROW,
                           activation_granularity=_EVERY_GRANULARITY,
                           dtype_slots=(("metadata", _NARROW_SCALES),)),
    impl_params=declines_an_untileable_row(),
    build=_torch_quantise_build,
))


register(Provider(
    library="torch", signature="quantise_activation",
    kernel_impl="torch_activation_dequantise", import_name="torch",
    # The mirror gate: here the QUANTISED format is what the launch reads
    # and the wide one is what it writes.
    serves=KernelSignature(signature="quantise_activation",
                           activation=_NARROW, output=_WIDE,
                           activation_granularity=_EVERY_GRANULARITY,
                           dtype_slots=(("metadata", _NARROW_SCALES),)),
    impl_params=declines_an_untileable_row(),
    build=_torch_dequantise_build,
))


# ---------------------------------------------------------------------------
# vLLM — v0.32.0
# ---------------------------------------------------------------------------

def _vllm_group_quant_build(use_ue8m0: bool):
    """vLLM's per-token-group fp8 quantiser.

    `vllm/model_executor/layers/quantization/utils/fp8_utils.py:632`

        def per_token_group_quant_fp8(
            x, group_size, eps=1e-10, dtype=None,
            column_major_scales=False, tma_aligned_scales=False,
            out_q=None, use_ue8m0=None) -> tuple[Tensor, Tensor]

    Returns `(x_q, x_s)` — e4m3 `[M, K]` and an fp32 scale `[M, K/g]`. The
    launch is `torch.ops._C.per_token_group_fp8_quant` (`:709`) on CUDA,
    with Triton fallbacks at `:727` / `:745`. What a server runs: the
    block-scaled linear kernel quantises here
    (`kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:119`, the
    `QuantFP8` built at `:68`) and GEMMs at `:132`; DeepSeek's sparse
    indexer calls it straight from the model forward
    (`models/deepseek_v2.py:843`), prefill and decode alike.

    `column_major_scales` / `tma_aligned_scales` are left at their defaults.
    They change the scale tensor's STRIDES and nothing else — the same
    elements, the same count, written for DeepGEMM's TMA
    (`kernels/linear/scaled_mm/deep_gemm.py:38`, itself behind
    `VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES`) — so measuring them as separate
    cells would split one cell over a deployment flag that moves no
    activation byte.
    """
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8)
        cast: ActivationCast = inputs["cast"]
        return BenchCase(
            run=Launch(per_token_group_quant_fp8, inputs["x"],
                       group_size=cast.group, eps=ACTIVATION_EPS,
                       use_ue8m0=use_ue8m0),
            to_canonical=cast.dequantise,
        )
    return build


register(Provider(
    library="vllm", signature="quantise_activation",
    kernel_impl="vllm_per_token_group_quant_fp8", import_name="vllm",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        # An exact fp32 scale is what `use_ue8m0=False` writes; a cell whose
        # scales are e8m0 is the sibling below, which rounds them to a power
        # of two and so quantises to other values.
        activation_granularity=("per-token", "per-group:128",
                                "per-group:32"),
        # The scale dtype the cell states (`metadata`) is the encoding the
        # oracle quantises to (`quant.activation_scale_encoding`): this entry
        # point writes exact fp32 scales, so an e8m0 cell is not its cell.
        dtype_slots=(("metadata", ("fp32",)),),
    ),
    impl_params=declines_an_untileable_row(),
    build=_vllm_group_quant_build(use_ue8m0=False),
))


register(Provider(
    library="vllm", signature="quantise_activation",
    kernel_impl="vllm_per_token_group_quant_fp8_ue8m0", import_name="vllm",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        activation_granularity=("per-group:128", "per-group:32"),
        # Its scales are powers of two (`metadata: ue8m0`), which an exact
        # fp32 cell does not ask for: the oracle would quantise the same row
        # to other values and blame the kernel.
        dtype_slots=(("metadata", ("ue8m0",)),),
    ),
    impl_params=declines_an_untileable_row(),
    build=_vllm_group_quant_build(use_ue8m0=True),
    # The scale VALUES are powers of two, which is what an e8m0 cell asks
    # for, but this entry point writes them in fp32 — four bytes where the
    # cell is billed one. The same approximation
    # `arch/activation_formats/fp8_token_group128_ue8m0.yaml` already
    # describes for SM90 stacks. The packed sibling below writes one byte.
    computes_at={"metadata": "fp32"},
))


def _vllm_packed_ue8m0_build(inv: Invocation, env: Env,
                             inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's UE8M0 quantiser that writes DeepGEMM's packed scale layout.

    `vllm/model_executor/layers/quantization/utils/fp8_utils.py:766`

        def per_token_group_quant_fp8_packed_for_deepgemm(
            x, group_size, eps=1e-10, use_ue8m0=None,
            out_q=None) -> tuple[Tensor, Tensor]

    Returns `(x_q, x_s_packed)`: e4m3 elements and an **int32** scale
    tensor, four UE8M0 exponent bytes to a word, logically
    `[mn, ceil(groups_per_row / 4)]` with TMA strides. One byte per scale,
    which is the cell's own `metadata: ue8m0`. The launch is
    `torch.ops._C.per_token_group_fp8_quant_packed` (`:826`). A server
    reaches it through `QuantFP8.forward_cuda`
    (`input_quant_fp8.py:96`) and from the MXFP8 DeepGEMM BMM
    (`kernels/linear/mxfp8/deep_gemm.py:75`).

    Bringing it back to the canonical layout means undoing the packing:
    each int32 holds four exponents, and an exponent `e` is the scale
    `2**(e-127)`. Done here rather than in the oracle because it is this
    kernel's storage, not the cell's math.
    """
    _require_cuda()
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm)
    cast: ActivationCast = inputs["cast"]
    groups = int(inv.dims["d_model"]) // cast.group
    rows = inputs["x"].shape[0]

    def unpack(pair):
        elements, packed = pair[0], pair[1]
        exponents = (packed[:rows].contiguous().view(torch.uint8)
                     .reshape(rows, -1)[:, :groups].float())
        return cast.dequantise((elements[:rows],
                                torch.exp2(exponents - 127.0)))

    return BenchCase(
        run=Launch(per_token_group_quant_fp8_packed_for_deepgemm, inputs["x"],
                   group_size=cast.group, eps=ACTIVATION_EPS, use_ue8m0=True),
        to_canonical=unpack,
    )


register(Provider(
    library="vllm", signature="quantise_activation",
    kernel_impl="vllm_per_token_group_quant_fp8_packed_for_deepgemm",
    import_name="vllm",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        activation_granularity=("per-group:128", "per-group:32"),
        # Its scales are powers of two (`metadata: ue8m0`), which an exact
        # fp32 cell does not ask for: the oracle would quantise the same row
        # to other values and blame the kernel.
        dtype_slots=(("metadata", ("ue8m0",)),),
    ),
    impl_params=declines_an_untileable_row(),
    build=_vllm_packed_ue8m0_build,
))


def _vllm_per_token_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's dynamic per-token fp8 quantiser — one scale for the whole row.

    `vllm/_custom_ops.py:1823`

        def scaled_fp8_quant(
            input, scale=None, num_token_padding=None, scale_ub=None,
            use_per_token_if_dynamic=False, output=None,
            group_shape=None) -> tuple[Tensor, Tensor]

    With `scale=None, use_per_token_if_dynamic=True` it allocates an fp32
    `[M, 1]` scale and launches
    `torch.ops._C.dynamic_per_token_scaled_fp8_quant` (`:1882`). This is the
    kernel `QuantFP8` dispatches to for `GroupShape.PER_TOKEN`
    (`input_quant_fp8.py:123`), which the per-token/per-tensor fp8 linear
    runs at `kernels/linear/scaled_mm/ScaledMMLinearKernel.py:161` before
    its GEMM at `:162`.

    `num_token_padding` is left unset: it pads the ROW COUNT to a GEMM's
    tile, so passing it would measure a cast over rows the cell does not
    carry, and `launch_trace` reads exactly that.
    """
    _require_cuda()
    from vllm import _custom_ops as ops
    cast: ActivationCast = inputs["cast"]
    return BenchCase(
        run=Launch(ops.scaled_fp8_quant, inputs["x"],
                   use_per_token_if_dynamic=True),
        to_canonical=cast.dequantise,
    )


register(Provider(
    library="vllm", signature="quantise_activation",
    kernel_impl="vllm_dynamic_per_token_scaled_fp8_quant",
    import_name="vllm",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        # One scale for the row and nothing else: this entry point writes
        # `[M, 1]`, so a grouped cell is not its cell.
        activation_granularity=("per-token",),
        # The scale dtype the cell states (`metadata`) is the encoding the
        # oracle quantises to (`quant.activation_scale_encoding`): this entry
        # point writes exact fp32 scales, so an e8m0 cell is not its cell.
        dtype_slots=(("metadata", ("fp32",)),),
    ),
    build=_vllm_per_token_build,
))


def _vllm_fp4_build(inv: Invocation, env: Env,
                    inputs: Dict[str, Any]) -> BenchCase:  # pragma: no cover
    """vLLM's NVFP4 activation quantiser — registered, not yet measured.

    `vllm/_custom_ops.py:1513`

        def scaled_fp4_quant(
            input, input_global_scale, is_sf_swizzled_layout=True,
            backend="none", padded_n=None) -> tuple[Tensor, Tensor]

    Returns packed `uint8 [m, n//2]` elements and an e4m3 block scale per 16
    channels, written in a swizzled (128x4) layout; the launch is
    `torch.ops._C.scaled_fp4_quant.out` (`:1579`). A server runs it right
    before the GEMM (`kernels/linear/nvfp4/cutlass.py:54`, GEMM at `:62`).
    """
    raise NotImplementedError("see disabled_reason")


register(Provider(
    library="vllm", signature="quantise_activation",
    kernel_impl="vllm_scaled_fp4_quant", import_name="vllm",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp4_e2m1",),
        activation_granularity=("per-group:16",),
        # NVFP4's block scale is e4m3 (beside one fp32 global).
        dtype_slots=(("metadata", ("fp8_e4m3",)),),
    ),
    build=_vllm_fp4_build,
    disabled_reason=(
        "the elements come back two to a byte and the block scales swizzled "
        "into a 128x4 tile, and this harness has no way to bring either to "
        "the canonical layout — so the row could be timed but not checked, "
        "and an unchecked cast is a cast that may have measured another "
        "granularity. Needs an unswizzle and an fp4 unpack, both of which "
        "are ours to write, not vLLM's to export."),
))


# ---------------------------------------------------------------------------
# sglang — v0.5.22
# ---------------------------------------------------------------------------

def _sglang_group_quant_build(scale_ue8m0: bool):
    """sglang's per-token-group fp8 quantiser.

    `python/sglang/kernels/ops/quantization/fp8_kernel.py:599`

        def sglang_per_token_group_quant_fp8(
            x, group_size, eps=1e-10, column_major_scales=False,
            scale_tma_aligned=False, scale_ue8m0=False,
            fuse_silu_and_mul=False, masked_m=None, *,
            output_q=None, output_s=None)

    Returns `(x_q, x_s)`. On CUDA the launch is the JIT-compiled C++ kernel
    `per_token_group_quant` (`:581`, wrapper
    `kernels/ops/quantization/per_token_group_quant.py:174`); the Triton
    path (`:230`) is the fallback. With `scale_ue8m0=True` and the
    row-major default it writes fp32 POWERS OF TWO matching
    `deep_gemm.ceil_to_ue8m0` (`:478-492`) — the same elements an e8m0 cell
    asks for, in four bytes rather than one, which is why that variant
    declares `computes_at`.

    What a server runs: the block-fp8 linear quantises at
    `srt/layers/quantization/fp8_utils.py:1355` (DeepGEMM backend, the
    `scale_ue8m0=True` row-major form on SM90) or `:1359`, and DeepSeek-V2
    calls it once per step from the model forward
    (`srt/models/deepseek_v2.py:1824`).

    `column_major_scales` / `scale_tma_aligned` stay at their defaults for
    the reason vLLM's do: they restride the scale tensor without changing
    an activation byte.
    """
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8)
        cast: ActivationCast = inputs["cast"]
        return BenchCase(
            run=Launch(sglang_per_token_group_quant_fp8, inputs["x"],
                       cast.group, ACTIVATION_EPS,
                       scale_ue8m0=scale_ue8m0),
            to_canonical=cast.dequantise,
        )
    return build


register(Provider(
    library="sglang", signature="quantise_activation",
    kernel_impl="sglang_per_token_group_quant_fp8", import_name="sglang",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        activation_granularity=("per-token", "per-group:128",
                                "per-group:32"),
        # The scale dtype the cell states (`metadata`) is the encoding the
        # oracle quantises to (`quant.activation_scale_encoding`): this entry
        # point writes exact fp32 scales, so an e8m0 cell is not its cell.
        dtype_slots=(("metadata", ("fp32",)),),
    ),
    impl_params=declines_an_untileable_row(),
    build=_sglang_group_quant_build(scale_ue8m0=False),
))


register(Provider(
    library="sglang", signature="quantise_activation",
    kernel_impl="sglang_per_token_group_quant_fp8_ue8m0",
    import_name="sglang",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        activation_granularity=("per-group:128", "per-group:32"),
        # Its scales are powers of two (`metadata: ue8m0`), which an exact
        # fp32 cell does not ask for: the oracle would quantise the same row
        # to other values and blame the kernel.
        dtype_slots=(("metadata", ("ue8m0",)),),
        # The fp32-power-of-two scale this form writes is not the JIT
        # `per_token_group_quant` the exact-scale sibling runs: it stays on
        # `per_token_group_quant_8bit_v2` (kernels/ops/quantization/
        # fp8_kernel.py:545-562 at v0.5.22), which offsets the input in
        # int32. Bisected on an H100: M x K = 2**31 runs (524,288 x 4096 and
        # 1,048,576 x 2048, groups 32 and 128) and one more row is an illegal
        # address — or, short of faulting, garbage scales, which is how
        # DeepSeek-V4.1-Flash's 526,920 x 4096 decode canary failed its
        # cross-check. The exact-scale sibling runs 524,289 x 4096 cleanly.
        max_product=((("B", "s_query", "d_model"), 2 ** 31),),
    ),
    impl_params=declines_an_untileable_row(),
    build=_sglang_group_quant_build(scale_ue8m0=True),
    computes_at={"metadata": "fp32"},
))


def _sglang_per_token_build(inv: Invocation, env: Env,
                            inputs: Dict[str, Any]) -> BenchCase:
    """sglang's dynamic per-token fp8 quantiser — one scale per row.

    `python/sglang/kernels/ops/quantization/fp8_kernel.py:815`

        def sglang_per_token_quant_fp8(
            x, dtype=fp8_dtype, *, output_q=None, output_s=None)

    Returns `(x_q, x_s)` with an fp32 `[M, 1]` scale; the launch is the
    compiled `sgl_per_token_quant_fp8` (`:838`, wrapper
    `kernels/ops/quantization/__init__.py:76`). A server runs it in
    `apply_fp8_linear` at `srt/layers/quantization/fp8_utils.py:2381`,
    before the GEMM. It is also what the grouped entry point short-circuits
    to when the group is the whole row (`fp8_kernel.py:633`), so the two
    compete on a per-token cell as one kernel reached two ways.
    """
    _require_cuda()
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_quant_fp8)
    cast: ActivationCast = inputs["cast"]
    return BenchCase(
        run=Launch(sglang_per_token_quant_fp8, inputs["x"]),
        to_canonical=cast.dequantise,
    )


register(Provider(
    library="sglang", signature="quantise_activation",
    kernel_impl="sglang_per_token_quant_fp8", import_name="sglang",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp8_e4m3",),
        activation_granularity=("per-token",),
        # The scale dtype the cell states (`metadata`) is the encoding the
        # oracle quantises to (`quant.activation_scale_encoding`): this entry
        # point writes exact fp32 scales, so an e8m0 cell is not its cell.
        dtype_slots=(("metadata", ("fp32",)),),
    ),
    build=_sglang_per_token_build,
))


def _sglang_fp4_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:  # pragma: no cover
    """sglang's NVFP4 activation quantiser — registered, not yet measured.

    `python/sglang/srt/layers/quantization/fp4_utils.py:31`, exported as the
    custom op `fp4_quantize` at `:84`:

        def _flashinfer_fp4_quantize_impl(
            input, global_scale=None, sf_vec_size=16, sf_use_ue8m0=False,
            is_sf_swizzled_layout=True, is_sf_8x4_layout=False,
            enable_pdl=None) -> tuple[Tensor, Tensor]

    FlashInfer's kernel wrapped by sglang, not an sglang kernel. A server
    runs it inside the W4A4 linear
    (`srt/layers/quantization/compressed_tensors/schemes/
    compressed_tensors_w4a4_nvfp4.py:145`).
    """
    raise NotImplementedError("see disabled_reason")


register(Provider(
    library="sglang", signature="quantise_activation",
    kernel_impl="sglang_flashinfer_fp4_quantize", import_name="sglang",
    serves=KernelSignature(
        signature="quantise_activation",
        activation=("bf16", "fp16"), output=("fp4_e2m1",),
        activation_granularity=("per-group:16",),
        # NVFP4's block scale is e4m3 (beside one fp32 global).
        dtype_slots=(("metadata", ("fp8_e4m3",)),),
    ),
    build=_sglang_fp4_build,
    disabled_reason=(
        "same gap as vLLM's `scaled_fp4_quant`: packed fp4 elements and "
        "swizzled e4m3 block scales, with no canonicalizer here to check "
        "the row against the oracle."),
))
