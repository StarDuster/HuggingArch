"""GEMM providers — the compute bulk of an LLM (all linear/qkv/mlp projections).

x[M=B·Sq, K] @ w[N, K]^T → [M, N]. Providers:
  * torch (cuBLAS/cuBLASLt via PyTorch dispatch) — bf16/fp16 baseline, works
    anywhere torch does.
  * DeepGEMM — fp8 blockwise GEMM (Hopper); only applies to fp8 activations.
  * vLLM cutlass_scaled_mm — W8A8 fp8.
  * Marlin — weight-only: int4 / NVFP4 / fp8-e4m3 weights against an f16
    activation, one provider per format.
  * humming — sglang's JIT kernel library.

WHAT EVERY PROVIDER HERE PROMISES

  1. The timed call is the FRAMEWORK's dispatch, not the raw op underneath it.
     Repack layouts, workspace sizing and the act-order / k-full / fp32-reduce
     decisions are part of what a server runs; calling beneath them times a
     configuration nobody deploys.

  2. Setup is outside the timed call. Quantizing and repacking a weight is
     load-time work (`process_weights_after_loading`); including it would
     measure checkpoint loading. JIT compilation is absorbed by the timing
     loop's warmup for the same reason.

  3. `applies_to` accepts exactly the set the build handles. A gate wider than
     the implementation is how a cell gets measured under the wrong
     quantization and recorded as if it were the right one — silently, since
     nothing raises. `fp8_e4m3` vs `fp8_e5m2` and NVFP4 vs MXFP4 are separate
     formats here for that reason, not variants.

  4. Every provider on a cell starts from the SAME canonical bf16 inputs and
     is checked against the same fp32 reference, which is what makes their
     latencies comparable. Two providers with different DEPLOYMENT semantics
     (W8A16 weight-only vs W8A8) may both apply to one cell — that comparison
     is meaningful, but it is a comparison of configurations, not of two
     implementations of the same thing.

Import-guarded; GPU bodies run in the matching provider env.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.arch.dtypes import dtype_bytes

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda, _torch_dtype
from .quantise import declines_an_untileable_row


_FP8_MAX = 448.0          # e4m3 max magnitude (used as the amax → scale divisor)
_BLK = 128                # DeepGEMM blockwise tile: 128 along K; 128×128 for weights
# The activation scale that tile implies: one F32 per token per 128 channels.
_DEEPGEMM_ACTIVATION_GRANULARITY = f"per-group:{_BLK}"


def _is_fp8(inv: Invocation) -> bool:
    """e4m3 only, on purpose. Everything below quantizes with `_FP8_MAX = 448`
    and `torch.float8_e4m3fn`; e5m2 trades three mantissa bits for one exponent
    bit and tops out at 57344, so a prefix match would hand an e5m2 cell to a
    kernel scaled 128× wrong and record the result as that cell's latency —
    silently, since nothing here would raise. No e5m2 checkpoint is in the plan
    today, which is the only reason the prefix has not already produced bad
    rows. Widening this means adding the e5m2 scale, not the match."""
    return inv.dtype_env.get("activation", "") == "fp8_e4m3"


def _torch_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """`F.linear` as the cuBLAS GEMM it dispatches to, with its output buffer
    handed to the call (`mm(..., out=)`) — the same kernel, writing where it
    is told.

    A promoted cell can compute wider than its edge stores. Eager aten has
    no spelling for that epilogue, so the build states both facts explicitly:
    operands and the product are the compute width, while the returned and
    recorded output is the cell's declared width. A bare `F.linear` names
    only its inputs, and a projection whose output is wider than its input
    and weight (`d_out` past `d_reduce` at many rows) then looked to the
    launch gate like a launch over part of the cell — every such cell
    refused, ~440 a pass on the H100. The moe_permute gather is the same
    lesson (`_torch_moe_permute_build`)."""
    import torch
    compute = _torch_gemm_compute_dtype(inv)
    dt = _torch_dtype(compute)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)
    rows = x.reshape(-1, x.shape[-1])
    out_dt = _torch_dtype(_torch_gemm_output_dtype(inv))
    out = torch.empty(rows.shape[0], w.shape[0], dtype=out_dt,
                      device=x.device)
    shape = (*x.shape[:-1], w.shape[0])
    if out_dt == dt:
        return BenchCase(run=Launch(torch.mm, rows, w.t(), out=out),
                         to_canonical=lambda o: o.view(shape))
    # aten's `mm.out` requires the output dtype to match its operands.  A
    # promoted product that narrows as it stores therefore spells the real
    # kernel's two facts as an eager product followed by the declared store:
    # the multiply stays at the cell's compute width, while the only tensor
    # this build hands back (and the only output the launch gate compares)
    # has the edge's declared dtype.
    product = torch.empty_like(out, dtype=dt)
    return BenchCase(
        run=Launch(_torch_mm_stores_declared, rows, w.t(), product, out),
        to_canonical=lambda o: o.view(shape))


def _torch_mm_stores_declared(x, matrix, product, out):
    """One promoted eager multiply, stored at the cell's declared width."""
    import torch
    torch.mm(x, matrix, out=product)
    out.copy_(product)
    return out


def _torch_gemm_params(inv: Invocation) -> str:
    """The width this floor multiplies at when that is not its store width."""
    weight = inv.dtype_env.get("weight", "")
    activation = inv.dtype_env.get("activation", "")
    output = inv.dtype_env.get("output", "")
    if weight and activation and output and (
            _torch_gemm_compute_dtype(inv) != output):
        return f"compute={_torch_gemm_compute_dtype(inv)}"
    return ""


def _torch_gemm_compute_dtype(inv: Invocation) -> str:
    """The width a torch GEMM's operands are promoted to.

    An fp32 router weight runs its fp32 matmul; a narrow weight is first
    dequantised to the activation width. BF16 and FP16 have no total order,
    and torch promotes a mixed pair to FP32 — the conservative peak, never a
    faster family charged for the slower launch.
    """
    activation, weight = inv.dtype, inv.dtype_for("weight")
    if activation == weight:
        return activation
    if dtype_bytes(activation) != dtype_bytes(weight):
        # The wider of the two, by the declared widths (`arch.dtypes`).
        return max((activation, weight), key=dtype_bytes)
    return "fp32"


def _torch_gemm_output_dtype(inv: Invocation) -> str:
    """The width the cell's edge declares this GEMM stores."""
    return inv.dtype_env.get("output") or _torch_gemm_compute_dtype(inv)


def _torch_gemm_computes_at(inv: Invocation) -> Dict[str, str]:
    actual = _torch_gemm_compute_dtype(inv)
    if actual == inv.dtype and actual == inv.dtype_for("weight"):
        return {}
    out: Dict[str, str] = {}
    if actual != inv.dtype:
        out["activation"] = actual
    if actual != inv.dtype_for("weight"):
        out["weight"] = actual
    return out


def _scale_tile(inv: Invocation) -> int:
    """The width of the weight's scale block along K and N, from the
    granularity the walk already states (`per-tile:<n>x<k>`).

    Read rather than assumed, because a checkpoint picks it: DeepSeek-V3 and
    the other blockwise fp8 releases store `weight_block_size: [128, 128]`,
    DeepSeek-V4.1 stores `[32, 32]`, and the two are different launches of the
    same kernel — DeepGEMM compiles its 1d2d GEMM with `SF_GRAN` 128 or 32
    (`sm90_fp8_gemm_1d2d.cuh`). Quantizing a 32-block checkpoint in 128-wide
    blocks would measure a different numeric recipe and report it under the
    checkpoint's name.

    Falls back to 128 for a cell that carries no granularity at all (planned
    before the slot existed) — the recipe those rows were measured with.
    """
    gran = inv.weight_granularity or ""
    if not gran.startswith("per-tile:"):
        return _BLK
    n_s, _, k_s = gran[len("per-tile:"):].partition("x")
    if n_s != k_s:
        raise NotImplementedError(
            f"non-square scale tile {gran}: both the quantizer here and "
            f"DeepGEMM's recipe take one granularity for N and K")
    return int(k_s)


#: The amax floor this helper has always used. NOT the one a serving cast
#: kernel uses (`quant.ACTIVATION_EPS`, 1e-10, which is what vLLM and sglang
#: pass): this mirrors vLLM's `per_token_cast_to_fp8` BENCHMARK helper, and
#: the 11,667 DeepGEMM rows already measured through it were measured at
#: this number. Kept explicit rather than silently adopted from the kernels,
#: because the two are different claims and only the GEMM setup makes this
#: one.
_SETUP_EPS = 1e-4


def _per_token_cast_to_fp8(x, tile: int = _BLK):
    """Activation → (e4m3 [M,K], fp32 scales [M, K//tile]). Per-token, per-
    `tile` input channels — mirrors vLLM `per_token_cast_to_fp8` / sglang
    benchmark, whose group matches the weight's block along K.

    The grid itself is `quant.quantise_groups`, which is also what the
    activation-cast providers time and what their oracle runs
    (`providers/quantise.py`). It used to be spelled out again here, and
    that was the same quantiser written twice: the GEMM builds treat it as
    untimed setup while a server pays for the launch, so the two had to end
    up being one function or drift into two answers about one cast."""
    from ..quant import FORMATS, quantise_groups
    M, K = x.shape
    q, scale = quantise_groups(x, FORMATS["fp8_e4m3"], tile, "exact",
                               x.dtype, eps=_SETUP_EPS)
    return q.reshape(M, K), scale


# Every FP8-activation GEMM here writes bf16 and says so at its own
# declaration (`output=("bf16",)`): each build allocates its output bf16
# (`torch.empty(M, N, dtype=torch.bfloat16)`, `out_dtype=torch.bfloat16`),
# the model's dtype downstream of the dequantising epilogue.


def _per_block_cast_to_fp8(w, tile: int = _BLK):
    """Weight → (e4m3 [N,K], fp32 scales [⌈N/tile⌉, ⌈K/tile⌉]). Per-
    tile×tile block — mirrors vLLM `per_block_cast_to_fp8`, which zero-pads to
    whole blocks: a checkpoint whose N is off the tile (DeepSeek's fused
    q_a+kv_a, 2112) stores a partial last block with a scale of its own."""
    import torch
    N, K = w.shape
    Np, Kp = -(-N // tile) * tile, -(-K // tile) * tile
    padded = w if (Np, Kp) == (N, K) else torch.nn.functional.pad(
        w, (0, Kp - K, 0, Np - N))
    wv = padded.reshape(Np // tile, tile, Kp // tile, tile).float()
    amax = wv.abs().amax(dim=(1, 3)).clamp_min(1e-4)     # [⌈N/tile⌉, ⌈K/tile⌉]
    q = (wv / (amax / _FP8_MAX)[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(Np, Kp)[:N, :K].contiguous(), (amax / _FP8_MAX)


def _deepgemm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # Real DeepGEMM fp8 blockwise GEMM: out = lhs @ rhs^T (NT). Derived from
    # vLLM `vllm/utils/deep_gemm.py` + sglang `deep_gemm_wrapper` call sites:
    #   deep_gemm.fp8_gemm_nt((a_q[M,K], a_s[M,K//128]),
    #                         (b_q[N,K], b_s[N//128,K//128]), out[M,N] bf16)
    # The activation scale must be MN-major / TMA-aligned; the weight scale is
    # row-major. Canonical x/w arrive lossless (bf16); we quantize here.
    import torch
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

    x, w = inputs["x"], inputs["w"]         # [M,K], [N,K]
    M, N = x.shape[0], w.shape[0]
    tile = _scale_tile(inv)
    x_q, x_s = _per_token_cast_to_fp8(x, tile)
    w_q, w_s = _per_block_cast_to_fp8(w, tile)
    x_s = get_mn_major_tma_aligned_tensor(x_s)          # activation scale → TMA-aligned
    # The recipe is the scale granularity (M, N, K) the kernel is compiled
    # for. DeepGEMM infers the default (1, 128, 128) from the DTYPES alone
    # (`get_default_recipe`), which is right for a 128-block checkpoint and
    # wrong for any other, so a non-default tile has to be stated. Left unset
    # at 128 so that path stays byte-identical to the 11,667 rows already
    # measured through it.
    recipe = None if tile == _BLK else (1, tile, tile)
    # `empty` is right HERE and only here: a dense GEMM writes every element of
    # [M,N], so there is nothing uninitialized left to compare. The grouped and
    # masked paths below write a subset and must be zeroed. Evidence for the
    # distinction: this kernel has 0 cross-check rejections over 11,667 rows,
    # the masked one 1,768 with inf and nan among them.
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    return BenchCase(
        run=lambda: deep_gemm.fp8_gemm_nt((x_q, x_s), (w_q, w_s), out,
                                          recipe=recipe),
        to_canonical=lambda _out: out,
    )


register(Provider(
    library="torch", signature="gemm", kernel_impl="torch_cublas_linear",
    import_name="torch",
    serves=KernelSignature(signature="gemm", weight=TORCH_FLOAT,
                           activation=TORCH_FLOAT, output=TORCH_FLOAT),
    build=_torch_build,
    impl_params=_torch_gemm_params,
    computes_at_for=_torch_gemm_computes_at,
))


def _torch_per_head_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    x = inputs["x"].to(_torch_dtype(inv.dtype))
    w = inputs["w"].to(_torch_dtype(inv.dtype))
    return BenchCase(
        run=lambda: torch.einsum("bhsk,hnk->bhsn", x, w),
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="per_head_gemm",
    kernel_impl="torch_per_head_einsum",
    import_name="torch",
    serves=KernelSignature(signature="per_head_gemm", weight=TORCH_FLOAT,
                           activation=TORCH_FLOAT),
    build=_torch_per_head_build,
))


def _torch_per_head_dequantised_build(inv: Invocation, env: Env,
                                      inputs: Dict[str, Any]) -> BenchCase:
    """A quantised per-head weight dequantised at load, then one bf16 BMM.

    The configuration every framework actually serves for these projections:
    vLLM and sglang dequantize an absorbed MLA projection in
    `process_weights_after_loading` and run `torch.bmm`, and sglang does the
    same for DeepSeek-V4.1's grouped output LoRA — `_dequant_fp8_wo_a_
    streaming` at load, then `wo_a_bf16_gemv` / `wo_a_bf16_small_batch` /
    `torch.bmm` / `einsum` on CUDA (its fp8 route,
    `_hip.wo_a_fp8_grid_matmul`, is gfx95-only).

    It is ONE launch for every head, which is the point of recording it
    beside the fp8 providers: those keep the weight small (a quarter of the
    bytes at fp8, an eighth at the 4-bit formats) but reach the same shape
    either through a masked grouped kernel or, at a 32-wide block, through
    one launch per head. Which configuration wins is then a measured
    question rather than an assumed one.

    The weight goes through the cell's own quantisation grid and back, so the
    numbers multiplied are the ones a server holds after loading — not the
    canonical bf16, which would be a weight no deployment has.
    """
    import torch

    x, w = inputs["x"], inputs["w"]              # [B,H,S,K], [H,N,K] bf16
    weight = _weight_dtype(inv)
    gran = inv.weight_granularity or ""
    deq = torch.empty_like(w, dtype=torch.bfloat16)
    for h in range(w.shape[0]):                  # load-time work, not timed
        if weight == "fp8_e4m3" and gran.startswith("per-tile:"):
            tile = _scale_tile(inv)
            q, s = _per_block_cast_to_fp8(w[h], tile)
            scale = s.repeat_interleave(tile, 0).repeat_interleave(tile, 1)
            deq[h] = (q.float() * scale[:w.shape[1], :w.shape[2]]).to(torch.bfloat16)
        else:                                    # per-channel / per-tensor fp8
            amax = (w[h].float().abs().amax(dim=1, keepdim=True)
                    if gran != "per-tensor"
                    else w[h].float().abs().amax().reshape(1, 1)).clamp_min(1e-4)
            q = (w[h].float() / (amax / _FP8_MAX)).to(torch.float8_e4m3fn)
            deq[h] = (q.float() * (amax / _FP8_MAX)).to(torch.bfloat16)
    xb = x.to(torch.bfloat16)
    return BenchCase(
        run=lambda: torch.einsum("bhsk,hnk->bhsn", xb, deq),
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="per_head_gemm",
    kernel_impl="torch_per_head_bmm_dequantised", import_name="torch",
    # Every quantised fp8 layout, because the dequantisation at load does not
    # care which one the checkpoint stored — it ends in bf16 either way. The
    # row records `torch_per_head_bmm_dequantised` so nobody reads a bf16
    # latency as an fp8 kernel's.
    serves=KernelSignature(signature="per_head_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           weight_granularity=("per-tile:128x128",
                                               "per-tile:32x32",
                                               "per-channel", "per-tensor")),
    build=_torch_per_head_dequantised_build,
    # The fallback itself, on the row: the cell is fp8 and this BMM is bf16.
    computes_at={"weight": "bf16", "activation": "bf16"},
))


def _torch_embedding_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    # `F.embedding` is `index_select` on the table (aten's own lowering);
    # called as that, with its output buffer handed to it, because an
    # expanding gather's largest operand is its output and a bare
    # `F.embedding` names only the table and the ids — the launch gate then
    # read a small vocabulary's table as a launch over part of the cell.
    dt = _torch_dtype(inv.dtype)
    w = inputs["w"].to(dt)
    ids = inputs["ids"]
    flat = ids.reshape(-1)
    out = torch.empty(flat.shape[0], w.shape[-1], dtype=dt, device=w.device)
    shape = (*ids.shape, w.shape[-1])
    return BenchCase(
        run=Launch(torch.index_select, w, 0, flat, out=out),
        to_canonical=lambda o: o.view(shape),
    )


register(Provider(
    library="torch", signature="embedding",
    kernel_impl="torch_embedding",
    import_name="torch",
    serves=KernelSignature(signature="embedding", weight=TORCH_FLOAT),
    build=_torch_embedding_build,
))
def _vllm_scaled_mm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # vLLM CUTLASS fp8 GEMM: out = (sa·a) @ (sb·b), b column-major [K,N].
    # One op, three scale layouts — whichever the checkpoint stores
    # (`inv.weight_granularity`), so the cell measures the kernel a
    # deployment of that checkpoint runs:
    #   * per-tile 128x128 — vLLM `W8A8BlockFp8LinearOp`'s cutlass path:
    #     activation per token per 128 channels (scale [M, K/128], column-
    #     major), weight per 128x128 block (scale [N/128, K/128], passed .t());
    #   * per-channel / per-tensor — `scaled_fp8_quant`: activation per token,
    #     weight per output channel or one scale.
    # A cell planned before the slot existed carries none and keeps the
    # per-channel recipe it was measured with.
    import torch
    from vllm import _custom_ops as ops

    x, w = inputs["x"].contiguous(), inputs["w"].contiguous()   # [M,K], [N,K] bf16
    # vLLM pads M on Hopper — "cutlass blockwise gemm limitation for hopper",
    # `apply_w8a8_block_fp8_linear` — by 4 - M % 4 rows, so a whole extra 4
    # when M is already a multiple; the padded launch is what serving runs.
    pad = 0
    if (inv.weight_granularity == f"per-tile:{_BLK}x{_BLK}"
            and torch.cuda.get_device_capability(x.device) == (9, 0)):
        pad = 4 - x.shape[0] % 4
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    if inv.weight_granularity == f"per-tile:{_BLK}x{_BLK}":
        a_q, a_s = _per_token_cast_to_fp8(x)                    # [M,K], [M,K/128]
        a_s = a_s.t().contiguous().t()                          # column-major
        w_q, w_s = _per_block_cast_to_fp8(w)                    # [N,K], [N/128,K/128]
    else:
        a_q, a_s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)   # [M,1]
        w_q, w_s = ops.scaled_fp8_quant(
            w, use_per_token_if_dynamic=inv.weight_granularity != "per-tensor")
    # b must be column-major [K,N]: w_q.t() is [K,N] with stride (1,K). The
    # weight scale follows it as a CONTIGUOUS [K/128, N/128] for the block
    # path — a transposed view of [N/128, K/128] is read with the wrong
    # strides and the result is merely close (cosine 0.995 against the
    # dequantized product on an H100, 0.999999 contiguous). A per-channel
    # [N,1] or per-tensor scale is the same either way.
    return BenchCase(
        run=lambda: ops.cutlass_scaled_mm(
            a_q, w_q.t(), scale_a=a_s, scale_b=w_s.t().contiguous(),
            out_dtype=torch.bfloat16),
        to_canonical=lambda out: out[:out.shape[0] - pad],
    )


def _vllm_scaled_mm_params(inv: Invocation) -> str:
    """Decline a block-scaled cell whose K is off the 128 tile before inputs
    are built: the activation is quantized in 128-wide groups along K, and
    vLLM refuses such a shard at load. N may end in a partial block — that
    is the shape vLLM sends here instead of DeepGEMM."""
    if (inv.weight_granularity or "").startswith("per-tile:") and (
            int(inv.dims["d_reduce"]) % _BLK):
        raise NotImplementedError("block-scaled FP8 needs K on the 128 tile")
    return ""


def _deepgemm_params(inv: Invocation) -> str:
    """Decline a shape this tile's kernel does not compile for, before inputs
    are built. The bound DEPENDS on the tile, which is why it is a gate here
    and not a `divisible` in the signature: DeepGEMM needs K and N on the
    128x128 scale tile for a 128-block checkpoint (production gates N%64; the
    128-aligned case is what gets benched — every transformer projection width
    is 128-aligned — and the rest is declined rather than padded into a
    different measurement), while the 32-granularity path states its own
    narrower contract, `n % 8 == 0 and k % 32 == 0`
    (`transform_sf_into_required_layout`)."""
    tile = _scale_tile(inv)
    k, n = int(inv.dims["d_reduce"]), int(inv.dims["d_out"])
    mod_k, mod_n = (tile, tile) if tile == _BLK else (32, 8)
    if k % mod_k or n % mod_n:
        raise NotImplementedError(
            f"DeepGEMM's {tile}-granularity fp8 GEMM needs K%{mod_k}==0 and "
            f"N%{mod_n}==0; this cell is K={k} N={n}")
    return ""


#: DeepGEMM has kernels for sm90, sm100 and sm120 and nothing older: every
#: implementation under `deep_gemm/include/deep_gemm/impls/` is one of
#: `sm90_*`, `sm100_*`, `sm120_*` (or the arch-agnostic `smxx_*` helpers,
#: which still call `cudaGridDependencySynchronize`, PTX `griddepcontrol`,
#: smxx_layout.cuh:36). On an Ada card the import works
#: — sglang ships `sgl-deep-gemm` in its env — and the first launch JITs,
#: then ptxas refuses ("Instruction 'griddepcontrol' requires .target sm_90
#: or higher", 10,407 times on the 4090), and in DeepSeek-V3/V3.1/V4-Flash's
#: sglang env the next block-scaled linear then hung past the 900 s
#: watchdog. The floor is the kernel's availability, like the import.
_DEEPGEMM_MIN_SM = 90


register(Provider(
    library="deepgemm", signature="gemm", kernel_impl="deepgemm_fp8_blockwise",
    import_name="deep_gemm",
    # 128x128 ONLY, and the 32-wide tile is deliberately not claimed here:
    # DeepGEMM's sm90 kernel template does compile `SF_GRAN == 32`
    # (`sm90_fp8_gemm_1d2d.cuh`), but no host path reaches it — every form of
    # a 32 recipe (`recipe=(1,32,32)`, `recipe_a=(1,32)/recipe_b=(32,32)`,
    # `(1,1,32)`, fp32 scales, packed UE8M0 scales) is refused at
    # `csrc/apis/layout.hpp:57` "Unknown SF transformation", measured on an
    # H100 with deep_gemm 2.8.2. The 1d1d family states why: `arch_major == 9
    # and gran_k == 128`, so 32 is Blackwell-only there. A 32-block
    # checkpoint is served by the Triton provider below instead.
    serves=KernelSignature(signature="gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=(
                               _DEEPGEMM_ACTIVATION_GRANULARITY,),
                           divisible=(("d_reduce", _BLK), ("d_out", _BLK)),
                           weight_granularity=(f"per-tile:{_BLK}x{_BLK}",)),
    build=_deepgemm_build,
    impl_params=_deepgemm_params,
    min_sm=_DEEPGEMM_MIN_SM,
))
# One build, two providers, because the card decides whether CUTLASS has a
# BLOCK-scaled W8A8 kernel at all: vLLM asks
# `cutlass_scaled_mm_supports_block_fp8(capability)` and on Ada (8.9) the
# answer is False — it runs its Triton blockwise GEMM there instead. The op
# still accepts the call and returns numbers: measured on an RTX 4090 the
# block path came back at cosine 0.45-0.78 against its reference and
# poisoned the CUDA context, which then failed the NEXT cell. The
# per-channel and per-tensor recipes have no such floor and keep serving
# every card.
register(Provider(
    library="vllm", signature="gemm",
    kernel_impl="vllm_cutlass_scaled_mm_blockwise", import_name="vllm",
    # CUTLASS W8A8 tolerates a 16-element tile, so it serves shapes DeepGEMM
    # declines — the two overlap on the aligned ones, which is the comparison.
    serves=KernelSignature(signature="gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=(
                               _DEEPGEMM_ACTIVATION_GRANULARITY,),
                           divisible=(("d_reduce", 16), ("d_out", 16)),
                           weight_granularity=(f"per-tile:{_BLK}x{_BLK}",)),
    build=_vllm_scaled_mm_build,
    impl_params=_vllm_scaled_mm_params,
    min_sm=90,
))
register(Provider(
    library="vllm", signature="gemm", kernel_impl="vllm_cutlass_scaled_mm",
    import_name="vllm",
    serves=KernelSignature(signature="gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=("per-token",),
                           divisible=(("d_reduce", 16), ("d_out", 16)),
                           weight_granularity=("per-channel", "per-tensor")),
    build=_vllm_scaled_mm_build,
    impl_params=_vllm_scaled_mm_params,
))


# ---------------------------------------------------------------------------
# grouped_gemm (MoE): per-expert weights, tokens grouped per expert. A torch
# bmm baseline (runs anywhere, bf16) + DeepGEMM's masked grouped fp8 kernel.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weight-only quantized GEMM (W4A16 / W8A16).
#
# The corpus is full of these — an fp4_e2m1 weight against a bf16 activation is
# how every 4-bit checkpoint runs — and until now nothing measured them AS
# weight-only. torch upcasts the weight and measures a bf16 GEMM; DeepGEMM and
# cutlass_scaled_mm both require an fp8 ACTIVATION, so they decline. The whole
# point of W4A16 is that the weight read shrinks 4× while the math stays f16,
# and that is exactly the axis no provider was exercising.
#
# Both go through the framework's own dispatch (`apply_*_marlin_linear`), not
# the raw `marlin_gemm` op: the repack layout, workspace sizing and the
# is_k_full / act-order decisions are part of what the framework runs, and a
# hand-rolled call would measure a configuration nobody serves.
# ---------------------------------------------------------------------------

# Marlin quantizes along K in groups; 128 is what every production config uses.
_MARLIN_GROUP = 128


def _weight_dtype(inv: Invocation) -> str:
    return inv.dtype_env.get("weight", "")


# 4-bit weight-only comes in two formats that are NOT interchangeable, and
# Marlin has a separate path for each:
#   * int4 / GPTQ  — uniformly spaced levels, -8..7, `scalar_types.uint4b8`
#   * fp4_e2m1     — E2M1 floating point: 0, .5, 1, 1.5, 2, 3, 4, 6 (×sign)
# A per-group scale absorbs a common factor, NOT the difference in spacing, so
# running an E2M1 checkpoint through the GPTQ kernel measures a quantization
# nobody deployed. They get separate providers because they are separate
# kernels, and a row must be able to say which one it timed.
def _is_int4_w4a16(inv: Invocation) -> bool:
    return (_weight_dtype(inv) == "int4"
            and inv.dtype_env.get("activation", "") in ("bf16", "fp16"))


def _is_nvfp4_w4a16(inv: Invocation) -> bool:
    """NVFP4 only. MXFP4 is a DIFFERENT format that happens to share the e2m1
    element encoding: its block is 32 elements with an e8m0 (power-of-two)
    scale and no global scale, against NVFP4's 16 elements, e4m3 block scale
    and one fp32 global scale. vLLM keeps them on separate paths for that
    reason (`prepare_fp4_layer_for_marlin` vs the mxfp4 preparers), so lumping
    them here would repack an MXFP4 checkpoint into NVFP4's layout and time a
    kernel nobody dispatches."""
    return (_weight_dtype(inv) == "fp4_e2m1"
            and inv.dtype_env.get("activation", "") in ("bf16", "fp16"))


def _is_w8a16(inv: Invocation) -> bool:
    """e4m3 only. e5m2 trades three mantissa bits for one exponent bit — a
    different dynamic range and a different scale, not a variant of the same
    thing — and `apply_fp8_marlin_linear` quantizes to e4m3. Matching on the
    `fp8` prefix would hand it an e5m2 cell and record the result as if that
    were the kernel for it."""
    return (_weight_dtype(inv) == "fp8_e4m3"
            and inv.dtype_env.get("activation", "") in ("bf16", "fp16"))


#: The scale groups Marlin compiles for, from vLLM's own
#: `MARLIN_SUPPORTED_GROUP_SIZES` (-1 is per-channel, handled by the W8A16
#: build). A checkpoint picks one — GPTQ and AWQ releases ship 128, Kimi-K2-
#: Thinking's INT4 ships 32 — and the group is part of the measurement: the
#: same 4-bit kernel reads 4× as many scales at 32 and resolves the weight
#: differently.
_MARLIN_GROUPS = (32, 64, 128)


def _marlin_group(inv: Invocation) -> int:
    """The scale group this cell's checkpoint stores, from the granularity the
    walk states (`per-group:<g>`), defaulting to 128 for a cell planned before
    the slot existed — the group those rows were measured with."""
    gran = inv.weight_granularity or ""
    if not gran.startswith("per-group:"):
        return _MARLIN_GROUP
    group = int(gran[len("per-group:"):])
    if group not in _MARLIN_GROUPS:
        raise NotImplementedError(
            f"Marlin compiles group sizes {_MARLIN_GROUPS}, not {group}")
    return group





def _marlin_int4_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """GPTQ int4 weight-only GEMM, W4A16.

    WHAT IS TIMED
        `apply_gptq_marlin_linear` — vLLM's own dispatch entry, the same call
        `GPTQMarlinLinearMethod.apply` makes when serving a GPTQ checkpoint.
        Not the raw `ops.marlin_gemm`: the workspace sizing, the `is_k_full`
        and act-order decisions and the fp32-reduce choice are part of what the
        framework runs, and calling underneath them would time a configuration
        no deployment uses.

    WHAT IS SET UP OUTSIDE THE TIMED CALL
        Quantization and the Marlin repack. Both are load-time work in a real
        server (`process_weights_after_loading`), so including them would
        measure checkpoint loading, not inference. `gptq_marlin_repack` is the
        framework's, not ours — the packed layout is an implementation detail
        of the kernel and must not be reimplemented here.

    QUANTIZATION
        Per-group symmetric int4 over K, group = 128, levels −8..7 offset by
        +8 into the `uint4b8` scalar type Marlin expects. The canonical bf16
        weight is the source, so every provider on this cell starts from the
        same numbers.

    WHY THE CELL MAY BE DECLINED
        K not divisible by the group, or N not divisible by 64 — see
        the `divisible` each Provider declares. Production gates the same
        divisibility rather than padding — a shape that does not divide is
        declined and said so, not padded into a different measurement.

    COMPARABILITY
        The reference is the shared fp32 GEMM oracle, so the cross-check
        measures quantization error, which is why `_XCHECK_RTOL` for 4-bit
        weights is loose. A latency here is comparable to torch/cuBLAS on the
        same cell only in the weight-only sense: the math is still f16, but
        the weight read is 4× smaller, which is the whole point of the format.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx, marlin_make_workspace_new)
    from vllm.scalar_type import scalar_types

    x, w = inputs["x"], inputs["w"]                       # [M,K] bf16, [N,K] bf16
    dt = _torch_dtype(inv.dtype)
    x = x.to(dt)
    N, K = w.shape
    dev = x.device
    wtype = scalar_types.uint4b8                          # GPTQ 4-bit (zp folded into b8)

    # Quantize the canonical bf16 weight to 4 bits per group, then hand it to
    # the framework's repacker — the packed layout is Marlin's, not ours.
    group = _marlin_group(inv)
    wg = w.float().reshape(N, K // group, group)
    scale = wg.abs().amax(dim=2).clamp_min(1e-6) / 7.0    # int4 signed range
    q = (wg / scale.unsqueeze(2)).round().clamp(-8, 7).to(torch.int32) + 8
    q = q.reshape(N, K)
    # int32 container, 8 values per word, K-major — the GPTQ on-disk layout
    # `gptq_marlin_repack` expects.
    qw = torch.zeros(K // 8, N, dtype=torch.int32, device=dev)
    qt = q.t().contiguous()                               # [K,N]
    for i in range(8):
        qw |= (qt[i::8] & 0xF) << (4 * i)
    g_idx = torch.empty(0, dtype=torch.int32, device=dev)
    sort_idx = torch.empty(0, dtype=torch.int32, device=dev)
    packed = ops.gptq_marlin_repack(qw, sort_idx, K, N, wtype.size_bits)
    scales = scale.t().contiguous().to(dt)                # [K//group, N]
    workspace = marlin_make_workspace_new(dev)
    zp = marlin_make_empty_g_idx(dev)

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear)
    return BenchCase(
        run=lambda: apply_gptq_marlin_linear(
            x, packed, scales, zp, g_idx, sort_idx, workspace, wtype,
            output_size_per_partition=N, input_size_per_partition=K,
            is_k_full=True, bias=None),
        to_canonical=lambda out: out,
    )


def _marlin_w8a16_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """fp8-e4m3 weight-only GEMM, W8A16.

    WHAT IS TIMED
        `apply_fp8_marlin_linear`. This is the weight-only fp8 deployment: the
        weight is stored e4m3 and the math runs in f16. It is a DIFFERENT
        configuration from `vllm_cutlass_scaled_mm` and `deepgemm_fp8_blockwise`
        on the same cell, which are W8A8 — activation quantized too. Comparing
        this row against those measures the cost of keeping the activation in
        f16, not two implementations of one thing.

    WHAT IS SET UP OUTSIDE THE TIMED CALL
        Per-tensor quantization and `prepare_fp8_layer_for_marlin`, the
        framework's own repack, on a stand-in `nn.Module` because that is the
        interface it takes. Load-time work in a server.

    WHY THE CELL MAY BE DECLINED
        Same divisibility as every Marlin path; e5m2 weights are excluded
        because the quantization here is e4m3 (`_is_w8a16`).
    """
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

    x, w = inputs["x"], inputs["w"]
    dt = _torch_dtype(inv.dtype)
    x = x.to(dt)
    N, K = w.shape
    dev = x.device
    amax = w.float().abs().amax().clamp_min(1e-6)
    w_q = (w.float() / (amax / _FP8_MAX)).to(torch.float8_e4m3fn)
    w_scale = (amax / _FP8_MAX).reshape(1).to(torch.float32).to(dev)

    # The framework repacks fp8 weights into Marlin layout on a layer object;
    # mirror that rather than reimplement the permutation.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_q.t().contiguous(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_scale, requires_grad=False)
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N
    layer.orig_dtype = dt
    prepare_fp8_layer_for_marlin(layer, size_k_first=True)
    workspace = marlin_make_workspace_new(dev)
    return BenchCase(
        run=lambda: apply_fp8_marlin_linear(
            x, layer.weight, layer.weight_scale, workspace, N, K, None,
            input_dtype=dt),
        to_canonical=lambda out: out,
    )


def _marlin_fp4_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """NVFP4 (E2M1) weight-only GEMM, W4A16.

    WHAT IS TIMED
        `apply_fp4_marlin_linear` — Marlin's own fp4 path, NOT the GPTQ one.
        The two differ in the value grid, not just the bit width: E2M1 spaces
        its levels 0, .5, 1, 1.5, 2, 3, 4, 6 while int4 spaces them uniformly,
        and a per-group scale absorbs a common factor rather than that
        difference. Routing an E2M1 checkpoint through the GPTQ kernel would
        time a quantization nobody deployed.

    WHAT IS SET UP OUTSIDE THE TIMED CALL
        Quantization and `prepare_fp4_layer_for_marlin`, which owns the repack.
        Load-time work in a server, so not part of the measurement.

    QUANTIZATION
        NVFP4's two-level scale, built here only so the framework's preparer
        has its expected inputs: an e4m3 block scale every 16 elements along K,
        one fp32 global scale, two e2m1 values packed per byte. MXFP4 shares
        the element encoding but blocks 32 elements under an e8m0 scale with no
        global scale — a different schema, gated out by `_is_nvfp4_w4a16` and
        served by different preparers in vLLM.

    WHY THE CELL MAY BE DECLINED
        Same divisibility as every Marlin path, declared in `serves`.

    COMPARABILITY
        Directly comparable to `marlin_gptq_int4_w4a16` on the same cell: same
        bit width, same memory traffic, different value grid — so a latency gap
        between them is the kernel, not the format's size. Against
        `torch_cublas_linear` it is the weight-only win being measured.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin)

    x, w = inputs["x"], inputs["w"]
    dt = _torch_dtype(inv.dtype)
    x = x.to(dt)
    N, K = w.shape
    dev = x.device

    # NVFP4 packs two e2m1 values per byte with an e4m3 scale every 16 elements
    # and one fp32 global scale — the layout the framework's repacker reads.
    blk = 16
    wg = w.float().reshape(N, K // blk, blk)
    amax = wg.abs().amax(dim=2).clamp_min(1e-6)
    gscale = (amax.max() / (6.0 * 448.0)).clamp_min(1e-12)      # e2m1 max 6, e4m3 max 448
    bscale = (amax / 6.0 / gscale).to(torch.float8_e4m3fn)
    q = (wg / (bscale.float().unsqueeze(2) * gscale)).clamp(-6, 6)
    lv = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=dev)
    idx = (q.abs().unsqueeze(-1) - lv).abs().argmin(-1).to(torch.uint8)
    idx |= (q < 0).to(torch.uint8) << 3
    idx = idx.reshape(N, K)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()  # [N, K//2] uint8

    # The preparer reads `params_dtype` off the layer and CREATES `workspace` on
    # it — passing our own would be a second, unused buffer while the kernel
    # read the one it made.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(bscale, requires_grad=False)
    # Named `weight_global_scale`, not anything else: the preparer decides the
    # format by ATTRIBUTE NAME — `is_nvfp4 = hasattr(layer,
    # "weight_global_scale")` — and that choice sets group_size to 16 (nvfp4)
    # or 32 (mxfp4). A different name silently gets the mxfp4 repack for an
    # nvfp4 packing, which surfaces one layer down as `Invalid thread config`.
    layer.weight_global_scale = torch.nn.Parameter(
        gscale.reshape(1).to(torch.float32), requires_grad=False)
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N
    layer.params_dtype = dt
    layer.group_size = blk
    prepare_fp4_layer_for_marlin(layer)
    workspace = getattr(layer, "workspace", None)
    if workspace is None:
        workspace = marlin_make_workspace_new(dev)
    # `apply_fp4_marlin_linear` switches format on whether a global scale is
    # passed (`is_nvfp4 = weight_global_scale is not None`), and the kernel
    # rejects one unless the WEIGHT was repacked as nvfp4. Passing a global
    # scale alongside a plain `prepare_fp4_layer_for_marlin` repack is the
    # mismatch `global_scale can only be used for nvfp4 format` names. Handing
    # over exactly what the preparer produced keeps the two ends agreeing.
    return BenchCase(
        # Both ends choose the format the same way and must be told
        # consistently: the preparer by the attribute's presence
        # (`is_nvfp4 = hasattr(layer, "weight_global_scale")`, which sets
        # group_size to 16 rather than mxfp4's 32), `apply` by this argument
        # being non-None. Giving one and not the other packs the weight one way
        # and reads it the other.
        run=lambda: apply_fp4_marlin_linear(
            x, layer.weight, layer.weight_scale, layer.weight_global_scale,
            workspace, N, K, bias=None, input_dtype=dt),
        to_canonical=lambda out: out,
    )


register(Provider(
    library="marlin", signature="gemm", kernel_impl="marlin_gptq_int4_w4a16",
    # Per-group symmetric int4 over K, group 128 — what `_marlin_int4_build`
    # quantizes with, declared so the harness rounds to the same grid.
    weight_group=_MARLIN_GROUP,
    # `scales = scale.t().contiguous().to(dt)` — the scale is held in the
    # ACTIVATION dtype, not fp32.
    weight_scale="activation",
    # DISABLED, not deleted: its cells stay visible as a gap rather than as
    # coverage nobody wanted. Two reasons now, and the second is the bigger one.
    #
    # ACCURACY. On an H100 this kernel sits at cosine 0.982-0.987 against the
    # fp32 reference while marlin's fp4 and fp8 paths, on the same grid
    # mechanism, reach 1.00000. The weight is NOT the cause: the round trip
    # through this build's own quantizer reproduces the placed grid exactly
    # (max|delta| = 0.0), and the error is flat across 1, 2, 4 and 8 scale
    # groups, so it is not the per-group scale layout either. That leaves the
    # GPTQ path this one alone takes — the int32 nibble packing or the uint4b8
    # zero point.
    #
    # AND IT NO LONGER FITS THE AUDITED RELEASE. `_marlin_int4_build` is
    # written against vLLM's pre-0.30 GPTQ Marlin API, and all three of the
    # entry points it touches changed when act-order plumbing moved inside the
    # repacked weight (verified against v0.32.0 on disk, and the symbol check
    # run in the box's marlin env at 0.32.0):
    #   * `marlin_make_empty_g_idx` is now `marlin_make_empty`
    #     (marlin_utils.py:508). The two names never coexist: 0.29.0 has only
    #     the old one, 0.30.0 only the new.
    #   * `ops.gptq_marlin_repack` lost its `perm` argument —
    #     `(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False)` at
    #     _custom_ops.py:1091, where this build passes `sort_idx` second.
    #   * `apply_gptq_marlin_linear` dropped `g_idx`, `g_idx_sort_indices` and
    #     `is_k_full` and gained `input_global_scale` / `input_dtype`
    #     (marlin_utils.py:752), so this build's positional call lands its
    #     arguments on the wrong parameters.
    # So re-enabling it is a re-authoring against the new API, not an import
    # fix: renaming the one moved symbol would turn an ImportError into a
    # silently misaligned call, which is strictly worse than failing to load.
    # Deliberately NOT routed through `linear_ssm._relocated` for the same
    # reason — that helper resolves a kernel whose MODULE moved while its
    # contract held, and nothing here held.
    #
    # The sibling marlin providers are unaffected: they import only symbols
    # that are where this file already names them at 0.32.0, and both measure
    # and cross-check there (fp4 and fp8, 2880x4096 at s_query 128 and 1024).
    disabled_reason=(
        "written against vLLM's pre-0.30 GPTQ Marlin API — marlin_make_empty"
        "_g_idx renamed, gptq_marlin_repack lost `perm`, apply_gptq_marlin_"
        "linear lost g_idx/is_k_full — so it does not load or call correctly "
        "at the audited release; and at 0.11.0, where it did, cosine "
        "0.982-0.987 against the fp32 reference with the weight round-trip "
        "exact and the error group-count independent, leaving the GPTQ int32 "
        "packing / uint4b8 zero point unaccounted for"),
    import_name="vllm",
    build=_marlin_int4_build,
    serves=KernelSignature(
        signature="gemm", weight=("int4",), activation=("bf16", "fp16"),
        divisible=(("d_reduce", _MARLIN_GROUP), ("d_out", 64))),
))
register(Provider(
    library="marlin", signature="gemm", kernel_impl="marlin_fp4_e2m1_w4a16",
    # NVFP4: an e4m3 block scale every 16 elements. NOT 128 — the same E2M1
    # element encoding at a different block size is a different grid.
    weight_group=16,
    # Two levels: `bscale = (amax/6/gscale).to(e4m3)` times one fp32 global.
    weight_scale="e4m3_global",
    import_name="vllm",
    build=_marlin_fp4_build,
    serves=KernelSignature(
        signature="gemm", weight=("fp4_e2m1",), activation=("bf16", "fp16"),
        divisible=(("d_reduce", _MARLIN_GROUP), ("d_out", 64))),
))
register(Provider(
    library="marlin", signature="gemm", kernel_impl="marlin_fp8_w8a16",
    # One scale for the whole tensor (`amax / 448`), hence group 0.
    weight_group=0,
    import_name="vllm",
    # W8A16: an FP8 weight multiplied against a 16-bit activation, with one
    # scale for the whole weight. Every FP8 checkpoint in this corpus
    # quantises the activation too, and none keeps a single scale: all
    # 121,100 FP8-weight `gemm` cells of the plans carry an FP8 activation
    # with per-tile 128x128, per-tile 32x32 or per-channel weight scales. So
    # both halves of what this kernel reads are absent, and the kernel is
    # still right: the gap is the corpus's, not the declaration's.
    unreached_reason=("no corpus checkpoint stores an FP8 weight beside a "
                      "16-bit activation, and none scales one per tensor; "
                      "every FP8 model here is W8A8 with tile or channel "
                      "scales"),
    build=_marlin_w8a16_build,
    serves=KernelSignature(
        signature="gemm", weight=("fp8_e4m3",), activation=("bf16", "fp16"),
        divisible=(("d_reduce", _MARLIN_GROUP), ("d_out", 64))),
))


# ---------------------------------------------------------------------------
# humming — sglang's JIT-compiled quantized GEMM (sglang 0.5.16 depends on it).
# Entry is `HummingLayer`, which owns the packing: `load_from_unquantized` takes
# a plain f16 weight and produces whatever the configured quant method needs, so
# the measured call is the same one sglang issues.
# ---------------------------------------------------------------------------

# humming is a QUANTIZED GEMM: its kernel asserts `b_dtype.num_bits <= 8`, so a
# 16-bit weight is outside what it implements, not merely untuned. Listing
# bf16/fp16 here made the gate wider than the build and every such cell failed
# on an assertion with no message.
_HUMMING_WEIGHT_DTYPE = {
    "fp4_e2m1": "float4e2m1", "int4": "int4",
    "fp8_e4m3": "float8e4m3", "fp8_e5m2": "float8e5m2",
}
# `mxfp4` is deliberately absent: humming names its element type
# (`float4e2m1`), but MXFP4's 32-element e8m0 block scale is a different
# weight schema from NVFP4's, and mapping both onto one name would quantize
# the weight one way and label it another.



def _humming_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """humming quantized GEMM — sglang 0.5.16's kernel library.

    WHAT IS TIMED
        `HummingLayer.forward`, the same entry sglang calls. humming launches
        through its own `libhumming_launcher.so` rather than a registered torch
        op, so nothing about this call goes through torch dispatch — which also
        means a torch-level profile shows nothing and only CUPTI sees the
        kernel.

    WHAT IS SET UP OUTSIDE THE TIMED CALL
        `load_from_unquantized` (quantize + pack from the canonical bf16
        weight) and `transform()`. `transform()` is easy to miss and not
        optional: it builds `humming_metas`, the per-sublayer kernel metadata
        `forward` indexes, and every call raises `'HummingLayer' object has no
        attribute 'humming_metas'` without it. It also repacks the weight into
        the layout the kernel reads, so it is load-time work either way.

    KERNEL SELECTION
        humming is JIT: the kernel is compiled per configuration on first use.
        The timing loop's warmup absorbs that compile — a profiled or timed
        FIRST call would record compilation, not the kernel.

    WHY THE CELL MAY BE DECLINED
        A weight dtype outside `_HUMMING_WEIGHT_DTYPE` (notably `mxfp4`, whose
        32-element e8m0 block schema differs from the `float4e2m1` element name
        it would otherwise map onto), a non-f16 activation, or a shape that
        does not divide.

    COMPARABILITY
        At bf16 weights this lands on the same cells as
        `torch_cublas_linear` — sglang's GEMM against cuBLAS, which is the
        comparison worth having. The quantized paths land beside the Marlin
        providers on their respective formats.
    """
    from humming.layer import HummingLayer

    x, w = inputs["x"], inputs["w"]
    dt = _torch_dtype(inv.dtype)
    x = x.to(dt)
    N, K = w.shape
    layer = HummingLayer(
        shape_n=N, shape_k=K,
        weight_config={"dtype": _HUMMING_WEIGHT_DTYPE[_weight_dtype(inv)],
                       "group_size": _MARLIN_GROUP},
        torch_dtype=dt,
    ).to(x.device)
    layer.load_from_unquantized(w.to(dt))
    # `transform()` is what builds `humming_metas` — the per-sublayer kernel
    # metadata `forward` indexes. Constructing the layer and loading weights is
    # not enough: without it every call raises `'HummingLayer' object has no
    # attribute 'humming_metas'`. It is also where the weight is repacked into
    # the layout the kernel reads, so it belongs here in build, not in the
    # timed call.
    layer.transform()
    return BenchCase(run=lambda: layer.forward(x), to_canonical=lambda out: out)


register(Provider(
    library="humming", signature="gemm", kernel_impl="humming_gemm",
    # humming quantizes inside `load_from_unquantized` and the element rule
    # lives in the compiled `ops.quant_weight`, so this grid could not be read
    # off the source. It is DERIVED from the readable half — `weight_config`
    # passes `group_size=128` and the scale dtype is `bs_dtype or f16_dtype`,
    # i.e. the activation dtype — and then CHECKED against the kernel: with the
    # weight placed on this grid the residual has to collapse, because a correct
    # quantizer of an already-representable weight is the identity. A guess that
    # failed that check would be visible immediately rather than quietly
    # rounding the weight somewhere humming does not.
    weight_group=_MARLIN_GROUP,
    weight_scale="activation",
    import_name="humming",
    build=_humming_build,
    serves=KernelSignature(
        signature="gemm", weight=tuple(_HUMMING_WEIGHT_DTYPE),
        activation=("bf16", "fp16"),
        divisible=(("d_reduce", _MARLIN_GROUP), ("d_out", 64))),
))


def _torch_grouped_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)        # [E,Sq,K], [E,N,K]
    counts = inv.moe_token_counts
    # ONE ragged launch, torch's own grouped GEMM: every expert's routed
    # rows packed contiguously (the layout a grouped kernel is handed, as
    # the DeepGEMM contiguous build packs them), the whole weight stack, and
    # the row offsets. Only routed rows are computed — the walk's FLOPs
    # follow them, not the padded buffer.
    #
    # This was a Python loop of one matmul per expert, and every launch in
    # it read ONE expert's weight: the launch gate refused all of them ("the
    # launch's largest operand holds N·K elements where the cell's holds
    # E·N·K — it measures a part of the cell"), and its wall clock was host
    # dispatch besides.
    rows = _packed_rows(x, counts)
    offsets = torch.tensor(counts, dtype=torch.int32,
                           device=x.device).cumsum(0, dtype=torch.int32)
    return BenchCase(
        run=Launch(torch._grouped_mm, rows, w.transpose(1, 2), offs=offsets),
        to_canonical=lambda flat: _ragged_to_padded(flat, counts, x.shape[1]))


def _packed_rows(x, counts):
    """Each expert's routed rows of the padded `[E, Sq, K]` x, packed into
    `[sum(counts), K]` — the layout a ragged grouped kernel is handed.

    A view when no expert is padded (every count is `Sq`, which a balanced
    distribution usually is), a copy only when there is padding to drop.
    Copied regardless, the harness held the activation twice — the canonical
    input the cross-check keeps, and the packed copy — where serving holds
    it once: at the H100's largest planned prefill cell (2,621,440 rows of
    8,192) the copy was a second 40 GiB, and the cell whose operands fit
    the card ran out of memory in the build."""
    E, Sq, K = x.shape
    if all(int(count) == Sq for count in counts):
        return x.reshape(E * Sq, K)
    import torch
    return torch.cat([x[expert, :count]
                      for expert, count in enumerate(counts)], dim=0)


def _ragged_to_padded(flat, counts, maximum):
    out = flat.new_zeros((len(counts), maximum, flat.shape[-1]))
    offset = 0
    for expert, count in enumerate(counts):
        out[expert, :count] = flat[offset:offset + count]
        offset += count
    return out


def _expected_m(counts) -> int:
    """DeepGEMM's `expected_m`: the AVERAGE rows per expert, not the buffer's.

    A third quantity beside `max_m` (how many rows the buffer holds) and
    `masked_m[e]` (how many of them expert e really has). sglang derives it
    from the routing, never from the allocation:

        expected_m = (topk_ids.numel() - 1) // num_local_experts + 1
            — srt/layers/moe/ep_moe/kernels.py, and the same quantity again
              in srt/layers/moe/token_dispatcher/mooncake.py

    which is ceil(routed rows / local experts). The buffer in that same
    function is sized separately (`m_max = (tokens // 256 + 1) * 256`), so
    the source itself keeps the two apart. It matters because `expected_m`
    is a JIT config key — `deep_gemm_execution_hook(expected_m, n, k, …)` in
    srt/layers/deep_gemm_wrapper/entrypoint.py — so it SELECTS the kernel.
    Passing the padded width would compile the schedule for a tile count the
    launch does not have, and the row would time a config production never
    runs. Once `masked_m` carries the real ragged counts, this must follow
    them or the two disagree about the same launch.
    """
    return -(-sum(counts) // len(counts)) if counts else 0


def _deepgemm_grouped_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # DeepGEMM masked grouped fp8: m_grouped_fp8_gemm_nt_masked(  (DeepGEMM
    # 2.x's name; 1.x called it fp8_m_grouped_gemm_nt_masked)
    #   (lhs_q[E,Sq,K], lhs_s[E,Sq,K//128]), (rhs_q[E,N,K], rhs_s[E,N//128,K//128]),
    #   out[E,Sq,N] bf16, masked_m[E], expected_m). Per-expert per-token (act) +
    #   per-128×128-block (weight) fp8 scales — same prep as the dense path,
    #   applied within each expert. From sglang deep_gemm_wrapper call sites.
    import torch
    import deep_gemm

    x, w = inputs["x"], inputs["w"]                       # [E,Sq,K], [E,N,K] bf16
    E, Sq, K = x.shape
    N = w.shape[1]

    # `max_m` and `masked_m` are DIFFERENT quantities, and conflating them is
    # what broke this provider. The buffers carry `max_m` rows — DeepGEMM's own
    # tests size them well above the token count and pad — while `masked_m[e]`
    # says how many of those rows are real. Setting both to Sq produced garbage
    # at every shape smaller than a tile: E=8, Sq=1 came back with cosine 0.0
    # and a relative error of 1e37. Padded to the 128-row tile with masked_m
    # left at the true count, every case measured lands at cosine 0.9992+ and
    # rel_err ~0.2, which is the fp8 band. That is also the production shape: a
    # buffer sized for capacity, with the per-expert token count in masked_m.
    max_m = -(-Sq // _BLK) * _BLK
    xp = x if max_m == Sq else torch.zeros(E, max_m, K, dtype=x.dtype,
                                           device=x.device)
    if max_m != Sq:
        xp[:, :Sq] = x

    # Per GROUP, and with NO scale-layout transform. DeepGEMM's k-major grouped
    # path takes a plain contiguous [E, max_m, K/128] fp32 scale — its own
    # `grouped_cast_fp8_fp4_with_major` builds exactly that and applies nothing
    # else. `get_mn_major_tma_aligned_tensor` is the DENSE 2-D helper; used here
    # it returns a strided view that happens to be right for the first group and
    # wrong for the rest, which is why expert 0 always agreed and the others did
    # not.
    x_q = torch.empty(E, max_m, K, dtype=torch.float8_e4m3fn, device=x.device)
    x_s = torch.empty(E, max_m, K // _BLK, dtype=torch.float32, device=x.device)
    for e in range(E):                                   # setup (not timed)
        x_q[e], x_s[e] = _per_token_cast_to_fp8(xp[e])
        x_s[e, Sq:] = 0        # DeepGEMM zeroes SFA rows past masked_m
    w_q = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=x.device)
    w_s = torch.empty(E, N // _BLK, K // _BLK, dtype=torch.float32, device=x.device)
    for e in range(E):
        w_q[e], w_s[e] = _per_block_cast_to_fp8(w[e])
    out = torch.zeros(E, max_m, N, dtype=torch.bfloat16, device=x.device)
    counts = inv.moe_token_counts
    masked_m = torch.tensor(counts, dtype=torch.int32, device=x.device)
    for expert, count in enumerate(counts):
        x_s[expert, count:] = 0
    return BenchCase(
        run=lambda: deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (x_q, x_s), (w_q, w_s), out, masked_m=masked_m,
            expected_m=_expected_m(counts)),
        # Only the real rows: the padding exists for the kernel's tile, not for
        # the comparison.
        to_canonical=lambda _out: out[:, :Sq],
    )


def _deepgemm_grouped_contig_params(inv: Invocation) -> str:
    counts = inv.moe_token_counts
    if any(count % _BLK for count in counts):
        raise NotImplementedError(
            f"contiguous grouped GEMM requires every expert count divisible "
            f"by {_BLK}; got {counts}")
    return ""


def _deepgemm_grouped_contig_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # DeepGEMM CONTIGUOUS grouped fp8 (m_grouped_fp8_gemm_nt_contiguous) — the
    # other half of DeepGEMM's contiguous-vs-masked switch (sglang
    # DeepGemmRunnerCore.use_masked_gemm). Tokens are laid out contiguously
    # across experts (M=E·Sq rows), `m_indices[M]` gives each row's expert.
    import torch
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

    x, w = inputs["x"], inputs["w"]                      # [E,Sq,K], [E,N,K] bf16
    E, Sq, K = x.shape
    N = w.shape[1]
    counts = inv.moe_token_counts
    source = _packed_rows(x, counts)
    a_q, a_s = _per_token_cast_to_fp8(source)
    a_s = get_mn_major_tma_aligned_tensor(a_s)
    w_q = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=x.device)
    w_s = torch.empty(E, N // _BLK, K // _BLK, dtype=torch.float32, device=x.device)
    for e in range(E):
        w_q[e], w_s[e] = _per_block_cast_to_fp8(w[e])
    out = torch.zeros(sum(counts), N, dtype=torch.bfloat16, device=x.device)
    m_indices = torch.arange(E, dtype=torch.int32,
                             device=x.device).repeat_interleave(
                                 torch.tensor(counts, dtype=torch.int64,
                                              device=x.device))
    return BenchCase(
        # `grouped_layout` positionally, not `m_indices=`. DeepGEMM renamed the
        # parameter (it is `m_indices` for m-grouped and `k_indices` for
        # k-grouped, so the shared entry point takes the neutral name), and the
        # keyword form raises TypeError against 2.6.1 — reported under the
        # `m_grouped_fp8_fp4_...` alias, which is why it read as an fp4 problem
        # on an fp8 cell. Swapping the PyPI cu13 `sgl-deep-gemm` for a source
        # build is what moved the API under us.
        run=lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_q, a_s), (w_q, w_s), out, m_indices),
        to_canonical=lambda _out: _ragged_to_padded(out, counts, Sq),
    )


#: Routed rows past which the stitched reference is not measured. It is a
#: CORRECTNESS ANCHOR, not an envelope candidate. Set when one launch was a
#: Python loop of `n_experts` matmuls (now one `torch._grouped_mm`, which the
#: launch gate can judge — the loop it refused on every cell), so its
#: numbers are the loop's: the wall clock was host dispatch rather than
#: device time — a cell at 256 experts recorded p50=2.4ms and cost 3.2s to
#: sample 75 times, 60x a plain GEMM cell of the same plan, for a number no
#: deployment pays (the real kernels are DeepGEMM's grouped launch and
#: vLLM's Triton fused MoE, which carry those cells here). Checking the
#: arithmetic once where it is cheap is the whole of what this build is for,
#: and small cells of every expert count and width remain to do it: 18,863
#: of the H100 plans' 43,617 grouped-GEMM cells are at or under this.
_RAGGED_ANCHOR_ROWS = 2048


register(Provider(
    library="torch", signature="grouped_gemm", kernel_impl="torch_ragged_matmul",
    import_name="torch",
    serves=KernelSignature(
        signature="grouped_gemm",
        # A product, not a per-axis bound: the cost is `n_experts` launches
        # of `s_expert_rows` rows each, and no combination of per-axis ceilings
        # describes that. And a BUDGET, not a capability — the loop runs any
        # size, nobody wants to pay 3.2s a cell to watch host dispatch — so
        # the 10,166 rows measured above it before it existed stay.
        samples_up_to=((("n_experts", "s_expert_rows"), _RAGGED_ANCHOR_ROWS),),
        # `torch._grouped_mm` wants every row stride a multiple of 16 bytes
        # ("strides should be multiple of 16 bytes"): 8 elements at the
        # 2-byte dtypes this floor runs, and so also at fp32.
        divisible=(("d_reduce", 8), ("d_out", 8)),
        weight=TORCH_FLOAT, activation=TORCH_FLOAT),
    # `torch._grouped_mm` runs on every device from 2.9: 2.8 dispatched it
    # to CUDA only and refused below SM90 (native_functions.yaml carries
    # just `CUDA: _grouped_mm_cuda`); 2.9 adds the CompositeExplicitAutograd
    # kernel and a fallback inside `_grouped_mm_cuda` for a card or dtype
    # off the SM90/SM100 bf16 fast path (aten/src/ATen/native/cuda/
    # Blas.cpp:1784-1803 at v2.9.0).
    version_specifier=">=2.9",
    build=_torch_grouped_build,
    fused_op="grouped_gemm_moe",
))
register(Provider(
    library="deepgemm", signature="grouped_gemm",
    kernel_impl="deepgemm_fp8_m_grouped_contiguous",
    import_name="deep_gemm",
    build=_deepgemm_grouped_contig_build,
    impl_params=_deepgemm_grouped_contig_params,
    fused_op="grouped_gemm_moe",
    # `s_expert_rows % 128` is the constraint the old gate was missing, and it
    # did not fail loudly: DeepGEMM's contiguous m-grouped layout wants each
    # expert's rows aligned to the 128-row tile, and when they are not the
    # kernel reads across the boundary into the next expert's rows. It returns
    # numbers either way — cosine against the fp32 oracle drops from 0.99932 to
    # 0.851, which reads like quantization noise rather than a wrong answer.
    # Measured at 128/256/512/1024 rows per expert (all 0.99932) against 421
    # (0.851).
    serves=KernelSignature(
        signature="grouped_gemm",
        weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
        # The launch quantises A per token per 128 channels; the empty
        # declaration refused exactly that cell and hid the row-alignment
        # refusal behind "activation_granularity".
        activation_granularity=(_DEEPGEMM_ACTIVATION_GRANULARITY,),
        divisible=(("d_reduce", _BLK), ("d_out", _BLK), ("s_expert_rows", _BLK)),
        weight_granularity=(f"per-tile:{_BLK}x{_BLK}",)),
    min_sm=_DEEPGEMM_MIN_SM,
))
register(Provider(
    library="deepgemm", signature="grouped_gemm",
    kernel_impl="deepgemm_fp8_m_grouped_masked",
    import_name="deep_gemm",
    serves=KernelSignature(signature="grouped_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=(
                               _DEEPGEMM_ACTIVATION_GRANULARITY,),
                           divisible=(("d_reduce", _BLK), ("d_out", _BLK)),
                           weight_granularity=(f"per-tile:{_BLK}x{_BLK}",)),
    build=_deepgemm_grouped_build,
    fused_op="grouped_gemm_moe",
    min_sm=_DEEPGEMM_MIN_SM,
))


def _deepgemm_per_head_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # Absorbed MLA's per-head GEMMs (W_UK on the q side, W_UV on the o side)
    # as sglang runs them with SGL_USE_DEEPGEMM_BMM: kv_b_proj keeps its fp8
    # 128x128 blocks, each head is one group of DeepGEMM's masked grouped
    # GEMM, and every head has all B·s_query rows (sglang
    # `per_token_group_quant_mla_deep_gemm_masked_fp8` then
    # `deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked`,
    # srt/models/deepseek_v2.py). vLLM and sglang's default instead
    # dequantize kv_b_proj at load and run bf16 torch.bmm.
    import torch
    import deep_gemm

    x, w = inputs["x"], inputs["w"]               # [B,H,S,K], [H,N,K] bf16
    B, H, S, K = x.shape
    N = w.shape[1]
    M = B * S
    max_m = -(-M // _BLK) * _BLK                  # the 128-row tile
    rows = x.permute(1, 0, 2, 3).reshape(H, M, K)
    x_q = torch.zeros(H, max_m, K, dtype=torch.float8_e4m3fn, device=x.device)
    x_s = torch.zeros(H, max_m, K // _BLK, dtype=torch.float32, device=x.device)
    w_q = torch.empty(H, N, K, dtype=torch.float8_e4m3fn, device=x.device)
    w_s = torch.empty(H, N // _BLK, K // _BLK, dtype=torch.float32, device=x.device)
    for h in range(H):                            # setup (not timed)
        x_q[h, :M], x_s[h, :M] = _per_token_cast_to_fp8(rows[h])
        w_q[h], w_s[h] = _per_block_cast_to_fp8(w[h])
    out = torch.zeros(H, max_m, N, dtype=torch.bfloat16, device=x.device)
    masked_m = torch.full((H,), M, dtype=torch.int32, device=x.device)
    return BenchCase(
        run=Launch(deep_gemm.m_grouped_fp8_gemm_nt_masked,
                   (x_q, x_s), (w_q, w_s), out, masked_m=masked_m,
                   expected_m=M),
        to_canonical=lambda _out: out[:, :M].reshape(H, B, S, N).permute(1, 0, 2, 3),
    )


register(Provider(
    library="deepgemm", signature="per_head_gemm",
    kernel_impl="deepgemm_fp8_per_head_masked",
    import_name="deep_gemm",
    serves=KernelSignature(signature="per_head_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=(
                               _DEEPGEMM_ACTIVATION_GRANULARITY,),
                           divisible=(("d_reduce", _BLK), ("d_out", _BLK)),
                           weight_granularity=(f"per-tile:{_BLK}x{_BLK}",)),
    build=_deepgemm_per_head_build,
    min_sm=_DEEPGEMM_MIN_SM,
))


def _vllm_block_scaled_mm():
    """vLLM's Triton block-scaled fp8 GEMM, under whichever name this
    release calls it.

    Same module, same kernel, same arguments — renamed between releases
    (`w8a8_block_fp8_matmul` through 0.11, `w8a8_triton_block_scaled_mm` by
    0.32). Resolved at build time rather than pinned, because the boxes run
    different vLLMs on purpose: the H100's env is 0.11 and the dev box's is
    0.32, and importing only the newer name failed 1,041 cells of
    Kimi-K2-Instruct on the H100 with the kernel sitting right there.
    """
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    for name in ("w8a8_triton_block_scaled_mm", "w8a8_block_fp8_matmul"):
        fn = getattr(fp8_utils, name, None)
        if fn is not None:
            return fn
    raise ImportError(
        "vllm.model_executor.layers.quantization.utils.fp8_utils has neither "
        "`w8a8_triton_block_scaled_mm` nor `w8a8_block_fp8_matmul`")


def _vllm_triton_block_fp8(tile: int):
    """vLLM's Triton block-scaled fp8 GEMM, bound to one scale tile.

    `w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, out_dtype)` takes
    the block as an ARGUMENT, which is why it serves a tile no CUTLASS or
    DeepGEMM path here does: DeepSeek-V4.1-Flash's `weight_block_size: [32,
    32]`. vLLM names that checkpoint itself at the call — "Checkpoints with
    exponent-only UE8M0 scales (e.g. DeepSeek-V4) are decoded to fp32 before
    launching the kernel" — so this is both the arithmetic the cell asks for
    and the launch a server makes for it. It is Triton, so it runs on Ada as
    well as Hopper, unlike the two block-scaled CUDA paths.
    """
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch

        w8a8_triton_block_scaled_mm = _vllm_block_scaled_mm()
        x, w = inputs["x"], inputs["w"]                  # [M,K], [N,K] bf16
        a_q, a_s = _per_token_cast_to_fp8(x, tile)       # [M,K], [M,⌈K/t⌉]
        w_q, w_s = _per_block_cast_to_fp8(w, tile)       # [N,K], [⌈N/t⌉,⌈K/t⌉]
        return BenchCase(
            run=lambda: w8a8_triton_block_scaled_mm(
                a_q, w_q, a_s, w_s, [tile, tile], torch.bfloat16),
            to_canonical=lambda out: out,
        )
    return build


def _vllm_triton_block_per_head(tile: int):
    """The same Triton GEMM, once per head — the only route to a per-head
    GEMM at a 32-wide block.

    The kernel is 2D, so H heads are H launches, and the name says `_loop`:
    for DeepSeek-V4.1-Flash's `attn.wo_a` (8 output-LoRA groups) one row is
    eight launches and a decode row is launch-bound. That is the shape of
    this computation on this hardware — DeepGEMM's grouped kernels have no
    32-granularity host path at all (measured; see the provider above) — and
    the arithmetic is the cell's own.
    """
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch

        w8a8_triton_block_scaled_mm = _vllm_block_scaled_mm()
        x, w = inputs["x"], inputs["w"]          # [B,H,S,K], [H,N,K] bf16
        B, H, S, K = x.shape
        N = w.shape[1]
        rows = x.permute(1, 0, 2, 3).reshape(H, B * S, K)
        a_q, a_s, w_q, w_s = [], [], [], []
        for h in range(H):                       # load-time work, not timed
            q, s = _per_token_cast_to_fp8(rows[h].contiguous(), tile)
            a_q.append(q)
            a_s.append(s)
            q, s = _per_block_cast_to_fp8(w[h], tile)
            w_q.append(q)
            w_s.append(s)
        out = torch.empty(H, B * S, N, dtype=torch.bfloat16, device=x.device)

        def _run():
            for h in range(H):
                out[h] = w8a8_triton_block_scaled_mm(
                    a_q[h], w_q[h], a_s[h], w_s[h], [tile, tile],
                    torch.bfloat16)
            return out

        return BenchCase(
            run=_run,
            to_canonical=lambda _o: out.reshape(H, B, S, N).permute(1, 0, 2, 3),
        )
    return build


#: vLLM's Triton block-scaled GEMM indexes every operand in int32: its row
#: and column offsets are `pid * BLOCK + tl.arange(...)` and its strides are
#: plain int arguments, so `offs_am * stride_am` (A, M x K), `offs_bn *
#: stride_bn` (B, N x K) and `stride_cm * offs_cm` (C, M x N) wrap once the
#: tensor passes 2**31 elements (`_w8a8_triton_block_scaled_mm`,
#: model_executor/layers/quantization/utils/fp8_utils.py:886-919 at v0.32.0).
#: A wrapped offset is an illegal address — a sticky error that fails every
#: later cell of the process — or, inside a neighbouring allocation, a
#: silent wrong write. 14 cells faulted exactly there on the 4090 (M x N from
#: 2.7e9 to 5.4e9, DeepSeek-V4-Flash, MiMo-V2-Flash, GLM-5.3-Flash), and no
#: cell under the bound did. A ceiling on products, so `max_product`; the
#: per-head loop launches the same kernel once per head at M = B·s_query.
_TRITON_BLOCK_FP8_INT32 = (
    (("B", "s_query", "d_reduce"), 2 ** 31),
    (("B", "s_query", "d_out"), 2 ** 31),
    (("d_out", "d_reduce"), 2 ** 31),
)


#: The scale tiles a block-scaled fp8 checkpoint ships: 128x128 (DeepSeek-V3
#: and every release since) and 32x32 (DeepSeek-V4.1-Flash). One provider per
#: tile rather than one reading the cell, because the tile is the kernel's
#: compile-time block here and belongs in the recorded identity.
for _tile in (_BLK, 32):
    register(Provider(
        library="vllm", signature="gemm",
        kernel_impl=f"vllm_triton_block_scaled_mm_{_tile}", import_name="vllm",
        serves=KernelSignature(signature="gemm",
                               weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                               activation_granularity=(f"per-group:{_tile}",),
                               weight_granularity=(f"per-tile:{_tile}x{_tile}",),
                               divisible=(("d_reduce", _tile),),
                               max_product=_TRITON_BLOCK_FP8_INT32),
        build=_vllm_triton_block_fp8(_tile),
    ))
register(Provider(
    library="vllm", signature="per_head_gemm",
    kernel_impl="vllm_triton_block_scaled_per_head_loop_32", import_name="vllm",
    serves=KernelSignature(signature="per_head_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           activation_granularity=("per-group:32",),
                           weight_granularity=("per-tile:32x32",),
                           divisible=(("d_reduce", 32),),
                           max_product=_TRITON_BLOCK_FP8_INT32),
    build=_vllm_triton_block_per_head(32),
))


def _torch_scaled_per_head_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    """Per-head fp8 GEMM whose weight scale is per output channel (or one per
    tensor) — `torch._scaled_grouped_mm` in its BATCHED form.

    The cell is an absorbed-MLA projection of a checkpoint that stores
    per-channel fp8 (Ling-2.6-1T's `q_absorbed_nope` / `v_absorbed`).
    DeepGEMM declines it — its grouped fp8 kernels read 128x128 blocks — and
    until now nothing else claimed an fp8 weight on this signature, so the
    cells had no route at all.

    `_scaled_grouped_mm` takes 3D x 3D with no `offs` as a batch of
    independent GEMMs (`_meta_registrations.meta_scaled_grouped_mm`: `offs`
    is required only when a multiplicand is 2D), which is what a per-head
    GEMM is. The activation is quantized per token and the weight per output
    channel, both fp32 scales with the batch dimension matching — the rowwise
    recipe those checkpoints ship.

    This is NOT what the frameworks launch for these cells: vLLM and sglang
    dequantize an absorbed projection at load and run bf16 `torch.bmm`
    (`torch_per_head_einsum` measures that configuration). It is the fp8
    kernel the CELL asks for, measured as fp8, which is the comparison the
    plan was written for — a deployment that keeps the weight in fp8 pays
    this, and the bf16 row beside it says what dequantizing costs instead.
    """
    _require_cuda()
    import torch

    x, w = inputs["x"], inputs["w"]              # [B,H,S,K] bf16, [H,N,K] bf16
    B, H, S, K = x.shape
    N = w.shape[1]
    per_channel = inv.weight_granularity != "per-tensor"
    rows = x.permute(1, 0, 2, 3).reshape(H, B * S, K)
    # Per-token activation scales, per-channel (or per-tensor) weight scales:
    # amax over the axis that is NOT reduced, in fp32, as the frameworks'
    # `scaled_fp8_quant` does.
    a_amax = rows.float().abs().amax(dim=2).clamp_min(1e-4)          # [H, M]
    a_q = (rows.float() / (a_amax / _FP8_MAX).unsqueeze(2)).to(torch.float8_e4m3fn)
    w_amax = (w.float().abs().amax(dim=2) if per_channel
              else w.float().abs().amax(dim=(1, 2), keepdim=True).expand(H, N)
              ).clamp_min(1e-4)                                      # [H, N]
    w_q = (w.float() / (w_amax / _FP8_MAX).unsqueeze(2)).to(torch.float8_e4m3fn)
    # mat_b must be column-major in its last two dims: [H,K,N] viewing [H,N,K].
    b = w_q.transpose(1, 2)
    a_s = (a_amax / _FP8_MAX).contiguous()
    b_s = (w_amax / _FP8_MAX).contiguous()
    return BenchCase(
        run=lambda: torch._scaled_grouped_mm(
            a_q, b, a_s, b_s, out_dtype=torch.bfloat16),
        to_canonical=lambda out: out.reshape(H, B, S, N).permute(1, 0, 2, 3),
    )


register(Provider(
    library="torch", signature="per_head_gemm",
    kernel_impl="torch_scaled_grouped_per_head", import_name="torch",
    serves=KernelSignature(signature="per_head_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           weight_granularity=("per-channel", "per-tensor"),
                           # 16-byte alignment on the last two strides, which
                           # for e4m3 is 16 elements (`check_valid_strides`).
                           divisible=(("d_reduce", 16), ("d_out", 16))),
    build=_torch_scaled_per_head_build,
    # `torch._scaled_grouped_mm is only supported on CUDA devices with
    # compute capability = [9.0, 10.0]` — its own refusal, verified on an
    # RTX 4090 (8.9), where it raises rather than running slowly.
    min_sm=90,
))


_MXFP4_GROUP = 32          # mxfp4: one e8m0 scale per 32 weights along K


def _to_mxfp4(w):
    """bf16 → (packed [..., K/2] uint8, e8m0 scales [..., K/32] uint8).

    The OCP mxfp4 pair: E2M1 values two to a byte, and one power-of-two
    exponent per 32 of them along the reduction axis, stored biased by 127 —
    what a gpt-oss checkpoint holds and what vLLM's repacker reads."""
    import torch
    *lead, K = w.shape
    g = w.float().reshape(*lead, K // _MXFP4_GROUP, _MXFP4_GROUP)
    amax = g.abs().amax(-1).clamp_min(1e-8)
    exp = torch.floor(torch.log2(amax / 6.0)).clamp(-127, 127)   # e2m1 max 6
    scale = torch.pow(2.0, exp)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=w.device)
    q = (g / scale.unsqueeze(-1)).clamp(-6, 6)
    idx = (q.abs().unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
    idx |= (q < 0).to(torch.uint8) << 3
    idx = idx.reshape(*lead, K)
    packed = (idx[..., 0::2] | (idx[..., 1::2] << 4)).contiguous()
    return packed, (exp + 127).to(torch.uint8)


def _marlin_moe_dispatch(counts, E: int, M: int, dev):
    """The token→expert map, thread-block size and workspace a Marlin expert
    GEMM launches with — one per cell, shared by every weight format that
    kernel reads (mxfp4 here, int4 below), because the dispatch is the MoE's
    and not the format's.

    The block is vLLM's own choice (`marlin_moe.MarlinExperts`) rather than a
    constant: the kernel refuses a thread config whose m-blocks outrun the
    rows ("Invalid thread config" at M=1 with a 64-row block), so it takes
    the smallest of 8/16/32/48/64 that the per-expert rows fill to 90%,
    floored at 16. `moe_align_block_size` builds the row map the kernel
    gathers A through, which is why that gather is inside the measurement.
    """
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    topk_ids = torch.arange(E, dtype=torch.int32, device=dev).repeat_interleave(
        torch.tensor(counts, dtype=torch.int64, device=dev)).reshape(M, 1)
    topk_weights = torch.ones(M, 1, dtype=torch.float32, device=dev)
    block_m = 64
    for candidate in (8, 16, 32, 48, 64):
        if M * 1 / E / candidate < 0.9:
            block_m = candidate
            break
    block_m = max(block_m, 16)
    sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids, block_m, E)
    workspace = torch.zeros(M * 4 + 1024, dtype=torch.int32, device=dev)
    return topk_weights, sorted_ids, expert_ids, num_pad, block_m, workspace


def _marlin_mxfp4_grouped_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """One expert-GEMM launch of an mxfp4 MoE, as vLLM runs it on Ada and on
    Hopper without FlashInfer/DeepGEMM: `moe_wna16_marlin_gemm`, the op its
    MarlinExperts calls twice per MoE (w13 then w2), so one call is this
    cell's grouped GEMM rather than the whole mixer.

    The pack and the scale layout are the framework's own
    (`prepare_moe_mxfp4_layer_for_marlin`), as with the other Marlin
    providers: a hand-rolled layout would measure a packing nobody serves.
    The dispatch (row map, block size, workspace) is `_marlin_moe_dispatch`,
    shared with the int4 build below."""
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mu
    from vllm.scalar_type import scalar_types

    x, w = inputs["x"], inputs["w"]                 # [E,Sq,K] bf16, [E,N,K] bf16
    E, _Sq, K = x.shape
    N = w.shape[1]
    dev = x.device
    counts = inv.moe_token_counts
    M = sum(counts)
    a = _packed_rows(x, counts).to(torch.bfloat16)
    # Marlin's thread tiles need (N%64, K%128) or (N%128, K%64) and refuse
    # the launch outright otherwise ("Invalid thread config"): gpt-oss's 2880
    # satisfies neither. vLLM reaches a legal pair by ALLOCATING the expert
    # weights at the padded width (`Mxfp4MoEMethod` rounds the hidden dim up;
    # `marlin_padded_nk` is the same arithmetic) and padding/slicing the
    # activation and output on every forward. So the padded launch is what
    # serving runs; the padded region decodes to 0.0 and adds nothing.
    pad_n, pad_k = mu.marlin_padded_nk(N, K, _MXFP4_GROUP)
    if (pad_n, pad_k) != (N, K):
        w = torch.nn.functional.pad(w, (0, pad_k - K, 0, pad_n - N))
    packed, scales = _to_mxfp4(w)
    # The framework's repacker takes the w13/w2 PAIR of a whole MoE and
    # derives one shape from the other (`repack_weight`: n = w13.shape[1]/2,
    # k = w13.shape[2]*2). This cell is ONE of the two GEMMs, so it goes in
    # the w13 slot and the partner is zeros of the shape that pairing
    # implies — never launched, and only the w13 half is read back. Passing
    # the same tensor twice fails w2's own shape assertion.
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    partner = torch.zeros(E, pad_k, pad_n // 4, dtype=packed.dtype, device=dev)
    partner_s = torch.zeros(E, pad_k, (pad_n // 2) // _MXFP4_GROUP,
                            dtype=scales.dtype, device=dev)
    qw, _qw2, sc, _sc2, _b1, _b2 = mu.prepare_moe_mxfp4_layer_for_marlin(
        layer, packed, partner, scales, partner_s, None, None)
    topk_weights, sorted_ids, expert_ids, num_pad, block_m, workspace = \
        _marlin_moe_dispatch(counts, E, M, dev)
    # The launch runs at the padded width; the row's identity is the cell's.
    a_pad = a if pad_k == K else torch.nn.functional.pad(a, (0, pad_k - K))
    out = torch.zeros(M, pad_n, dtype=torch.bfloat16, device=dev)
    return BenchCase(
        # The last three are the framework's own choices for an expert GEMM,
        # not defaults: `MarlinExperts` passes use_atomic_add=False,
        # use_fp32_reduce=True, is_zp_float=False for both the w13 and the w2
        # launch. (The DENSE linear path picks atomic-add by shape through
        # `should_use_atomic_add_reduce`; the MoE path never does, and an
        # atomic reduce is a different kernel configuration from the split-K
        # one serving runs.)
        run=lambda: ops.moe_wna16_marlin_gemm(
            a_pad, out, qw, None, sc, None, None, None, workspace,
            sorted_ids, expert_ids, num_pad, topk_weights, block_m, 1, False,
            scalar_types.float4_e2m1f, M, pad_n, pad_k, False, True, False),
        # OUT of the correctness gate, and this is why: the signature's
        # oracle is fp32 math on the UNQUANTISED weight, and mxfp4's own
        # resolution puts a correct kernel at rel 0.9-1.3 / cos 0.98 against
        # it (measured over five shapes on an RTX 4090), far outside any
        # band that still catches a wrong kernel. The band table has no
        # 4-bit entry at all, which is also why the int4 and nvfp4 Marlin
        # providers have never landed a row.
        #
        # Verified instead against the oracle that models the SAME
        # quantisation — the dequantised weights, which is what this kernel
        # computes: rel 0.024, cos 0.99999 at [E=8, M=64, K=512, N=1024] on
        # an RTX 4090. A provider-supplied oracle would make that the
        # recorded check; `BenchCase` has no such slot yet.
        to_canonical=None,
    )


register(Provider(
    library="vllm", signature="grouped_gemm",
    kernel_impl="vllm_marlin_mxfp4_grouped_gemm", import_name="vllm",
    serves=KernelSignature(signature="grouped_gemm",
                           weight=("mxfp4", "fp4_e2m1"),
                           activation=("bf16", "fp16"),
                           weight_granularity=(f"per-group:{_MXFP4_GROUP}",),
                           # Only the scale group: the thread tiles are
                           # reached by padding, as vLLM reaches them.
                           divisible=(("d_reduce", _MXFP4_GROUP),
                                      ("d_out", 4),)),
    build=_marlin_mxfp4_grouped_build,
))


def _marlin_int4_grouped_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """One expert-GEMM launch of a W4A16 int4 MoE — Kimi-K2-Thinking's INT4
    release and the other compressed-tensors / GPTQ MoE checkpoints, which
    vLLM serves through `MarlinExperts` (`CompressedTensorsWNA16MarlinMoE
    Method`): the same `moe_wna16_marlin_gemm` the mxfp4 build above calls,
    with the GPTQ grid instead of E2M1.

    The group comes from the cell (`_marlin_group`): 32 for Kimi-K2-Thinking,
    128 for a GPTQ release, and the kernel reads 4× as many scales at 32, so
    the two are different measurements of one kernel rather than one cell.

    Setup — quantize, pack, repack, permute — is load-time work in a server
    (`process_weights_after_loading`) and stays outside the timed call; the
    packed layouts are the framework's own repackers, never hand-rolled.
    """
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales)
    from vllm.scalar_type import scalar_types

    x, w = inputs["x"], inputs["w"]                 # [E,Sq,K] bf16, [E,N,K] bf16
    E, _Sq, K = x.shape
    N = w.shape[1]
    dev = x.device
    group = _marlin_group(inv)
    counts = inv.moe_token_counts
    M = sum(counts)
    a = _packed_rows(x, counts).to(torch.bfloat16)

    # Per-group symmetric int4 over K, levels -8..7 offset by +8 into the
    # `uint4b8` scalar type (the zero point folded into the container), the
    # same grid `_marlin_int4_build` places for the dense case.
    wg = w.float().reshape(E, N, K // group, group)
    scale = wg.abs().amax(dim=3).clamp_min(1e-6) / 7.0          # [E,N,K/group]
    q = (wg / scale.unsqueeze(3)).round().clamp(-8, 7).to(torch.int32) + 8
    q = q.reshape(E, N, K).permute(0, 2, 1).contiguous()        # [E,K,N]
    # GPTQ on-disk container: int32, 8 nibbles per word down K.
    qw = torch.zeros(E, K // 8, N, dtype=torch.int32, device=dev)
    for i in range(8):
        qw |= (q[:, i::8] & 0xF) << (4 * i)
    packed = ops.gptq_marlin_moe_repack(qw, K, N, 4)
    scales = marlin_moe_permute_scales(
        scale.permute(0, 2, 1).contiguous().to(torch.bfloat16), K, N, group)
    _topk_w, sorted_ids, expert_ids, num_pad, block_m, workspace = \
        _marlin_moe_dispatch(counts, E, M, dev)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    return BenchCase(
        run=lambda: ops.moe_wna16_marlin_gemm(
            a, out, packed, None, scales, None, None, None, workspace,
            sorted_ids, expert_ids, num_pad, _topk_w, block_m, 1, False,
            scalar_types.uint4b8, M, N, K, False, True, False),
        # Same reason as the mxfp4 build: the signature's oracle is fp32 math
        # on the UNQUANTISED weight, and the cross-check band table has no
        # 4-bit entry — which is exactly why the dense `marlin_gptq_int4_w4a16`
        # provider has never landed a row. int4's own resolution puts this
        # kernel at rel 0.097 / cos 0.9953 against that oracle, outside any
        # band that would still catch a wrong kernel.
        #
        # Verified instead against the oracle that models the SAME
        # quantisation — the dequantised weights, which is what this kernel
        # computes: rel 0.0026, cos 0.999997 at [E=6, M=132, K=2048, N=7168]
        # on an RTX 4090. A provider-supplied oracle would make that the
        # recorded check; `BenchCase` has no such slot yet.
        to_canonical=None,
    )


def _marlin_int4_grouped_params(inv: Invocation) -> str:
    """Decline a shard whose reduction width is not a whole number of scale
    groups. The bound depends on the group the CELL carries, which is why it
    is a gate rather than a `divisible`: a rank holding K=48 of a group-32
    checkpoint has a partial group, and no Marlin launch reads one."""
    group = _marlin_group(inv)              # raises for an unsupported group
    k = int(inv.dims["d_reduce"])
    if k % group:
        raise NotImplementedError(
            f"int4 group {group} needs K%{group}==0; this rank holds K={k}")
    return f"group_size={group}"


register(Provider(
    library="vllm", signature="grouped_gemm",
    kernel_impl="vllm_marlin_int4_grouped_gemm", import_name="vllm",
    impl_params=_marlin_int4_grouped_params,
    serves=KernelSignature(signature="grouped_gemm",
                           weight=("int4",), activation=("bf16", "fp16"),
                           weight_granularity=tuple(
                               f"per-group:{g}" for g in _MARLIN_GROUPS),
                           # `gptq_marlin_moe_repack` asserts K%16; N rides
                           # the 64-wide thread tile as in the dense build.
                           divisible=(("d_reduce", 16), ("d_out", 64))),
    build=_marlin_int4_grouped_build,
))


def _deepgemm_fp4_grouped_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """One expert GEMM of an mxfp4 MoE whose ACTIVATION is fp8 — W4A8.

    This is the launch DeepSeek-V4 / V4.1-Flash's MoE runs, and the reason
    those cells had no provider at all: the Marlin build above reads a bf16
    activation (W4A16), so it declines them, and nothing else claimed the
    pair. vLLM serves it as `Mxfp4MoeBackend.DEEPGEMM_MXFP4`, whose
    activation key is `kFp8Dynamic128Sym` — fp8 e4m3, per token, per 128
    channels along K — against the checkpoint's 32-wide fp4 blocks.

    Derived from `DeepGemmFP4Experts.apply`
    (`fused_moe/experts/deep_gemm_moe.py:654-681`, vLLM 0.32.0): the fp4
    weight goes in as int8-viewed packed pairs, the two granularities are
    stated as `recipe_a=(1, 128)` / `recipe_b=(1, 32)`, and the weight's
    e8m0 block scales are upcast to fp32 and laid out by
    `transform_sf_into_required_layout` (what
    `deepgemm_post_process_weight_scale_block` wraps). The upcast is done
    here rather than imported because this runs in the deepgemm env, which
    has no vLLM: `2**(e8m0 − 127)` is the whole of it.

    NOT YET RUN ON HARDWARE, and it cannot be from this machine: the kernel
    is Blackwell-only (`sm100_fp8_fp4_gemm_1d1d.cuh` / `sm120_…`; DeepGEMM
    ships no sm90 fp8×fp4 GEMM), and both boxes here are sm90 and sm89. It
    is registered so the capability is declared where the audit reads it —
    2,598 planned cells on the two DeepSeek-V4 models currently read as "no
    kernel exists", which is false — and so the first Blackwell box measures
    them instead of discovering the gap again. A build error there is
    recorded as a failure, never as a number.
    """
    _require_cuda()
    import torch
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

    x, w = inputs["x"], inputs["w"]                      # [E,Sq,K], [E,N,K] bf16
    E, _Sq, K = x.shape
    N = w.shape[1]
    counts = inv.moe_token_counts
    source = _packed_rows(x, counts)
    a_q, a_s = _per_token_cast_to_fp8(source, _BLK)      # fp8 [M,K], [M,K/128]
    a_s = get_mn_major_tma_aligned_tensor(a_s)
    w_q = torch.empty(E, N, K // 2, dtype=torch.uint8, device=x.device)
    w_e8m0 = torch.empty(E, N, K // _MXFP4_GROUP, dtype=torch.uint8,
                         device=x.device)
    for e in range(E):                                   # load-time work
        w_q[e], w_e8m0[e] = _to_mxfp4(w[e])
    w_s = deep_gemm.transform_sf_into_required_layout(
        torch.exp2(w_e8m0.float() - 127.0), mn=N, k=K,
        recipe=(1, 1, _MXFP4_GROUP), num_groups=E, is_sfa=False)
    out = torch.zeros(sum(counts), N, dtype=torch.bfloat16, device=x.device)
    m_indices = torch.arange(E, dtype=torch.int32, device=x.device
                             ).repeat_interleave(
        torch.tensor(counts, dtype=torch.int64, device=x.device))
    return BenchCase(
        run=lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a_q, a_s), (w_q.view(torch.int8), w_s), out, m_indices,
            recipe_a=(1, _BLK), recipe_b=(1, _MXFP4_GROUP)),
        # Same ground as the Marlin mxfp4 build: the signature's oracle is
        # fp32 on the unquantised weight, and no band distinguishes a correct
        # 4-bit kernel from a wrong one there. Unlike that build, this one has
        # no hand-verification either — there is no card here that runs it —
        # so the first rows it lands should be checked against the
        # dequantised-weight oracle before they are read as an envelope.
        to_canonical=None,
    )


register(Provider(
    library="deepgemm", signature="grouped_gemm",
    kernel_impl="deepgemm_fp8_fp4_m_grouped_contiguous",
    import_name="deep_gemm",
    serves=KernelSignature(signature="grouped_gemm",
                           weight=("mxfp4", "fp4_e2m1"),
                           activation=("fp8_e4m3",), output=("bf16",),
                           weight_granularity=(f"per-group:{_MXFP4_GROUP}",),
                           # The launch quantises A at 128 (`recipe_a`), even
                           # when the checkpoint's declared activation group is
                           # 32. Until that discrepancy is measured and ruled
                           # on, a group-32 cell is a visible gap, not a row
                           # silently measured at another granularity.
                           activation_granularity=(
                               _DEEPGEMM_ACTIVATION_GRANULARITY,),
                           # The activation's own 128-wide group along K is
                           # the binding constraint; the weight's 32 divides
                           # it. N rides the kernel's tile.
                           divisible=(("d_reduce", _BLK),),
                           # A group-128 key alone is only the kernel's numeric
                           # contract. The selector fact — vLLM's normalized
                           # `moe_backend=deep_gemm` request — must travel with
                           # the cell before even this scope can claim it.
                           requires_params=("moe_backend",)),
    build=_deepgemm_fp4_grouped_build,
    # `serves` is the kernel's numeric scope, not a claim that vLLM selected
    # it. In the audited auto order FlashInfer TRT-LLM MXFP8 comes first
    # (oracle/mxfp4.py:425-449); DeepGEMM needs an explicit `deep_gemm`
    # request (oracle/mxfp4.py:754-779, :347-350) or a proved FlashInfer
    # failure. Neither deployment fact reaches an invocation yet, so no
    # group-128 cell is accepted while its selector is unknown.
    disabled_reason=("SM100 DeepSeek auto launch is FlashInfer TRT-LLM MXFP8 "
                     "group 32; DeepGEMM group 128 needs an explicit or "
                     "proved-fallback selector"),
    # Blackwell: DeepGEMM compiles fp8×fp4 for sm100 and sm120 only.
    min_sm=100,
))


def _vllm_fused_moe_kernel():
    """vLLM's Triton expert GEMM, under this release's name and ARITY.

    Renamed `invoke_fused_moe_kernel` → `invoke_fused_moe_triton_kernel`
    between 0.11 and 0.32, and the older one takes one more operand: a
    weight zero-point (`B_zp`) that the newer dropped. So the call is made
    by KEYWORD and the zero point added only when the callee has a
    parameter for it — asking the function rather than the version number,
    the same way `Launch` asks for its parameter names.

    Until this existed the provider imported the 0.32 name unconditionally
    and so had never run on the H100 box at all (its env is 0.11): every
    MoE model's Triton grouped-GEMM cells failed, 413 of them on
    GLM-4.5-Air alone, and the sweep reported it as the model failing.
    """
    import inspect
    from vllm.model_executor.layers.fused_moe import fused_moe as _fm
    for name in ("invoke_fused_moe_triton_kernel", "invoke_fused_moe_kernel"):
        fn = getattr(_fm, name, None)
        if fn is not None:
            try:
                takes_zp = "B_zp" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                takes_zp = False
            return fn, takes_zp
    raise ImportError(
        "vllm.model_executor.layers.fused_moe.fused_moe has neither "
        "`invoke_fused_moe_triton_kernel` nor `invoke_fused_moe_kernel`")


def _vllm_triton_moe_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # vLLM/sglang Triton fused_moe_kernel — the OTHER MoE fusion style: the
    # token GATHER is fused INTO the grouped GEMM (kernel indexes A rows via
    # sorted_token_ids), so this measures grouped-GEMM-with-gather, distinct
    # from DeepGEMM/CUTLASS's separate-permute + pure-GEMM. bf16 gate_up GEMM,
    # top_k=1, equal tokens per expert. From vLLM fused_experts_impl call site.
    import torch
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size, try_get_optimal_moe_config,
    )
    invoke, takes_zp = _vllm_fused_moe_kernel()
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)        # [E,Sq,K], [E,N,K]
    E, Sq, _K = x.shape
    N = w.shape[1]
    counts = inv.moe_token_counts
    M = sum(counts)
    a = _packed_rows(x, counts)
    topk_ids = torch.arange(E, dtype=torch.int32, device=x.device) \
        .repeat_interleave(torch.tensor(counts, dtype=torch.int64,
                                       device=x.device)).reshape(M, 1)
    config = try_get_optimal_moe_config(tuple(w.shape), tuple(w.shape), 1, None, M)
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)
    # Zeroed for the same reason as the DeepGEMM paths: `moe_align_block_size`
    # pads `sorted_ids` to the block, so the triton kernel writes the valid rows
    # and leaves the rest — uninitialized, they are compared anyway.
    out = torch.zeros(M, 1, N, dtype=dt, device=x.device)
    ctype = {torch.bfloat16: tl.bfloat16,
             torch.float16: tl.float16}.get(dt, tl.bfloat16)
    return BenchCase(
        run=lambda: invoke(
            a, w, out, None, None, **({"B_zp": None} if takes_zp else {}),
            topk_weights=None, sorted_token_ids=sorted_ids,
            expert_ids=expert_ids, num_tokens_post_padded=num_pad,
            mul_routed_weight=False, top_k=1, config=config,
            compute_type=ctype, use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False),
        to_canonical=lambda _out: _ragged_to_padded(out[:, 0], counts, Sq),
    )


register(Provider(
    library="vllm", signature="grouped_gemm",
    kernel_impl="vllm_triton_fused_moe",
    import_name="vllm",
    # The unquantised path (`use_fp8_w8a8=False`, `compute_type` bf16 or
    # fp16): weights and activations at the cell's 16-bit dtype.
    serves=KernelSignature(signature="grouped_gemm", weight=("bf16", "fp16"),
                           activation=("bf16", "fp16")),
    build=_vllm_triton_moe_build,
    fused_op="grouped_gemm_moe",   # gather⊕gemm boundary (refined in Stage 3)
))


def _vllm_triton_moe_fp8_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """The same Triton expert GEMM as above, in its W8A8 fp8 configuration —
    `use_fp8_w8a8`, the path vLLM takes for an fp8 MoE whose scales are
    per-channel or per-tensor (`Fp8MoEMethod` without the blockwise
    DeepGEMM/CUTLASS backends: Ling-2.6-1T and the other
    per-output-channel fp8 releases).

    A separate build rather than a branch in the bf16 one, because what it
    measures is a different launch: the activation is quantized per token,
    the weight scale rides the kernel's inner loop, and the accumulate is
    fp8×fp8→fp32. The quantization of the ACTIVATION is part of a server's
    forward (there is no fp8 activation on the wire), so it stays inside the
    timed call — unlike the weight's, which is load-time.
    """
    import torch
    import triton.language as tl
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size, try_get_optimal_moe_config,
    )
    invoke, takes_zp = _vllm_fused_moe_kernel()

    x, w = inputs["x"], inputs["w"]                     # [E,Sq,K], [E,N,K] bf16
    E, Sq, _K = x.shape
    N = w.shape[1]
    counts = inv.moe_token_counts
    M = sum(counts)
    a = _packed_rows(x, counts).to(torch.bfloat16)
    per_channel = inv.weight_granularity != "per-tensor"
    # Load-time work: the checkpoint already holds the quantized weight.
    w_q = torch.empty(E, N, _K, dtype=torch.float8_e4m3fn, device=x.device)
    w_s = torch.empty(E, N if per_channel else 1, 1,
                      dtype=torch.float32, device=x.device)
    for e in range(E):
        q, s = vllm_ops.scaled_fp8_quant(
            w[e].to(torch.bfloat16), use_per_token_if_dynamic=per_channel)
        w_q[e], w_s[e] = q, s.reshape(-1, 1)
    config = try_get_optimal_moe_config(tuple(w.shape), tuple(w.shape), 1, None, M)
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids := torch.arange(E, dtype=torch.int32, device=x.device)
        .repeat_interleave(torch.tensor(counts, dtype=torch.int64,
                                        device=x.device)).reshape(M, 1),
        config["BLOCK_SIZE_M"], E)
    del topk_ids
    out = torch.zeros(M, 1, N, dtype=torch.bfloat16, device=x.device)

    def run():
        a_q, a_s = vllm_ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
        return invoke(
            a_q, w_q, out, a_s, w_s, **({"B_zp": None} if takes_zp else {}),
            topk_weights=None, sorted_token_ids=sorted_ids,
            expert_ids=expert_ids, num_tokens_post_padded=num_pad,
            mul_routed_weight=False, top_k=1, config=config,
            compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=per_channel)

    return BenchCase(run=run,
                     to_canonical=lambda _o: _ragged_to_padded(out[:, 0],
                                                               counts, Sq))


register(Provider(
    library="vllm", signature="grouped_gemm",
    kernel_impl="vllm_triton_fused_moe_fp8", import_name="vllm",
    serves=KernelSignature(signature="grouped_gemm",
                           weight=("fp8_e4m3",), activation=("fp8_e4m3",), output=("bf16",),
                           weight_granularity=("per-channel", "per-tensor"),
                           # Triton tiles over K; the kernel's own config
                           # picks the block, so only the 16-wide load the
                           # fp8 path needs is required here.
                           divisible=(("d_reduce", 16),)),
    build=_vllm_triton_moe_fp8_build,
    fused_op="grouped_gemm_moe",
))


# ---------------------------------------------------------------------------
# fused_moe — the WHOLE MoE expert block.
#
# Canonical tensors: x[M,H], topk_ids/weights[M,K], w1[E,2I,H] gate|up,
# w2[E,H,I]. Math is FC1 + SwiGLU + FC2 + weighted reduce.
#
# torch is the stitched floor. FlashInfer `cutlass_fused_moe` is the
# single-GPU kernel (vLLM flashinfer_cutlass). DeepGEMM MegaMoE fuses
# EP dispatch/combine on symmetric memory — multi-process, body pending.
# CuteDSL NVFP4 / SonicMoE / cuDNN grouped-GEMM+SwiGLU are other
# granularities and are not registered here.
# ---------------------------------------------------------------------------

def _torch_fused_moe_build(inv, env, inputs) -> BenchCase:
    import torch
    import torch.nn.functional as F
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    ids = inputs["topk_ids"].long()
    scales = inputs["topk_weights"].to(x.dtype)
    w1 = inputs["w1"].to(dt)
    w2 = inputs["w2"].to(dt)
    i = w1.shape[1] // 2

    def run():
        out = x.new_zeros(x.shape)
        for e in range(w1.shape[0]):
            mask = ids == e
            if not mask.any():
                continue
            tok, slot = torch.where(mask)
            gate, up = w1[e, :i], w1[e, i:]
            hid = x[tok]
            inter = F.silu(hid @ gate.T) * (hid @ up.T)
            out[tok] += scales[tok, slot, None] * (inter @ w2[e].T)
        return out

    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="fused_moe", kernel_impl="torch_fused_moe",
    import_name="torch",
    serves=KernelSignature(signature="fused_moe", weight=TORCH_FLOAT,
                           activation=TORCH_FLOAT),
    build=_torch_fused_moe_build,
    fused_op="fused_moe",
))


def _flashinfer_cutlass_fused_moe_build(inv, env, inputs) -> BenchCase:
    # Timed call is FlashInfer's own fused-MoE entry — the same function
    # vLLM `FlashInferExperts` / `flashinfer_cutlass_moe` dispatches.
    # Setup: swap canonical gate|up into the library's up|gate packing
    # (`w3, w1 = chunk(w31, 2)` in flashinfer tests). Not timed.
    _require_cuda()
    import torch
    from flashinfer.fused_moe import cutlass_fused_moe
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).contiguous()
    ids = inputs["topk_ids"].to(torch.int)
    scales = inputs["topk_weights"].float().contiguous()
    w1 = inputs["w1"].to(dt).contiguous()
    i = w1.shape[1] // 2
    w31 = torch.cat((w1[:, i:], w1[:, :i]), dim=1).contiguous()
    w2 = inputs["w2"].to(dt).contiguous()
    out = torch.empty_like(x)

    def run():
        return cutlass_fused_moe(
            x, ids, scales, w31, w2, x.dtype,
            output=out, quant_scales=None,
        )

    return BenchCase(run=run, to_canonical=lambda o: out if o is None else o)


register(Provider(
    library="flashinfer", signature="fused_moe",
    kernel_impl="flashinfer_cutlass_fused_moe",
    import_name="flashinfer",
    serves=KernelSignature(signature="fused_moe",
                           weight=("bf16", "fp16"),
                           activation=("bf16", "fp16")),
    build=_flashinfer_cutlass_fused_moe_build,
    fused_op="fused_moe",
    min_sm=100,
))


def _deepgemm_megamoe_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # API is pinned but NOT runnable off a multi-process symmetric-memory
    # launch: fp4 staging + symm-buffer + weight transforms are SM100 and
    # DeepGEMM/vLLM-internal. Catalog entry so MegaMoE surfaces; the
    # single-GPU measurement is FlashInfer cutlass above. Declared disabled
    # (`disabled_reason`) so no cell is offered to it until the body exists.
    raise NotImplementedError("see disabled_reason")


register(Provider(
    library="deepgemm", signature="fused_moe",
    kernel_impl="deepgemm_fp8_fp4_mega_moe",
    import_name="deep_gemm",
    # What its name and MegaMoE's staging say it runs (FP8 activations,
    # FP4 experts); disabled, so it serves no cell until a body exists.
    serves=KernelSignature(signature="fused_moe",
                           weight=("mxfp4", "fp4_e2m1"),
                           activation=("fp8_e4m3",), output=("bf16",)),
    build=_deepgemm_megamoe_build,
    fused_op="fused_moe",
    min_sm=100,
    disabled_reason=(
        "no runnable body yet: deep_gemm.fp8_fp4_mega_moe needs SM100 and a "
        "multi-process symmetric-memory launch (fp4 staging, symm buffer, "
        "weight transforms); the single-GPU fused MoE is FlashInfer cutlass"),
))


# ---------------------------------------------------------------------------
# MoE token dispatch (permute / combine) — the memory-bound token shuffle
# around the expert GEMMs. Dropped as zero-cost until Stage 2; now first-class
# so the whole MoE chain is measurable. torch is the stitched baseline; a fused
# kernel (triton fused_moe / DeepEP) absorbs permute into the GEMM launch.
# ---------------------------------------------------------------------------
def _torch_moe_permute_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x, idx = inputs["x"].to(dt), inputs["idx"]
    # The dispatch's gather: the step's token rows in, one routed row out per
    # (token, expert). The output buffer is part of the CALL (`out=`), because
    # a bare `x[idx]` names neither the expansion nor the buffer it lands in —
    # the row's launch signature and the launch-trace gate both read what the
    # call was handed, and an expanding gather's largest operand is its output.
    out = torch.empty(idx.shape[0], x.shape[-1], dtype=dt, device=x.device)
    return BenchCase(run=Launch(torch.index_select, x, 0, idx, out=out),
                     to_canonical=lambda o: o)


def _moved_record_scope(signature: str) -> KernelSignature:
    """The scope of a launch that MOVES a quantised activation record: one
    narrow format on both dtype slots, every granularity an activation
    format declares, its scale dtypes. One reader of `quantise`'s own
    declarations rather than a second list of formats.

    Refuses at load for a signature whose launch computes its output
    (`quant.gathers_a_record`, read from the op's own `gather_source`): the
    scope says "same record in and out", and only a gather makes that a
    move rather than an unmeasured computation.
    """
    from ..quant import gathers_a_record
    from .quantise import _EVERY_GRANULARITY, _NARROW, _NARROW_SCALES
    if not gathers_a_record(signature):
        raise ValueError(
            f"{signature!r} computes its output, so a kernel cannot be "
            f"registered as moving its operand's record")
    return KernelSignature(signature=signature, activation=_NARROW,
                           output=_NARROW,
                           activation_granularity=_EVERY_GRANULARITY,
                           dtype_slots=(("metadata", _NARROW_SCALES),))


def _torch_quantised_permute_build(inv: Invocation, env: Env,
                                   inputs: Dict[str, Any]) -> BenchCase:
    """The same dispatch gather when the rows it moves are a quantised
    activation: the record's elements AND the scales that belong to each row
    (both frameworks permute the pair — vLLM's
    `moe_permute`/`_moe_permute_fp8` moves `a1q` beside `a1q_scale`, sglang's
    `ep_moe` dispatch likewise), so the launch moves one byte an element and
    four per scale group, not two bytes an element.

    Quantising is setup, not the measurement: a server's dispatch reads
    elements an earlier cast wrote."""
    import torch
    from ..quant import ActivationCast
    cast = ActivationCast.moved_by(inv)
    elements, scales = cast.quantise(inputs["x"])
    rows = elements.reshape(scales.shape[0], -1)
    idx = inputs["idx"]
    out = torch.empty(idx.shape[0], rows.shape[-1], dtype=rows.dtype,
                      device=rows.device)
    out_scales = torch.empty(idx.shape[0], scales.shape[-1],
                             dtype=scales.dtype, device=scales.device)

    def run():
        torch.index_select(rows, 0, idx, out=out)
        torch.index_select(scales, 0, idx, out=out_scales)
        return out, out_scales
    return BenchCase(run=run, to_canonical=cast.dequantise)


def _torch_moe_combine_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x, w, idx = inputs["x"].to(dt), inputs["w"], inputs["idx"]
    # The inverse REDUCES: every routed copy of a token is weighted and summed
    # into that token's row. Out-of-place, because an in-place `index_add_`
    # would accumulate across the timing loop and the check would then compare
    # the oracle against N passes.
    #
    # Storage stays the cell's dtype — that is the one thing a latency may not
    # be read at some other precision than (the byte law charges bf16 reads and
    # a bf16 write; fp32 buffers moved 2.4x those bytes and the measurement was
    # systematically, silently slow). But the PRODUCTS may not be rounded to
    # that dtype before the sum: bf16's 8-bit mantissa maps a cancelling pair
    # onto one value — measured on the canary cell, -1.62123 and +1.63074 both
    # land on 1.625 and sum to exactly 0 where the fp32 oracle keeps +0.0095
    # (worst 0.157 here, 0.229 on the H100, band 0.1). A real combine (vLLM
    # `moe_sum`) multiplies and accumulates in registers and never rounds the
    # product to storage precision. Torch cannot scatter-add a narrow source
    # into a wide accumulator (`index_add` wants one dtype), so the product is
    # computed in fp32 at build time and SPLIT: `hi` is its bf16 rounding, `lo`
    # the bf16 rounding of what that dropped, and the two index_adds together
    # carry ~16 mantissa bits — the cancellation survives (worst 0.007 on the
    # same cell). The adds themselves still round bf16 per routed row; measured
    # worst on deep cells (~20 rows/token) is 0.03-0.05, inside the band. The
    # price is a second scatter-add per call (~2.4x the single-add time) — the
    # floor trades speed for a number that is true.
    prod = x.float() * w[:, None]
    hi = prod.to(dt)
    lo = (prod - hi.float()).to(dt)
    tokens, width = int(inputs["tokens"]), x.shape[-1]
    zeros = torch.zeros(tokens, width, dtype=dt, device=x.device)

    def run():
        return (torch.index_add(zeros, 0, idx, hi)
                + torch.index_add(zeros, 0, idx, lo))

    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="moe_permute", kernel_impl="torch_index_permute",
    import_name="torch",
    # The dispatch's gather: the signature declares BOTH row counts (the
    # token-ordered `x` it reads, the routed-long `idx` it writes with), so
    # `cell_check` compares the build against what the launch moves.
    serves=KernelSignature(signature="moe_permute", activation=TORCH_FLOAT),
    build=_torch_moe_permute_build,
))
register(Provider(
    library="torch", signature="moe_permute",
    kernel_impl="torch_index_permute_quantised", import_name="torch",
    # The quantised dispatch: the cell reads and writes the same record, so
    # both dtype slots are the narrow one and the scale layout is the cell's.
    serves=_moved_record_scope("moe_permute"),
    # The moved row is `d_model` wide: a width its scale groups do not tile
    # is declined before any input is allocated, as every other launch
    # holding such a row declines it.
    impl_params=declines_an_untileable_row(),
    build=_torch_quantised_permute_build,
))
register(Provider(
    library="torch", signature="moe_combine", kernel_impl="torch_weighted_combine",
    import_name="torch",
    serves=KernelSignature(signature="moe_combine", activation=TORCH_FLOAT),
    build=_torch_moe_combine_build,
))


def _torch_silu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    # SwiGLU baseline (memory-bound): silu(gate) * up over x=[M, 2I] → [M, I].
    # The vLLM / FlashInfer fused kernels are the envelope; torch guarantees the
    # op is covered even when those envs won't provision (vLLM config-context
    # churn, FlashInfer JIT needs nvcc).
    import torch.nn.functional as F
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    i = x.shape[-1] // 2
    return BenchCase(run=lambda: F.silu(x[..., :i]) * x[..., i:],
                     to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="silu_and_mul", kernel_impl="torch_silu_and_mul",
    import_name="torch",
    # Writes at the cell's dtype: not the FP8 output a cast-fused SwiGLU
    # epilogue writes.
    serves=KernelSignature(signature="silu_and_mul", activation=TORCH_FLOAT),
    build=_torch_silu_and_mul_build,
))


def _torch_gelu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch.nn.functional as F
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    i = x.shape[-1] // 2
    return BenchCase(run=lambda: F.gelu(x[..., :i]) * x[..., i:],
                     to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="gelu_and_mul", kernel_impl="torch_gelu_and_mul",
    import_name="torch",
    serves=KernelSignature(signature="gelu_and_mul", activation=TORCH_FLOAT),
    build=_torch_gelu_and_mul_build,
))


def _torch_situ_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    i = x.shape[-1] // 2
    gate, up = x[..., :i], x[..., i:]
    # `situ(x) = beta * tanh(x / beta) * sigmoid(x)`, and the up branch is
    # softly clipped the same way when the kernel is given a linear beta
    # (vLLM `SituAndMul.forward_native`; <= 0 is its "unset", which passes
    # up through). Both come off the cell, which carries what the model
    # publishes — at a beta of this harness's choosing the eager floor and
    # the kernel race on a function nobody runs.
    from ..refs import situ_betas
    beta, linear = situ_betas(inputs)
    if linear > 0:
        up = linear * torch.tanh(up / linear)
    return BenchCase(
        run=lambda: beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * up,
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="situ_and_mul", kernel_impl="torch_situ_and_mul",
    import_name="torch",
    serves=KernelSignature(signature="situ_and_mul",
                           activation=TORCH_FLOAT,
                           requires_params=("situ_beta",)),
    build=_torch_situ_and_mul_build,
    fused_op="situ_and_mul",
))
