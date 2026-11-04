"""RoPE providers — rotary embedding on Q+K, the fused QK-norm+RoPE kernel,
and the vision towers' two-axis rotation.

Three signatures live here because they share the rotary math:

  * `rope` — rotate Q[tokens,N·D] and K[tokens,Nk·D] in one launch. torch is
    the always-available baseline; flashinfer / vLLM / sglang each ship a real
    in-place kernel (`apply_rope_with_cos_sin_cache_inplace` /
    `_custom_ops.rotary_embedding`).
  * `fused_qk_norm_rope` — per-head RMSNorm(Q,K) then RoPE over one packed
    qkv tensor. sglang (`sgl_kernel.fused_qk_norm_rope`, base+YARN) and vLLM
    (`_custom_ops.fused_qk_norm_rope`, precomputed cos/sin cache) ship it.
  * `rope_2d` — one tensor x[tokens,H,D] rotated by per-token cos/sin rows a
    vision tower built from (h, w). torch baseline and vLLM `ApplyRotaryEmb`
    (vllm-flash-attn Triton `apply_rotary_emb`).

Import-guarded: bodies import the library lazily inside `build`, so `discover()`
runs on a CPU box; `available()` gates execution on the library being present.
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from backend.inference.benchmark_identity import PLAIN
from . import TORCH_FLOAT, BenchCase, KernelSignature, Launch, Provider, register
from ..refs import records_on_grid
from .attention import _require_cuda, _torch_dtype


# Every in-place rotary launch here dispatches on torch's float types, so
# each declares `TORCH_FLOAT` — the same set, named once, rather than a
# per-kernel spelling of it: vLLM's `rotary_embedding`
# (VLLM_STABLE_DISPATCH_FLOATING_TYPES — Float, Half, BFloat16;
# csrc/pos_encoding_kernels.cu:172, csrc/dispatch_utils.h:39-42) and
# sglang's JIT `FusedRopeKernel`, templated on q/k's dtype with the table
# fp32 (kernels/ops/attention/rope.py:60-74). Both measured fp32 on the
# H100 (1,653 and 1,467 rows).


def _record_writer(floats):
    """`(output tokens, formats)` of a floor that writes the record the cell
    names through `rows_on_record` and the plain row at `floats` where it
    names none: the records on the grid, and their element dtypes beside
    the floats. A token list cannot say "the activation's dtype, or the
    record's element", so it admits the union."""
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    records = records_on_grid()
    elements = sorted({corpus_token(CACHE_FORMATS[n].dtype)
                       for n in records if n is not PLAIN})
    return (*floats, *elements), records


# --- rope: torch baseline (always available) -------------------------------


def _mrope_launch_params(inv: Invocation):
    from ..refs import _mrope_sections
    sections, interleaved = _mrope_sections(inv)
    return list(sections), bool(interleaved)


def _mrope_declines_bad_sections(inv: Invocation) -> str:
    _mrope_launch_params(inv)
    return ""


def _production_qkv_views(inputs, torch, dt):
    """Rebuild q and k as the strided rows of one fused qkv projection — the
    packed [q|k|v] pitch, with the unread V tail keeping the row stride wider
    than either operand. Production applies qk-norm BEFORE rotary at the
    Qwen call sites (vLLM qwen3_moe.py:337-343; sglang qwen3_5.py:1477), so
    what the rotary module is handed there is the norm's fresh contiguous
    rows, not these views; the strided form is the fused projection's own
    layout, which the kernels equally admit — `triton_mrope` reads
    `q.stride(0)` (mrope.py:194-195, :217), as does sglang's fused kernel.
    The launch measured is the same `triton_mrope` / `triton_mrope_fused`."""
    q3, k3 = inputs["q"].to(dt), inputs["k"].to(dt)
    tokens = q3.shape[0]
    q_width, k_width, v_width = q3.numel() // tokens, k3.numel() // tokens, k3.numel() // tokens
    packed = torch.zeros(tokens, q_width + k_width + v_width,
                         dtype=dt, device=q3.device)
    packed[:, :q_width] = q3.reshape(tokens, q_width)
    packed[:, q_width:q_width + k_width] = k3.reshape(tokens, k_width)
    return (packed[:, :q_width], packed[:, q_width:q_width + k_width],
            q3, k3)


def _torch_qk_mrope_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import get_ref
    reference = get_ref(inv.signature).reference
    return BenchCase(run=lambda: reference(inputs), to_canonical=lambda o: o)


def _vllm_qk_mrope_build(inv: Invocation, env: Env,
                         inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's own path: gather cos/sin from the three positions, then hand the
    strided q/k rows to `triton_mrope` (mrope.py:449-468). The wrapper's
    `cos.contiguous()` / `sin.contiguous()` are production code (:203-204)."""
    _require_cuda()
    import torch
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
    dt = _torch_dtype(inv.dtype)
    q, k, q3, k3 = _production_qkv_views(inputs, torch, dt)
    sections, interleaved = _mrope_launch_params(inv)
    d_head, rd = int(q3.shape[-1]), int(inputs["d_rope"])
    cos_sin = inputs["cos_sin_cache"].to(dt)[inputs["positions"]]
    cos, sin = cos_sin.chunk(2, dim=-1)

    def canonical(out):
        q_out, k_out = out
        return _canon_qk(q_out, k_out, q3.shape[-2], k3.shape[-2], d_head)

    return BenchCase(
        run=Launch(triton_mrope, q, k, cos, sin, sections, d_head, rd,
                   interleaved, True),
        to_canonical=canonical,
    )


def _sgl_qk_mrope_build(inv: Invocation, env: Env,
                        inputs: Dict[str, Any]) -> BenchCase:
    """sglang's in-place fused kernel reads the cache and all three positions
    directly (mrope.py:278-289); q/k stay the production split views."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
    dt = _torch_dtype(inv.dtype)
    q, k, q3, k3 = _production_qkv_views(inputs, torch, dt)
    sections, interleaved = _mrope_launch_params(inv)
    d_head, rd = int(q3.shape[-1]), int(inputs["d_rope"])

    def canonical(_):
        return _canon_qk(q, k, q3.shape[-2], k3.shape[-2], d_head)

    return BenchCase(
        run=Launch(triton_mrope_fused, q, k,
                   inputs["cos_sin_cache"].to(dt), inputs["positions"],
                   sections, d_head, rd, interleaved, False, True, None),
        to_canonical=canonical,
    )


_MROPE_GEOMETRY = (("d_head", (128, 256)), ("d_rope", (64, 128)))
# The section widths and `mrope_interleaved` are structural axes of the
# signature, so every cell carries them; Qwen3.5's packed sections are
# `mrope_interleaved=0`, which a presence gate on that axis used to refuse.

for _lib, _impl, _imp, _build in (
    ("torch", "torch_fused_qk_mrope", "torch", _torch_qk_mrope_build),
    ("vllm", "vllm_triton_mrope", "vllm", _vllm_qk_mrope_build),
    ("sglang", "sglang_triton_mrope_fused",
     "sglang.kernels.ops.attention.rotary_triton", _sgl_qk_mrope_build),
):
    register(Provider(
        library=_lib, signature="fused_qk_mrope", kernel_impl=_impl,
        import_name=_imp,
        serves=KernelSignature(signature="fused_qk_mrope",
                               activation=("bf16", "fp16"),
                               enum=_MROPE_GEOMETRY),
        impl_params=_mrope_declines_bad_sections,
        build=_build, fused_op="fused_qk_mrope",
    ))


def _torch_qk_norm_partial_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import get_ref
    reference = get_ref(inv.signature).reference
    return BenchCase(run=lambda: reference(inputs), to_canonical=lambda o: o)


def _sgl_qk_norm_partial_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """sglang's dense-layer launch: one [q|k|v] projection row, normed and
    rotated in place (minimax_m3.py:1373-1389)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope)
    dt = _torch_dtype(inv.dtype)
    qkv = inputs["qkv"].to(dt).clone()
    N = int(inv.dims["n_heads"])
    Nk = int(inv.dims["n_kv_heads"])
    D = int(inv.dims["d_head"])

    def canonical(_):
        return _canon_qk(qkv[:, :N * D], qkv[:, N * D:(N + Nk) * D],
                         N, Nk, D)

    return BenchCase(
        run=Launch(minimax_qknorm_rope, qkv,
                   inputs["q_norm_w"].to(torch.bfloat16),
                   inputs["k_norm_w"].to(torch.bfloat16),
                   inputs["cos_sin_cache"].float(), inputs["positions"],
                   N, Nk, Nk, inputs["eps"]),
        to_canonical=canonical,
    )


def _vllm_sparse_operands(inv, inputs, torch, dt):
    """The sparse row, output buffers, both caches and real slot mappings the
    NVIDIA attention hands the full launch (model.py:613-649)."""
    dense = inputs["qkv"].to(dt)
    tokens, dense_width = dense.shape
    D = int(inv.dims["d_head"])
    n_index = int(inv.dims["n_index_heads"])
    page = int(inv.dims["kv_block_size"])
    sparse_width = dense_width + (n_index + 1) * D
    qkv = torch.zeros(tokens, sparse_width, dtype=dt, device=dense.device)
    qkv[:, :dense_width] = dense
    blocks = -(-tokens // page) + 1
    cache = torch.zeros(blocks, int(inv.dims["n_kv_heads"]), page, 2 * D,
                        dtype=dt, device=dense.device)
    slots = inputs["slots"].to(dense.device)
    index_slots = inputs["index_slots"].to(dense.device)
    index_cache = torch.zeros(blocks, page, D, dtype=dt, device=dense.device)
    q_out = torch.empty(tokens, int(inv.dims["n_heads"]) * D,
                        dtype=dt, device=dense.device)
    index_q = torch.empty(tokens, n_index * D, dtype=dt,
                          device=dense.device)
    index_w = inputs["index_w"].to(dt)
    return (qkv, q_out, index_q, cache, index_cache, index_w,
            slots, index_slots)


def _vllm_qk_norm_partial_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """The full sparse production launch: main Q/K norm+RoPE, main K/V cache
    insert, and the index branch, all with the buffers allocated by the
    attention at model.py:613-649."""
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    dt = _torch_dtype(inv.dtype)
    (qkv, q_out, index_q, cache, index_cache, index_w,
     slots, index_slots) = _vllm_sparse_operands(inv, inputs, torch, dt)
    N = int(inv.dims["n_heads"])
    Nk = int(inv.dims["n_kv_heads"])
    D = int(inv.dims["d_head"])
    page = int(inv.dims["kv_block_size"])

    def canonical(_):
        return _canon_qk(q_out, qkv[:, N * D:(N + Nk) * D], N, Nk, D)

    return BenchCase(
        run=Launch(ops.fused_minimax_m3_qknorm_rope_kv_insert,
                   qkv, inputs["q_norm_w"].to(dt),
                   inputs["k_norm_w"].to(dt),
                   inputs["cos_sin_cache"].to(dt), inputs["positions"],
                   N, Nk, int(inputs["d_rope"]), inputs["eps"],
                   index_w[0], index_w[1],
                   int(inv.dims["n_index_heads"]), slots, index_slots,
                   cache, index_cache, page, q_out, index_q, "auto"),
        to_canonical=canonical,
    )


_MINIMAX_GEOMETRY = (("d_head", (128,)), ("d_rope", (64,)),
                     ("kv_block_size", (16,)))

register(Provider(
    library="torch", signature="fused_qk_norm_partial_rope",
    kernel_impl="torch_fused_qk_norm_partial_rope", import_name="torch",
    # The oracle's dense chain: no index branch and no cache written, so it
    # serves neither variant axis (the sparse launch is a wider one). It
    # reads the gains at whatever dtype the cell draws them (`.float()`).
    serves=KernelSignature(signature="fused_qk_norm_partial_rope",
                           activation=("bf16", "fp16"),
                           weight=TORCH_FLOAT,
                           enum=_MINIMAX_GEOMETRY[:2]),
    build=_torch_qk_norm_partial_build,
    fused_op="fused_qk_norm_partial_rope",
))
register(Provider(
    library="sglang", signature="fused_qk_norm_partial_rope",
    kernel_impl="sglang_minimax_qknorm_rope",
    import_name="sglang.kernels.ops.attention.minimax_qknorm_rope",
    # No variant axis: sglang's sparse rows use `minimax_qknorm_rope_grouped`
    # and fold the index branch into the same launch (minimax_m3.py:
    # 1353-1368); that is a wider span, not this dense [q|k|v] launch, which
    # writes no cache. The build hands the kernel bf16 gains.
    serves=KernelSignature(signature="fused_qk_norm_partial_rope",
                           activation=("bf16",), weight=("bf16",),
                           enum=_MINIMAX_GEOMETRY[:2]),
    build=_sgl_qk_norm_partial_build,
    fused_op="fused_qk_norm_partial_rope",
))
for _dt in ("bf16", "fp16"):
    register(Provider(
        library="vllm", signature="fused_qk_norm_partial_rope",
        kernel_impl=f"vllm_minimax_qknorm_rope_{_dt}", import_name="vllm._C_stable_libtorch",
        serves=KernelSignature(
            signature="fused_qk_norm_partial_rope", activation=(_dt,),
            # "q/k norm weight dtype must match qkv" (.cu:1002-1004).
            weight=(_dt,),
            enum=_MINIMAX_GEOMETRY,
            # The sparse launch only: the index branch, and both paged
            # caches written at the cell's page (`_vllm_sparse_operands`
            # reads both dims; the oracle draws the slot maps only when
            # both are present).
            requires_dims=("n_index_heads", "kv_block_size"),
            dtype_slots=(("kv", (_dt,)),),
            secondary_outputs=(("index_q", _dt), ("kv_cache", _dt),
                               ("index_cache", _dt))),
        build=_vllm_qk_norm_partial_build,
        fused_op="fused_qk_norm_partial_rope",
    ))

def _canon_qk(q, k, N: int, Nk: int, D: int):
    """In-place kernels mutate q/k; canonicalize to the [q‖k] float concat the
    reference produces."""
    import torch
    return torch.cat([q.reshape(q.shape[0], N * D).float(),
                      k.reshape(k.shape[0], Nk * D).float()], dim=1)


def _torch_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The eager torch RoPE, timed as vLLM's `forward_native` runs it.

    WHAT IS TIMED
        The rotation of Q and K, and nothing else. Two things used to be inside
        the timed call and are not any more, because neither is what a torch
        RoPE does:

          * an explicit `.float()` materialization of Q and K. The cos/sin
            table is fp32, so `x * cos` promotes anyway — the copy bought
            nothing and cost a full-size temporary of both tensors.
          * the `[q‖k]` concatenation. `forward_native` returns the two tensors
            separately; the concat exists only so the cross-check can compare
            one array, which is what `to_canonical` is for — and it is exactly
            where flashinfer/vLLM/sglang already do it, OUTSIDE their timing.

        Measured on an RTX 4090 at 16k tokens: 5353us before, 3690us after —
        1.45x. It is the always-available baseline, so every rope speedup in
        the corpus was inflated by that factor.

        The cos/sin table is cast to the query's dtype once, outside the call,
        as vLLM's `forward_native` does (`_match_cos_sin_cache_dtype`,
        rotary_embedding/base.py:105-125, v0.32.0), so the rotation runs at the
        cell's precision. It used to stay fp32: every product then promoted to
        fp32 and the launch gate refused the floor on every decode cell ("the
        launch's largest operand is float32 where the cell declares bfloat16",
        the fp32 table being the widest thing it read; ~340 a pass on the
        H100). The cast was once avoided because a max-relative-error check
        rejected it at cosine 1.00000; the RMS/worst-element band that replaced
        that check admits it.

    COMPARABILITY
        Now measures the same computation as the in-place kernels, at the same
        dtype, with the same work outside the timed region.
    """
    from ..refs import _rope_rotate
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["rotary_dim"]
    pos, neox = inputs["positions"], inputs["is_neox"]
    dt = _torch_dtype(inv.dtype)
    q, k = inputs["q"].to(dt), inputs["k"].to(dt)
    cache = inputs["cos_sin_cache"].to(dt)

    def _run():
        return (_rope_rotate(q.view(-1, N, D), pos, cache, rd, neox),
                _rope_rotate(k.view(-1, Nk, D), pos, cache, rd, neox))

    return BenchCase(
        run=_run,
        to_canonical=lambda o: _canon_qk(o[0], o[1], N, Nk, D))


register(Provider(
    library="torch", signature="rope", kernel_impl="torch_rope",
    import_name="torch",
    serves=KernelSignature(signature="rope", activation=TORCH_FLOAT),
    build=_torch_rope_build,
))


# --- rope: real in-place kernels (flashinfer / vLLM / sglang) ---------------

def _fi_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    dt = _torch_dtype(inv.dtype)
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    q = inputs["q"].to(dt).clone(); k = inputs["k"].to(dt).clone()
    cache = inputs["cos_sin_cache"].float()               # flashinfer wants fp32
    pos = inputs["positions"]

    def _run():
        apply_rope_with_cos_sin_cache_inplace(
            positions=pos, query=q, key=k, head_size=D,
            cos_sin_cache=cache, is_neox=inputs["is_neox"])
        return q, k
    return BenchCase(run=_run, to_canonical=lambda _: _canon_qk(q, k, N, Nk, D))


register(Provider(
    library="flashinfer", signature="rope", kernel_impl="flashinfer_rope",
    import_name="flashinfer",
    # q/k fp16 or bf16: `apply_rope_pos_ids_cos_sin_cache` dispatches the
    # query's dtype through DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16 (flashinfer
    # 0.7.2 csrc/rope.cu:247, the macro at csrc/tvm_ffi_utils.h:86-96 has
    # cases for f16 and bf16 only and otherwise fails "failed to dispatch
    # data type"). Gemma-2's fp32 cells: ~930 failed builds per H100 pass.
    #
    # Head widths 64/128/256/512 only: those are the widths
    # `BatchQKApplyRotaryPosIdsCosSinCache` runs its own rope kernel at
    # (include/flashinfer/pos_enc.cuh:1501-1502, 0.7.2.post1). Any other
    # width is routed to `RopeQuantize` (:1572-1590), whose NeoX path loads
    # the partner half of each head from memory another thread writes
    # (`vec_apply_llama_rope_cos_sin`, :130-148) with no barrier between —
    # and the in-place entry point hands it q as its own output
    # (flashinfer/rope.py:1278-1287). At 192
    # a head is 12 threads of 16 elements, so some heads straddle two warps
    # and the race shows: Step-5-Preview's d_rope 192 cell failed the
    # cross-check (typical 0.0251, worst 3.28, cosine 0.99923; 2026-10-28)
    # where every 64/128/256/512 cell passed. flashinfer main still routes
    # it so, so there is no release to move to.
    serves=KernelSignature(signature="rope", activation=("bf16", "fp16"),
                           enum=(("d_rope", (64, 128, 256, 512)),)),
    build=_fi_rope_build,
))


def _vllm_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm import _custom_ops as ops
    dt = _torch_dtype(inv.dtype)
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    q = inputs["q"].to(dt).clone(); k = inputs["k"].to(dt).clone()
    cache = inputs["cos_sin_cache"].to(dt)                # vLLM casts cache to query dtype
    pos = inputs["positions"]

    def _run():
        ops.rotary_embedding(pos, q, k, D, cache, inputs["is_neox"])
        return q, k
    return BenchCase(run=_run, to_canonical=lambda _: _canon_qk(q, k, N, Nk, D))


register(Provider(
    library="vllm", signature="rope", kernel_impl="vllm_c_rotary_embedding",
    import_name="vllm",
    serves=KernelSignature(signature="rope", activation=TORCH_FLOAT),
    build=_vllm_rope_build,
))


def _sgl_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """sglang v0.5.22's CUDA path for a head of 64/128/256/512: its own
    `sglang.kernels.ops.attention.rope.apply_rope_with_cos_sin_cache_inplace`
    on `[tokens, heads, head_size]` views, the cache kept fp32
    (srt/layers/rotary_embedding/base.py:112-139, `forward_cuda` :390-410).

    This used to look the kernel up in `sgl_kernel` as if 0.4.x had only
    renamed it to `rotary_embedding`. It had not: that is sglang's FALLBACK
    kernel — the one base.py:127-138 takes for any other head size, "the same
    implementation" as vLLM's `rotary_embedding`, which wants the cache at the
    query's dtype — so every cell failed "expected scalar type BFloat16 but
    found Float" (≈1,700 failures a pass on the H100, 2026-10-24) and no row
    of the kernel sglang actually runs was measured."""
    _require_cuda()
    from sglang.kernels.ops.attention.rope import (
        apply_rope_with_cos_sin_cache_inplace,
    )
    dt = _torch_dtype(inv.dtype)
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    q = inputs["q"].to(dt).clone(); k = inputs["k"].to(dt).clone()
    cache = inputs["cos_sin_cache"].float()               # sglang keeps it fp32
    pos = inputs["positions"]
    tokens = int(pos.shape[0])
    q_view, k_view = q.view(tokens, N, D), k.view(tokens, Nk, D)

    def _run():
        apply_rope_with_cos_sin_cache_inplace(
            q=q_view, k=k_view, cos_sin_cache=cache, positions=pos,
            is_neox=inputs["is_neox"])
        return q, k
    return BenchCase(run=_run, to_canonical=lambda _: _canon_qk(q, k, N, Nk, D))


register(Provider(
    library="sglang", signature="rope", kernel_impl="sgl_apply_rope_cos_sin_cache",
    import_name="sglang",
    # sglang's fused kernel covers only these head sizes — an enumeration, not
    # a divisibility, which is why the declaration language has that form.
    serves=KernelSignature(signature="rope", activation=TORCH_FLOAT,
                           enum=(("d_rope", (64, 128, 256, 512)),)),
    build=_sgl_rope_build,
))


# --- rope_partial / rope_partial_q: partial rotation on whole heads ---------
#
# The fused spans `fused_qk_rope_partial` (GQA: the first `d_rope` of each
# `d_head` head of q and k) and `fused_qk_rope_partial_q` (MLA: the last
# `d_rope` of each q head, the rope key whole), each one in-place launch on the
# heads where they lie (vLLM v0.32.0 / sglang v0.5.22):
#
#   vLLM   `RotaryEmbedding.forward_cuda` (model_executor/layers/
#          rotary_embedding/base.py) → `_custom_ops.rotary_embedding(positions,
#          q, k, head_size, cos_sin_cache, is_neox)`, cache cast to the query
#          dtype; the kernel rotates `cache.size(1)` of each `head_size`-strided
#          head. GQA hands it the whole q / k (`q, k = self.rotary_emb(...)`);
#          MLA hands it the strided tail view and k_pe
#          (model_executor/layers/mla.py:228).
#   sglang `RotaryEmbedding.forward_cuda` (srt/layers/rotary_embedding/
#          base.py:389-412) → `apply_rope_with_cos_sin_cache_inplace`
#          (kernels/ops/attention/rope.py:183) on `q[..., :rotary_dim]` views
#          for GQA; MLA's `forward_mha` passes the split-off q_pe view and
#          k_pe. Heads whose `head_size` is not 64/128/256/512 take sglang's
#          `rotary_embedding` fallback instead (base.py:117-136), a different
#          kernel, so the sglang providers declare that enum.
#
# NeoX vs GPT-J interleaving (DeepSeek's MLA is interleaved) is a kernel
# constexpr over the same bytes: the cells are measured NeoX, as `rope` is.

def _canon_q_kpe(q, k_pe, N: int, D: int, rd: int):
    import torch
    return torch.cat([q.reshape(q.shape[0], N * D).float(),
                      k_pe.reshape(k_pe.shape[0], rd).float()], dim=1)


def _sgl_rope_partial_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.attention.rope import (
        apply_rope_with_cos_sin_cache_inplace,
    )
    dt = _torch_dtype(inv.dtype)
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["rotary_dim"]
    q = inputs["q"].to(dt).clone(); k = inputs["k"].to(dt).clone()
    cache = inputs["cos_sin_cache"].float()               # sglang keeps it fp32
    pos = inputs["positions"]
    tokens = int(pos.shape[0])
    q_rope = q.view(tokens, N, D)[..., :rd]
    k_rope = k.view(tokens, Nk, D)[..., :rd]

    def _run():
        apply_rope_with_cos_sin_cache_inplace(
            q=q_rope, k=k_rope, cos_sin_cache=cache, positions=pos,
            is_neox=inputs["is_neox"])
        return q, k
    return BenchCase(run=_run, to_canonical=lambda _: _canon_qk(q, k, N, Nk, D))


def _vllm_rope_partial_q_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm import _custom_ops as ops
    dt = _torch_dtype(inv.dtype)
    N, D, rd = inputs["N"], inputs["D"], inputs["rotary_dim"]
    q = inputs["q"].to(dt).clone(); k_pe = inputs["k_pe"].to(dt).clone()
    cache = inputs["cos_sin_cache"].to(dt)                # vLLM casts cache to query dtype
    pos = inputs["positions"]
    tokens = int(pos.shape[0])
    q_tail = q.view(tokens, N, D)[..., D - rd:]           # the strided view mla.py:228 passes
    k_view = k_pe.view(tokens, 1, rd)

    def _run():
        ops.rotary_embedding(pos, q_tail, k_view, rd, cache, inputs["is_neox"])
        return q, k_pe
    return BenchCase(run=_run,
                     to_canonical=lambda _: _canon_q_kpe(q, k_pe, N, D, rd))


def _sgl_rope_partial_q_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.attention.rope import (
        apply_rope_with_cos_sin_cache_inplace,
    )
    dt = _torch_dtype(inv.dtype)
    N, D, rd = inputs["N"], inputs["D"], inputs["rotary_dim"]
    q = inputs["q"].to(dt).clone(); k_pe = inputs["k_pe"].to(dt).clone()
    cache = inputs["cos_sin_cache"].float()
    pos = inputs["positions"]
    tokens = int(pos.shape[0])
    q_pe = q.view(tokens, N, D)[..., D - rd:]             # forward_mha's split view
    k_view = k_pe.view(tokens, 1, rd)

    def _run():
        apply_rope_with_cos_sin_cache_inplace(
            q=q_pe, k=k_view, cos_sin_cache=cache, positions=pos,
            is_neox=inputs["is_neox"])
        return q, k_pe
    return BenchCase(run=_run,
                     to_canonical=lambda _: _canon_q_kpe(q, k_pe, N, D, rd))


def _torch_rope_partial_q_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's `forward_native` on MLA's operands: the strided view of each
    query head's last `d_rope` and the rope key, rotated at the cell's
    precision (the cos/sin table cast once, here, as vLLM caches its cast).

    The timed call is the two rotations and nothing else. Serving never
    re-joins q — `mla.py:227-230` assigns the rotated tail back onto its own
    view — so the `[q_nope | q_pe]` join and the `[q | k_pe]` flattening the
    cross-check compares are canonicalisation, in `to_canonical`. This used
    to time the oracle itself, fp32 arithmetic and joins included."""
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    N, D, rd = inputs["N"], inputs["D"], inputs["rotary_dim"]
    pos, neox = inputs["positions"], inputs["is_neox"]
    cache = inputs["cos_sin_cache"].to(dt)
    q = inputs["q"].to(dt).view(-1, N, D)
    k_pe = inputs["k_pe"].to(dt).view(-1, 1, rd)
    q_tail = q[..., D - rd:]                    # the view mla.py:228 passes

    def _run():
        return (_rope_rotate(q_tail, pos, cache, rd, neox),
                _rope_rotate(k_pe, pos, cache, rd, neox))

    def _canonical(out):
        import torch
        q_rot, k_rot = out
        joined = torch.cat([q[..., :D - rd], q_rot], dim=-1)
        return torch.cat([joined.reshape(joined.shape[0], N * D),
                          k_rot.reshape(k_rot.shape[0], rd)], dim=1)

    return BenchCase(run=_run, to_canonical=_canonical)


_SGL_ROPE_HEADS = (64, 128, 256, 512)

for _impl, _lib, _imp, _build, _serves in (
    # The `rope` builds already take `D` (head width) and `rotary_dim`
    # (cache width) apart, which is all a partial rotation is to them.
    ("torch_rope_partial", "torch", "torch", _torch_rope_build,
     KernelSignature(signature="rope_partial", activation=TORCH_FLOAT)),
    ("vllm_c_rotary_embedding_partial", "vllm", "vllm", _vllm_rope_build,
     KernelSignature(signature="rope_partial", activation=TORCH_FLOAT)),
    ("sglang_apply_rope_partial", "sglang", "sglang", _sgl_rope_partial_build,
     KernelSignature(signature="rope_partial", activation=TORCH_FLOAT,
                     enum=(("d_head", _SGL_ROPE_HEADS),))),
):
    register(Provider(library=_lib, signature="rope_partial", kernel_impl=_impl,
                      import_name=_imp, serves=_serves, build=_build,
                      fused_op="fused_qk_rope_partial"))

for _impl, _lib, _imp, _build, _serves in (
    ("torch_rope_partial_q", "torch", "torch", _torch_rope_partial_q_build,
     KernelSignature(signature="rope_partial_q", activation=TORCH_FLOAT)),
    ("vllm_c_rotary_embedding_partial_q", "vllm", "vllm",
     _vllm_rope_partial_q_build,
     KernelSignature(signature="rope_partial_q", activation=TORCH_FLOAT)),
    # sglang's rotary module for MLA has `head_size` = the rope width.
    ("sglang_apply_rope_partial_q", "sglang", "sglang",
     _sgl_rope_partial_q_build,
     KernelSignature(signature="rope_partial_q", activation=TORCH_FLOAT,
                     enum=(("d_rope", _SGL_ROPE_HEADS),))),
):
    register(Provider(library=_lib, signature="rope_partial_q",
                      kernel_impl=_impl, import_name=_imp, serves=_serves,
                      build=_build, fused_op="fused_qk_rope_partial_q"))


# --- fused_qk_rmsnorm_rope_gate: Qwen's gated attention prologue -----------

def _torch_qk_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The eager floor: the [q|gate] split, both norms, both partial
    rotations, the gate carried through."""
    import torch
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["d_rope"]
    qg = inputs["q_gate"].to(dt)
    k_in = inputs["k"].to(dt)
    qw, kw = inputs["q_norm_w"].to(dt), inputs["k_norm_w"].to(dt)
    eps, pos = inputs["eps"], inputs["positions"]
    # At the cell's precision throughout, the table included (the eager
    # rotary path's model-dtype cache, vLLM rotary_embedding/base.py:58-61).
    cache = inputs["cos_sin_cache"].to(dt)

    def _rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def _run():
        v = qg.view(-1, N, 2, D)
        q, gate = v[:, :, 0, :], v[:, :, 1, :]
        k = k_in.view(-1, Nk, D)
        q = _rope_rotate(_rms(q, qw), pos, cache, rd, True)
        k = _rope_rotate(_rms(k, kw), pos, cache, rd, True)
        return q, k, gate
    return BenchCase(run=_run, to_canonical=_canon_qk_gate)


def _canon_qk_gate(out):
    """(q, k, gate) as the oracle's one [q | k | gate] float row."""
    import torch
    return torch.cat([t.reshape(t.shape[0], -1).float() for t in out[:3]],
                     dim=1)


def _qkv_gate_views(inputs, torch, dt):
    """q|gate and k as the strided column blocks of one fused qkv projection
    row, as both frameworks hand them: `qkv.split([2·q_size, kv_size,
    kv_size])` (vLLM qwen3_next.py:407-409; sglang qwen3_5.py:1431-1433).
    The unread V tail keeps the row pitch wider than either operand; both
    kernels index rows by `stride(0)`."""
    qg, k = inputs["q_gate"].to(dt), inputs["k"].to(dt)
    tokens = qg.shape[0]
    qg_w, k_w = qg.numel() // tokens, k.numel() // tokens
    qkv = torch.zeros(tokens, qg_w + 2 * k_w, dtype=dt, device=qg.device)
    qkv[:, :qg_w] = qg.reshape(tokens, qg_w)
    qkv[:, qg_w:qg_w + k_w] = k.reshape(tokens, k_w)
    return qkv[:, :qg_w], qkv[:, qg_w:qg_w + k_w]


def _gemma_gain(w):
    """The checkpoint's Gemma form of an effective gain `g`: `g - 1`, which
    both kernels shift back by one in fp32 (vLLM `norm_beta`, sglang
    `WEIGHT_SHIFT`). Exact for the oracle's gains on [0.5, 2)
    (`refs._qk_gate_inputs`)."""
    return w - 1


def _vllm_qk_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's one launch (model_executor/layers/fused_qk_norm_rope.py:138),
    called as `Qwen3NextAttention._project_qkv_gate` does
    (models/qwen3_next.py:406-433): the split views of qkv, the GemmaRMSNorm
    weights with `norm_beta=1.0` (a constexpr, :41 — the kernel compiled for
    another beta is another kernel), the model-dtype cos/sin cache
    (rotary_embedding/base.py:58-61) and the rotary width. 1-D positions:
    the cell does not key the 3-axis form, which reads two more position
    rows and nothing else."""
    _require_cuda()
    import torch
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope_gate)
    dt = _torch_dtype(inv.dtype)
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["d_rope"]
    qg, k = _qkv_gate_views(inputs, torch, dt)
    qw, kw = _gemma_gain(inputs["q_norm_w"]), _gemma_gain(inputs["k_norm_w"])
    cache = inputs["cos_sin_cache"].to(dt)
    pos = inputs["positions"].to(torch.int64)
    return BenchCase(
        run=Launch(fused_qk_rmsnorm_rope_gate, qg, k, qw, kw, cache, pos,
                   float(inputs["eps"]), N, Nk, D, rd, norm_beta=1.0),
        to_canonical=_canon_qk_gate)


def _sgl_qk_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """sglang's one launch, `fused_qk_gemma_rmsnorm_rope_gate`
    (kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:149), called as
    `Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused` does
    (srt/models/qwen3_5.py:1428-1456): the split views of qkv, the raw
    GemmaRMSNorm weights (the kernel adds 1.0, :64, :92), the rotary
    module's cos/sin cache — fp32 on CUDA (rotary_embedding/base.py:112-117;
    this path never calls `_match_cos_sin_cache_dtype`) — and
    `has_gate=True`. 1-D positions, so `mrope_axis_map=None`
    (qwen3_5.py:1452): the cell does not key the 3-axis form, which reads
    two more position rows and the lane map and nothing else."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate)
    dt = _torch_dtype(inv.dtype)
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["d_rope"]
    qg, k = _qkv_gate_views(inputs, torch, dt)
    qw, kw = _gemma_gain(inputs["q_norm_w"]), _gemma_gain(inputs["k_norm_w"])
    cache = inputs["cos_sin_cache"].float()
    pos = inputs["positions"].to(torch.int64)
    return BenchCase(
        run=Launch(fused_qk_gemma_rmsnorm_rope_gate, qg, k, qw, kw, cache,
                   pos, float(inputs["eps"]), N, Nk, D, rd, has_gate=True),
        to_canonical=_canon_qk_gate)


register(Provider(
    library="torch", signature="fused_qk_rmsnorm_rope_gate",
    kernel_impl="torch_qk_rmsnorm_rope_gate", import_name="torch",
    serves=KernelSignature(signature="fused_qk_rmsnorm_rope_gate",
                           activation=TORCH_FLOAT),
    build=_torch_qk_gate_build,
    fused_op="fused_qk_rmsnorm_rope_gate",
))
# Both launches write k beside q, as fresh activation-dtype rows; the cache
# insert is the attention backend's own launch, so a row whose key cache is
# stored in another format is not this launch (`kv`).
register(Provider(
    library="vllm", signature="fused_qk_rmsnorm_rope_gate",
    kernel_impl="vllm_fused_qk_rmsnorm_rope_gate", import_name="vllm",
    # The kernel's own constraint: bf16/fp16 and an even rotary dim
    # (fused_qk_norm_rope.py:176-185).
    serves=KernelSignature(signature="fused_qk_rmsnorm_rope_gate",
                           activation=("bf16", "fp16"),
                           dtype_slots=(("kv", ("bf16", "fp16")),),
                           divisible=(("d_rope", 2),)),
    build=_vllm_qk_gate_build,
    fused_op="fused_qk_rmsnorm_rope_gate",
))
register(Provider(
    library="sglang", signature="fused_qk_rmsnorm_rope_gate",
    kernel_impl="sglang_fused_qk_gemma_rmsnorm_rope_gate",
    import_name="sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate",
    # An FP16 or a BF16 variant (`FP16=`, fused_qk_rmsnorm_rope_gate.py:228;
    # anything else stores bf16 over its input). The rotary width is halved
    # (`half_rotary = rotary_dim // 2`, :197), so an odd one drops a lane.
    serves=KernelSignature(signature="fused_qk_rmsnorm_rope_gate",
                           activation=("bf16", "fp16"),
                           dtype_slots=(("kv", ("bf16", "fp16")),),
                           divisible=(("d_rope", 2),)),
    build=_sgl_qk_gate_build,
    fused_op="fused_qk_rmsnorm_rope_gate",
))


# --- fused_qnorm_rope / fused_norm_rope: norm then partial rotation ---------

def _norm_rope_floor(rows_of, canonical=lambda o: o):
    """The eager floor for a norm+rope launch at the cell's dtype: the RMS
    accumulated in fp32 (as an unfused `rms_norm` does, without an fp32
    copy of the operand), the gain where the kernel reads one, then the
    last `d_rope` rotated. `rows_of` lays the operand out as the kernel's
    token rows `[tokens, H, d]`, and `canonical` takes the result back to
    the oracle's layout.

    A cell that names the record its launch writes (`output_format`) gets
    the rows written as that record (`refs.rows_on_record`), so the floor
    pays the launch's quantisation: normed with one rounding (the
    record's), the tail rotated from the unrounded row as the store does."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch
        from ..refs import _rope_rotate, rows_on_record
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        w = inputs["w"].to(dt) if "w" in inputs else None
        rd = int(inputs["d_rope"])
        eps = float(inputs["eps"])
        pos = inputs["positions"]
        cache = inputs["cos_sin_cache"]
        record = inv.output_format
        rows_shape = rows_of(x).shape
        wide = torch.promote_types(dt, torch.float32)
        record_row = (torch.empty(rows_shape, dtype=wide, device=x.device)
                      if record is not None else None)
        plain_row = (torch.empty(rows_shape, dtype=dt, device=x.device)
                     if record is None else None)

        def _run():
            r = torch.rsqrt(x.square().mean(-1, keepdim=True,
                                            dtype=torch.float32) + eps)
            if record is not None:
                n = rows_of(x * r if w is None else x * r * w)
                rotated = _rope_rotate(n[..., -rd:], pos, cache, rd,
                                       is_neox=False)
                record_row[..., :-rd] = n[..., :-rd]
                record_row[..., -rd:] = rotated
                return rows_on_record(record_row, record, dt, tail=rd)
            n = x * r.to(dt)
            if w is not None:
                n = n * w
            n = rows_of(n)
            plain_row[..., :-rd] = n[..., :-rd]
            plain_row[..., -rd:] = _rope_rotate(
                n[..., -rd:], pos, cache, rd, is_neox=False).to(dt)
            return plain_row
        return BenchCase(run=_run,
                         to_canonical=lambda o: canonical(o).float())
    return build


_FLOOR_OUTPUT, _FLOOR_FORMATS = _record_writer(TORCH_FLOAT)
_FLOOR_RECORDS = dict(output=_FLOOR_OUTPUT, output_format=_FLOOR_FORMATS)

register(Provider(
    library="torch", signature="fused_qnorm_rope",
    kernel_impl="torch_fused_qnorm_rope", import_name="torch",
    # The floor writes the record the cell names through `rows_on_record`
    # and the row where it names none: those records and the plain row.
    serves=KernelSignature(signature="fused_qnorm_rope",
                           activation=TORCH_FLOAT, **_FLOOR_RECORDS),
    # [B, n_heads, s_query, d_head] → token-major [tokens, n_heads, d_head].
    build=_norm_rope_floor(lambda n: n.permute(0, 2, 1, 3).flatten(0, 1)),
    fused_op="fused_qnorm_rope",
))
register(Provider(
    library="torch", signature="fused_norm_rope",
    kernel_impl="torch_fused_norm_rope", import_name="torch",
    # 255 rows measured at the FlashMLA record (output fp8_e4m3).
    serves=KernelSignature(signature="fused_norm_rope",
                           activation=TORCH_FLOAT, **_FLOOR_RECORDS),
    # One row per token: canonical `[tokens, d]`, rotated as one head.
    build=_norm_rope_floor(lambda n: n[:, None, :], lambda o: o[:, 0]),
    fused_op="fused_norm_rope",
))


# sglang v0.5.22 `fused_q_norm_rope` (kernels/ops/attention/dsv4/
# elementwise.py:171; JIT `FusedQNormRopeKernel`, kernels/jit/csrc/
# deepseek_v4/main_norm_rope.cuh:201-262, kernel :84-198), as
# `DeepseekV4Attention._compute_q_b` calls it when `q_head_norm`
# (srt/models/deepseek_v4.py:1414-1418), in prefill and decode alike: on CUDA
# every prepare path reaches `_compute_q_b` (:1541, :1590, :1803, :2069; the
# fused-qk alternatives are HIP-only, :1253 and unified_kv_kernels/
# env_gate.py:13-16).
#
# What production hands it:
#   q_input   the `wq_b` GEMM output viewed `[tokens, n_heads, d_head]`
#             (:1388-1389) — contiguous; the kernel requires the inner two
#             strides (`{-1, kHeadDim, 1}`, .cuh:219-223).
#   q_output  bf16 (V4's FP8 query buffer exists only for V4.1, whose query
#             has no head norm — :2294-2305 — so the wrapper's FP8 branch,
#             elementwise.py:180-186, is never this span's). Either
#             `empty_like(q)` (:1415) or, when `_kernel_num_heads` pads the
#             heads (:1034-1054: attention TP > 1, `64 if n_local_heads <=
#             64`), the leading-heads view of a `[tokens, 64, d_head]`
#             buffer (:2337-2356) — row stride wider than the heads written.
#   freqs_cis the complex64 `[max_pos, d_rope / 2]` table, `polar(1, θ)`
#             (kernels/ops/attention/deepseek_v4_rope.py:66-68); the wrapper
#             views it as interleaved (re, im) fp32 (elementwise.py:187).
#
# The rotation: lane j reads the adjacent pair (2j, 2j+1) of the rope tail
# as (real, imag) and multiplies by (cos θ_j, sin θ_j) (.cuh:171-197) —
# GPT-J interleaved, the oracle's `is_neox=False`.

def _freqs_cis_from_cos_sin(cache):
    """The oracle's `[max_pos, d_rope]` table, `[cos | sin]` halves, as the
    complex64 `freqs_cis` the V4 layer holds: `cos + i·sin` per frequency.
    The values themselves, not recomputed angles."""
    import torch
    half = cache.shape[-1] // 2
    return torch.complex(cache[:, :half].float().contiguous(),
                         cache[:, half:].float().contiguous())


#: The head width the serving buffer pads the query heads up to under
#: attention TP (`_kernel_num_heads`, srt/models/deepseek_v4.py:1054, the
#: FlashMLA decode kernel's head count). sglang's constant, stated once.
_SGL_DSV4_PADDED_HEADS = 64


def _sgl_qnorm_rope_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    """sglang's warp-per-(token, head) launch through its own wrapper, with
    the operands `_compute_q_b` passes (see the block above)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_q_norm_rope,
    )
    dt = _torch_dtype(inv.dtype)
    # [B, n_heads, s_query, d_head] → the GEMM's token-major rows.
    q = inputs["x"].to(dt).permute(0, 2, 1, 3).flatten(0, 1).contiguous()
    tokens, n_heads, d_head = q.shape
    if n_heads < _SGL_DSV4_PADDED_HEADS:
        q_out = torch.empty(tokens, _SGL_DSV4_PADDED_HEADS, d_head, dtype=dt,
                            device=q.device)[:, :n_heads]
    else:
        q_out = torch.empty_like(q)
    freqs_cis = _freqs_cis_from_cos_sin(inputs["cos_sin_cache"])
    positions = inputs["positions"].to(torch.int64)
    eps = float(inputs["eps"])

    def run():
        fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)
        return q_out
    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="sglang", signature="fused_qnorm_rope",
    kernel_impl="sglang_fused_q_norm_rope",
    import_name="sglang.kernels.ops.attention.dsv4.elementwise",
    serves=KernelSignature(
        signature="fused_qnorm_rope",
        # The wrapper's FP8 path asserts a bf16 input (elementwise.py:182)
        # and V4 serves bf16; the output is the input's dtype.
        activation=("bf16",),
        # The template's static asserts (.cuh:87-95): one (re, im) pair per
        # lane, so the rotation is exactly 64 wide; and the head splits into
        # whole 16-byte vectors per lane, which for a 2-byte dtype is any
        # multiple of 256 (64 and 128 also compile, as one partial-width
        # vector per lane; the declaration has no union form, so those are
        # declined rather than claimed).
        enum=(("d_rope", (64,)),),
        divisible=(("d_head", 256),)),
    build=_sgl_qnorm_rope_build,
    fused_op="fused_qnorm_rope",
))


# sglang v0.5.22 runs `fused_norm_rope` (one row per token: RMSNorm with the
# norm's gain, then the GPT-J rotation of the row's last 64 channels) as two
# launches, one per kind of row the plan matches:
#
# 1. In place, the row written back in its own dtype: the JIT
#    `FusedNormRopeKernel` (kernels/jit/csrc/deepseek_v4/fused_norm_rope.cuh:
#    170-251, kernel :41-167; module `norm_rope_module`, kernels/ops/
#    attention/dsv4/compress_old.py:60). Two wrappers reach it:
#      * `compress_fused_norm_rope_inplace` (compress_old.py:271), on the
#        compressor's pooled entry (srt/layers/attention/dsv4/compressor.py:
#        141-151, `kv_compressed` the `[num_q_tokens, head_dim]` output of
#        `compress_forward`, compress_old.py:243), in both phases. Its handle
#        is the compressor's plan (`make_compressor_plan`, compressor.py:
#        223-241): prefill, `CompressorPrefillPlan.compress_plan` — one
#        16-byte `PrefillPlan {ragged_id, batch_id, position, window_len}`
#        per pooled entry (include/sgl_kernel/deepseek_v4/compress.cuh:16-21),
#        the kernel rotating row `ragged_id` at `position + 1 -
#        compress_ratio` (.cuh:71-76); decode, `CompressorDecodePlan.seq_lens`
#        int32, row i rotated at `seq_len - compress_ratio` and skipped, before
#        any load, unless `seq_len % compress_ratio == 0` (.cuh:77-81).
#      * `fused_norm_rope_inplace` (compress_old.py:291), on the latent KV in
#        prefill context parallelism (`_compute_kv_bf16`, srt/models/
#        deepseek_v4.py:1464-1482, from :2074 / :2083): mode 2, an int64
#        position per row (.cuh:82-84).
#    All three modes run the same body (.cuh:89-164) on a row they reach; the
#    mode only says where the row and its position come from. So the
#    compressor's form IS this span's function, rotated at the position its
#    plan names — the pooled entry's window start. `compress_ratio` enters
#    only that arithmetic and the decode skip; the provider builds the plan at
#    a ratio of 1 (`_SGL_COMPRESSOR_PLAN_RATIO`), which makes every row of the
#    cell a completed entry and its plan position the oracle's own, rather
#    than production's 4 / 128 (`Literal[4, 128]`), at which the oracle's
#    consecutive positions are not window starts. Every row the cell carries
#    is one the graph bills (the compressor's pooled rows are the windows a
#    step completes), so the rows a production decode step skips are rows no
#    cell counts. The pooled rows sit densely here; production leaves them at
#    their `ragged_id` in a `num_q_tokens`-row buffer — the same rows read
#    and written, at other addresses.
#    The wrapper's `head_dim` is the template's (the row width, 512 for the
#    attention's compressor and 128 for the indexer's); the rope width is
#    64 only (.cuh:56).
#
# 2. Into the SWA cache, no bf16 row: `fused_k_norm_rope_flashmla`
#    (kernels/ops/attention/dsv4/elementwise.py:307; JIT
#    `FusedKNormRopeFlashMLAKernel`, kernels/jit/csrc/deepseek_v4/
#    main_norm_rope.cuh:437-559, kernel :289-436), from
#    `set_swa_key_buffer_radix_fused_norm_rope`
#    (srt/mem_cache/deepseek_v4_memory_pool.py:2041, call :2065), reached
#    from `_compute_kv_to_cache` (srt/models/deepseek_v4.py:1420-1462) on
#    every CUDA prepare path in both phases (:1531, :1604, :1804, :2122);
#    the default `SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE=False`
#    (srt/environ.py:1588) keeps it off the bf16 path. It writes the V4
#    FlashMLA record (`KVLayout.V4`, include/sgl_kernel/deepseek_v4/
#    kv_layout.cuh:43-50): 448 e4m3 channels each 64 of which share one
#    UE8M0 exponent computed from the fp32 normed values (main_norm_rope.cuh
#    :423-434), then the 64 rotated channels as bf16 (:413-422), and a scale
#    row of 7 exponents + 1 pad byte. That record is byte-identical, within a
#    page, to the one vLLM's insert writes (`_dsv4_swa_record_bytes` /
#    `_dsv4_swa_rows` below): data rows at `slot * 576` and scale rows from
#    `page_size * 576` (kv_layout.cuh:102-111), the exponent byte biased by
#    127 (fp8_utils.cuh:31-41). The one difference is the page STRIDE —
#    sglang pads a page up to a multiple of 576 bytes (`kv_page_bytes`,
#    kv_layout.cuh:78-83; the pool allocates it so, deepseek_v4_memory_pool
#    .py:271-292) — and the decoder addresses within a page, so it reads
#    either. Production's `kv` is usually `qkv_a[..., q_lora_rank:]`, a
#    strided view of the fused `wqkv_a` output (`SGLANG_OPT_FUSE_WQA_WKV`
#    defaults on, srt/environ.py:1665; the kernel takes the row stride,
#    main_norm_rope.cuh:275-278); here the rows are contiguous — the same
#    1 KiB per row read, at another stride.

#: The compress ratio the compressor's plan is built at (block above): 1, at
#: which each pooled row's plan position is the oracle's own position.
_SGL_COMPRESSOR_PLAN_RATIO = 1

#: Tokens per page of sglang's DeepSeek-V4 SWA cache on CUDA: the scheduler's
#: page, asserted to be 256 (srt/mem_cache/kv_cache_configurator.py:1767-1769)
#: and a template argument of the store (main_norm_rope.cuh:445-448). The span
#: reads no cache (no `role="kv"` operand), so no page keys its cell and the
#: build pages at the one value sglang runs.
_SGL_DSV4_SWA_PAGE = 256


def _sgl_compressor_plan_handle(phase: str, positions):
    """The handle `compress_fused_norm_rope_inplace` passes the kernel
    (`plan[1]`), built so that the kernel's own arithmetic, at
    `_SGL_COMPRESSOR_PLAN_RATIO`, lands every row on `positions`:

      decode   int32 `seq_lens`, `seq_len - ratio` the position (.cuh:79-81);
      prefill  `[rows, 16]` uint8, one `PrefillPlan {ragged_id = row,
               batch_id = 0, position = position + ratio - 1, window_len = 0}`
               per row (compress.cuh:16-21; the kernel reads `ragged_id` and
               `position`, .cuh:72-74).
    """
    import torch
    ratio = _SGL_COMPRESSOR_PLAN_RATIO
    pos = positions.to(torch.int64)
    if phase == "decode":
        return (pos + ratio).to(torch.int32)
    rows = pos.shape[0]
    plan = torch.zeros(rows, 4, dtype=torch.int32, device=pos.device)
    plan[:, 0] = torch.arange(rows, dtype=torch.int32, device=pos.device)
    plan[:, 2] = (pos + ratio - 1).to(torch.int32)
    return plan.view(torch.uint8)


def _sgl_row_norm_rope_operands(inv: Invocation, inputs: Dict[str, Any]):
    """`(rows, gain, freqs_cis, positions, eps)` as both launches take them:
    contiguous `[tokens, d]` rows and the gain at the activation's dtype (the
    kernels match the weight's dtype to the row's, fused_norm_rope.cuh:196-199,
    main_norm_rope.cuh:502-505), the layer's complex `freqs_cis`, int64
    positions."""
    import torch
    dt = _torch_dtype(inv.dtype)
    return (inputs["x"].to(dt).contiguous(), inputs["w"].to(dt).contiguous(),
            _freqs_cis_from_cos_sin(inputs["cos_sin_cache"]),
            inputs["positions"].to(torch.int64), float(inputs["eps"]))


def _sgl_norm_rope_inplace_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    """The compressor's in-place launch through its own wrapper, with the
    plan its phase runs (block above)."""
    _require_cuda()
    from sglang.kernels.ops.attention.dsv4.compress_old import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
        compress_fused_norm_rope_inplace,
    )
    x, w, freqs_cis, positions, eps = _sgl_row_norm_rope_operands(inv, inputs)
    handle = _sgl_compressor_plan_handle(inv.phase, positions)
    ratio = _SGL_COMPRESSOR_PLAN_RATIO
    if inv.phase == "decode":
        plan = CompressorDecodePlan(ratio, handle)
    else:
        # `write_plan` is the store's; this launch reads `plan[1]` only
        # (compress_old.py:280-288).
        plan = CompressorPrefillPlan(ratio, handle, handle[:0])
    kv = x.clone()
    return BenchCase(
        run=Launch(compress_fused_norm_rope_inplace, kv, w, eps, freqs_cis,
                   plan),
        to_canonical=lambda _o: kv.float())


def _sgl_norm_rope_flashmla_build(inv: Invocation, env: Env,
                                  inputs: Dict[str, Any]) -> BenchCase:
    """The SWA store as `set_swa_key_buffer_radix_fused_norm_rope` calls it:
    a fresh request's consecutive slots in a `[pages, page bytes]` uint8
    cache laid out as the pool allocates it; read back by decoding the
    record."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_k_norm_rope_flashmla,
    )
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    x, w, freqs_cis, positions, eps = _sgl_row_norm_rope_operands(inv, inputs)
    tokens, d_head = x.shape
    d_rope = int(inputs["d_rope"])
    page = _SGL_DSV4_SWA_PAGE
    pages = -(-tokens // page)
    cache = torch.zeros(pages, KVLayout.V4.page_bytes(page),
                        dtype=torch.uint8, device=x.device)
    slots = torch.arange(tokens, dtype=torch.int32, device=x.device)
    return BenchCase(
        run=Launch(fused_k_norm_rope_flashmla, x, w, eps, freqs_cis,
                   positions, slots, cache, page, KVLayout.V4),
        to_canonical=lambda _o: _dsv4_swa_rows(
            cache, slots.to(torch.int64), page, d_head, d_rope))


register(Provider(
    library="sglang", signature="fused_norm_rope",
    kernel_impl="sglang_compress_fused_norm_rope_inplace",
    import_name="sglang.kernels.ops.attention.dsv4.compress_old",
    serves=KernelSignature(
        signature="fused_norm_rope",
        # V4 serves bf16 and the row is written back in its own dtype.
        activation=("bf16",), weight=("bf16",), output=("bf16",),
        # The template's static asserts (fused_norm_rope.cuh:46-56): the
        # rotation is one (re, im) pair per lane, so exactly 64; the row
        # splits into `kLocalSize` vectors of `min(8, d / 32)` elements per
        # lane, the rope tail within the last of them and a whole number of
        # them — which for a 2-byte dtype is the powers of two from 64 to
        # 2048.
        enum=(("d_rope", (64,)),
              ("d_head", (64, 128, 256, 512, 1024, 2048)))),
    build=_sgl_norm_rope_inplace_build,
    fused_op="fused_norm_rope",
))
register(Provider(
    library="sglang", signature="fused_norm_rope",
    kernel_impl="sglang_fused_k_norm_rope_flashmla",
    import_name="sglang.kernels.ops.attention.dsv4.elementwise",
    serves=KernelSignature(
        signature="fused_norm_rope",
        activation=("bf16",), weight=("bf16",),
        # What it writes: the FlashMLA V4 record, FP8 e4m3 with a UE8M0
        # exponent per 64 and a bf16 rope tail — no bf16 row. The planner
        # keys a cached-tensor writer's `output` by the record's element
        # dtype and `output_format` by the record; a cell keyed at bf16
        # there is the in-place launch's.
        output=("fp8_e4m3",),
        output_format=("fp8_ue8m0_group64_bf16_rope_tail",),
        # "FlashMLA layout requires (512, 64)" (main_norm_rope.cuh:450).
        enum=(("d_head", (512,)), ("d_rope", (64,)))),
    build=_sgl_norm_rope_flashmla_build,
    fused_op="fused_norm_rope",
))


# --- fused_inv_rope: DeepSeek-V4's attention output, un-rotated -------------

def _torch_inv_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The eager un-rotation: the always-available floor for the launch vLLM
    fuses with its FP8 quant."""
    import torch
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    # The walk's [B, n_heads, s_query, d_head] as the kernel's token rows.
    o = inputs["o"].to(dt).permute(0, 2, 1, 3).flatten(0, 1).contiguous()
    rd = int(inputs["d_rope"])
    cache = inputs["cos_sin_cache"]
    half = cache.shape[-1] // 2
    conj = torch.cat([cache[:, :half], -cache[:, half:]], dim=-1)
    pos = inputs["positions"]
    out = torch.empty(o.shape, dtype=torch.promote_types(o.dtype,
                                                        torch.float32),
                      device=o.device)

    def _run():
        out[..., :-rd] = o[..., :-rd]
        out[..., -rd:] = _rope_rotate(o[..., -rd:], pos, conj, rd,
                                      is_neox=False)
        return out
    return BenchCase(run=_run, to_canonical=lambda out: out.float())


register(Provider(
    library="torch", signature="fused_inv_rope", kernel_impl="torch_inv_rope",
    import_name="torch",
    serves=KernelSignature(signature="fused_inv_rope",
                           activation=TORCH_FLOAT),
    build=_torch_inv_rope_build,
    fused_op="fused_inv_rope",
))


def _vllm_inv_rope_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's fused un-rotation. `quantize=False` measures the span the
    fusion plan draws — o in, o out: the FP8 the serving path also writes is
    the precision profile's choice, not part of this boundary, and the plan's
    cost is the boundary's."""
    _require_cuda()
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant)
    import torch
    dt = _torch_dtype(inv.dtype)
    # The kernel's operand dtypes are not one dtype: the activation is the
    # cell's (bf16), the rope table is fp32 and the positions int64, and it
    # asserts each (`assert cos_sin_cache.dtype == torch.float32`). Casting
    # the table to the activation dtype, as the rest of this file does for
    # kernels that read it at the activation's, trips that assertion.
    o = inputs["o"].to(dt).permute(0, 2, 1, 3).flatten(0, 1).contiguous()
    rd = int(inputs["d_rope"])
    n_heads = int(inv.dims["n_heads"])
    d_head = int(inv.dims["d_head"])
    nope = d_head - rd
    pos = inputs["positions"].to(torch.int64)
    cache = inputs["cos_sin_cache"].float()
    tokens = int(o.shape[0])

    def _run():
        out, _scales = fused_inv_rope_fp8_quant(
            o, pos, cache, n_groups=1, heads_per_group=n_heads,
            nope_dim=nope, rope_dim=rd, quantize=False)
        return out
    # The kernel returns `[tokens, n_groups, heads_per_group · head_dim]`
    # (its internal buffer transposed); the contract is one row per head,
    # which is what the reference computes and what `o` came in as.
    return BenchCase(
        run=_run,
        to_canonical=lambda out: out.reshape(tokens, n_heads, d_head).float())


def _vllm_inv_rope_quant_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's quantizing form: inverse RoPE and the activation's FP8/UE8M0
    record in one launch. The scale layout is the wrapper's row-major form,
    which is both SM90 serving and the local Ada smoke surface; SM100's packed
    representation is another ABI and is not claimed here."""
    _require_cuda()
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant)
    import torch
    cast = inputs["cast"]
    o = inputs["o"].to(_torch_dtype(cast.wide)).permute(
        0, 2, 1, 3).flatten(0, 1).contiguous()
    rd = int(inputs["d_rope"])
    n_heads = int(inv.dims["n_heads"])
    d_head = int(inv.dims["d_head"])
    groups = int(inputs["n_groups"])
    heads_per_group, remainder = divmod(n_heads, groups)
    if remainder or d_head % cast.group:
        raise NotImplementedError(
            f"the launch groups {n_heads} heads as {groups} and {d_head} "
            f"channels by {cast.group}; neither division is whole")
    nope = d_head - rd
    pos = inputs["positions"].to(torch.int64)
    cache = inputs["cos_sin_cache"].float()

    def _run():
        out, _scales = fused_inv_rope_fp8_quant(
            o, pos, cache, n_groups=groups,
            heads_per_group=heads_per_group, nope_dim=nope, rope_dim=rd,
            quant_group_size=cast.group, tma_aligned_scales=False,
            quantize=True)
        return out, _scales

    def _canonical(pair):
        # This launch keeps its output groups on their own axis; the cast's
        # grid is over each group's flattened row, so both sides are read as
        # rows before the one dequantiser sees them.
        out, scales = pair
        return cast.dequantise((out.reshape(-1, out.shape[-1]),
                                scales.reshape(-1, scales.shape[-1])))
    return BenchCase(run=_run, to_canonical=_canonical)


register(Provider(
    library="vllm", signature="fused_inv_rope",
    kernel_impl="vllm_fused_inv_rope", import_name="vllm",
    # The kernel's own shape assumptions: the rope tail is 64 wide and the
    # head splits into whole quant groups.
    # `quantize=False` writes `o.dtype` (fused_inv_rope_fp8_quant.py:515);
    # V4 serves bf16 and that is all it was measured at (959 rows).
    serves=KernelSignature(signature="fused_inv_rope",
                           activation=("bf16",),
                           enum=(("d_rope", (64,)),),
                           divisible=(("d_head", 128),)),
    build=_vllm_inv_rope_build,
    fused_op="fused_inv_rope",
))

register(Provider(
    library="vllm", signature="fused_inv_rope",
    kernel_impl="vllm_fused_inv_rope_quant", import_name="vllm",
    serves=KernelSignature(signature="fused_inv_rope",
                           activation=("bf16",),
                           output=("fp8_e4m3",),
                           activation_granularity=("per-group:32",
                                                   "per-group:128"),
                           dtype_slots=(("metadata", ("ue8m0",)),),
                           enum=(("d_rope", (64,)),),
                           divisible=(("d_head", 128),)),
    build=_vllm_inv_rope_quant_build,
    fused_op="fused_inv_rope",
))


# --- fused_qk_norm_rope -----------------------------------------------------

def _sgl_qknr_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from sgl_kernel import fused_qk_norm_rope
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    qkv = inputs["qkv"].clone()                           # in-place; keep a fresh copy
    q_w, k_w = inputs["q_norm_w"], inputs["k_norm_w"]
    pos = inputs["positions"].to(torch.int32)

    def _run():
        # Neutral YARN (factor=1) so the measured kernel matches the plain
        # norm+rope reference; low/high/attention_factor pass-through.
        fused_qk_norm_rope(qkv, N, Nk, Nk, D, inputs["eps"], q_w, k_w,
                           inputs["base"], inputs["is_neox"], pos,
                           1.0, 0.0, 0.0, 1.0)
        return qkv
    return BenchCase(run=_run, to_canonical=lambda o: o.float())


register(Provider(
    library="sglang", signature="fused_qk_norm_rope",
    kernel_impl="sgl_fused_qk_norm_rope",
    import_name="sgl_kernel",
    # bf16 only, head_dim a multiple of 64.
    serves=KernelSignature(signature="fused_qk_norm_rope",
                           activation=("bf16",),
                           divisible=(("d_rope", 64),)), build=_sgl_qknr_build,
))


def _vllm_qknr_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    qkv = inputs["qkv"].clone()
    q_w, k_w = inputs["q_norm_w"], inputs["k_norm_w"]
    cache = inputs["cos_sin_cache"].to(qkv.dtype)         # vLLM takes precomputed cos/sin
    pos = inputs["positions"].to(torch.int64)

    def _run():
        ops.fused_qk_norm_rope(qkv, N, Nk, Nk, D, inputs["eps"], q_w, k_w,
                               cache, inputs["is_neox"], pos)
        return qkv
    return BenchCase(run=_run, to_canonical=lambda o: o.float())


register(Provider(
    library="vllm", signature="fused_qk_norm_rope",
    kernel_impl="vllm_c_fused_qk_norm_rope",
    import_name="vllm",
    # bf16 only, head_dim a multiple of 64.
    serves=KernelSignature(signature="fused_qk_norm_rope",
                           activation=("bf16",),
                           divisible=(("d_rope", 64),)), build=_vllm_qknr_build,
))


def _torch_qknr_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import _qk_norm_rope_reference
    dt = _torch_dtype(inv.dtype)
    packed = dict(inputs)
    packed["qkv"] = inputs["qkv"].to(dt)
    packed["q_norm_w"] = inputs["q_norm_w"].to(dt)
    packed["k_norm_w"] = inputs["k_norm_w"].to(dt)

    def _run():
        return _qk_norm_rope_reference(packed)
    return BenchCase(run=_run, to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="fused_qk_norm_rope",
    kernel_impl="torch_fused_qk_norm_rope",
    import_name="torch",
    serves=KernelSignature(signature="fused_qk_norm_rope",
                           activation=TORCH_FLOAT),
    build=_torch_qknr_build,
))


# --- rope_2d: the vision towers' two-axis rotation ---------------------------

def _torch_rope_2d_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """`ApplyRotaryEmb.forward_native` (rotary_embedding/common.py): the eager
    NeoX rotation, cos/sin cast to x's dtype as vLLM's `forward_static`
    does."""
    from ..refs import _rope_2d_rotate
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    cos, sin = inputs["cos"].to(dt), inputs["sin"].to(dt)
    return BenchCase(run=lambda: _rope_2d_rotate(x, cos, sin),
                     to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="rope_2d", kernel_impl="torch_rope_2d",
    import_name="torch",
    serves=KernelSignature(signature="rope_2d", activation=TORCH_FLOAT),
    build=_torch_rope_2d_build,
))


def _vllm_rope_2d_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """`ApplyRotaryEmb.forward_cuda`: vllm-flash-attn's Triton
    `apply_rotary_emb(x, cos, sin, interleaved=False)` on x [1, tokens, H, D]
    — the call the Qwen-VL and MiMo-V2 omni vision towers make."""
    _require_cuda()
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).unsqueeze(0)
    cos, sin = inputs["cos"].to(dt), inputs["sin"].to(dt)
    return BenchCase(run=lambda: apply_rotary_emb(x, cos, sin, False),
                     to_canonical=lambda o: o.squeeze(0).float())


# vllm-flash-attn's Triton `apply_rotary` checks only that cos/sin match
# x's dtype (ops/triton/rotary.py:173-177), which the build arranges, and
# computes in fp32 registers, storing at x's dtype — torch's float types,
# `TORCH_FLOAT`.

register(Provider(
    library="vllm", signature="rope_2d", kernel_impl="vllm_apply_rotary_emb",
    import_name="vllm",
    serves=KernelSignature(signature="rope_2d", activation=TORCH_FLOAT),
    build=_vllm_rope_2d_build,
))


# --- xdrope: the same two kernels, a different cell --------------------------
#
# XD-RoPE's rotation IS `rope_2d`'s — one NeoX rotation over the whole head by
# per-token cos/sin rows — and vLLM's `XDRotaryEmbedding.forward_cuda` says so
# by assembling its rows and then calling the same `ApplyRotaryEmb`. What is
# its own is where the rows come from: four position streams (p, w, h, t), one
# `xdrope_section` channel section each, which the cell carries in `cos`/`sin`
# (`refs._xdrope_inputs`).
#
# So the builders are shared and the SIGNATURE is what differs, which is the
# right way round: a separate signature because the cell is a different one,
# the same provider body because the launch is the same launch. Without these
# two registrations HunyuanOCR's 215 planned xdrope cells had a signature, a
# reference and nowhere to run.

register(Provider(
    library="torch", signature="xdrope", kernel_impl="torch_xdrope",
    import_name="torch",
    serves=KernelSignature(signature="xdrope", activation=TORCH_FLOAT),
    build=_torch_rope_2d_build,
))

register(Provider(
    library="vllm", signature="xdrope", kernel_impl="vllm_xdrope_apply_rotary_emb",
    import_name="vllm",
    serves=KernelSignature(signature="xdrope", activation=TORCH_FLOAT),
    build=_vllm_rope_2d_build,
))


# --- fused_qnorm_rope_kv_insert: the DeepSeek-V4 attention prologue ---------
#
# One launch prepares the whole attention input: the query heads normalised
# (where the span norms them) and rotated, and the latent-KV row rotated,
# quantised and written into the paged sliding-window cache. vLLM runs it as
# `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
# (csrc/torch_bindings.cpp:425-431, kernel
# csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1134), and the two
# layers that call it differ only in `apply_q_norm`:
#
#   V4    models/deepseek_v4/attention.py:759 `_fused_qnorm_rope_kv_insert`,
#         called at :593, the op at :800 — nine positional arguments, so the
#         four trailing flags take their defaults (`apply_q_norm=True`,
#         `kv_mxfp8=False`, `apply_q_rope=True`, `is_q_interleaved=False`).
#   V4.1  models/deepseek_v41/attention.py:1011, from
#         `project_query_and_cache_kv` :856-860, the op at :1072 with `False`
#         for `apply_q_norm` at :1082.
#
# The paged (uint8) overload is the one both reach: vLLM dispatches on the
# cache tensor's dtype (:791 / :1056) and both specs declare the sliding-window
# cache as block-scaled FP8. `kv_mxfp8` stays False — V4 never passes it, and
# V4.1's value is `current_platform.is_device_capability_family(100)`
# (deepseek_v41/attention.py:128-129, :512), which is SM100, where that span's
# binding does not bind (`below_compute_capability: "10.0"`). So on every card
# either plan binds on, the record written is V4's: `d_head - d_rope` FP8
# bytes, `d_rope` bf16 values, and one UE8M0 exponent per 64 quantised
# channels in a per-block tail (.cu:29-34, :355-420).

#: Channels that share one UE8M0 exponent in that record (`kQuantBlock`,
#: .cu:99). The kernel's own compiled geometry, like the 512-wide head and
#: 64-wide rotation its checks require — the providers below decline any cell
#: that is not exactly that, so the whole record follows from it.
_DSV4_SWA_QUANT_BLOCK = 64

#: Pages the sliding-window cache is allocated in. Its backend declares
#: `MultipleOf(32)` and prefers 256
#: (vllm/v1/attention/backends/mla/sparse_swa.py:154, :158) — the insert
#: itself takes the page as a runtime argument (:106-107) — so these are the
#: multiples of its base up to that preference. vLLM's global CacheConfig
#: default of 16 is not among them: this cache never runs at 16, and a cell
#: keyed there is declined rather than measured at a paging layout that
#: never happened.
_DSV4_SWA_BLOCK_SIZES = (32, 64, 128, 256)


def _dsv4_swa_record_bytes(d_head: int, d_rope: int):
    """`(data bytes, scale bytes)` of one token's record in that cache."""
    nope = d_head - d_rope
    groups = nope // _DSV4_SWA_QUANT_BLOCK
    # The scale tail carries one pad byte after the real exponents
    # (`kScaleBytesPerToken`, .cu:101).
    return nope + d_rope * 2, groups + 1


def _dsv4_swa_rows(cache, slots, page: int, d_head: int, d_rope: int):
    """The `[tokens, d_head]` rows a launch left in the paged cache, in fp32.

    Decodes the stored record: FP8 e4m3 channels scaled by the exponent byte
    the kernel wrote for their group, then the bf16 rotated tail. The grid is
    not re-derived here — the exponents are read back, so a kernel that chose
    them differently disagrees with the oracle, which writes the record by
    the registered format (`refs.rows_on_record`) where the cell names it.
    """
    import torch
    data_bytes, scale_bytes = _dsv4_swa_record_bytes(d_head, d_rope)
    nope, groups = d_head - d_rope, scale_bytes - 1
    block, within = slots // page, slots % page
    rows = cache[block]                                  # [tokens, block bytes]
    index = torch.arange(data_bytes, device=cache.device)
    record = torch.gather(
        rows, 1, (within[:, None] * data_bytes + index[None, :]))
    quantised = (record[:, :nope].contiguous()
                 .view(torch.float8_e4m3fn).float()
                 .view(-1, groups, _DSV4_SWA_QUANT_BLOCK))
    tail = record[:, nope:].contiguous().view(torch.bfloat16).float()
    index = torch.arange(groups, device=cache.device)
    exponent = torch.gather(
        rows, 1, (page * data_bytes + within[:, None] * scale_bytes
                  + index[None, :])).float() - 127.0
    return torch.cat(
        [(quantised * torch.exp2(exponent)[:, :, None]).flatten(1), tail],
        dim=-1)


def _vllm_qnorm_rope_kv_insert_build(inv: Invocation, env: Env,
                                     inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's fused insert, called as the two DeepSeek-V4 layers call it.

    The canonical paged cache (one `d_head` record per slot) becomes the
    packed uint8 block the op takes: `[blocks, page x (data + scale) bytes]`,
    viewed 2-D exactly as the layer views it (`swa_kv_cache.view(shape[0], -1)`,
    attention.py:798). `q_head_padded` is the cell's own head count: the
    launch then writes the heads this span bills and no pad slots, which is
    what vLLM runs wherever its own padding is the identity (the head counts
    the providers accept).
    """
    _require_cuda()
    import torch
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt).contiguous()
    kv = inputs["kv"].to(dt).contiguous()
    _tokens, n_heads, d_head = q.shape
    d_rope = int(inputs["d_rope"])
    page = int(inputs["kv_block_size"])
    data_bytes, scale_bytes = _dsv4_swa_record_bytes(d_head, d_rope)
    blocks = int(inputs["kv_cache"].shape[0])
    cache = torch.zeros(blocks, page * (data_bytes + scale_bytes),
                        dtype=torch.uint8, device=q.device)
    slots = inputs["slot_mapping"].to(torch.int64)
    positions = inputs["positions"].to(torch.int64)
    # fp32, as the op's own check requires (.cu:1170-1173).
    table = inputs["cos_sin_cache"].float()
    norm = bool(inputs["q_norm"])

    def run():
        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, cache, slots, positions, table, n_heads,
            float(inputs["eps"]), page, norm, False, True, False)

    def to_canonical(out):
        return torch.cat(
            [out.float().flatten(1),
             _dsv4_swa_rows(cache, slots, page, d_head, d_rope)], dim=-1)

    return BenchCase(run=run, to_canonical=to_canonical)


def _torch_qnorm_rope_kv_insert_build(inv: Invocation, env: Env,
                                      inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and AT THE CELL'S PRECISION — a floor
    beneath the fused kernel on any card.

    Not the oracle's routine: that one computes in fp32 to be a trusted
    reference, and timing it reports another precision's latency (the driver
    refuses it — "the launch's largest operand is float32 where the cell
    declares bfloat16"). Same chain, run at the width the cell declares: the
    per-head RMSNorm without a gain where this span norms, then the GPT-J
    rotation of the last `d_rope` channels of every query head and of the
    latent row. The reduction behind the norm is accumulated in fp32, which
    is what an unfused `rms_norm` does too. A cell that names the record the
    cache keeps (`kv_format`) gets the latent row written as that record
    (`refs.rows_on_record`), so the floor pays the launch's quantisation;
    any other gets the row at the cell's dtype.
    """
    import torch
    from ..refs import _rope_rotate, rows_on_record
    dt = _torch_dtype(inv.dtype)
    q, kv = inputs["q"].to(dt), inputs["kv"].to(dt)
    rd = int(inputs["d_rope"])
    pos, table, eps = inputs["positions"], inputs["cos_sin_cache"], inputs["eps"]
    norm = bool(inputs["q_norm"])
    record = inv.kv_format

    def rotate_into(x, out):
        """The rotation written into a full-width output buffer.

        The fused launch ALLOCATES AND RETURNS the whole query (it has to:
        it zero-fills vLLM's padding head slots), and the span bills that
        row in and out. A chain that only rebuilt the rotated 64 channels
        would never touch the operand the cell is keyed on — the driver says
        so, "the launch's largest operand holds 4194304 elements where the
        cell's holds 67108864" — and would be a cheaper launch than the one
        the plan prices. So the untouched head is copied and the rotated
        tail written beside it, which is what an unfused rope does.
        """
        nope = x.shape[-1] - rd
        out[..., :nope] = x[..., :nope]
        out[..., nope:] = _rope_rotate(x[..., nope:], pos, table, rd,
                                       is_neox=False).to(dt)
        return out

    q_out, kv_out = torch.empty_like(q), torch.empty_like(kv[:, None, :])
    kv_row = (torch.empty(kv[:, None].shape, dtype=torch.float32,
                          device=kv.device)
              if record is not None else None)

    def run():
        x = q
        if norm:
            # fp32 ACCUMULATION without an fp32 copy of the whole tensor:
            # `x.float()` would materialise one, and the driver then sees
            # a launch whose largest operand is another precision's.
            x = x * torch.rsqrt(
                x.square().mean(-1, keepdim=True, dtype=torch.float32)
                + eps).to(dt)
        q_rot = rotate_into(x, q_out)
        if record is None:
            return q_rot, rotate_into(kv[:, None, :], kv_out)
        # The record's tail is rounded once, from the fp32 rotation, as the
        # kernel stores it; its quantised channels are the unrotated row.
        kv_row[..., :-rd] = kv[:, None, :-rd]
        kv_row[..., -rd:] = _rope_rotate(kv[:, None, -rd:].float(), pos,
                                         table, rd, is_neox=False)
        return q_rot, rows_on_record(kv_row, record, dt, tail=rd)

    return BenchCase(
        run=run,
        to_canonical=lambda o: torch.cat(
            [o[0].float().flatten(1), o[1].float().flatten(1)], dim=-1))


for _sig in ("fused_qnorm_rope_kv_insert",
             "fused_qnorm_rope_kv_insert_no_q_norm"):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_c_{_sig}",
        # The module that has to import for THIS kernel to run is the
        # compiled op library, not the package: importing `vllm` alone
        # leaves `torch.ops._C` empty, and vLLM's own CUDA platform gets the
        # ops by importing this module (`platforms/cuda.py:23`, `:240`). A
        # release that ships no such module ships no such op, which is the
        # availability fact this gate is for.
        import_name="vllm._C_stable_libtorch", fused_op=_sig,
        # sm_80+: the op refuses the launch below Ampere (.cu:676-683).
        min_sm=80,
        serves=KernelSignature(
            signature=_sig,
            # bf16, which is what `VLLM_STABLE_DISPATCH_HALF_TYPES` takes
            # here and what both models serve at.
            activation=("bf16",),
            # The kernel's geometry is compiled in: HEAD_DIM 512, ROPE_DIM 64,
            # NOPE_DIM 448 (.cu:19-27, checked at :1159-1161).
            enum=(("d_head", (512,)), ("d_rope", (64,)),
                  # The head counts at which vLLM's own q-buffer padding is
                  # the identity (`get_padded_num_q_heads` is 64 or 128,
                  # deepseek_v4/nvidia/flashmla.py:78-85), so the launch
                  # measured is the launch the span bills. A narrower TP
                  # shard is padded up and writes more heads than the span
                  # bills; that is a question for what the plan bills, not
                  # one to answer by measuring a different launch.
                  ("n_heads", (64, 128))),
            kv_block_sizes=_DSV4_SWA_BLOCK_SIZES,
            # vLLM has THREE overloads of this op and dispatches on the
            # cache tensor's dtype (attention.py:791 / :1056): the packed
            # `uint8` UE8M0 path, a plain bf16-row path, and a per-tensor-
            # fp8 one. This build always allocates the packed layout — V4's
            # FlashMLA record, FP8 e4m3 with one UE8M0 per 64 and the RoPE
            # tail bf16 (.cu:29-34, :355-420) — so a cell keyed at another
            # cache is the other overload's, declined through the
            # declaration where smoke and the published capability record
            # read it: the bf16-row overload by the `kv` dtype, the
            # per-tensor-FP8 one (also `kv=fp8_e4m3`) by the record's name.
            dtype_slots=(("kv", ("fp8_e4m3",)),),
            kv_format=("fp8_ue8m0_group64_bf16_rope_tail",)),
        build=_vllm_qnorm_rope_kv_insert_build,
    ))
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch", fused_op=_sig,
        # The eager chain writes no paged cache: it reads the contract's
        # record straight, so no page is part of what it measured. It
        # writes the latent as the record the cell names
        # (`rows_on_record`), or the row where it names none.
        serves=KernelSignature(signature=_sig, kv_block_sizes=(),
                               activation=TORCH_FLOAT,
                               dtype_slots=(("kv", _FLOOR_OUTPUT),),
                               kv_format=_FLOOR_FORMATS),
        build=_torch_qnorm_rope_kv_insert_build,
    ))


# --- fused_dsa_qkv_norm_rope{,_indexer}: the DSA attention prologue ---------
#
# vLLM v0.32.0 `fused_norm_rope` (models/deepseek_v32/common/kernels.py:509,
# `_fused_norm_rope_kernel` :118 over (tokens, 4) programs), as
# `DeepseekV32Attention.forward` calls it (deepseek_v32/attention.py:378):
# pid 2 norms the query, pid 1 the latent and rotates k_pe, pid 0 the
# indexer key, pid 3 resets the top-k index buffer.
#
# The timed launch is serving's, cache-writing branch: `slot_mapping` set
# (attention.py:394, :401), so pid 1 also writes the normed latent and the
# rotated key into the paged MLA cache (kernels.py:357-405) and, on an
# indexer layer, pid 0 writes the FP8 index-K record into the paged index
# cache (:488-506). The caches are the ones serving passes:
#   * the MLA cache, a bf16 `[blocks, page, d_kv_lora + d_rope]` pool at
#     the `auto` cache dtype (an `fp8_ds_mla` deployment takes the
#     quantising branch at :368-380 instead — a different launch the cell's
#     dtype envelope, which carries no kv slot, cannot state);
#   * the index-K cache, the packed `[blocks, page, d_indexer_head + 4]`
#     uint8 record buffer `DeepseekV32IndexerCache` is allocated as
#     (model_executor/models/deepseek_v2.py:731-736: 128 FP8 values plus
#     one fp32 scale a token; `uses_shuffled_layout` False, :671-673).
# The three normed/rotated inputs are the strided splits of the one
# `qkv_lora` projection row serving passes (attention.py:305-307 — the
# wrapper carries each row's stride, kernels.py:658-672), and the indexer
# key the leading slice of the fused wk/weights row (attention.py:310-312,
# the projection sized `head_dim + n_head`, deepseek_v2.py:715-722 — the
# head count the bound model's, `_VLLM_INDEXER_HEADS`).
#
# One stated boundary: serving's non-PCP launch passes `index_k_out=None`
# (attention.py:350) and keeps the indexer key only in the FP8 record,
# while the oracle computes the bf16 row — so the build also passes
# `index_k_out` (serving's PCP launch passes exactly it, :350) and reads
# the canonical row from it. The launch measured is the production
# cache-writing launch plus the one bf16 index-key row store the PCP form
# also performs; the FP8 record is written, as serving writes it, and not
# read back.

#: vLLM's MLA page for the DSA models (the FlashMLA-family backends'
#: block size; `cache_config.block_size`, deepseek_v2.py:660).
_DSA_CACHE_PAGE = 64

#: The indexer head count beside the key in the fused wk/weights row
#: (`head_dim + n_head`, deepseek_v2.py:715-722), keyed by the cell's
#: `d_q_lora` — the dims the two bound models' cells differ in:
#: DeepSeek-V3.2 (d_q_lora 1536) runs 64 (`config.index_n_heads  # 64`,
#: deepseek_v2.py:700-701), GLM-5.x (d_q_lora 2048) 32
#: (snapshots/zai-org__GLM-5.2.json `index_n_heads`). The row stride is a
#: runtime argument (`index_k.stride(0)`, kernels.py:675), not a compiled
#: constant; a cell outside both geometries is built at DeepSeek-V3.2's.
_VLLM_INDEXER_HEADS = {1536: 64, 2048: 32}


def _vllm_dsa_norm_rope_build(inv: Invocation, env: Env,
                              inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope
    dt = _torch_dtype(inv.dtype)
    q0, kv0, k_pe0 = (inputs[n].to(dt) for n in ("q", "kv", "k_pe"))
    tokens, dq = q0.shape
    dkv, dr = int(kv0.shape[1]), int(k_pe0.shape[1])
    # One `qkv_lora` row, split as attention.py:305-307 splits it.
    qkv = torch.empty(tokens, dq + dkv + dr, dtype=dt, device=q0.device)
    qkv[:, :dq] = q0
    qkv[:, dq:dq + dkv] = kv0
    qkv[:, dq + dkv:] = k_pe0
    q, kv, k_pe = qkv.split([dq, dkv, dr], dim=-1)
    indexer = "ik" in inputs
    positions = inputs["positions"].to(torch.int64)
    table = inputs["cos_sin_cache"]
    eps = float(inputs["eps"])
    kv_out, k_pe_out = torch.empty_like(kv0), torch.empty_like(k_pe0)
    # Serving's slot mapping: each token its own cache slot.
    slots = torch.arange(tokens, dtype=torch.int64, device=q0.device)
    page = _DSA_CACHE_PAGE
    blocks = -(-tokens // page)
    # The paged MLA cache (bf16 at `auto`): kv_c and k_pe side by side.
    mla_cache = torch.zeros(blocks, page, dkv + dr, dtype=dt,
                            device=q0.device)
    if indexer:
        ik0 = inputs["ik"].to(dt)
        dik = int(ik0.shape[1])
        # The fused wk/weights row, the key its leading slice
        # (attention.py:310-312).
        heads = _VLLM_INDEXER_HEADS.get(dq, 64)
        kw = torch.empty(tokens, dik + heads, dtype=dt, device=q0.device)
        kw[:, :dik] = ik0
        ik = kw[:, :dik]
        ik_out = torch.empty_like(ik0)
        # The packed index-K record cache (FP8 values + fp32 scale).
        index_cache = torch.zeros(blocks, page, dik + 4, dtype=torch.uint8,
                                  device=q0.device)
        # The buffer the indexer layer resets — the cell's own top-k width.
        topk = torch.empty(tokens, int(inputs["n_topk_keys"]),
                           dtype=torch.int32, device=q0.device)
        extra = dict(index_k=ik, index_k_layer_norm_w=inputs["ik_weight"],
                     index_k_layer_norm_bias=inputs["ik_bias"],
                     index_k_rope_cos_sin_cache=table,
                     index_rope_interleave=bool(
                         inputs["index_rope_interleave"]),
                     indexer_slot_mapping=slots,
                     indexer_k_cache=index_cache,
                     index_k_out=ik_out)
    else:
        ik_out = None
        # A shared-indexer layer reads only the buffer's width (pid 3
        # returns at once, :203-206) and writes no index cache.
        topk = torch.empty(tokens, 1, dtype=torch.int32, device=q0.device)
        extra = dict(index_k=None, index_k_layer_norm_w=None,
                     index_k_layer_norm_bias=None,
                     index_k_rope_cos_sin_cache=None)

    def run():
        return fused_norm_rope(
            positions, q, inputs["q_weight"], eps, kv, inputs["kv_weight"],
            eps, k_pe, table, extra["index_k"], extra["index_k_layer_norm_w"],
            extra["index_k_layer_norm_bias"], eps,
            extra["index_k_rope_cos_sin_cache"], topk,
            slot_mapping=slots,
            indexer_slot_mapping=extra.get("indexer_slot_mapping"),
            indexer_k_cache=extra.get("indexer_k_cache"),
            indexer_cache_shuffled=False,
            mla_kv_cache=mla_cache,
            mla_kv_cache_dtype="auto",
            mla_k_scale=None,
            has_indexer=indexer,
            index_rope_interleave=extra.get("index_rope_interleave", False),
            kv_c_out=kv_out, k_pe_out=k_pe_out,
            index_k_out=extra.get("index_k_out"))

    def to_canonical(q_out):
        parts = [q_out, kv_out, k_pe_out] + ([ik_out] if indexer else [])
        return torch.cat([p.float() for p in parts], dim=-1)

    return BenchCase(run=run, to_canonical=to_canonical)


def _dsa_norm_rope_chain(i, torch):
    """The plan's nodes as separate eager ops at the cell's dtype."""
    import torch.nn.functional as F
    from ..refs import _rope_rotate
    from .norm_act import _rms_at
    eps, pos = i["eps"], i["positions"]
    table = i["cos_sin_cache"]
    rd = int(table.shape[-1])
    out = (_rms_at(i["q"], i["q_weight"], eps),
           _rms_at(i["kv"], i["kv_weight"], eps),
           _rope_rotate(i["k_pe"][:, None, :], pos, table, rd,
                        is_neox=False)[:, 0].to(i["k_pe"].dtype))
    if "ik" in i:
        ik = F.layer_norm(i["ik"], (i["ik"].shape[-1],), i["ik_weight"],
                          i["ik_bias"], eps)
        out += (_rope_rotate(ik[:, None, :], pos, table, rd,
                             is_neox=not i["index_rope_interleave"])
                [:, 0].to(ik.dtype),)
    return out


#: The extents a `tl.arange` over a whole row can take.
_POWERS_OF_TWO = tuple(2 ** k for k in range(4, 14))


def _register_dsa_norm_rope() -> None:
    from .norm_act import _cell_floor
    for sig in ("fused_dsa_qkv_norm_rope", "fused_dsa_qkv_norm_rope_indexer"):
        register(Provider(
            library="vllm", signature=sig, kernel_impl=f"vllm_{sig}",
            import_name="vllm.models.deepseek_v32.common.kernels",
            serves=KernelSignature(
                signature=sig,
                # `tl.arange(0, KV_DIM)` (:237) and `tl.arange(0,
                # KPE_HALF_ROT_DIM)` (:52): the latent width and the rotation
                # width are powers of two; the query and the indexer key are
                # masked to the next one.
                enum=(("d_kv_lora", _POWERS_OF_TWO), ("d_rope", _POWERS_OF_TWO)),
                # bf16, what both bound models serve and all it was measured
                # at (897 rows).
                activation=("bf16",),
                # The build writes the MLA cache at `auto` — bf16 rows, no
                # scale (`mla_kv_cache_dtype="auto"`, `mla_k_scale=None`);
                # the `fp8_ds_mla` record and a per-tensor FP8 cache take
                # the quantising branch (kernels.py:368-380), another
                # launch this build never makes.
                kv_format=(PLAIN,)),
            build=_vllm_dsa_norm_rope_build, fused_op=sig,
        ))
        register(Provider(
            library="torch", signature=sig, kernel_impl=f"torch_{sig}",
            import_name="torch",
            serves=KernelSignature(signature=sig, activation=TORCH_FLOAT),
            build=_cell_floor(_dsa_norm_rope_chain), fused_op=sig,
        ))


_register_dsa_norm_rope()
