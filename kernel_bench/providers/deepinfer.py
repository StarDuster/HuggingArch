"""DeepInfer providers — DeepSeek's DeepSeek-V4.1-Flash runtime.

DeepInfer is a source-built stack, not a single wheel: its providers gate on
the installed `deepinfer` package and then call the vendored TileKernels /
DeepGEMM entrypoint the model runtime calls, at the release the binding
audits (`framework_bindings/deepinfer.yaml`). Builds are lazy, so catalog
discovery imports none of the accelerator libraries.

The torch floors of the DeepInfer-only spans live with their families
(`compress_store.py`, `moe_gate.py`, `mhc.py`).
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda, _torch_dtype
from .norm_act import _engram_canonical

#: DeepInfer resolves V4.1's cache at 128 tokens a block
#: (`ModelMetadata.resolved_tokens_per_block`, model_metadata_utils.py:134-135).
_DEEPINFER_PAGES = (128,)


def _deepinfer_page_only(inv: Invocation) -> str:
    """A cell keyed at another page is declined rather than measured at a
    paging layout DeepInfer never allocates."""
    page = inv.dims.get("kv_block_size")
    if page is not None and int(page) not in _DEEPINFER_PAGES:
        raise NotImplementedError(
            f"DeepInfer resolves V4.1 at page {_DEEPINFER_PAGES[0]}; this "
            f"cell is keyed at {page}")
    return ""


# ---------------------------------------------------------------------------
# mhc_mega_post_pre — `deep_gemm.mega_mhc`, as HCLayerMixin._mega_mhc calls
# it (v4_modules/hc.py:229-298).
# ---------------------------------------------------------------------------

def _mega_mhc_build(mxfp8: bool):
    def build(inv: Invocation, env: Env,
              inputs: Dict[str, Any]) -> BenchCase:
        """The one-launch shifted mHC seam. It writes the normed input (BF16
        always; with the MXFP8 reader also FP8 with per-32 UE8M0 scales,
        allocated as hc.py:262-266 does), the new residual and the three
        new coefficient sets. The graph leaves the coefficient split a
        billed node, so the canonical form derives only that node's `mixes`
        operand from the new residual, beside the two primary writes. The
        MXFP8 arm's canonical reads the BF16 row the same launch writes."""
        _require_cuda()
        import torch
        import deepinfer.model_executor.models.v4_modules.hc  # noqa: F401  — stack gate
        import deep_gemm

        t, hc, d = inputs["residual"].shape
        device = inputs["residual"].device
        new_residual = torch.empty(t, hc, d, dtype=torch.bfloat16,
                                   device=device)
        new_prev_mix = torch.empty(t, hc, 1, dtype=torch.float32,
                                   device=device)
        new_post_mix = torch.empty(t, hc, 1, dtype=torch.float32,
                                   device=device)
        new_comb_mix = torch.empty(t, hc, hc, dtype=torch.float32,
                                   device=device)
        y_bf16 = torch.empty(t, d, dtype=torch.bfloat16, device=device)
        y_fp8 = y_gemm_sf = None
        if mxfp8:
            y_fp8 = torch.empty(t, d, dtype=torch.float8_e4m3fn,
                                device=device)
            m_pad = (t + 3) // 4 * 4
            y_gemm_sf = torch.empty(d // 128, m_pad, dtype=torch.int32,
                                    device=device).mT[:t]
        fn = inputs["fn"].float().contiguous()
        launch = Launch(
            deep_gemm.mega_mhc,
            inputs["sublayer"].to(torch.bfloat16).contiguous(),
            inputs["residual"].to(torch.bfloat16).contiguous(),
            inputs["pre"].float().unsqueeze(-1).contiguous(),
            inputs["post"].float().unsqueeze(-1).contiguous(),
            inputs["comb"].float().contiguous(),
            fn=fn, mix_scales=inputs["scale"].float().contiguous(),
            mix_bases=inputs["base"].float().contiguous(), hc_mult=int(hc),
            hc_norm_eps=float(inputs["rms_eps"]),
            hc_pre_eps=float(inputs["hc_pre_eps"]),
            hc_post_scale=float(inputs["hc_post_mult_value"]),
            sinkhorn_eps=float(inputs["hc_sinkhorn_eps"]),
            num_sinkhorn_iters=int(inputs["sinkhorn_repeat"]),
            rmsnorm_weight=inputs["norm_weight"].to(torch.bfloat16)
            .contiguous(),
            rmsnorm_eps=float(inputs["norm_eps"]), rmsnorm_scale=1.0,
            new_residual=new_residual, new_prev_mix=new_prev_mix,
            new_post_mix=new_post_mix, new_comb_res_mix=new_comb_mix,
            y_bf16=y_bf16, y_fp8=y_fp8, y_gemm_sf=y_gemm_sf,
            y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0,
        )

        def canonical(_out):
            from ..refs import hc_normalised_mixes
            flat = new_residual.reshape(t, hc * d)
            mixes = hc_normalised_mixes(flat, fn, float(inputs["rms_eps"]))
            return torch.cat([y_bf16.float(), mixes, flat.float()], dim=-1)

        return BenchCase(run=launch, to_canonical=canonical)
    return build


# The eligibility `_mega_mhc` is taken under (hc.py:99-101): four copies and
# a hidden size that splits into 4 x 1024-wide K blocks.
_MEGA_MHC_SHAPE = dict(enum=(("n_hidden_copies", (4,)),),
                       divisible=(("d_model", 1024),))

register(Provider(
    library="deepinfer", signature="mhc_mega_post_pre",
    kernel_impl="deepinfer_mega_mhc", import_name="deepinfer",
    serves=KernelSignature(signature="mhc_mega_post_pre",
                           activation=("bf16",), output=("bf16",),
                           **_MEGA_MHC_SHAPE),
    build=_mega_mhc_build(False), fused_op="mhc_mega_post_pre", min_sm=100,
))
register(Provider(
    library="deepinfer", signature="mhc_mega_post_pre",
    kernel_impl="deepinfer_mega_mhc_mxfp8", import_name="deepinfer",
    # The cast absorbed (the plan's optional `quant`): the normed input as
    # FP8 E4M3 with one UE8M0 scale per 32 (`get_fp8_casting_granularity()
    # == 32`, hc.py:262), and the BF16 row the launch always writes.
    serves=KernelSignature(signature="mhc_mega_post_pre",
                           activation=("bf16",), output=("fp8_e4m3",),
                           activation_granularity=("per-group:32",),
                           secondary_outputs=(("y_bf16", "bf16"),),
                           **_MEGA_MHC_SHAPE),
    build=_mega_mhc_build(True), fused_op="mhc_mega_post_pre", min_sm=100,
))


# ---------------------------------------------------------------------------
# fused_engram_gate — TileKernels' Engram gate (v4_modules/engram.py:151).
# ---------------------------------------------------------------------------

def _engram_build(inv: Invocation, env: Env,
                  inputs: Dict[str, Any]) -> BenchCase:
    """TileKernels' Engram gate through DeepInfer's production call.

    DeepInfer fuses the q/k weights once at load (`fused_weight`,
    engram.py:134); that setup is outside the timed call. The launch is
    in-place on x, so the build clones x once for a stable timed
    invocation.
    """
    _require_cuda()
    import tile_kernels.engram
    import deepinfer.model_executor.models.v4_modules.engram  # noqa: F401

    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).flatten(0, 1).unflatten(
        -1, inputs["q_weight"].shape).contiguous().clone()
    kv = inputs["kv"].to(dt).flatten(0, 1).contiguous()
    fused_weight = tile_kernels.engram.fused_weight(
        inputs["q_weight"].to(dt).contiguous(),
        inputs["k_weight"].to(dt).contiguous())
    launch = Launch(
        tile_kernels.engram.engram_gate_fwd, x, kv, fused_weight,
        float(inputs["eps"]), 1e-6, save_for_backward=False,
        image_token_mask=None, inplace=True)
    return BenchCase(
        run=launch,
        to_canonical=lambda out: _engram_canonical(out[0], inputs),
    )


register(Provider(
    library="deepinfer", signature="fused_engram_gate",
    kernel_impl="deepinfer_fused_engram_gate", import_name="deepinfer",
    serves=KernelSignature(signature="fused_engram_gate",
                           activation=("bf16",), weight=("bf16",)),
    build=_engram_build, fused_op="fused_engram_gate", min_sm=100,
))


# ---------------------------------------------------------------------------
# fused_norm_rope / fused_compress_norm_rope_store — DeepInfer's one
# transform wrapper, `fused_norm_rope_hadamard_cast_store_kvcache`
# (layers/tilelang_ops/fused_transform.py:189): RMSNorm, partial GPT-J RoPE,
# the cache record's cast and the paged store.
# ---------------------------------------------------------------------------

def _transform_store_build(row: str, gain: str, layout_name: str):
    def build(inv: Invocation, env: Env,
              inputs: Dict[str, Any]) -> BenchCase:
        """The row normed, rotated, cast to the cell's record and written to
        a fresh request's consecutive slots, as the store paths call it
        (attn.py:433, index_attn.py:146); read back by decoding the
        record."""
        _require_cuda()
        import torch
        from deepinfer.model_executor.layers.tilelang_ops import cache_common
        from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
            CastArgs, KVCacheOutArgs, NormArgs, RopeArgs,
            fused_norm_rope_hadamard_cast_store_kvcache,
        )
        # DeepInfer's own layout object for the record this provider
        # declares it writes (`serves.output_format`, one per provider):
        # named beside that declaration, so the two cannot part.
        layout = getattr(cache_common, layout_name)
        page = _DEEPINFER_PAGES[0]
        kv = inputs[row].to(torch.bfloat16).contiguous()
        rows, d = kv.shape
        cache = cache_common.alloc_kv_cache(layout, -(-rows // page), page,
                                            device=kv.device)
        slots = torch.arange(rows, dtype=torch.int32, device=kv.device)
        rd = int(inputs["d_rope"])
        launch = Launch(
            fused_norm_rope_hadamard_cast_store_kvcache, kv,
            norm_args=NormArgs(inputs[gain].to(torch.bfloat16).contiguous(),
                               float(inputs["eps"])),
            rope_args=RopeArgs(
                inputs["positions"].to(torch.int32).contiguous(),
                inputs["cos_sin_cache"].float().contiguous(), rd, False,
                offset=d - rd),
            cast_args=CastArgs.from_cache_format(layout),
            out_args=KVCacheOutArgs(cache, slots, layout=layout),
        )

        def canonical(_out):
            _page, data = cache_common.split_kv_cache(cache, layout)
            out = cache_common.ref_cast_to_bf16(data, layout).reshape(-1, d)
            return out[slots.long()].float()

        return BenchCase(run=launch, to_canonical=canonical)
    return build


# The SWA latent row. Decode only: in prefill DeepInfer casts the row back
# to BF16 and stores it in a second launch (attn.py:411-418, :855), which the
# binding leaves unfused.
register(Provider(
    library="deepinfer", signature="fused_norm_rope",
    kernel_impl="deepinfer_swa_norm_rope_store", import_name="deepinfer",
    serves=KernelSignature(
        signature="fused_norm_rope", activation=("bf16",), weight=("bf16",),
        output=("fp8_e4m3",), output_format=("fp8_ue8m0_group32",),
        phases=("decode",), divisible=(("d_head", 32),)),
    # DeepInfer writes this record through `V41_FORMAT` (cache_common.py:109:
    # E4M3, one UE8M0 per 32).
    build=_transform_store_build("x", "w", "V41_FORMAT"),
    fused_op="fused_norm_rope",
    min_sm=100,
))

# The index key (MXFP4, 128 wide), as MQA.store_index_kv writes it
# (index_attn.py:146). The compressed main row's launch (NVFP4, 512 wide,
# compress.py:148) has no plan: V4.1's graph rotates that row after its
# store marker (`csa2_compress_store`), which `fused_compress_norm_rope_store`
# does not match, so no cell reaches it and it is not registered.
register(Provider(
    library="deepinfer", signature="fused_compress_norm_rope_store",
    kernel_impl="deepinfer_v41_index_fp4_norm_rope_store",
    import_name="deepinfer",
    serves=KernelSignature(
        signature="fused_compress_norm_rope_store", activation=("bf16",),
        output=("fp4_e2m1",), output_format=("mxfp4",),
        enum=(("d_head", (128,)),), kv_block_sizes=_DEEPINFER_PAGES),
    impl_params=_deepinfer_page_only,
    # ... through `FP4_INDEX_KV_FORMAT` (cache_common.py:129: E2M1, one
    # UE8M0 per 32 — MXFP4).
    build=_transform_store_build("kv", "norm_weight", "FP4_INDEX_KV_FORMAT"),
    fused_op="fused_compress_norm_rope_store", min_sm=100,
))


# ---------------------------------------------------------------------------
# fused_compress_pool_state — `compress_and_update_state_{prefill,decode}`
# (layers/tilelang_ops/compress.py:20, :74), from
# Compressor.forward_pre_norm (v4_modules/compress.py:98, :111).
# ---------------------------------------------------------------------------

def _pool_operands(inv: Invocation, inputs: Dict[str, Any]):
    """The fp32 [score, kv] halves the launch asserts (compress.py:95); the
    cell bills them at the activation dtype and the build upcasts untimed,
    as the vLLM compressor arm does."""
    dt = _torch_dtype(inv.dtype)
    kv = inputs["kv"].to(dt).float().contiguous()
    score = inputs["score"].to(dt).float().contiguous()
    return kv, score, int(inputs["compress_ratio"])


def _pool_prefill_build(inv: Invocation, env: Env,
                        inputs: Dict[str, Any]) -> BenchCase:
    """A fresh chunk-aligned prefill: `B` requests of `s_query` tokens each,
    every closed group pooled off the raw rows and the trailing rows saved
    to each request's state slot."""
    _require_cuda()
    import torch
    from deepinfer.model_executor.layers.tilelang_ops.compress import (
        compress_and_update_state_prefill,
    )
    kv, score, cr = _pool_operands(inv, inputs)
    tokens, d = kv.shape
    batch = int(inv.dims["B"])
    per_req = tokens // batch
    pools = max(per_req // cr, 1)
    dev = kv.device
    # The ring holds one window plus a speculative row
    # (compress.py:98-99 asks >= ratio + max_seqlen_q - 1); only the
    # trailing rows of each request are saved into it.
    state = torch.zeros(batch, cr + 1, 2, d, dtype=torch.float32, device=dev)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    lens = torch.full((batch,), per_req, dtype=torch.int32, device=dev)
    cu = torch.cat((zero, lens.cumsum(0, dtype=torch.int32)))
    cu_pool = torch.arange(0, (batch + 1) * pools, pools, dtype=torch.int32,
                           device=dev)
    mapping = torch.arange(batch, dtype=torch.int32,
                           device=dev).repeat_interleave(pools)
    state_idx = torch.arange(batch, dtype=torch.int32, device=dev)
    return BenchCase(
        run=Launch(compress_and_update_state_prefill, score, kv, state,
                   state_idx, mapping, cu, cu.clone(), cu_pool, cr, None),
        to_canonical=lambda out: out.float())


def _pool_decode_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """A decode step where every request's token closes its group: token
    `t`'s state slot already holds rows `t - ratio + 1 .. t - 1` (the cell's
    preceding rows, score at index 0 and kv at 1, compress.py:229-232), so
    each request's pooled row at `t = ratio - 1 (mod ratio)` is the oracle's
    group of consecutive rows."""
    _require_cuda()
    import torch
    from deepinfer.model_executor.layers.tilelang_ops.compress import (
        compress_and_update_state_decode,
    )
    kv, score, cr = _pool_operands(inv, inputs)
    tokens, d = kv.shape
    dev = kv.device
    state = torch.zeros(tokens, cr + 1, 2, d, dtype=torch.float32,
                        device=dev)
    for back in range(1, cr):
        state[:, cr - 1 - back, 0] = score.roll(back, 0)
        state[:, cr - 1 - back, 1] = kv.roll(back, 0)
    state_idx = torch.arange(tokens, dtype=torch.int32, device=dev)
    context = torch.full((tokens,), cr, dtype=torch.int32, device=dev)
    return BenchCase(
        run=Launch(compress_and_update_state_decode, score, kv, state,
                   state_idx, context, 1, cr),
        to_canonical=lambda out: out[cr - 1::cr].float())


for _phase, _build in (("prefill", _pool_prefill_build),
                       ("decode", _pool_decode_build)):
    register(Provider(
        library="deepinfer", signature="fused_compress_pool_state",
        kernel_impl=f"deepinfer_compress_pool_{_phase}",
        import_name="deepinfer",
        # The launch runs only where the compressor pools (ratio 2;
        # compress.py:89 returns the raw row at ratio 1), and writes BF16
        # pooled rows (compress.py:51).
        serves=KernelSignature(takes_state=True, signature="fused_compress_pool_state",
                               activation=("bf16",), phases=(_phase,),
                               enum=(("compress_ratio", (2,)),)),
        build=_build, fused_op="fused_compress_pool_state", min_sm=100,
    ))


# ---------------------------------------------------------------------------
# moe_mega_gate_sqrtsoftplus — DeepGEMM `bf16_mega_gate`
# (DeepseekMoE.gate_forward, v4_modules/mlp.py:398).
# ---------------------------------------------------------------------------

def _mega_gate_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    """MegaGate with no EPLB duplication: the logical-to-physical map is the
    identity, so the launch's expert ids compare directly with the dense
    router oracle."""
    _require_cuda()
    import torch
    import deep_gemm
    import deepinfer.model_executor.models.v4_modules.mlp  # noqa: F401
    from ..refs import _moe_gate_dense

    weight = inputs["weight"].to(torch.bfloat16).contiguous()
    launch = Launch(
        deep_gemm.bf16_mega_gate,
        inputs["x"].to(torch.bfloat16).contiguous(), weight,
        int(inputs["topk"]), False, 0,
        float(inputs["routed_scaling_factor"]), 0,
        scoring_func="sqrtsoftplus", bias=inputs["bias"].float().contiguous())
    return BenchCase(
        run=launch,
        to_canonical=lambda out: _moe_gate_dense(out[1], out[0],
                                                 int(weight.shape[0])))


register(Provider(
    library="deepinfer", signature="moe_mega_gate_sqrtsoftplus",
    kernel_impl="deepinfer_bf16_mega_gate", import_name="deepinfer",
    # The launch's host asserts (DeepGEMM csrc/apis/mega_gate.hpp:80-109):
    # BF16 x and weight, a hidden size in 256-wide steps, at most 512
    # experts in lanes of 4, at most 32 picked.
    serves=KernelSignature(
        signature="moe_mega_gate_sqrtsoftplus", activation=("bf16",),
        weight=("bf16",),
        divisible=(("d_model", 256), ("n_experts", 4)),
        max_dims=(("n_experts", 512), ("n_experts_active", 32))),
    build=_mega_gate_build, fused_op="moe_mega_gate_sqrtsoftplus",
    min_sm=100,
))


# ---------------------------------------------------------------------------
# Registered, disabled: launches whose operands only a DeepInfer worker
# provides.
# ---------------------------------------------------------------------------

def _worker_only(inv: Invocation, env: Env,
                 inputs: Dict[str, Any]) -> BenchCase:
    raise NotImplementedError("see disabled_reason")


register(Provider(
    library="deepinfer", signature="fused_moe",
    kernel_impl="deepinfer_mega_moe", import_name="deepinfer",
    # What V4.1's MegaMoE runs (FP8 activations, FP4 experts, BF16 out),
    # as DeepGEMM's own MegaMoE arm states it.
    serves=KernelSignature(signature="fused_moe",
                           weight=("mxfp4", "fp4_e2m1"),
                           activation=("fp8_e4m3",), output=("bf16",)),
    build=_worker_only, fused_op="fused_moe", min_sm=100,
    disabled_reason=(
        "DeepInfer's MegaMoE launches from a DeepGEMM SymmBuffer created by "
        "the engine's expert-parallel process group; a one-process harness "
        "buffer is not the serving launch. Provision a DeepInfer worker env "
        "and measure the boundary through its runtime before enabling rows"),
))

register(Provider(
    library="deepinfer", signature="fused_mla_norm_rope_attn_rope_cast",
    kernel_impl="deepinfer_flashmla_mega_attention", import_name="deepinfer",
    serves=KernelSignature(
        signature="fused_mla_norm_rope_attn_rope_cast",
        activation=("bf16",), output=("fp8_e4m3",),
        enum=(("n_heads", (64, 128)), ("d_head", (512,)),
              ("d_rope", (64,))),
        kv_block_sizes=_DEEPINFER_PAGES),
    impl_params=_deepinfer_page_only,
    build=_worker_only,
    fused_op="fused_mla_norm_rope_attn_rope_cast", min_sm=100,
    disabled_reason=(
        "FlashMLA's mega-attention ABI requires q_b and Wv weights permuted "
        "at model load (`permute_q_b_proj` / `permute_wv_proj`). A canonical "
        "q/c/indices cell does not carry those weights, so calling the bare "
        "kernel would measure a different head/group layout; measure through "
        "a provisioned DeepInfer worker before enabling rows"),
))
