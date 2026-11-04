"""Compress/store fused spans — the KV-compressing attention families.

DeepSeek-V4 / V4.1's KV compressors, GLM-5.3-Flash's k-pool indexer
compressor and Qwen QSA's indexer preparation, each one launch in the
frameworks, called the way serving calls it at the audited releases (vLLM
v0.32.0, sglang v0.5.22). A torch floor runs each span's naive chain at the
cell's precision, so there is a floor beneath the fused launch on any card.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register
from ..refs import records_on_grid
from .attention import _require_cuda, _torch_dtype
from .rope import _dsv4_swa_rows


#: vLLM's compressor-state ring holds the open group's rows for one request:
#: `max(8, pow2(num_speculative + 2))` rows a block
#: (deepseek_v41/compressor.py:151-153) — 8 at no speculation, the width the
#: sweep's deployments run.
_DSV41_RING_ROWS = 8

#: The packed fp8_ds_mla record every compressed-cache store below writes on
#: an H100: 448 FP8 NoPE bytes + 64 bf16 RoPE values (576) and 7 UE8M0
#: exponents + a pad byte (8) — V4's record, which V4.1 keeps everywhere but
#: SM100 (`_use_v41_mxfp8_kv_record`, deepseek_v41/attention.py:128-129).
_DSV4_PACKED_RECORD_BYTES = 584

#: Pages vLLM allocates the MLA-family caches in on CUDA: FlashMLA's 64
#: (vllm/platforms/cuda.py forces it; the store kernels take the page as a
#: runtime shape). A cell keyed at another page is declined rather than
#: measured at a paging layout that never happened.
_MLA_CACHE_PAGES = (64,)


#: DeepSeek-V4's FlashMLA record (`cache_formats/
#: fp8_ue8m0_group64_bf16_rope_tail.yaml`): 448 FP8 values, 64 BF16 RoPE
#: channels and the UE8M0 scales (7 and a pad byte) a 512-wide row, the
#: packed 584 B record the 512-wide stores below write. A cell keyed at
#: another record — the plain bf16 row, a per-tensor FP8 cache — is another
#: kernel (`_rope_plain_insert_kernel`; vLLM dispatches on the cache's
#: format, deepseek_v41/compressor.py:307-316), declined through the
#: declaration (`KernelSignature.output_format`).
_FLASHMLA_V4_RECORD = "fp8_ue8m0_group64_bf16_rope_tail"



def _record_floor_output() -> tuple:
    """What a floor writing through `rows_on_record` writes: the row at the
    cell's dtype where the cell names no record, else the element dtype of
    the record it names — read off the same registry `records_on_grid`
    reads, so the two declarations cannot drift."""
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    return TORCH_FLOAT + tuple(sorted(
        {corpus_token(CACHE_FORMATS[name].dtype)
         for name in records_on_grid() if name is not None}))


def _served_page_only(pages):
    """Decline a cell keyed at a page the kernels never run. `accepts()`
    does not read `kv_block_sizes`, so without this gate a planned cell at
    the binding's default page (16) would be measured by a provider whose
    `serves` claims only these pages — a paging layout that never
    happened. Resolved at `impl_id`, before any input is allocated."""
    def check(inv: Invocation) -> str:
        page = inv.dims.get("kv_block_size")
        if page is not None and int(page) not in pages:
            raise NotImplementedError(
                f"this arm serves the cache at page {tuple(pages)}; the "
                f"cell is keyed at {page} — no deployment pages it so")
        return ""
    return check


def _decline_v41_sm100_record(capability) -> None:
    """On SM100, V4.1's fp8_ds_mla cache takes the MXFP8 record (528 B,
    `_rope_quant_insert_mxfp8_kernel`; the choice is
    `_use_v41_mxfp8_kv_record`, deepseek_v41/attention.py:128-129) — a
    different kernel from the 584 B one this arm builds."""
    if tuple(capability) >= (10, 0):
        raise NotImplementedError(
            "on SM100 this store is the MXFP8-record kernel "
            "(_rope_quant_insert_mxfp8_kernel), which this arm does not "
            "build; the 584 B record is the pre-SM100 launch")


def _dsv41_fused_kv_score(kv: Any, score: Any, cr: int, torch: Any) -> Any:
    """The fused wkv|wgate GEMM row the launch reads: fp32, kv the leading
    `d_head` columns (`fused_compress_quant_cache.py:51`, the row width
    asserted at :65). At ratio 1 the row is kv alone — the group program
    reads only it (:203-204) — so the gate has no columns there."""
    tokens, d_head = kv.shape
    if cr == 1:
        return kv.float().contiguous()
    kv_score = torch.empty(tokens, d_head * cr, dtype=torch.float32,
                           device=kv.device)
    kv_score[:, :d_head] = kv
    kv_score[:, d_head:] = score
    return kv_score


# ---------------------------------------------------------------------------
# DeepSeek-V4.1 CSA2 (vLLM v0.32.0): `fused_save_compress_norm` and
# `fused_compress_rope_store`.
# ---------------------------------------------------------------------------

def _vllm_save_compress_norm_build(inv: Invocation, env: Env,
                                   inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `fused_save_compress_norm` (models/deepseek_v41/common/ops/
    fused_compress_quant_cache.py:22), as `DeepseekCompressor.forward` calls
    it (deepseek_v41/compressor.py:283): one fused `[tokens, 512·ratio]`
    fp32 row — the fused wkv|wgate GEMM's output, kv its leading half
    (compressor.py:250-253) — the positions, the fp32 ring, the slot /
    request tables, the bf16 gain, and the bf16 latent it writes.

    The cell bills the kv and gate halves at the activation dtype; the
    launch upcasts them into the fp32 row the kernel requires (:64), which
    is the precision the GEMM emits in production. One ring block and one
    request program PER REQUEST (the grid is `num_reqs + cdiv(tokens,
    ratio)`, :84-94): `B` request programs commit their own ring tails
    here, and every closed group is pooled by its pair program off the raw
    rows — a chunk-aligned prefill launch, where no request program closes
    a cross-chunk group. That one path (the group a chunk's first token
    closes against the predecessor's ring row, :140-179) shares the kernel
    and differs by one ring-row read per request; it is not the launch a
    chunk-aligned cell bills."""
    _require_cuda()
    import torch
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm,
    )
    dt = _torch_dtype(inv.dtype)
    kv, score = inputs["kv"].to(dt), inputs["score"].to(dt)
    tokens, d_head = kv.shape
    cr = int(inputs["compress_ratio"])
    batch = int(inv.dims["B"])
    per_req = tokens // batch
    kv_score = _dsv41_fused_kv_score(kv, score, cr, torch)
    positions = inputs["positions"].to(torch.int64)
    slots = torch.arange(tokens, dtype=torch.int64, device=kv.device)
    if cr > 1:
        ring = torch.zeros(batch, _DSV41_RING_ROWS, d_head * cr,
                           dtype=torch.float32, device=kv.device)
        # One ring block a request, `block · rows + position % rows`
        # (compressor.py:39-48).
        ring_slots = ((slots // per_req) * _DSV41_RING_ROWS
                      + (slots % per_req) % _DSV41_RING_ROWS)
        qsl = torch.arange(0, tokens + 1, per_req, dtype=torch.int32,
                           device=kv.device)
        t2r = (torch.arange(batch, dtype=torch.int32, device=kv.device)
               .repeat_interleave(per_req))
    else:
        # Ratio 1 keeps no ring; the slots only mark valid tokens (:86).
        ring = qsl = t2r = None
        ring_slots = slots
    latent = torch.empty(tokens, d_head, dtype=torch.bfloat16,
                         device=kv.device)
    w = inputs["norm_weight"].to(torch.bfloat16)
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: fused_save_compress_norm(
            kv_score, positions, ring, ring_slots, qsl, t2r, w, eps, cr,
            latent),
        # The kernel writes only the group-closing rows (:60); the groups
        # are cr-aligned per request, so the closing rows read flat.
        to_canonical=lambda _o: latent[cr - 1::cr].float())


def _torch_save_compress_norm_build(inv: Invocation, env: Env,
                                    inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: per closed
    group, softmax over the gate (fp32 accumulation, as the kernel's own
    fp32 rows give it), the weighted sum of kv, and the RMSNorm written at
    the latent's bf16."""
    import torch
    dt = _torch_dtype(inv.dtype)
    kv, score = inputs["kv"].to(dt), inputs["score"].to(dt)
    w = inputs["norm_weight"].to(dt)
    cr, eps = int(inputs["compress_ratio"]), float(inputs["eps"])
    tokens, d_head = kv.shape
    groups = tokens // cr

    def run():
        if cr == 1:
            # A group of one is the row itself: no pool, just the norm
            # (fp32 accumulation in the reduction, no fp32 copy of the
            # operand — the launch gate reads the widest tensor's dtype).
            x = kv
            r = x * torch.rsqrt(
                x.square().mean(-1, keepdim=True, dtype=torch.float32)
                + eps).to(dt)
            return r * w
        kvg = kv[:groups * cr].view(groups, cr, d_head)
        scg = score[:groups * cr].view(groups, cr, d_head)
        # The softmax stays at the row's dtype; the weighted sum reduces in
        # fp32 onto a tensor 1/cr the operand's size — the fp32 the kernel's
        # own rows give it, without a full-width fp32 copy.
        pooled = (kvg * torch.softmax(scg, dim=1)).sum(
            1, dtype=torch.float32)
        r = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True)
                                 + eps)
        return (r * w.float()).to(dt)

    return BenchCase(run=run, to_canonical=lambda out: out.float())


register(Provider(
    library="vllm", signature="fused_save_compress_norm",
    kernel_impl="vllm_fused_save_compress_norm",
    import_name="vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache",
    serves=KernelSignature(takes_state=True,
        signature="fused_save_compress_norm",
        # The kernel's rows are compiled 512 wide (`tl.arange(0, 512)`,
        # :138), the latent bf16 (:67), the ratio 1 or 2 (:63).
        activation=("bf16",),
        enum=(("d_head", (512,)), ("compress_ratio", (1, 2)))),
    build=_vllm_save_compress_norm_build, fused_op="fused_save_compress_norm",
))
register(Provider(
    library="torch", signature="fused_save_compress_norm",
    kernel_impl="torch_fused_save_compress_norm", import_name="torch",
    serves=KernelSignature(signature="fused_save_compress_norm",
                           activation=TORCH_FLOAT),
    build=_torch_save_compress_norm_build, fused_op="fused_save_compress_norm",
))


def _torch_compress_pool_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """DeepInfer's pool (`fused_compress_pool_state`) as its naive chain:
    the floor above without the norm — per closed group the softmax at the
    row's dtype and the weighted sum reduced in fp32 onto the 1/ratio-wide
    pooled rows, written at the row's dtype."""
    import torch
    dt = _torch_dtype(inv.dtype)
    kv, score = inputs["kv"].to(dt), inputs["score"].to(dt)
    cr = int(inputs["compress_ratio"])
    tokens, d_head = kv.shape
    groups = tokens // cr
    kvg = kv[:groups * cr].view(groups, cr, d_head)
    scg = score[:groups * cr].view(groups, cr, d_head)
    return BenchCase(
        run=lambda: (kvg * torch.softmax(scg, dim=1)).sum(
            1, dtype=torch.float32).to(dt),
        to_canonical=lambda out: out.float())


register(Provider(
    library="torch", signature="fused_compress_pool_state",
    kernel_impl="torch_fused_compress_pool_state", import_name="torch",
    serves=KernelSignature(signature="fused_compress_pool_state",
                           activation=TORCH_FLOAT),
    build=_torch_compress_pool_build, fused_op="fused_compress_pool_state",
))


def _pooled_store_launch(rows: Any, cr: int, per_req_pools: int,
                         torch: Any):
    """The full-token launch shape the store kernels are invoked with:
    production hands every token's position and slot and gates inside the
    program (`(position + 1) % ratio != 0 → return`, plus the slot's < 0
    marker), so the grid is `pools · ratio` programs, not `pools`
    (fused_compress_quant_cache.py:249-263, :316-322;
    sparse_attn_compress_cutedsl.py:1640-1652).

    Returns `(rows_all, positions, slots)`: the pooled rows laid at their
    closing tokens' rows of a `[tokens, d]` buffer (the non-boundary rows
    are never read — the gate precedes the load), each token's position
    restarting per request, and each token's compressed-cache slot (-1
    unless it closes a group)."""
    pools, d = rows.shape
    batch = pools // per_req_pools
    per_req = per_req_pools * cr
    tokens = pools * cr
    device = rows.device
    rows_all = torch.zeros(tokens, d, dtype=rows.dtype, device=device)
    positions = torch.arange(per_req, dtype=torch.int64,
                             device=device).repeat(batch)
    boundary = positions % cr == cr - 1
    rows_all[boundary] = rows
    slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
    slots[boundary] = torch.arange(pools, dtype=torch.int64, device=device)
    return rows_all, positions, slots


def _vllm_compress_rope_store_build(inv: Invocation, env: Env,
                                    inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `rope_quant_insert` (fused_compress_quant_cache.py:224), as
    `DeepseekCompressor.insert_cache` calls it (deepseek_v41/
    compressor.py:319) with the paged fp8_ds_mla cache: the latent at one
    row per token (production allocates it so, compressor.py:277-282), the
    full token grid with the boundary gate inside the program
    (`_pooled_store_launch`), the fp32 table, the packed `[blocks, page,
    584]` uint8 cache (V4's record, which V4.1 writes off SM100 —
    attention.py:122-129), each token's slot."""
    _require_cuda()
    import torch
    _decline_v41_sm100_record(torch.cuda.get_device_capability())
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )
    dt = _torch_dtype(inv.dtype)
    latent = inputs["latent"].to(dt).contiguous()
    page = int(inputs["kv_block_size"])
    blocks = int(inputs["kv_cache"].shape[0])
    cache = torch.zeros(blocks, page, _DSV4_PACKED_RECORD_BYTES,
                        dtype=torch.uint8, device=latent.device)
    cr = int(inputs["compress_ratio"])
    d_rope = int(inputs["d_rope"])
    per_req = int(inv.dims["s_slots"])
    latent_all, positions, slots = _pooled_store_launch(
        latent, cr, per_req, torch)
    table = inputs["cos_sin_cache"].float()
    written = slots[slots >= 0]
    return BenchCase(
        run=lambda: rope_quant_insert(latent_all, positions, table, cache,
                                      slots, cr),
        to_canonical=lambda _o: _dsv4_swa_rows(
            cache.view(blocks, -1), written, page, int(latent.shape[1]),
            d_rope))


def _torch_compress_rope_store_build(inv: Invocation, env: Env,
                                     inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the last
    `d_rope` channels rotated (the only slice ever widened, as the qnorm
    floor states it), written back at the activation dtype beside the
    untouched NoPE channels, then written as the record the cell names
    (`refs.rows_on_record`), so the floor pays the launch's quantisation.
    No cache: the floor reads the contract's row straight, so no page is
    part of what it measured."""
    import torch
    from ..refs import _rope_rotate, rows_on_record
    dt = _torch_dtype(inv.dtype)
    latent = inputs["latent"].to(dt)
    rd, cr = int(inputs["d_rope"]), int(inputs["compress_ratio"])
    pos = inputs["positions"] // cr * cr
    table = inputs["cos_sin_cache"]
    record = inv.output_format
    wide = torch.promote_types(dt, torch.float32)
    record_row = torch.empty(latent.shape, dtype=wide, device=latent.device)
    out = torch.empty_like(latent)

    def run():
        rotated = _rope_rotate(latent[:, None, -rd:].float(), pos, table, rd,
                               is_neox=False)[:, 0]
        if record is not None:
            record_row[:, :-rd] = latent[:, :-rd]
            record_row[:, -rd:] = rotated
            return rows_on_record(record_row, record, dt, tail=rd)
        out.copy_(latent)
        out[:, -rd:] = rotated.to(dt)
        return out

    return BenchCase(run=run, to_canonical=lambda out: out.float())


for _sig, _vllm_build, _torch_build in (
    ("fused_compress_rope_store", _vllm_compress_rope_store_build,
     _torch_compress_rope_store_build),
):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_{_sig}",
        import_name=("vllm.models.deepseek_v41.common.ops."
                     "fused_compress_quant_cache"),
        serves=KernelSignature(
            signature=_sig,
            # The kernel's geometry: 512-wide latents, the last 64 rotated
            # (:247, :323, :343), the ratio 1 or 2 (:246); the record it
            # writes, V4's FlashMLA one.
            activation=("bf16",), output=("fp8_e4m3",),
            output_format=(_FLASHMLA_V4_RECORD,),
            # The cache it writes is its output (the record above); a cell
            # keys no other cache, so `kv` reads as the activation's.
            dtype_slots=(("kv", ("bf16",)),),
            enum=(("d_head", (512,)), ("d_rope", (64,)),
                  ("compress_ratio", (1, 2))),
            kv_block_sizes=_MLA_CACHE_PAGES),
        impl_params=_served_page_only(_MLA_CACHE_PAGES),
        build=_vllm_build, fused_op=_sig,
    ))
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch",
        # Writes the record the cell names (`rows_on_record`) or the row.
        serves=KernelSignature(signature=_sig, kv_block_sizes=(),
                               activation=TORCH_FLOAT,
                               output=_record_floor_output(),
                               dtype_slots=(("kv", TORCH_FLOAT),),
                               output_format=records_on_grid()),
        build=_torch_build, fused_op=_sig,
    ))


# ---------------------------------------------------------------------------
# DeepSeek-V4 (vLLM v0.32.0): the overlap compressor (pool + norm + rope +
# store, one launch) and the no-overlap compressor's tail. One span, two
# kernels, keyed by the head width the row carries: the 512-wide main
# compressor is CuteDSL (`SparseAttnCompressNormRopeStoreC4Kernel`), the
# indexer's 128-wide one is Triton (`compress_norm_rope_store_triton`).
# ---------------------------------------------------------------------------

def _overlap_state_and_meta(inv: Invocation, inputs: Dict[str, Any],
                            torch: Any):
    """The fp32 state rows and the per-token metadata the launch reads,
    built the way serving's `_SAVE_PARTIAL_STATES_KERNEL` writes them
    (deepseek_v4/common/ops/save_partial_states.py:37): each position's
    `[Ca|Cb]` kv halves at the row's front and its gate (ape added) behind
    them, the launch's tokens every position of every request, and the
    state paged at production's 4 rows a block for the overlap compressor
    (compressor.py:167-169 — the state shares the KV cache's page
    geometry), each request's blocks its own.

    Production's state rows are the fp32 upcast of the same rows the cell
    bills at the activation dtype: the fused wkv|wgate GEMM emits fp32
    (deepseek_v4/attention.py:694-699, `out_dtype=torch.float32`) and the
    walk bills its halves bf16 — the split is stated here because the
    launch's input precision is fp32 either way.
    """
    dims = inv.dims
    batch, s_slots = int(dims["B"]), int(dims["s_slots"])
    cape, d = int(dims["Cape"]), int(dims["d_head"])
    length = s_slots * cape
    dt = _torch_dtype(inv.dtype)
    kv, score = inputs["kv"].to(dt), inputs["score"].to(dt)
    state = torch.zeros(batch, length, 4 * d, dtype=torch.float32,
                        device=kv.device)
    state[..., :2 * d] = kv.view(batch, length, 2 * d)
    state[..., 2 * d:] = score.view(batch, length, 2 * d)
    blocks_per_req = length // cape
    state = state.view(batch * blocks_per_req, cape, 4 * d)
    device = kv.device
    tokens = batch * length
    positions = torch.arange(length, dtype=torch.int64,
                             device=device).repeat(batch)
    state_slots = torch.arange(tokens, dtype=torch.int64, device=device)
    requests = torch.arange(batch, dtype=torch.int32,
                            device=device).repeat_interleave(length)
    block_table = (torch.arange(blocks_per_req, dtype=torch.int32,
                                device=device)[None]
                   + torch.arange(batch, dtype=torch.int32,
                                  device=device)[:, None] * blocks_per_req)
    # One compressed slot per closed window, in (request, window) order;
    # every other token carries the kernel's own not-a-boundary marker.
    kv_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
    kv_slots[cape - 1::cape] = torch.arange(batch * s_slots,
                                            dtype=torch.int64, device=device)
    return state, positions, state_slots, requests, block_table, kv_slots


def _segregated_record_rows(cache, slots, page: int, d: int):
    """The `[rows, d]` entries a launch left in a paged FP8-record cache,
    in fp32 — the layout every one of these store kernels writes (vLLM
    `kpool_compress.py:237-248` and `fused_compress_quant_cache.py:750-756`;
    sglang `kpool_fp8_index.py:1001-1020`): within a block, `page` records
    of `d` FP8 values come first, then `page` fp32 scales — record i's
    values at `i·d`, its scale at `page·d + i·4`. Read back, not
    re-derived: a kernel that chose the scale differently still disagrees
    with the oracle. `cache` is the `[blocks, page·(d+4)]` byte view."""
    import torch
    flat = cache.view(cache.shape[0], -1)
    block, within = slots // page, slots % page
    rows = flat[block]                                       # [n, block bytes]
    vidx = torch.arange(d, device=cache.device)
    values = torch.gather(rows, 1, within[:, None] * d + vidx[None, :]
                          ).contiguous().view(torch.float8_e4m3fn).float()
    sidx = torch.arange(4, device=cache.device)
    scale = torch.gather(rows, 1, page * d + within[:, None] * 4
                         + sidx[None, :]).contiguous().view(torch.float32)
    return values * scale


def _vllm_overlap_c4_build(inv: Invocation, env: Env,
                           inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `SparseAttnCompressNormRopeStoreC4Kernel`
    (sparse_attn_compress_cutedsl.py:78), as `DeepseekCompressor` launches
    it for the 512-wide overlap compressor (compressor.py:449-475): the
    fp32 state rows, the per-token metadata, the bf16 gain, the fp32
    table, the packed 584 B/token cache, one compressed slot per closed
    window."""
    _require_cuda()
    import torch
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL as kernel,
    )
    (state, positions, state_slots, requests, block_table,
     kv_slots) = _overlap_state_and_meta(inv, inputs, torch)
    page = int(inputs["kv_block_size"])
    d = state.shape[-1] // 4
    windows = int(inputs["slot_mapping"].numel())
    blocks = -(-windows // page)
    cache = torch.zeros(blocks, page, _DSV4_PACKED_RECORD_BYTES,
                        dtype=torch.uint8, device=state.device)
    w = inputs["norm_weight"].to(torch.bfloat16)
    table = inputs["cos_sin_cache"].float()
    eps = float(inputs["eps"])
    rd = int(inputs["d_rope"])
    written = kv_slots[kv_slots >= 0]
    return BenchCase(
        run=lambda: kernel(
            state_cache=state, token_to_req_indices=requests,
            positions=positions, slot_mapping=state_slots,
            block_table=block_table, block_size=int(state.shape[1]),
            rms_norm_weight=w, rms_norm_eps=eps, cos_sin_cache=table,
            kv_cache=cache, kv_slot_mapping=kv_slots,
            compress_ratio=int(inputs["Cape"]), head_dim=d, rope_head_dim=rd),
        to_canonical=lambda _o: _dsv4_swa_rows(cache.view(blocks, -1),
                                               written, page, d, rd))


def _vllm_overlap_indexer_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `compress_norm_rope_store_triton(head_dim=128)` — the indexer
    overlap compressor (`_fused_kv_compress_norm_rope_insert_indexer_attn`,
    common/ops/fused_compress_quant_cache.py:711, dispatched at
    compressor.py:469-473): the same state rows and metadata, the FP8
    index-K record cache (128 FP8 values + one fp32 scale a token,
    compressor.py:304-308)."""
    _require_cuda()
    import torch
    from types import SimpleNamespace
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )
    (state, positions, state_slots, requests, block_table,
     kv_slots) = _overlap_state_and_meta(inv, inputs, torch)
    page = int(inputs["kv_block_size"])
    d = state.shape[-1] // 4
    windows = int(inputs["slot_mapping"].numel())
    blocks = -(-windows // page)
    cache = torch.zeros(blocks, page, d + 4, dtype=torch.uint8,
                        device=state.device)
    w = inputs["norm_weight"].to(torch.bfloat16)
    table = inputs["cos_sin_cache"].float()
    eps = float(inputs["eps"])
    rd = int(inputs["d_rope"])
    meta = SimpleNamespace(slot_mapping=kv_slots)
    written = kv_slots[kv_slots >= 0]
    return BenchCase(
        run=lambda: compress_norm_rope_store_triton(
            state_cache=state, num_actual=int(positions.numel()),
            token_to_req_indices=requests, positions=positions,
            slot_mapping=state_slots, block_table=block_table,
            block_size=int(state.shape[1]), state_width=2 * d,
            cos_sin_cache=table, kv_cache=cache, k_cache_metadata=meta,
            pdl_kwargs={}, head_dim=d, rope_head_dim=rd,
            compress_ratio=int(inputs["Cape"]), overlap=True,
            use_fp4_cache=False, rms_norm_weight=w, rms_norm_eps=eps,
            quant_block=d, token_stride=d, scale_dim=4),
        to_canonical=lambda _o: _segregated_record_rows(cache, written, page,
                                                     d))


def _torch_overlap_build(inv: Invocation, env: Env,
                         inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the shift
    → concat pairs, the softmax pool (fp32 reduction onto a tensor a
    fraction of the operand), the norm, and the rotation of the last
    `d_rope` channels — window 0 pooling its own Cb half alone, as the
    kernel's gather mask states."""
    import torch
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    # The launch reads the compressor STATE, held at the cell's `kv` dtype
    # (fp32 in serving, `_overlap_state_and_meta`): pooling it at the
    # activation dtype rounds the softmax weights the kernels keep wide,
    # which a coarse record (mxfp4) turns into whole grid steps.
    sdt = _torch_dtype(inv.dtype_for("kv"))
    kv, score = inputs["kv"].to(sdt), inputs["score"].to(sdt)
    w = inputs["norm_weight"].to(dt)
    cape, eps = int(inputs["Cape"]), float(inputs["eps"])
    rd = int(inputs["d_rope"])
    pos = inputs["positions"] // cape * cape
    table = inputs["cos_sin_cache"]
    d = kv.shape[1] // 2
    # Per request, as the kernel gathers through each request's own block
    # table: a request's first window pools its own Cb half alone.
    per_req = int(inputs["windows_per_request"])
    kvw = kv.view(-1, per_req, cape, 2 * d)
    scw = score.view(-1, per_req, cape, 2 * d)
    prev_k = torch.cat([torch.zeros_like(kvw[:, :1, :, :d]),
                        kvw[:, :-1, :, :d]], dim=1)
    prev_s = torch.cat([torch.full_like(scw[:, :1, :, :d], float("-inf")),
                        scw[:, :-1, :, :d]], dim=1)

    # The pooled pair each request's window reads is fixed by the inputs.
    pair_k = torch.cat([prev_k, kvw[..., d:]], dim=2)
    pair_s = torch.cat([prev_s, scw[..., d:]], dim=2)
    wide = torch.promote_types(dt, torch.float32)
    # One normed row a window: `[B·windows, d]`, the shape `y` has — the
    # record row the run hands `rows_on_record`, not kvw's.
    wide_row = torch.empty(kvw.shape[0] * kvw.shape[1], d, dtype=wide,
                           device=kv.device)

    def run():
        pooled = (pair_k * torch.softmax(pair_s, dim=2)).sum(
            2, dtype=torch.float32)
        y = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True)
                                 + eps)
        y = (y * w.float()).flatten(0, 1)
        # Rotated from the fp32 normed row, as both kernels rotate it.
        rotated = _rope_rotate(y[:, None, -rd:], pos, table, rd,
                               is_neox=False)[:, 0]
        if record is not None:
            # Written as the record the cell names (`rows_on_record`).
            wide_row[:, :-rd] = y[:, :-rd]
            wide_row[:, -rd:] = rotated
            return rows_on_record(wide_row, record, dt, tail=rd)
        y = y.to(dt)
        y[:, -rd:] = rotated.to(dt)
        return y

    from ..refs import rows_on_record
    record = inv.output_format
    return BenchCase(run=run, to_canonical=lambda out: out.float())


for _sig, _build, _impl, _import, _heads, _record in (
    ("fused_overlap_compress_norm_rope_store", _vllm_overlap_c4_build,
     "vllm_cutedsl_compress_norm_rope_store_c4",
     "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl",
     (512,), _FLASHMLA_V4_RECORD),
    # The FP8 index-key record (`use_fp4_cache=False`): 128 FP8 values and
    # one power-of-two fp32 scale a key (fused_compress_quant_cache.py:752,
    # :868-881).
    ("fused_overlap_compress_norm_rope_store", _vllm_overlap_indexer_build,
     "vllm_triton_kv_compress_norm_rope_insert_indexer",
     "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
     (128,), "fp8_pow2_group128"),
):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=_impl,
        import_name=_import,
        serves=KernelSignature(takes_state=True,
            signature=_sig,
            # The kernels' geometry: the overlap pool is compiled at
            # compress_ratio 4 (the CuteDSL dispatch refuses any other,
            # sparse_attn_compress_cutedsl.py:2437-2441; the Triton
            # `overlap = compress_ratio == 4`), the head at 512 or 128, the
            # rotation over the last 64; the record each writes.
            activation=("bf16",), output=("fp8_e4m3",),
            output_format=(_record,),
            # Both read the fp32 compressor state (`_overlap_state_and_meta`;
            # 19 measured rows at kv=fp32).
            dtype_slots=(("kv", ("fp32",)),),
            enum=(("d_head", _heads), ("d_rope", (64,)), ("Cape", (4,))),
            kv_block_sizes=_MLA_CACHE_PAGES),
        impl_params=_served_page_only(_MLA_CACHE_PAGES),
        build=_build, fused_op=_sig,
    ))
register(Provider(
    library="torch", signature="fused_overlap_compress_norm_rope_store",
    kernel_impl="torch_fused_overlap_compress_norm_rope_store",
    import_name="torch",
    # Writes the record the cell names (`rows_on_record`) or the row.
    serves=KernelSignature(signature="fused_overlap_compress_norm_rope_store",
                           kv_block_sizes=(),
                           activation=TORCH_FLOAT,
                           output=_record_floor_output(),
                           dtype_slots=(("kv", TORCH_FLOAT),),
                           output_format=records_on_grid()),
    build=_torch_overlap_build,
    fused_op="fused_overlap_compress_norm_rope_store",
))


def _vllm_compress_norm_rope_store_build(inv: Invocation, env: Env,
                                         inputs: Dict[str, Any]
                                         ) -> BenchCase:
    """vLLM `SparseAttnNormRopeStoreKernel`
    (sparse_attn_compress_cutedsl.py:1453), the second of the two launches
    the ratio-128 compressor runs (compressor.py:475). Production's pool
    launch leaves one fp32 row per STATE token in `compressed_kv`
    (:1347-1355) and the store launches over all of them, gating inside the
    program (:1640-1652) — so the bench lays the cell's pooled rows at
    their closing tokens of a full token grid (`_pooled_store_launch`), not
    a rows-only one. The cell bills the pooled rows at the activation
    dtype; the launch reads the fp32 the pool launch wrote (the upcast is
    the previous launch's output precision)."""
    _require_cuda()
    import torch
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL as kernel,
    )
    dt = _torch_dtype(inv.dtype)
    kv = inputs["kv"].to(dt)
    page = int(inputs["kv_block_size"])
    blocks = int(inputs["kv_cache"].shape[0])
    cache = torch.zeros(blocks, page, _DSV4_PACKED_RECORD_BYTES,
                        dtype=torch.uint8, device=kv.device)
    w = inputs["norm_weight"].to(torch.bfloat16)
    table = inputs["cos_sin_cache"].float()
    eps = float(inputs["eps"])
    cr = int(inputs["compress_ratio"])
    rd = int(inputs["d_rope"])
    d = int(kv.shape[1])
    per_req = int(inv.dims["s_slots"])
    compressed, positions, kv_slots = _pooled_store_launch(
        kv.float(), cr, per_req, torch)
    # The state-slot column only gates `>= 0` here — the store reads the
    # pooled rows, not the state.
    state_slots = torch.zeros(compressed.shape[0], dtype=torch.int64,
                              device=kv.device)
    written = kv_slots[kv_slots >= 0]
    return BenchCase(
        run=lambda: kernel(
            compressed_kv=compressed, positions=positions,
            slot_mapping=state_slots, rms_norm_weight=w, rms_norm_eps=eps,
            cos_sin_cache=table, kv_cache=cache, kv_slot_mapping=kv_slots,
            compress_ratio=cr, head_dim=d, rope_head_dim=rd),
        to_canonical=lambda _o: _dsv4_swa_rows(cache.view(blocks, -1),
                                               written, page, d, rd))


def _torch_compress_norm_rope_store_build(inv: Invocation, env: Env,
                                          inputs: Dict[str, Any]
                                          ) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the
    RMS accumulated in fp32 off the operand itself (no fp32 copy of it),
    the row scaled and gained and rounded to the activation dtype once, as
    the kernels round (a second rounding between the scale and the gain
    moves enough keys across a record's grid step to leave the bf16 band),
    then the last `d_rope` channels rotated and written back at the
    activation dtype. A cell whose `output_format` names the record its
    launch writes gets the row written as that record
    (`refs.rows_on_record`), so the floor pays the launch's quantisation;
    any other is the row."""
    import torch
    from ..refs import _rope_rotate, rows_on_record
    dt = _torch_dtype(inv.dtype)
    kv = inputs["kv"].to(dt)
    w = inputs["norm_weight"].to(dt)
    eps = float(inputs["eps"])
    rd, cr = int(inputs["d_rope"]), int(inputs["compress_ratio"])
    pos = inputs["positions"] // cr * cr
    table = inputs["cos_sin_cache"]
    record = inv.output_format
    # The chunked record's one destination, its width read off the record's
    # own renderer on a one-row probe (the width is a fact of the format,
    # not of the data).
    record_out = None
    if record is not None:
        probe = rows_on_record(kv[:1].to(dt), record, dt, tail=rd)
        record_out = torch.empty(kv.shape[0], probe.shape[1],
                                 dtype=probe.dtype, device=probe.device)

    def run():
        # The sum of squares in fp32 over the bf16 operand (squaring in
        # bf16 first would round each term).
        ss = torch.linalg.vector_norm(kv, dim=-1, keepdim=True,
                                      dtype=torch.float32)
        r = torch.rsqrt(ss.square() / kv.shape[-1] + eps)
        y = (kv * r * w).to(dt)
        y[:, -rd:] = _rope_rotate(y[:, None, -rd:].float(), pos, table, rd,
                                  is_neox=False)[:, 0].to(dt)
        if record is None:
            return y
        # The grid's level search holds one candidate per E2M1 magnitude
        # for every value (8x the row, `quant.quantise_groups`); in chunks
        # the workspace stays the row's size and the row stays the
        # launch's widest operand.
        chunk = max(1, y.shape[0] // 8)
        at = 0
        for part in y.split(chunk):
            block = rows_on_record(part, record, dt, tail=rd)
            record_out[at:at + block.shape[0]] = block
            at += block.shape[0]
        return record_out

    return BenchCase(run=run, to_canonical=lambda out: out.float())


register(Provider(
    library="vllm", signature="fused_compress_norm_rope_store",
    kernel_impl="vllm_cutedsl_norm_rope_store_c128",
    import_name="vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl",
    serves=KernelSignature(
        signature="fused_compress_norm_rope_store",
        # The kernel is the ratio-128 compressor's store (its sibling is
        # the ratio-4 one, a different compile of the same class); 512-wide
        # rows, the last 64 rotated, written as V4's FlashMLA record.
        activation=("bf16",), output=("fp8_e4m3",),
        output_format=(_FLASHMLA_V4_RECORD,),
        enum=(("d_head", (512,)), ("d_rope", (64,)),
              ("compress_ratio", (128,))),
        kv_block_sizes=_MLA_CACHE_PAGES),
    impl_params=_served_page_only(_MLA_CACHE_PAGES),
    build=_vllm_compress_norm_rope_store_build,
    fused_op="fused_compress_norm_rope_store",
))
# DeepSeek-V4.1's index key under the same plan: vLLM
# `indexer_k_norm_rope_store` (models/deepseek_v41/common/ops/
# indexer_k_store.py:29), the one Triton launch `_produce_k` makes
# (deepseek_v41/attention.py:1427). The record it writes is the launch's
# output: 128 FP8 values + one fp32 scale (132 B), or 64 packed E2M1 bytes +
# four UE8M0 exponents (68 B) when the deployment picks the MXFP4 index cache
# (`use_fp4_cache`, chosen once per deployment from `indexer_kv_dtype`,
# v1/attention/backends/mla/indexer.py:59-67) — one provider per record.

def _index_key_record(record: str, width: int) -> Tuple[int, int]:
    """Bytes of one index-key record of `width` channels, (values, scales),
    read off the cache format's own declaration
    (`backend/arch/cache_formats/<record>.yaml`) instead of restated here:
    the channels at the format's element dtype, then one scale per group at
    the scale's (indexer_k_store.py:66-71; `_indexer_k_cache_head_dim`,
    deepseek_v41/attention.py:92-100). FP8 with one power-of-two fp32 scale
    a 128-wide key is 132 B (:237-256); MXFP4 with a UE8M0 exponent per 32
    is 68 B (:210-235)."""
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import dtype_bytes
    fmt = CACHE_FORMATS[record]
    groups = -(-width // fmt.scale.group)
    return (int(width * dtype_bytes(fmt.dtype)),
            int(groups * dtype_bytes(fmt.scale.dtype)))


def _index_key_rows(cache: Any, slots: Any, page: int, record: str,
                    width: int, torch: Any) -> Any:
    """The `[rows, width]` keys a launch left in the paged index-K cache, in
    fp32, decoded from the bytes it wrote (the scales read back, not
    re-derived). A page is its `page` value records, then their scales
    (indexer_k_store.py:192-208). `record` is the cache-format name
    (`_index_key_record`)."""
    values_b, scale_b = _index_key_record(record, width)
    block, within = slots // page, slots % page
    flat = cache.view(cache.shape[0], -1)[block]          # [rows, page bytes]
    vidx = within[:, None] * values_b + torch.arange(values_b,
                                                     device=cache.device)
    sidx = (page * values_b + within[:, None] * scale_b
            + torch.arange(scale_b, device=cache.device))
    values = torch.gather(flat, 1, vidx).contiguous()
    scales = torch.gather(flat, 1, sidx).contiguous()
    if record == "fp8_pow2_group128":
        return (values.view(torch.float8_e4m3fn).float()
                * scales.view(torch.float32))
    from .indexer import _decode_mxfp4
    return _decode_mxfp4(values, scales.view(torch.int32), values.shape[0],
                         1, 2 * values_b).reshape(values.shape[0], -1)


def _vllm_indexer_k_store_build(record: str):
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        """vLLM `indexer_k_norm_rope_store` as `_produce_k` calls it: the bf16
        `wk(latent)` projection at one row per TOKEN (the latent is
        allocated per token, the store reads only boundary rows), every
        token's position and indexer slot (-1 off a boundary), the fp32
        table, the bf16 k_norm gain, the uint8 paged cache
        (`[blocks, page, record bytes]`). The grid is the full token range
        and the program gates (`_pooled_store_launch`); the cell's rows are
        the graph's, one per closed group."""
        _require_cuda()
        import torch
        from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
            indexer_k_norm_rope_store,
        )
        dt = _torch_dtype(inv.dtype)
        kv = inputs["kv"].to(dt)
        page = int(inputs["kv_block_size"])
        blocks = int(inputs["kv_cache"].shape[0])
        width = int(inv.dims["d_head"])
        values_b, scale_b = _index_key_record(record, width)
        cache = torch.zeros(blocks, page, values_b + scale_b,
                            dtype=torch.uint8, device=kv.device)
        cr = int(inputs["compress_ratio"])
        rows, positions, slots = _pooled_store_launch(
            kv, cr, int(inv.dims["s_slots"]), torch)
        written = slots[slots >= 0]
        return BenchCase(
            run=lambda: indexer_k_norm_rope_store(
                rows, positions, inputs["cos_sin_cache"].float(),
                inputs["norm_weight"].to(torch.bfloat16),
                float(inputs["eps"]), cache, slots, cr,
                record == "mxfp4"),
            to_canonical=lambda _o: _index_key_rows(cache, written, page,
                                                    record, width, torch))
    return build


# `kernel_impl` keeps the element dtype it was first registered under (the
# measured rows' identity); the record is `output_format`.
for _suffix, _output, _record in (
        ("fp8_e4m3", "fp8_e4m3", "fp8_pow2_group128"),
        ("mxfp4", "fp4_e2m1", "mxfp4")):
    register(Provider(
        library="vllm", signature="fused_compress_norm_rope_store",
        kernel_impl=f"vllm_indexer_k_norm_rope_store_{_suffix}",
        import_name="vllm.models.deepseek_v41.common.ops.indexer_k_store",
        serves=KernelSignature(
            signature="fused_compress_norm_rope_store",
            # The launch asserts a bf16, 128-wide key and a group of 1 or 2
            # (indexer_k_store.py:58-61) and rotates the last 64
            # (`ROPE_HEAD_DIM=64`, :107); the record is its output: the
            # element dtype and the record itself.
            activation=("bf16",), output=(_output,),
            output_format=(_record,),
            enum=(("d_head", (128,)), ("d_rope", (64,)),
                  ("compress_ratio", (1, 2))),
            kv_block_sizes=_MLA_CACHE_PAGES),
        impl_params=_served_page_only(_MLA_CACHE_PAGES),
        # The MXFP4 record is packed with `cvt.rn.satfinite.e2m1x2.f32`
        # (`_fp32x2_to_fp4x2`, deepseek_v4/common/ops/fused_indexer_q.py:39),
        # which ptxas rejects below SM100 (on an SM89 card: "not supported on
        # .target 'sm_89'"); the FP8 record compiles anywhere.
        min_sm=100 if _record == "mxfp4" else None,
        build=_vllm_indexer_k_store_build(_record),
        fused_op="fused_compress_norm_rope_store",
    ))


register(Provider(
    library="torch", signature="fused_compress_norm_rope_store",
    kernel_impl="torch_fused_compress_norm_rope_store", import_name="torch",
    # Writes the record the cell names (`rows_on_record`) or the row.
    serves=KernelSignature(signature="fused_compress_norm_rope_store",
                           kv_block_sizes=(),
                           activation=TORCH_FLOAT,
                           output=_record_floor_output(),
                           output_format=records_on_grid()),
    build=_torch_compress_norm_rope_store_build,
    fused_op="fused_compress_norm_rope_store",
))


# ---------------------------------------------------------------------------
# GLM-5.3-Flash's k-pool indexer compressor: one Triton launch in each
# framework, near-identical math (softmax(gate+ape) pool, two bf16
# round-trips, Hadamard-128, per-pool pow2-scale FP8, the index-K record).
#
# The record is FP8 by construction in BOTH frameworks (the kernels write
# nothing else, Hadamard-rotated, one pow2 F32 scale a pool), and the model
# ships it that way (`quant_context.cache_formats.index_key:
# fp8_pow2_hadamard128`), so the cell's `output_format` names it and the
# graph carries the rotation as the plan's `rotate` member. A cell keyed at
# another record (a profile's `fp8_per_tensor`, or no transform) is a
# launch neither kernel makes, and both decline it by their declaration.
# ---------------------------------------------------------------------------

def _kpool_candidate_windows(keys_flat: Any, gate_flat: Any, cr: int,
                             torch: Any):
    """Serving's per-token candidate windows: `_kpool_compress_insert`
    launches one program per TOKEN, forming the `cr`-window ending at each
    (`idx = (pos - (kpool-1)).clamp_min(0) + arange(kpool)`;
    glm5next/nvidia/sparse_indexer.py:79-88) and masking the candidates
    that complete no pool. Returns `(slot_k, slot_score, loc, write_mask)`
    for `tokens = pools·cr` candidates — the ragged tail of a chunk (never
    a completed pool, and no operand of this span) stays outside, as the
    plan bills it."""
    tokens = int(keys_flat.shape[0])
    device = keys_flat.device
    offs = torch.arange(cr, device=device)
    idx = ((torch.arange(tokens, device=device) - (cr - 1)).clamp_min(0)
           [:, None] + offs[None, :])
    slot_k = keys_flat[idx].contiguous()
    slot_score = gate_flat[idx].contiguous()
    completes = torch.arange(tokens, device=device) % cr == cr - 1
    loc = torch.where(
        completes, torch.arange(tokens, device=device) // cr,
        torch.full((tokens,), -1, dtype=torch.int64, device=device))
    return slot_k, slot_score, loc, completes


def _vllm_kpool_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `kpool_compress_and_write_cache` (models/glm5next/nvidia/ops/
    kpool_compress.py:259), as `_kpool_compress_insert` calls it
    (glm5next/nvidia/sparse_indexer.py:84-94): one candidate window per
    token with only the pool-completing ones marked to write (production
    does not compact; the masked programs still launch and read), the fp32
    ape, the `[blocks, page, d+4]` uint8 cache, `round_scale` — GLM-5.3's
    `scale_fmt` is "ue8m0" (glm5next/common/attention.py:252), the pow2
    scale."""
    _require_cuda()
    import torch
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_compress_and_write_cache,
    )
    dt = _torch_dtype(inv.dtype)
    cr = int(inputs["compress_ratio"])
    d = int(inputs["keys"].shape[-1])
    keys_flat = inputs["keys"].to(dt).reshape(-1, d)
    gate_flat = inputs["gate"].to(dt).reshape(-1, d)
    slot_k, slot_score, loc, write_mask = _kpool_candidate_windows(
        keys_flat, gate_flat, cr, torch)
    ape = inputs["ape"].float().contiguous()
    pools = int(inputs["slot_mapping"].shape[0])
    page = int(inputs["kv_block_size"])
    blocks = -(-pools // page)
    cache = torch.zeros(blocks, page, d + 4, dtype=torch.uint8,
                        device=keys_flat.device)
    written = loc[loc >= 0]
    return BenchCase(
        run=lambda: kpool_compress_and_write_cache(
            cache, slot_k, slot_score, ape, loc, cr,
            head_dim=d, write_mask=write_mask, round_scale=True,
            return_compressed=False, write_cache=True),
        to_canonical=lambda _o: _segregated_record_rows(cache, written,
                                                        page, d))


def _sgl_kpool_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    """sglang `kpool_softmax_rotate_write_cache` (srt/layers/attention/dsa/
    kpool_fp8_index.py:699), as the no-plan extend path calls it ONCE PER
    REQUEST (`_compress_write_extend`'s `for i in range(batch_size)`,
    dsa_indexer_kpool.py:506-569 → `_write_compressed_pooled_index_cache`
    :283): the request's own completed pools, unmasked, its write slots
    global. `pool` carries only the two layout constants the wrapper reads
    (`index_page_size`, which it asserts is the compiled 64, and
    `index_head_dim`) — the preshuffle tiling is ROCm-only
    (`_preshuffle_tile`, :17-21), so on CUDA the record offsets are the
    row-major ones the reader below decodes. The timed region is the batch
    loop, as serving runs it.
    """
    _require_cuda()
    import torch
    from types import SimpleNamespace
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        kpool_softmax_rotate_write_cache,
    )
    dt = _torch_dtype(inv.dtype)
    cr = int(inputs["compress_ratio"])
    d = int(inputs["keys"].shape[-1])
    batch = int(inv.dims["B"])
    pools = int(inputs["slot_mapping"].shape[0])
    per_req = pools // batch
    keys = inputs["keys"].to(dt).view(pools, cr, d)
    gate = inputs["gate"].to(dt).view(pools, cr, d)
    loc = inputs["slot_mapping"].to(torch.int64).contiguous()
    ape = inputs["ape"].float().contiguous()
    page = int(inputs["kv_block_size"])
    blocks = -(-pools // page)
    buf = torch.zeros(blocks, page * (d + 4), dtype=torch.uint8,
                      device=keys.device)
    pool = SimpleNamespace(index_page_size=page, index_head_dim=d)

    def run():
        for r in range(batch):
            kpool_softmax_rotate_write_cache(
                # Dim-0 slices of the contiguous view are contiguous:
                # no copy is made (or needed) per request.
                pool=pool, buf=buf,
                slot_k=keys[r * per_req:(r + 1) * per_req],
                slot_score=gate[r * per_req:(r + 1) * per_req],
                ape=ape, loc=loc[r * per_req:(r + 1) * per_req],
                write_mask=None, round_scale=True, return_compressed=False,
                write_cache=True)

    return BenchCase(
        run=run,
        to_canonical=lambda _o: _segregated_record_rows(
            buf, loc, page, d))


def _torch_kpool_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the
    softmax pool (fp32 reduction onto a tensor 1/ratio of the operand),
    the bf16 round-trip the kernels state. A cell whose `output_format`
    names the record its launch writes gets the pooled key written as that
    record (`refs.rows_on_record`): rotated exactly when the record's
    format carries the transform, then on its grid — so the floor pays
    the plan's `rotate` member and the launch's quantisation; any other is
    the pooled key."""
    import torch
    from ..refs import rows_on_record
    dt = _torch_dtype(inv.dtype)
    cr = int(inputs["compress_ratio"])
    gate = inputs["gate"].to(dt).view(-1, cr, inputs["gate"].shape[-1])
    keys = inputs["keys"].to(dt).view(-1, cr, inputs["keys"].shape[-1])
    ape = inputs["ape"].float()
    record = inv.output_format

    def run():
        pooled = (keys * torch.softmax(gate + ape[None].to(dt),
                                       dim=1)).sum(1, dtype=torch.float32)
        pooled = pooled.to(dt)
        if record is None:
            return pooled
        return rows_on_record(pooled, record, dt)

    return BenchCase(run=run, to_canonical=lambda out: out.float())


for _lib, _impl, _import, _build, _pages in (
    ("vllm", "vllm_kpool_compress_and_write_cache",
     "vllm.models.glm5next.nvidia.ops.kpool_compress", _vllm_kpool_build,
     _MLA_CACHE_PAGES),
    ("sglang", "sglang_kpool_softmax_rotate_write_cache",
     "sglang.srt.layers.attention.dsa.kpool_fp8_index", _sgl_kpool_build,
     # The wrapper asserts the page is the compiled BLOCK_SIZE_K
     # (kpool_fp8_index.py:718-719).
     (64,)),
):
    register(Provider(
        library=_lib, signature="fused_kpool_compress_store",
        kernel_impl=_impl, import_name=_import,
        serves=KernelSignature(
            signature="fused_kpool_compress_store",
            # The kernels' compiled geometry: the 128-wide index head
            # (INDEX_HEAD_DIM in both files; the Hadamard is written for
            # 128), the key windows bf16 (both wrappers assert it), and the
            # one record both write (see the family note above).
            activation=("bf16",), output=("fp8_e4m3",),
            output_format=("fp8_pow2_hadamard128",),
            # The cache they write is their output (the record); `kv`
            # reads as the activation's.
            dtype_slots=(("kv", ("bf16",)),),
            enum=(("d_indexer_head", (128,)),),
            kv_block_sizes=_pages),
        impl_params=_served_page_only(_pages),
        build=_build, fused_op="fused_kpool_compress_store",
    ))
register(Provider(
    library="torch", signature="fused_kpool_compress_store",
    kernel_impl="torch_fused_kpool_compress_store", import_name="torch",
    # The 128-wide head both kernels compile; at another width the span is
    # another function, which the floor declines the way the kernels do.
    # Writes the record the cell names (`rows_on_record`) or the pooled key.
    serves=KernelSignature(signature="fused_kpool_compress_store",
                           activation=TORCH_FLOAT,
                           output=_record_floor_output(),
                           dtype_slots=(("kv", TORCH_FLOAT),),
                           output_format=records_on_grid(),
                           enum=(("d_indexer_head", (128,)),),
                           kv_block_sizes=()),
    build=_torch_kpool_build, fused_op="fused_kpool_compress_store",
))


# ---------------------------------------------------------------------------
# Qwen QSA's indexer preparation. sglang runs each span as its own JIT
# launch (kernels/ops/attention/qsa_indexer.py); vLLM runs both as work
# items of the one `qsa_prepare` launch (models/qwen4_exp/nvidia/ops/
# qsa_prepare.py:504), which also prepares the main attention's q/k/gate
# and writes its K/V cache. Each cell here is handed the launch the way
# serving reaches it for that span's share:
#   * the query cell at `num_k_work == 0` — the launch every decode step
#     that completes no block makes (one block completes every
#     `compress_ratio` steps, so this is the common case);
#   * the key cell with the completing block's work items — the launch a
#     completing step makes, the step's query tiles included (they share
#     `num_tokens`; there is no flag that drops them).
# The main-attention operands sit at the smallest legal geometry (one q
# head, one KV head): the main chain is measured under
# `fused_qk_rmsnorm_rope_gate`, and a bigger main would bill its work to
# this cell a second time. The indexer work items themselves are
# byte-identical to production's.
# ---------------------------------------------------------------------------

#: vLLM's QSA attention backend consumes manager pages directly and takes
#: any multiple of 16 (qwen4_exp/nvidia/qsa.py:187-189); the compressed
#: cache's page is one of them (the binding's own default is 16, the
#: MLA-forced CUDA page 64).
_QSA_CACHE_PAGES = (16, 32, 64)


def _qsa_fused_qk(q: Any, d: int, torch: Any) -> Any:
    """The fused index-Q/K projection row serving splits
    (`projected_qk.split([HQ·d, d], -1)`, vLLM indexer_qsa.py:288-292;
    sglang qsa_indexer.py:168-169): one `[tokens, (HQ+1)·d]` buffer, the
    query its leading columns — a strided view, which is the layout both
    launches read (vLLM carries `q.stride(0)`; sglang takes the buffer
    whole). The trailing raw-key columns hold whatever the allocator had;
    the launch writes them to the raw-key ring, as serving's does, and
    nothing reads them back here."""
    tokens = int(q.shape[0])
    heads = int(q.shape[1]) if q.ndim == 3 else int(q.shape[-1]) // d
    buf = torch.empty(tokens, (heads + 1) * d, dtype=q.dtype,
                      device=q.device)
    buf[:, :heads * d] = q.reshape(tokens, heads * d)
    return buf


def _qsa_main_operands(tokens: int, d: int, dt: Any, weight: Any,
                       eps: float, device: Any, torch: Any):
    """`qsa_prepare`'s main-attention share at its smallest legal geometry:
    one q head and one KV head of the query's own width (the wrapper
    derives the head counts from these shapes, qsa_prepare.py:541-542)."""
    main_d = d
    main_qkv = torch.empty(tokens, 4 * main_d, dtype=dt, device=device)
    main_kv_cache = torch.zeros(1, 16, 1, 2 * main_d, dtype=dt,
                                device=device)
    # Slots stay inside the one block the bench allocates.
    main_slots = torch.arange(tokens, dtype=torch.int64,
                              device=device) % 16
    return dict(main_qkv=main_qkv, main_q_norm_weight=weight,
                main_k_norm_weight=weight, main_eps=eps,
                main_kv_cache=main_kv_cache, main_slot_mapping=main_slots,
                main_k_scale=1.0, main_v_scale=1.0)


def _vllm_qsa_q_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `qsa_prepare` (qwen4_exp/nvidia/ops/qsa_prepare.py:504), as
    `QSAIndexer.forward` calls it (indexer_qsa.py:310) on a step that
    completes no key block: the query the strided leading view of the
    fused projection row, the Gemma weight, the fp32 table, 1-D positions
    (text), no MRoPE sections asked — and no key work items, so the key
    side's operands are valid but unread. `QSA runs BF16 only`
    (qwen4_exp/nvidia/qsa.py:354-355).

    The arm serves DECODE cells only: a decode step that completes no
    block is exactly this launch (`num_k_work == 0`), and that is the
    decode step's dominant cadence (one block completes every
    `compress_ratio` steps, the completing launch measured on the key
    cell). A prefill chunk that completes blocks folds the key work into
    the same launch — no `qsa_prepare` call is the query work alone there,
    so rather than measure a reduced launch, the arm declines prefill and
    the sglang / torch arms carry it.
    """
    _require_cuda()
    import torch
    from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare
    dt = _torch_dtype(inv.dtype)
    q0 = inputs["q"].to(dt)
    tokens, d = int(q0.shape[0]), int(q0.shape[-1])
    heads = int(q0.shape[1])
    buf = _qsa_fused_qk(q0, d, torch)
    q, k = buf[:, :heads * d], buf[:, heads * d:]
    q_out = torch.empty(tokens, heads, d, dtype=dt, device=q.device)
    weight = inputs["weight"].to(dt)
    positions = inputs["positions"].to(torch.int64)
    table = inputs["cos_sin_cache"].float().contiguous()
    eps = float(inputs["eps"])
    state = torch.zeros(1, 4, d, dtype=dt, device=q.device)
    compressed = torch.zeros(1, 16, d, dtype=dt, device=q.device)
    empty_work = torch.empty(0, 2, dtype=torch.int32, device=q.device)
    return BenchCase(
        run=lambda: qsa_prepare(
            q, k, positions, table, weight, weight, eps, q_out,
            state, torch.zeros(tokens, dtype=torch.int32, device=q.device),
            torch.zeros(1, 1, dtype=torch.int32, device=q.device),
            torch.tensor([0, tokens], dtype=torch.int32, device=q.device),
            positions.to(torch.int32),
            compressed,
            torch.full((tokens,), -1, dtype=torch.int32, device=q.device),
            empty_work,
            compress_ratio=4,      # inert: no key work items
            mrope_section=None, rope_pos_offset=None,
            **_qsa_main_operands(tokens, d, dt, weight, eps, q.device,
                                 torch)),
        to_canonical=lambda _o: q_out.float().flatten(1))


def _sgl_qsa_q_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    """sglang `qsa_index_q_norm_rope_store` (kernels/ops/attention/
    qsa_indexer.py:69), as `QSAIndexer.project_qk` calls it (srt/layers/
    attention/qsa/qsa_indexer.py:188-204): the fused projection row whole,
    the fp32 table the fused-prep gate requires (:136-137), the all-t
    axis map of a text forward, the raw-key ring and the position buffer
    it also writes. `out_dtype` defaults to the projection's dtype — the
    bf16 store of a bf16 QSA cache; where the pool holds FP8 the same
    launch writes e4m3 rows, a store precision the cell's billing does
    not carry. The launch also writes the raw-key ring and the per-slot
    RoPE coordinates (the kernel has no variant without them); the plan
    bills those bytes on the key chain (fusion_plans/
    qsa_index_q_norm_rope.yaml: "the bytes are the same, they move
    between the two nodes"), so the measured launch carries a write the
    cell does not."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_q_norm_rope_store,
    )
    dt = _torch_dtype(inv.dtype)
    q0 = inputs["q"].to(dt)
    tokens, d = int(q0.shape[0]), int(q0.shape[-1])
    heads = int(q0.shape[1])
    buf = _qsa_fused_qk(q0, d, torch)
    rd = int(inputs["cos_sin_cache"].shape[-1])
    return BenchCase(
        run=lambda: qsa_index_q_norm_rope_store(
            buf, inputs["positions"].to(torch.int64),
            inputs["cos_sin_cache"].float().contiguous(),
            torch.zeros(rd // 2, dtype=torch.int32, device=q0.device),
            inputs["weight"].to(dt),
            torch.arange(tokens, dtype=torch.int64, device=q0.device),
            torch.zeros(tokens, d, dtype=dt, device=q0.device),
            torch.zeros(tokens, 3, dtype=torch.int64, device=q0.device),
            heads, rd, float(inputs["eps"]), True),
        to_canonical=lambda out: out.float().flatten(1))


for _sig, _lib, _impl, _import, _build, _acts in (
    ("qsa_index_q_norm_rope", "vllm", "vllm_qsa_prepare_q_norm_rope",
     "vllm.models.qwen4_exp.nvidia.ops.qsa_prepare", _vllm_qsa_q_build,
     ("bf16",)),
    ("qsa_index_q_norm_rope", "sglang", "sglang_qsa_index_q_norm_rope_store",
     "sglang.kernels.ops.attention.qsa_indexer", _sgl_qsa_q_build,
     ("bf16", "fp16")),
):
    register(Provider(
        library=_lib, signature=_sig, kernel_impl=_impl,
        import_name=_import,
        serves=KernelSignature(
            signature=_sig,
            activation=_acts,
            # The vLLM arm's launch is the non-completing decode step's —
            # see the build — so it serves decode cells; a prefill chunk's
            # production launch folds the completing blocks' key work in.
            phases=("decode",) if _lib == "vllm" else (),
            # The compiled geometries: vLLM's fused-prep gate is the
            # 128-wide head and 64-wide rotation of the binding's dims
            # (indexer_qsa.py:72-85); sglang's JIT compiles bf16/fp16 at
            # head 64/128/256 (qsa_indexer.py:39-49), the rotation width a
            # runtime argument.
            enum=((("d_indexer_head", (128,)), ("d_rope", (64,)))
                  if _lib == "vllm" else
                  (("d_indexer_head", (64, 128, 256)),))),
        build=_build, fused_op=_sig,
    ))


def _vllm_qsa_k_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `qsa_prepare`'s key work items (qsa_prepare.py:296-443), as a
    chunk completing whole blocks reaches them: the raw keys the trailing
    slice of the fused projection row (the strided view serving passes,
    indexer_qsa.py:288-292 — `k.stride(0)` is a runtime argument the
    kernel carries), the per-request chunk tables, one work item per
    completed block, the paged compressed cache in the cell's own cache
    dtype (vLLM chooses bf16 or fp8 at indexer_qsa.py:150-164; the kernel
    casts at the store).

    The step's query tiles run in the same launch — `num_tokens` is shared
    and no flag drops them (qsa_prepare.py:576-583) — and their output is
    allocated and not read back. They are the same work the
    `qsa_index_q_norm_rope` cell bills: the one launch does both spans,
    and this measurement includes the tile share. The tile count uses the
    Qwen4Exp family's four index heads (the binding's `architectures`
    gate names the family; the key span's own dims carry no head count).
    """
    _require_cuda()
    import torch
    from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare
    dt = _torch_dtype(inv.dtype)
    raw = inputs["raw_key"].to(dt)
    tokens, d = raw.shape
    cr = int(inputs["compress_ratio"])
    page = int(inputs["kv_block_size"])
    groups = tokens // cr
    batch = int(inv.dims["B"])
    per_req = tokens // batch
    heads = 4                       # the Qwen4Exp index head count; the
    # tiles it sizes are inert to the key work items this cell bills.
    weight = inputs["weight"].to(dt)
    positions = inputs["positions"].to(torch.int64)
    table = inputs["cos_sin_cache"].float().contiguous()
    eps = float(inputs["eps"])
    device = raw.device
    # The fused projection row as serving builds it: the raw keys its
    # trailing columns, the query tiles' input its leading ones (inert
    # content — the cell bills none of it and it is not read back).
    q_buf = torch.empty(tokens, (heads + 1) * d, dtype=dt, device=device)
    q_buf[:, heads * d:] = raw
    q, k = q_buf[:, :heads * d], q_buf[:, heads * d:]
    q_out = torch.empty(tokens, heads, d, dtype=dt, device=device)
    state = torch.zeros(1, cr, d, dtype=dt, device=device)
    qsl = torch.arange(0, tokens + 1, per_req, dtype=torch.int32,
                       device=device)
    work = torch.tensor([[r, i] for r in range(batch)
                         for i in range(per_req // cr)],
                        dtype=torch.int32, device=device)
    slots = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    boundary = torch.arange(cr - 1, tokens, cr, dtype=torch.int64,
                            device=device)
    slots[boundary] = torch.arange(groups, dtype=torch.int32,
                                   device=device)
    blocks = -(-groups // page)
    compressed = torch.zeros(blocks, page, d,
                             dtype=inputs["kv_cache"].dtype, device=device)
    kv_slots = inputs["slot_mapping"].to(torch.int64)
    return BenchCase(
        run=lambda: qsa_prepare(
            q, k, positions, table, weight, weight, eps, q_out,
            state, (positions % cr).to(torch.int32),
            torch.zeros(batch, 1, dtype=torch.int32, device=device),
            qsl, positions.to(torch.int32), compressed, slots, work,
            compress_ratio=cr,
            mrope_section=None, rope_pos_offset=None,
            **_qsa_main_operands(tokens, d, dt, weight, eps, device,
                                 torch)),
        to_canonical=lambda _o: compressed.view(-1, d)[
            kv_slots].float())


def _sgl_qsa_k_operands(inputs: Dict[str, Any], dt: Any, torch: Any):
    """The key-compression launch's operands, as `_fused_compress_store`
    passes them (srt/layers/attention/qsa/qsa_indexer.py:249-263): the raw
    keys as the pending ring keeps them (one row a slot, in order), each
    completed block's ring slots with the group-start slot in column 0, the
    per-slot RoPE coordinates (t=h=w for text), and the flat compressed
    ring in the cache's own dtype."""
    raw = inputs["raw_key"].to(dt)
    tokens, d = raw.shape
    cr = int(inputs["compress_ratio"])
    groups = tokens // cr
    device = raw.device
    ring = raw.contiguous()
    group_locs = torch.arange(tokens, dtype=torch.int32,
                              device=device).view(groups, cr).contiguous()
    rope_pos = (inputs["positions"].to(torch.int64)[:, None]
                .expand(tokens, 3).contiguous())
    # The compressed ring in the cell's own cache dtype — bf16 or fp8,
    # both production (`qsa_compressed_dtype`, qsa_kv_pool.py:147-156).
    buffer = torch.zeros(groups, d, dtype=inputs["kv_cache"].dtype,
                         device=device)
    return ring, group_locs, rope_pos, buffer, d, cr, groups


def _sgl_qsa_k_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    """sglang `qsa_index_k_compress_store` (kernels/ops/attention/
    qsa_indexer.py:143): the operands of `_sgl_qsa_k_operands`, the fp32
    table, the all-t axis map of a text forward, the Gemma weight, the flat
    ring — not a paged cache, so no page is part of what this arm
    measured."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_k_compress_store,
    )
    dt = _torch_dtype(inv.dtype)
    ring, group_locs, rope_pos, buffer, _d, cr, groups = (
        _sgl_qsa_k_operands(inputs, dt, torch))
    device = ring.device
    rd = int(inputs["cos_sin_cache"].shape[-1])
    return BenchCase(
        run=lambda: qsa_index_k_compress_store(
            ring, group_locs, rope_pos,
            inputs["cos_sin_cache"].float().contiguous(),
            torch.zeros(rd // 2, dtype=torch.int32, device=device),
            inputs["weight"].to(dt),
            torch.arange(groups, dtype=torch.int32, device=device),
            buffer, cr, rd, float(inputs["eps"]), True),
        to_canonical=lambda _o: buffer.float())


def _torch_qsa_q_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the
    Gemma norm (fp32 accumulation in the reduction), then the rotation of
    the first `d_rope` channels of each head written back at the
    activation dtype."""
    import torch
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt)
    w = inputs["weight"].to(dt)
    eps = float(inputs["eps"])
    rd = int(inputs["cos_sin_cache"].shape[-1])
    pos, table = inputs["positions"], inputs["cos_sin_cache"]
    out = torch.empty_like(q)

    def run():
        y = q * torch.rsqrt(
            q.square().mean(-1, keepdim=True, dtype=torch.float32)
            + eps).to(dt)
        y = y * (1.0 + w)
        out.copy_(y)
        out[..., :rd] = _rope_rotate(y[..., :rd].float(), pos, table, rd,
                                     is_neox=True).to(dt)
        return out

    return BenchCase(run=run, to_canonical=lambda out: out.float().flatten(1))


def _torch_qsa_k_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive chain, eager and at the cell's precision: the fp32
    block mean rounded to the storage dtype, the Gemma norm, the rotation
    at each block's first position. No cache: the floor reads the
    contract's row straight."""
    import torch
    from ..refs import _rope_rotate
    dt = _torch_dtype(inv.dtype)
    raw = inputs["raw_key"].to(dt)
    w = inputs["weight"].to(dt)
    cr = int(inputs["compress_ratio"])
    eps = float(inputs["eps"])
    rd = int(inputs["cos_sin_cache"].shape[-1])
    pos = inputs["positions"][::cr]
    table = inputs["cos_sin_cache"]
    d = int(raw.shape[-1])
    out = torch.empty(raw.shape[0] // cr, d, dtype=dt, device=raw.device)

    def run():
        # The block mean accumulates in fp32 (`dtype=`), as the kernels do,
        # without an fp32 copy of the raw keys: that copy was the launch's
        # widest operand, and the gate refused the floor as fp32 work.
        pooled = raw.view(-1, cr, d).mean(dim=1, dtype=torch.float32).to(dt)
        y = pooled * torch.rsqrt(
            pooled.square().mean(-1, keepdim=True, dtype=torch.float32)
            + eps).to(dt)
        y = y * (1.0 + w)
        out.copy_(y)
        out[:, :rd] = _rope_rotate(y[:, None, :rd].float(), pos, table,
                                   rd, is_neox=True)[:, 0].to(dt)
        return out

    return BenchCase(run=run, to_canonical=lambda out: out.float())


for _sig, _lib, _impl, _import, _build, _acts, _caches, _enum, _pages in (
    # vLLM stores bf16 or fp8 index keys (indexer_qsa.py:155-162), the
    # kernel casting at the store (`_to_dst_dtype`, qsa_prepare.py:84-88).
    ("qsa_index_k_compress_store", "vllm",
     "vllm_qsa_prepare_k_compress_store",
     "vllm.models.qwen4_exp.nvidia.ops.qsa_prepare", _vllm_qsa_k_build,
     ("bf16",), ("bf16", "fp8_e4m3"),
     (("d_indexer_head", (128,)), ("d_rope", (64,))), _QSA_CACHE_PAGES),
    # sglang's ring is the key's own dtype.
    ("qsa_index_k_compress_store", "sglang",
     "sglang_qsa_index_k_compress_store",
     "sglang.kernels.ops.attention.qsa_indexer", _sgl_qsa_k_build,
     ("bf16", "fp16"), ("bf16", "fp16"),
     (("d_indexer_head", (64, 128, 256)),), ()),
    ("qsa_index_k_compress_store", "torch", "torch_qsa_index_k_compress_store",
     "torch", _torch_qsa_k_build, TORCH_FLOAT, TORCH_FLOAT, (), ()),
):
    register(Provider(
        library=_lib, signature=_sig, kernel_impl=_impl,
        import_name=_import,
        # The framework kernels write the key into the open group's ring /
        # state rows (`_vllm_qsa_k_build`, `_sgl_qsa_k_build`); the torch
        # floor pools the cell's own keys and keeps no state.
        serves=KernelSignature(signature=_sig, activation=_acts,
                               dtype_slots=(("kv", _caches),),
                               enum=_enum, kv_block_sizes=_pages,
                               takes_state=_lib != "torch"),
        impl_params=_served_page_only(_pages) if _pages else None,
        build=_build, fused_op=_sig,
    ))
register(Provider(
    library="torch", signature="qsa_index_q_norm_rope",
    kernel_impl="torch_qsa_index_q_norm_rope", import_name="torch",
    serves=KernelSignature(signature="qsa_index_q_norm_rope",
                           activation=TORCH_FLOAT),
    build=_torch_qsa_q_build, fused_op="qsa_index_q_norm_rope",
))
