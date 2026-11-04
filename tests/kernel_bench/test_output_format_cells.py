"""A cell that writes a cached record as its output reaches the kernel that
writes THAT record, and only it.

The `output` slot is the record's element dtype, and dtypes cannot say how
the scales are laid out: DeepSeek-V4's FlashMLA record (one UE8M0 per 64,
the last 64 channels BF16) and an FP8 index key with one F32 scale are both
`fp8_e4m3`; an MXFP4 and an NVFP4 index key are both `fp4_e2m1`. The cell
names the record (`output_format`, a `backend/arch/cache_formats/` name) and
a kernel declares the records it writes (`KernelSignature.output_format`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.precision import AS_SHIPPED, ResolvedPrecision
from backend.arch.eval.program import Program
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.calibration_overlay import deployment_kv_block_size
from backend.inference.framework_bindings import (ServingDeployment,
                                                  get_binding)
from backend.inference.gpu_specs import get_gpu
from backend.inference.op_signatures import op_signature
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from kernel_bench.invocation import Invocation
from kernel_bench.providers import (KernelSignature, Provider,
                                    all_providers, register)
from tests.snapshots import resolve_snapshot, shared_loader


def _cells(model, framework, phase, op, where, precision=None):
    """The walked cells of `op` rows whose id contains `where`: the
    identity the plan keys them at (`RowIdentity`), as an Invocation.
    `precision` is a request's (`{"preset": ...}`); absent, as shipped."""
    from backend.arch.precision_profile import PrecisionProfile
    resolved = resolve_snapshot(model, shared_loader())
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="B200", activation_dtype=ResolvedPrecision(
            AS_SHIPPED, Program.of(resolved)).model_dtype))
    result = analyze(resolved, parallel=ParallelConfig(),
                     avg_input_tokens=1024,
                     avg_output_tokens=0 if phase == "prefill" else 64,
                     phase=phase, batch_size=8, gpu=get_gpu("B200"),
                     framework_binding=binding, breakdown=True,
                     precision=precision and PrecisionProfile.from_request(
                         precision))
    # The page the framework allocates the cache in: a cell that reads or
    # writes a paged cache is keyed at it (`deployment_kv_block_size`).
    identity = RowIdentity(result,
                           kv_block_size=deployment_kv_block_size(binding))
    out = []
    for row in result.per_op_timing:
        if row["op"] != op or where not in row["aggregation_id"]:
            continue
        kernel, shape = identity.of(row)
        env, declined = identity.leaf_dtype_env(row)
        assert not declined, declined
        sig = op_signature(kernel)
        out.append(Invocation(op=kernel, signature=sig, phase=phase,
                              dims=json.loads(shape), tp=1,
                              dtype_env=json.loads(env)))
    assert out, (model, op, where)
    return out


def _claimed(inv, library):
    return {p.kernel_impl for p in all_providers()
            if p.library == library and p.serves is not None
            and p.serves.accepts(inv)}


def test_v4s_latent_store_reaches_the_flashmla_launch():
    """sglang's latent-KV norm + rope + store writes the FlashMLA record:
    the cell names it, and the in-place bf16 launch is not this cell's."""
    for inv in _cells("deepseek-ai/DeepSeek-V4-Flash", "sglang", "decode",
                      "fused_norm_rope", "attn.kv_concat"):
        assert inv.output_format == "fp8_ue8m0_group64_bf16_rope_tail"
        assert _claimed(inv, "sglang") == {
            "sglang_fused_k_norm_rope_flashmla"}, inv.dtype_env


def test_v41s_index_key_reaches_the_kernel_of_its_declared_record():
    """DeepSeek-V4.1 declares its index key MXFP4: the vLLM launch that
    writes MXFP4 records measures it, the FP8 one does not."""
    for inv in _cells("deepseek-ai/DeepSeek-V4.1-Flash", "vllm", "prefill",
                      "fused_compress_norm_rope_store", ".indexer.k_cache"):
        assert (inv.dtype_env["output"], inv.output_format) == (
            "fp4_e2m1", "mxfp4")
        assert _claimed(inv, "vllm") == {
            "vllm_indexer_k_norm_rope_store_mxfp4"}, inv.dtype_env


def test_v41s_swa_row_reaches_deepinfers_store_of_its_record():
    """DeepInfer's V4.1 SWA latent row is cast to the spec's `kv` record
    (E4M3, one UE8M0 per 32) and stored in the same decode launch; the
    cell names that record and DeepInfer's arm claims it."""
    for inv in _cells("deepseek-ai/DeepSeek-V4.1-Flash", "deepinfer",
                      "decode", "fused_norm_rope", "attn.kv_concat"):
        assert (inv.dtype_env["output"], inv.output_format) == (
            "fp8_e4m3", "fp8_ue8m0_group32"), inv.dtype_env
        assert _claimed(inv, "deepinfer") == {
            "deepinfer_swa_norm_rope_store"}, inv.dtype_env


def _provider(**serves):
    return Provider(library="torch", signature="fused_norm_rope",
                    kernel_impl="test_output_format_probe",
                    import_name="torch",
                    serves=KernelSignature(signature="fused_norm_rope",
                                           **serves),
                    build=lambda inv, env, inputs: None)


def test_registration_refuses_a_record_the_registry_does_not_name():
    with pytest.raises(ValueError, match="is not a cache format"):
        register(_provider(output=("fp8_e4m3",),
                           output_format=("fp8_flashmla_v4",)))


def test_registration_refuses_a_record_its_output_dtype_cannot_hold():
    with pytest.raises(ValueError, match="stores fp4_e2m1 elements"):
        register(_provider(output=("fp8_e4m3",), output_format=("mxfp4",)))


def test_v4s_q_kv_insert_reaches_the_kernel_that_writes_its_record():
    """vLLM's DeepSeek-V4 q-norm / rope / KV insert writes the FlashMLA
    record beside the query it outputs: the cell names it in `kv_format`,
    and the packed-record overload this harness builds claims it."""
    for inv in _cells("deepseek-ai/DeepSeek-V4-Flash", "vllm", "decode",
                      "fused_qnorm_rope_kv_insert", "attn.q_concat"):
        assert (inv.kv_dtype, inv.kv_format) == (
            "fp8_e4m3", "fp8_ue8m0_group64_bf16_rope_tail"), inv.dtype_env
        assert _claimed(inv, "vllm") == {
            "vllm_c_fused_qnorm_rope_kv_insert"}, inv.dtype_env


def test_v32s_sparse_mla_reaches_the_kernel_that_reads_its_record():
    """DeepSeek-V3.2's FP8 latent is FlashMLA's `fp8_ds_mla` (four
    power-of-two F32 scales): the cell names it, and FlashMLA's FP8 sparse
    decode — which reads that record and nothing else — claims it."""
    for inv in _cells("deepseek-ai/DeepSeek-V3.2", "vllm", "decode",
                      "flash_dsa", "attn_apply_latent"):
        assert inv.kv_format == "fp8_pow2_group128", inv.dtype_env
        assert "flashmla_sparse_fp8" in _claimed(inv, "flashmla")
        per_tensor = Invocation(op=inv.op, signature=inv.signature,
                                phase=inv.phase, dims=inv.dims, tp=1,
                                dtype_env={**inv.dtype_env,
                                           "kv_format": "fp8_per_tensor"})
        assert "flashmla_sparse_fp8" not in _claimed(per_tensor, "flashmla")


def test_registration_refuses_a_cache_record_its_kv_dtype_cannot_hold():
    with pytest.raises(ValueError, match="kv_format 'mxfp4' stores fp4_e2m1"):
        register(_provider(dtype_slots=(("kv", ("fp8_e4m3",)),),
                           kv_format=("mxfp4",)))
    with pytest.raises(ValueError, match="kv_format 'fp8_flashmla' is not"):
        register(_provider(kv_format=("fp8_flashmla",)))


# --- A declaration that admits the plain tensor ----------------------------

def _format_refusals(kernel_impl, **dtype_env):
    """The format-slot clauses of `kernel_impl`'s declaration that refuse a
    cell of `dtype_env` (the other clauses are about the cell's shape)."""
    from backend.inference.benchmark_identity import FORMAT_SLOTS
    p = next(p for p in all_providers() if p.kernel_impl == kernel_impl)
    inv = Invocation(op=p.signature, signature=p.signature, phase="decode",
                     dims={}, tp=1, dtype_env={"activation": "bf16",
                                               **dtype_env})
    return [why for why in p.serves.refusals(inv)
            if why.split(":")[0] in FORMAT_SLOTS]


def test_registration_takes_plain_beside_names_and_refuses_it_twice():
    from backend.inference.benchmark_identity import PLAIN
    from kernel_bench.providers import _check_formats
    _check_formats(_provider(kv_format=(PLAIN,)))
    _check_formats(_provider(dtype_slots=(("kv", ("bf16", "fp8_e4m3")),),
                             kv_format=(PLAIN, "fp8_per_tensor")))
    _check_formats(_provider(output=("bf16",), output_format=(PLAIN,)))
    with pytest.raises(ValueError, match="names a format twice"):
        _check_formats(_provider(kv_format=(PLAIN, PLAIN)))
    with pytest.raises(ValueError, match="is not a cache format"):
        _check_formats(_provider(kv_format=(PLAIN, "plain")))


def test_a_plain_only_kernel_declines_a_record_cell_and_takes_a_plain_one():
    """FlashMLA's dense decode build pages the latent at its own dtype and
    passes no descale: a bf16 cache. A cell whose cache is a per-tensor FP8
    record is the FP8 entry point's, which this build never calls."""
    assert _format_refusals("flashmla_decode", kv="bf16") == []
    assert _format_refusals("flashmla_decode", kv="fp8_e4m3",
                            kv_format="fp8_per_tensor") == [
        "kv_format:fp8_per_tensor is not served (serves plain)"]


def test_a_plain_or_per_tensor_kernel_declines_a_grouped_record():
    """FlashInfer's paged decode reads a plain cache or an FP8 one with one
    k/v scale; FlashMLA's grouped `fp8_ds_mla` record is neither."""
    for impl in ("flashinfer_paged_decode", "flash_attention_v3_kvcache",
                 "torch_sdpa"):
        assert _format_refusals(impl, kv="bf16") == [], impl
        assert _format_refusals(impl, kv="fp8_e4m3",
                                kv_format="fp8_per_tensor") == [], impl
        assert _format_refusals(impl, kv="fp8_e4m3",
                                kv_format="fp8_pow2_group128") == [
            "kv_format:fp8_pow2_group128 is not served "
            "(serves plain, fp8_per_tensor)"], impl


def test_v32s_sparse_mla_record_is_not_the_gather_floors():
    """The gather floor reads the latent at the `kv` dtype with no scale and
    no bf16 tail: not the 656-byte record V3.2's cell names."""
    for inv in _cells("deepseek-ai/DeepSeek-V3.2", "vllm", "decode",
                      "flash_dsa", "attn_apply_latent"):
        assert inv.kv_format == "fp8_pow2_group128", inv.dtype_env
        assert _claimed(inv, "torch") == set(), inv.dtype_env


def test_smoke_meets_a_plain_admitting_kernel_with_the_plain_cell():
    from kernel_bench.smoke import cell_dtype_env
    p = next(p for p in all_providers()
             if p.kernel_impl == "flash_attention_v3_kvcache")
    env = cell_dtype_env(p, "", "bf16")
    assert "kv_format" not in env and p.serves.accepts(Invocation(
        op=p.signature, signature=p.signature, phase="decode",
        dims={"B": 1, "s_query": 1, "s_kv": 8, "n_heads": 2,
              "n_kv_heads": 1, "d_head": 64, "kv_block_size": 16}, tp=1,
        dtype_env=env)), env


def test_a_record_writing_floor_serves_the_records_its_oracle_grid_places():
    """The torch floors write the record the cell names through
    `rows_on_record`; a record it cannot place (per-tensor, a transform) is
    not theirs."""
    assert _format_refusals("torch_fused_norm_rope", output="fp8_e4m3",
                            output_format="fp8_ue8m0_group32") == []
    assert _format_refusals("torch_fused_norm_rope", output="fp8_e4m3",
                            output_format="fp8_per_tensor") == [
        "output_format:fp8_per_tensor is not served "
        "(serves plain, fp8_group128, fp8_pow2_group128, "
        "fp8_pow2_hadamard128, fp8_ue8m0_group32, "
        "fp8_ue8m0_group64_bf16_rope_tail, mxfp4, nvfp4)"]


# --- The dense MLA decode over a per-tensor FP8 latent ----------------------

def _mla_decode_cell(**over):
    """A dense MLA decode cell at FlashMLA's geometry and page, its latent
    cache a per-tensor FP8 record unless `over` says otherwise."""
    dims = {"B": 2, "s_query": 1, "s_kv": 300, "n_heads": 16,
            "d_head": 576, "d_v_head": 512, "kv_block_size": 64,
            **over.pop("dims", {})}
    env = {"activation": "bf16", "output": "bf16", "kv": "fp8_e4m3",
           "kv_format": "fp8_per_tensor", **over.pop("env", {})}
    env = {k: v for k, v in env.items() if v is not None}
    return Invocation(op="mla_attention", signature="mla_attention",
                      phase=over.pop("phase", "decode"), dims=dims, tp=1,
                      dtype_env=env)


def test_flashmlas_fp8_decode_claims_the_per_tensor_fp8_latent_only():
    """vLLM's FlashMLA backend reads an `--kv-cache-dtype fp8` latent through
    `flash_mla_with_kvcache_fp8`: e4m3, one descale a tensor, 576 / 512
    wide, 64-token pages, decode. A bf16 cache is the plain entry point's,
    a grouped record (`fp8_ds_mla`) the sparse kernel's, and another
    geometry or phase no launch of this kernel."""
    fp8 = next(p for p in all_providers()
               if p.kernel_impl == "vllm_flashmla_decode_fp8")
    assert fp8.serves.accepts(_mla_decode_cell())
    assert fp8.serves.accepts(_mla_decode_cell(dims={"s_query": 2}))
    declined = (
        _mla_decode_cell(env={"kv": "bf16", "kv_format": None}),
        _mla_decode_cell(env={"kv_format": None}),
        _mla_decode_cell(env={"kv_format": "fp8_pow2_group128"}),
        _mla_decode_cell(dims={"d_head": 192, "d_v_head": 128}),
        _mla_decode_cell(phase="prefill"),
    )
    for inv in declined:
        assert not fp8.serves.accepts(inv), inv


def test_a_per_tensor_fp8_mla_decode_cell_reaches_flashmlas_fp8_launch():
    """Under `kv_fp8` a dense-MLA model's decode attention reads the latent
    as `fp8_per_tensor`. The plain builds (bf16 latent, no descale) decline
    it, and vLLM's FlashMLA FP8 launch — the one vLLM runs for it on an
    H100 — claims it; the bf16 cell of the same model keeps the plain
    builds."""
    plain = {"flashmla_decode", "flashinfer_mla_decode",
             "cutlass_mla_decode"}
    model = "deepseek-ai/DeepSeek-V2-Lite"
    for inv in _cells(model, "vllm", "decode", "flash_attention",
                      "attn_apply_latent", precision={"preset": "kv_fp8"}):
        assert (inv.kv_dtype, inv.kv_format) == (
            "fp8_e4m3", "fp8_per_tensor"), inv.dtype_env
        claimed = {p.kernel_impl for p in all_providers()
                   if p.serves is not None and p.serves.accepts(inv)}
        assert claimed == {"vllm_flashmla_decode_fp8"}, claimed
    for inv in _cells(model, "vllm", "decode", "flash_attention",
                      "attn_apply_latent"):
        claimed = {p.kernel_impl for p in all_providers()
                   if p.serves is not None and p.serves.accepts(inv)}
        assert claimed == plain, claimed
