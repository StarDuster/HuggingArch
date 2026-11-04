"""A launch that writes a cached tensor as its output is keyed by that
tensor's format in its `output` slot, and not again under `kv`.

sglang runs DeepSeek-V4's latent-KV norm + rope + store as one launch
(`fused_k_norm_rope_flashmla`, plan `fused_norm_rope`) that writes the
FlashMLA FP8 record. Its cell used to say `output: bf16` — no walk fact
stated a row's output dtype — and carried the record's dtype as `kv`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.precision import AS_SHIPPED, ResolvedPrecision
from backend.arch.eval.program import Program
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.framework_bindings import (ServingDeployment,
                                                  get_binding)
from backend.inference.gpu_specs import get_gpu
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_loader


def test_the_latent_kv_store_launch_outputs_the_fp8_record():
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4-Flash",
                                shared_loader())
    binding = get_binding("sglang").deployed(ServingDeployment.at(
        gpu_sku="B200", activation_dtype=ResolvedPrecision(
            AS_SHIPPED, Program.of(resolved)).model_dtype))
    result = analyze(resolved, parallel=ParallelConfig(), avg_input_tokens=64,
                     avg_output_tokens=0, phase="decode", batch_size=8,
                     gpu=get_gpu("B200"), framework_binding=binding,
                     breakdown=True)
    identity = RowIdentity(result)
    stores = [row for row in result.per_op_timing
              if row["op"] == "fused_norm_rope"
              and row["aggregation_id"].split("|")[-1].startswith(
                  "attn.kv_concat")]
    assert stores
    for row in stores:
        env = json.loads(identity.leaf_dtype_env(row)[0])
        assert env["output"] == "fp8_e4m3", row["aggregation_id"]
        assert "kv" not in env, row["aggregation_id"]


def _rows(model, framework, *, precision=None, phase="decode"):
    resolved = resolve_snapshot(model, shared_loader())
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="B200", activation_dtype=ResolvedPrecision(
            AS_SHIPPED, Program.of(resolved)).model_dtype))
    result = analyze(resolved, parallel=ParallelConfig(), avg_input_tokens=64,
                     avg_output_tokens=0 if phase == "prefill" else 64,
                     phase=phase, batch_size=8, gpu=get_gpu("B200"),
                     framework_binding=binding, breakdown=True,
                     precision=precision)
    identity = RowIdentity(result)
    return [(row, *identity.leaf_dtype_env(row))
            for row in result.per_op_timing]


def test_the_latent_kv_store_names_the_flashmla_record():
    """FP8 E4M3 is what the record's elements are; how its scales are laid
    out (one UE8M0 per 64, the last 64 channels BF16) is the record, which
    an FP8 index key with one F32 scale shares the dtype with. The key
    states it by registry name, and only on a row that writes a cached
    tensor as its output."""
    rows = _rows("deepseek-ai/DeepSeek-V4-Flash", "sglang")
    stores = [(row, json.loads(env)) for row, env, _ in rows
              if row["op"] == "fused_norm_rope"
              and row["aggregation_id"].split("|")[-1].startswith(
                  "attn.kv_concat")]
    assert stores
    for row, env in stores:
        assert env["output_format"] == "fp8_ue8m0_group64_bf16_rope_tail", (
            row["aggregation_id"])
        assert env["output"] == "fp8_e4m3"
    # Rule 3: a slot that does not apply is omitted — every row whose
    # output is an activation keys no format.
    plain = [json.loads(env) for row, env, _ in rows
             if row.get("output_cache") is None]
    assert plain and not any("output_format" in env for env in plain)


def test_the_v41_index_key_names_its_mxfp4_record():
    """DeepSeek-V4.1 caches its index key MXFP4 (`index_key: mxfp4`): the
    element dtype is FP4 E2M1, which NVFP4 shares; the record says which."""
    rows = _rows("deepseek-ai/DeepSeek-V4.1-Flash", "vllm", phase="prefill")
    keys = [json.loads(env) for row, env, _ in rows
            if row["op"] == "fused_compress_norm_rope_store"
            and ".indexer.k_cache" in row["aggregation_id"]]
    assert keys
    for env in keys:
        assert (env["output"], env["output_format"]) == ("fp4_e2m1", "mxfp4")


def test_a_cache_stored_plain_keys_no_format_and_an_unnamed_one_declines():
    """A bf16 cache's dtype is its whole format: its writers key none (the
    MLA latent's norm). A profile may spell out a format the registry does
    not name; such a record has no key, and its writers are declined with
    the reason rather than keyed as some other record."""
    from backend.arch.precision_profile import PrecisionProfile
    model = "deepseek-ai/DeepSeek-V2-Lite"
    shipped = _rows(model, "vllm")
    writers = [env for row, env, _ in shipped if row.get("output_cache")]
    assert writers
    assert not any("output_format" in json.loads(env) for env in writers)
    named = PrecisionProfile.from_request({"preset": "kv_fp8"})
    keyed = [(json.loads(env), why)
             for row, env, why in _rows(model, "vllm", precision=named)
             if row.get("output_cache")]
    assert keyed and all(
        env.get("output_format") == "fp8_per_tensor" and not why
        for env, why in keyed), keyed
    spelled = PrecisionProfile.from_request({"canonical": {"cache": [{
        "select": {"slots": ["kv", "latent", "rope_key", "index_key",
                             "compressed_kv"]},
        "format": {"dtype": "F8_E4M3",
                   "scale": {"granularity": "per-group", "group": 64,
                             "dtype": "F32"}}}]}})
    declined = [(json.loads(env), why)
                for row, env, why in _rows(model, "vllm", precision=spelled)
                if row.get("output_cache")]
    assert declined
    for env, why in declined:
        assert "output_format" not in env
        assert "no registry cache format names" in why, why


def test_the_cache_beside_the_output_and_its_readers_name_the_record():
    """`kv` says a row touches an FP8 E4M3 cache; `kv_format` says which
    record, by the same naming rule as `output_format`. vLLM's V4 q-norm /
    rope / KV-insert launch writes the FlashMLA record beside the query it
    outputs, and the attention that reads it reads that record; a
    per-tensor-scaled FP8 cache is `kv=fp8_e4m3` too. The MXFP4 index key
    its scorer reads is named the same way; the compressor's fp32 state is
    plain and names none."""
    rows = _rows("deepseek-ai/DeepSeek-V4-Flash", "vllm")
    record = "fp8_ue8m0_group64_bf16_rope_tail"
    by_op = {}
    for row, env, why in rows:
        env = json.loads(env)
        if "kv" in env:
            by_op.setdefault(row["op"], set()).add(
                (env["kv"], env.get("kv_format"), why))
    assert by_op["fused_qnorm_rope_kv_insert"] == {("fp8_e4m3", record, "")}
    assert by_op["flash_attention"] == {("fp8_e4m3", record, "")}
    assert by_op["indexer_logits"] == {("fp4_e2m1", "mxfp4", "")}
    assert by_op["softmax_act"] == {("fp32", None, "")}
    # Rule 3: no cache, no format.
    assert not any("kv_format" in json.loads(env)
                   for _row, env, _ in rows if "kv" not in json.loads(env))


def test_the_dsa_latent_reader_names_the_flashmla_fp8_record():
    """DeepSeek-V3.2's sparse MLA reads its FP8 latent (four power-of-two
    F32 scales a row, `scale_fmt: ue8m0`: FlashMLA's `fp8_ds_mla`) beside a
    bf16 RoPE key; the key's `kv` is the latent, and so is its record."""
    rows = _rows("deepseek-ai/DeepSeek-V3.2", "vllm")
    dsa = {(json.loads(env)["kv"], json.loads(env).get("kv_format"))
           for row, env, _ in rows if row["op"] == "flash_dsa"}
    assert dsa == {("fp8_e4m3", "fp8_pow2_group128")}


def test_a_cache_readers_record_follows_the_precision_profile():
    """A named profile's cache record keys its readers; a spelled-out one
    the registry does not name declines them with the reason, as it
    declines the writers' `output_format`."""
    from backend.arch.precision_profile import PrecisionProfile
    model = "deepseek-ai/DeepSeek-V2-Lite"
    shipped = [json.loads(env) for row, env, _ in _rows(model, "vllm")
               if "kv" in json.loads(env)]
    assert shipped and not any("kv_format" in env for env in shipped)
    named = PrecisionProfile.from_request({"preset": "kv_fp8"})
    readers = [(json.loads(env), why)
               for row, env, why in _rows(model, "vllm", precision=named)
               if "kv" in json.loads(env)]
    assert readers and all(env.get("kv_format") == "fp8_per_tensor"
                           and not why for env, why in readers), readers
    spelled = PrecisionProfile.from_request({"canonical": {"cache": [{
        "select": {"slots": ["kv", "latent", "rope_key", "index_key",
                             "compressed_kv"]},
        "format": {"dtype": "F8_E4M3",
                   "scale": {"granularity": "per-group", "group": 64,
                             "dtype": "F32"}}}]}})
    declined = [(json.loads(env), why)
                for row, env, why in _rows(model, "vllm", precision=spelled)
                if "kv" in json.loads(env)]
    assert declined
    for env, why in declined:
        assert "kv_format" not in env
        assert "its cache is stored as" in why, why


def _envs(model, framework, op, **kwargs):
    return [(row["aggregation_id"], json.loads(env))
            for row, env, _ in _rows(model, framework, **kwargs)
            if row["op"] == op]


def test_the_glm_indexer_query_launch_writes_the_fp8_record():
    """The index-key cache's readers multiply its rows with a query
    quantised to their element type (`index_key.paired_operand`): the graft
    stands that cast on the query, and sglang's rope-first launch takes it
    in, so its cell keys the FP8 record it writes — one F32 scale a head
    row — not the BF16 edge every provider declined."""
    first = _envs("zai-org/GLM-5.3", "sglang",
                  "fused_indexer_q_rope_first_quant")
    assert first
    for where, env in first:
        assert (env["output"], env["activation_granularity"],
                env["metadata"]) == ("fp8_e4m3", "per-token", "fp32"), where
    # The scorer reads that record: its operand is FP8, as DeepGEMM's.
    logits = _envs("zai-org/GLM-5.3", "sglang", "indexer_logits")
    assert logits and all(env["activation"] == "fp8_e4m3"
                          for _where, env in logits)


def test_the_v4_indexer_query_follows_how_the_index_key_is_stored():
    """DeepSeek-V4 ships its index key MXFP4, so its query launch writes
    MXFP4 (vLLM's `use_fp4` form); a rule storing the key FP8 — both
    stacks' default — makes it the FP8 launch. One fact, the cache's
    format, decides both, under every framework."""
    from backend.arch.precision_profile import PrecisionProfile
    for fw in ("sglang", "vllm"):
        quant = _envs("deepseek-ai/DeepSeek-V4-Flash", fw,
                      "fused_indexer_q_rope_quant")
        assert quant, fw
        for where, env in quant:
            assert (env["output"], env["activation_granularity"]) == (
                "fp4_e2m1", "per-group:32"), where
    fp8_key = PrecisionProfile.from_request({"canonical": {"cache": [{
        "select": {"slots": ["index_key"]}, "format": "fp8_group128"}]}})
    quant = _envs("deepseek-ai/DeepSeek-V4-Flash", "sglang",
                  "fused_indexer_q_rope_quant", precision=fp8_key)
    assert quant
    for where, env in quant:
        assert (env["output"], env["activation_granularity"],
                env["metadata"]) == ("fp8_e4m3", "per-token", "fp32"), where


def test_the_v41_split_query_launch_writes_mxfp4():
    split = _envs("deepseek-ai/DeepSeek-V4.1-Flash", "vllm",
                  "fused_indexer_q_rope_quant_split")
    assert split
    for where, env in split:
        assert (env["output"], env["activation_granularity"]) == (
            "fp4_e2m1", "per-group:32"), where


def test_the_glm_index_key_store_names_the_record_the_kernel_writes():
    """sglang's fused K-store kernel writes `amax/448` fp32 scales
    (`indexer_k.cuh:330`), the `fp8_group128` record — the cell keys it, not
    the power-of-two grid the unfused fallback rounds to."""
    stores = _envs("zai-org/GLM-5.3", "sglang",
                   "fused_indexer_k_norm_rope_store")
    assert stores
    for where, env in stores:
        assert (env["output"], env["output_format"]) == (
            "fp8_e4m3", "fp8_group128"), where
