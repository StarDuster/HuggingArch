"""`rekey_lake.rekeyed`: an old row takes today's key only when that key is
the old one with slots dropped, and only one such key."""
import dataclasses
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernel_bench.rekey_lake import granularity_of, rekeyed

OLD = '{"activation":"bf16","kv":"bf16","metadata":"fp32","output":"bf16","weight":"bf16"}'


def test_slots_dropped_is_rekeyed():
    new = '{"activation":"bf16","output":"bf16","weight":"bf16"}'
    assert rekeyed(OLD, {new}) == (new, "")


def test_a_slot_the_old_key_states_otherwise_is_not():
    # A router bias the leaf keeps in fp32 was stored as bf16: another kernel.
    assert rekeyed(OLD, {'{"activation":"bf16","output":"bf16","weight":"fp32"}'}) == (None, "disagrees")


def test_a_slot_the_old_key_never_stated_is_not():
    new = ('{"activation":"bf16","metadata":"fp32","output":"bf16","weight":"bf16",'
           '"weight_granularity":"per-tile:128x128"}')
    assert rekeyed(OLD, {new}) == (None, "disagrees")


def test_two_agreeing_keys_are_ambiguous():
    assert rekeyed(OLD, {'{"activation":"bf16","output":"bf16"}',
                         '{"activation":"bf16","output":"bf16","weight":"bf16"}'}) == (None, "ambiguous")


def test_a_node_no_plan_has_is_left():
    assert rekeyed(OLD, set()) == (None, "no_node")


# The filler `output`: every row before the walk stated a leaf's output was
# stored at `output=bf16`. An fp16 model's row takes the fp16 its launch wrote.
FP16_OLD = '{"activation":"fp16","kv":"fp16","metadata":"fp32","output":"bf16","weight":"fp16"}'
FP16_NEW = '{"activation":"fp16","output":"fp16","weight":"fp16"}'


def test_the_filler_output_takes_what_the_launch_wrote():
    assert rekeyed(FP16_OLD, {FP16_NEW}, written="fp16") == (FP16_NEW, "output")


def test_the_filler_output_stays_when_the_launch_read_the_slot():
    # A provider that declares its output was handed the filler and wrote it.
    assert rekeyed(FP16_OLD, {FP16_NEW}, written=None) == (None, "disagrees")


def test_the_filler_output_stays_when_the_launch_wrote_another_dtype():
    # A cache record the launch quantises to only when the slot asks for it.
    old = '{"activation":"bf16","kv":"fp8_e4m3","output":"bf16"}'
    assert rekeyed(old, {'{"activation":"bf16","output":"fp8_e4m3"}'},
                   written="bf16") == (None, "disagrees")


def test_an_output_that_was_no_filler_does_not_change():
    old = FP16_OLD.replace('"output":"bf16"', '"output":"fp32"')
    assert rekeyed(old, {FP16_NEW}, written="fp16") == (None, "disagrees")


def test_no_other_slot_changes_beside_the_output():
    new = '{"activation":"fp16","output":"fp16","weight":"fp32"}'
    assert rekeyed(FP16_OLD, {new}, written="fp16") == (None, "disagrees")


def test_a_key_that_only_drops_slots_wins_over_the_output_rewrite():
    same = '{"activation":"fp16","output":"bf16","weight":"fp16"}'
    assert rekeyed(FP16_OLD, {same, FP16_NEW}, written="fp16") == (same, "")


def test_what_a_launch_writes_is_read_off_its_provider():
    from kernel_bench.providers import all_providers
    from kernel_bench.rekey_lake import written_by
    providers = [p for p in all_providers() if p.serves is not None]
    declared = next(p for p in providers if p.serves.output)
    plain = next(p for p in providers
                 if not p.serves.output and p.computes_at is None)
    assert written_by(declared.kernel_impl, FP16_OLD) is None
    assert written_by(plain.kernel_impl, FP16_OLD) == "fp16"
    assert written_by("no_such_kernel", FP16_OLD) is None
    # A launch that computes in another dtype declares what it writes
    # (registration requires it), so it read the slot like any declarer.
    dequantising = next(p for p in providers
                        if (p.computes_at or {}).get("activation"))
    assert dequantising.serves.output
    fp8 = '{"activation":"fp8_e4m3","output":"bf16","weight":"fp8_e4m3"}'
    assert written_by(dequantising.kernel_impl, fp8) is None


# `kv_format`: a row's `kv` said an FP8 cache before the key named its record.
V4_OLD = '{"activation":"bf16","kv":"fp8_e4m3","output":"bf16"}'
V4_NEW = ('{"activation":"bf16","kv":"fp8_e4m3",'
          '"kv_format":"fp8_ue8m0_group64_bf16_rope_tail","output":"bf16"}')


def test_the_record_a_provider_declares_is_added():
    assert rekeyed(V4_OLD, {V4_NEW},
                   record="fp8_ue8m0_group64_bf16_rope_tail") == (
        V4_NEW, "kv_format")


def test_no_declared_record_leaves_the_row():
    # A floor that names no record wrote a plain cast, or read the slot.
    assert rekeyed(V4_OLD, {V4_NEW}) == (None, "disagrees")


def test_another_declared_record_leaves_the_row():
    assert rekeyed(V4_OLD, {V4_NEW}, record="fp8_per_tensor") == (
        None, "disagrees")


def test_the_record_a_launch_touches_is_read_off_its_provider():
    from kernel_bench.rekey_lake import record_of
    assert record_of("vllm_c_fused_qnorm_rope_kv_insert") == (
        "fp8_ue8m0_group64_bf16_rope_tail")
    assert record_of("flashmla_sparse_fp8") == "fp8_pow2_group128"
    # Declares none, or several: it chose by the slot, or never read it.
    assert record_of("torch_fused_qnorm_rope_kv_insert") is None
    assert record_of("deepgemm_fp8_paged_mqa_logits") is None


# `activation_granularity`: the H100 row shapes below are one real row of
# each model named by the review (DeepSeek-V3, GLM-5.3 and MiMo-V2-Flash),
# each measured by DeepGEMM's masked grouped implementation before the key
# stated the launch's 128-wide activation-scale group.
DEEPGEMM_OLD = json.dumps(
    {"activation": "fp8_e4m3", "metadata": "fp32", "output": "bf16",
     "weight": "fp8_e4m3", "weight_granularity": "per-tile:128x128"},
    sort_keys=True, separators=(",", ":"))
H100_MASKED_ROWS = [
    ("deepseek-ai/DeepSeek-V3",
     {"d_out": 128, "d_reduce": 7168, "n_experts": 256, "n_tokens": 1024,
      "s_query": 4}),
    ("zai-org/GLM-5.3",
     {"d_out": 128, "d_reduce": 6144, "n_experts": 144, "n_tokens": 144,
      "s_expert_rows": 1}),
    ("XiaomiMiMo/MiMo-V2-Flash",
     {"d_out": 128, "d_reduce": 4096, "n_experts": 128, "n_tokens": 128,
      "s_expert_rows": 1}),
]


@pytest.mark.parametrize("_model, dims", H100_MASKED_ROWS)
def test_a_masked_h100_row_takes_its_provider_granularity_and_is_served(
        _model, dims):
    from kernel_bench.providers import all_providers
    from tools.unserved_rows import _verdict

    granularity = granularity_of("deepgemm_fp8_m_grouped_masked",
                                  DEEPGEMM_OLD)
    assert granularity == "per-group:128"
    new = json.dumps(
        {**json.loads(DEEPGEMM_OLD),
         "activation_granularity": granularity},
        sort_keys=True, separators=(",", ":"))
    assert rekeyed(DEEPGEMM_OLD, {new}, granularity=granularity) == (
        new, "activation_granularity")
    # The recorded launch, not today's graph, licenses this key-vocabulary
    # migration; a renamed or removed node must not strand the measurement.
    assert rekeyed(DEEPGEMM_OLD, set(), granularity=granularity) == (
        new, "activation_granularity")

    row = {"framework": "deepgemm",
           "kernel_impl": "deepgemm_fp8_m_grouped_masked",
           "op": "grouped_gemm", "phase": "decode",
           "shape_json": json.dumps(dims, sort_keys=True,
                                    separators=(",", ":")),
           "dtype_env_json": DEEPGEMM_OLD}
    providers = list(all_providers())
    assert _verdict(row, providers) == (
        "refused",
        "activation_granularity:- is not served (serves per-group:128)")
    row["dtype_env_json"] = new
    assert _verdict(row, providers) == ("served", "")


def test_a_key_that_already_states_a_non_128_granularity_is_not_migrated():
    # An MXFP4 grouped cell already says per-group:32. The FP8xFP4 launch's
    # own declaration is per-group:128; rekey must not replace the stated
    # value or manufacture a candidate for a key that launch refuses.
    old = json.dumps(
        {"activation": "fp8_e4m3",
         "activation_granularity": "per-group:32", "output": "bf16",
         "weight": "mxfp4", "weight_granularity": "per-group:32"},
        sort_keys=True, separators=(",", ":"))
    new = json.loads(old)
    new["activation_granularity"] = "per-group:128"
    new = json.dumps(new, sort_keys=True, separators=(",", ":"))
    assert granularity_of("deepgemm_fp8_fp4_m_grouped_contiguous", old) is None
    assert rekeyed(old, {new}) == (None, "disagrees")


def test_a_new_key_that_already_carries_the_slot_is_left_alone():
    new = json.dumps(
        {**json.loads(DEEPGEMM_OLD),
         "activation_granularity": "per-group:128"},
        sort_keys=True, separators=(",", ":"))
    assert rekeyed(new, {new},
                   granularity=granularity_of(
                       "deepgemm_fp8_m_grouped_masked", new)) == (new, "")


def test_several_declared_granularities_refuse_the_migration(monkeypatch):
    from kernel_bench import rekey_lake
    from kernel_bench.providers import all_providers

    provider = next(p for p in all_providers()
                    if p.kernel_impl == "deepgemm_fp8_m_grouped_masked")
    ambiguous = dataclasses.replace(
        provider,
        serves=dataclasses.replace(
            provider.serves,
            activation_granularity=("per-group:64", "per-group:128")))
    monkeypatch.setattr(rekey_lake, "_provider", lambda _name: ambiguous)
    with pytest.raises(ValueError, match=(
            "ambiguous activation_granularity declaration for "
            "deepgemm_fp8_m_grouped_masked: serves per-group:64, "
            "per-group:128")):
        granularity_of("deepgemm_fp8_m_grouped_masked", DEEPGEMM_OLD)
