"""The precision profile: how rules resolve, and what a profile changes.

Domain facts only: the §3.6 resolution examples of the design, and the
byte consequences a profile has on real checkpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import cache as cache_mod
from backend.arch.eval import capacity as capacity_mod
from backend.arch.eval.precision import (AS_SHIPPED, PRESETS, PrecisionError,
                                         PrecisionProfile)
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program, shared_walk


def _analyze(name: str, profile=None):
    return shared_walk(Deployment(shared_program(name),
                      precision=profile).at(batch=1, input_tokens=1))


def _weights(rules):
    return PrecisionProfile.canonicalize({"weights": rules})


# -- resolution (§3.6) ------------------------------------------------------

def test_same_selection_written_two_ways_is_one_rule():
    a = _weights([{"select": {"roles": "weight_class"},
                   "format": {"scheme": "fp8"}}])
    b = _weights([{"select": {"roles": ["attention", "attention_swa",
                                        "attention_indexer", "ffn_dense",
                                        "ffn_moe"]},
                   "format": {"scheme": "fp8", "block_h": 128}}])
    assert a.key == b.key


def test_more_specific_rule_wins_where_both_select():
    profile = _weights([
        {"select": {"roles": "weight_class"}, "format": {"scheme": "fp8"}},
        {"select": {"roles": ["ffn_moe"]},
         "format": {"scheme": "nvfp4", "activation": "w4a4"}}])
    assert len(profile.weights) == 2


def test_crossing_rules_with_different_formats_are_refused_at_their_meet():
    with pytest.raises(PrecisionError) as refused:
        _weights([
            {"select": {"roles": ["ffn_moe"]},
             "format": {"scheme": "nvfp4", "activation": "w4a4"}},
            {"select": {"stacks": ["aux:mtp"]}, "format": {"dtype": "BF16"}}])
    assert refused.value.errors[0]["intersection"] == {
        "roles": ["ffn_moe", "ffn_moe_latent", "ffn_moe_shared"],
        "stacks": ["aux:mtp"]}


def test_crossing_rules_with_one_format_gain_their_meet():
    profile = _weights([
        {"select": {"roles": ["ffn_moe"]}, "format": {"scheme": "fp8"}},
        {"select": {"stacks": ["aux:mtp"]}, "format": {"scheme": "fp8"}}])
    derived = [r for r in profile.weights if r.derived]
    assert [dict(r.sigma) for r in derived] == [
        {"roles": ("ffn_moe", "ffn_moe_latent", "ffn_moe_shared"),
         "stacks": ("aux:mtp",)}]


def test_activation_option_is_one_the_scheme_declares():
    with pytest.raises(PrecisionError):
        _weights([{"select": {"roles": "weight_class"},
                   "format": {"scheme": "awq", "activation": "w4a4"}}])


def test_a_storage_format_is_refused_for_token_ids():
    with pytest.raises(PrecisionError):
        PrecisionProfile.canonicalize({"cache": [
            {"select": {"slots": ["token_window"]}, "format": "fp8_per_tensor"}]})


def test_presets_have_stable_distinct_keys():
    keys = {pid: p.key for pid, p in PRESETS.items()}
    assert len(set(keys.values())) == len(keys)
    assert AS_SHIPPED.is_as_shipped
    assert PrecisionProfile.from_request(None) == AS_SHIPPED


# -- what a profile changes -------------------------------------------------

def _cache_bytes(analysis, kind):
    return sum(t.element_bytes * float(t.elements)
               for entry in cache_mod.derive(analysis).values.values()
               for t in entry.tensors if t.kind == kind)


def test_kv_fp8_halves_the_kv_slots_and_keeps_the_recurrent_state():
    name = "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    shipped = _analyze(name)
    fp8 = _analyze(name, PRESETS["kv_fp8"])
    assert _cache_bytes(fp8, "kv") * 2 == _cache_bytes(shipped, "kv")
    assert (_cache_bytes(fp8, "recurrent_state")
            == _cache_bytes(shipped, "recurrent_state"))


def test_a_declared_fp8_indexer_key_carries_its_group_scale():
    analysis = _analyze("deepseek-ai__DeepSeek-V3.2")
    keys = [t for entry in cache_mod.derive(analysis).values.values()
            for t in entry.tensors if t.kind == "index_key"]
    assert keys
    for key in keys:
        # One F32 scale per 128 channels of the 128-wide key, per position,
        # stored with that position: its size is the data and the scale.
        assert key.scale_position_bytes == 4
        assert key.position_bytes == key.data_position_bytes + 4
        assert key.bytes_at(1000) == (key.data_position_bytes + 4) * 1000


def test_a_checkpoint_wider_than_16_bits_is_served_at_bfloat16():
    """gemma-2 stores F32 and neither framework runs it at F32, so as
    shipped is as served (`dtypes.served_dtype`): the checkpoint keeps its
    stored format and every object of the walk runs 2 bytes wide."""
    analysis = _analyze("google__gemma-2-9b")
    precision = analysis.resolved_precision
    assert precision.shipped_model_dtype == "F32"     # what the file holds
    assert precision.model_dtype == "BF16"            # what is loaded
    assert precision.served_as_stored is False
    # Edges, cache and weights, all three from the one resolution.
    assert precision.default_edge_bytes == 2.0
    assert {t.dtype for entry in cache_mod.derive(analysis).values.values()
            for t in entry.tensors} == {"BF16"}
    reps = [row.rep for row in analysis.tensors()]
    assert {rep.saved.storage_dtype for rep in reps} == {"F32"}
    assert {rep.resident.storage_dtype for rep in reps} == {"BF16"}
    # And the file is still the file: the checkpoint projection reports the
    # stored width, or the byte gate compares the loaded one against the
    # header and calls a correct spec wrong by half.
    assert {row.rep.resident.storage_dtype
            for row in analysis.checkpoint_tensors()
            if row.rep is not None} == {"F32"}


def test_a_16_bit_checkpoint_is_served_exactly_as_it_is_stored():
    """The other half of the law, and the half a blanket downcast would
    break: Llama-2 ships F16, `llama` is in neither framework's
    float16-unsupported list, so nothing converts and no tensor carries an
    intent at all."""
    analysis = _analyze("meta-llama__Llama-2-7b-hf")
    precision = analysis.resolved_precision
    assert (precision.shipped_model_dtype, precision.model_dtype) == ("F16",
                                                                      "F16")
    assert precision.served_as_stored is True
    assert precision.weight_intent is None
    assert all(row.rep.intent is None for row in analysis.tensors())


def test_nvfp4_weights_count_their_scales_and_leave_the_embedding():
    shipped = _analyze("Qwen__Qwen3-32B")
    fp4 = _analyze("Qwen__Qwen3-32B", PRESETS["w_nvfp4_w4a4_kv_fp8"])
    report = fp4.precision_effective()
    rows = {(r["category"], r["selector"]): r for r in report["effective"]}
    assert rows[("weights", "outside_roles")]["changed"] is False
    assert rows[("weights", "ffn_dense")]["changed"] is True
    shipped_w, fp4_w = capacity_mod.weights(shipped), capacity_mod.weights(fp4)
    assert float(shipped_w.metadata) == 0 < float(fp4_w.metadata)
    # 31.2 G matrix parameters at half a byte, an E4M3 scale per 16, and
    # the BF16 embedding and LM head (§7.6): about 20.7 GB.
    assert 20.0 < float(fp4_w.resident) / 1e9 < 21.5


def test_as_shipped_reports_nothing_overridden():
    report = _analyze("deepseek-ai__DeepSeek-V3", AS_SHIPPED).precision_effective()
    assert report["overridden"] is False
    report = _analyze("deepseek-ai__DeepSeek-V3",
                      PRESETS["kv_fp8"]).precision_effective()
    assert report["overridden"] is True


def test_a_canonical_profile_sent_back_is_the_same_profile():
    """The server's canonical JSON round-trips: flags, rules and key."""
    first = _weights([
        {"select": {"roles": ["ffn_moe"]}, "format": {"scheme": "fp8"}},
        {"select": {"stacks": ["aux:mtp"]}, "format": {"scheme": "fp8"}}])
    again = PrecisionProfile.from_request({"canonical": first.canonical()})
    assert again.canonical() == first.canonical()
    assert again.key == first.key
    for preset in PRESETS.values():
        assert PrecisionProfile.from_request(
            {"canonical": preset.canonical()}).key == preset.key
