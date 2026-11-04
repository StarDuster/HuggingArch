"""`SpecLoader.synthesize_block` is lowering's block synthesis, public: it
yields the name and graph an author writing that binding would get, and it
never writes into the spec (or the loader) it was handed."""
from __future__ import annotations

import pytest

from backend.arch.loader import SpecLoader
from tests.snapshots import resolve_snapshot

_MOE = "pre_norm__attn-mla_attention_ffn-moe_ffn_gate_routing_bias_norm-rmsnorm"


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return SpecLoader()


def test_rebinding_a_param_gives_the_authored_name(loader):
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    spec = base.spec
    before = (dict(spec.blocks), dict(spec.components), dict(spec.component_roles))
    moe = spec.get_block(_MOE)
    params = {**moe.synth_params, "attn": "mla_output_gate_attention"}
    new_spec, name = loader.synthesize_block(spec, moe.synth_template, params)
    assert name == ("pre_norm__attn-mla_output_gate_attention_ffn-"
                    "moe_ffn_gate_routing_bias_norm-rmsnorm")
    block = new_spec.get_block(name)
    assert block.synth_params == params
    assert block.graph["self_attn"].op == "mla_output_gate_attention"
    assert new_spec.component_roles["mla_output_gate_attention"] == "attention"
    # nothing written back: the handed spec and the loader's own copy
    assert (dict(spec.blocks), dict(spec.components),
            dict(spec.component_roles)) == before
    assert name not in (loader.get(spec.name).blocks or {})
    assert spec.get_block(_MOE) is moe


def test_an_existing_binding_is_reused(loader):
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    moe = base.spec.get_block(_MOE)
    new_spec, name = loader.synthesize_block(
        base.spec, moe.synth_template, dict(moe.synth_params))
    assert name == _MOE
    assert new_spec.get_block(name) is moe


def test_registration_keeps_a_role_the_spec_gave(loader):
    # gpt-oss binds swa_bias_sink_attention as attention_swa; synthesizing a
    # block that binds it must not reset it to the library's `attention`.
    base = resolve_snapshot("openai/gpt-oss-20b", loader)
    swa = next(b for n, b in base.spec.blocks.items()
               if b.synth_params.get("attn") == "swa_bias_sink_attention")
    new_spec, name = loader.synthesize_block(
        base.spec, swa.synth_template, {**swa.synth_params, "ffn": "swiglu_ffn"})
    assert name not in base.spec.blocks
    assert new_spec.component_roles["swa_bias_sink_attention"] == "attention_swa"
    assert new_spec.component_roles["swiglu_ffn"] == "ffn_dense"
