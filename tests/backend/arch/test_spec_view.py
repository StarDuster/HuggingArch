"""The spec's `view:` block — display facts that are not dims.

Architecture-view fields that are not counts (RoPE base/scaling, the FFN
nonlinearity, the norm epsilon, the SWA-variant head widths) used to be read
off the raw HF config by the renderer, alias by alias. They are now bound in
the spec (`ArchSpec.VIEW_KEYS`), resolved by the loader in their native
types, and the renderer reads `ResolvedSpec.view_values` and nothing else.
The published context length is the dim `s_max`, bound only by specs whose
models publish one.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dims
from backend.arch.loader import ResolvedSpec, SpecLoader
from backend.arch.schema.spec import VIEW_KEYS
from backend.rendering.builders import build_model_config
from tests.snapshots import resolve_snapshot, snapshot_config

MODELS = PROJECT_ROOT / "backend" / "arch" / "models"


def _spec_with_view(tmp_path, view_yaml: str):
    text = (MODELS / "llama.yaml").read_text()
    assert "\nview:" in text
    head, _sep, tail = text.partition("\nview:")
    rest = tail.split("\n\n", 1)[1] if "\n\n" in tail else ""
    (tmp_path / "llama.yaml").write_text(head + "\nview:\n" + view_yaml + "\n\n" + rest)
    return SpecLoader().load_model_spec(str(tmp_path / "llama.yaml"))


def test_a_view_key_outside_the_vocabulary_is_a_load_error(tmp_path):
    with pytest.raises(ValueError, match="view: unknown key"):
        _spec_with_view(tmp_path, "  rope_base: config.rope_theta")


def test_a_view_binding_must_be_a_config_path(tmp_path):
    with pytest.raises(ValueError, match="must be a config path"):
        _spec_with_view(tmp_path, "  norm.eps: 1e-5")


def test_view_values_keep_their_native_types():
    r = resolve_snapshot("deepseek-ai__DeepSeek-V3.json")
    v = r.view_values
    assert isinstance(v["rope_scaling"], dict)
    assert isinstance(v["hidden_act"], str)
    assert isinstance(v["norm.eps"], float)
    assert set(v) <= set(VIEW_KEYS)


def test_a_path_the_config_does_not_publish_is_absent_not_defaulted():
    # Step-3.7-Flash publishes neither `hidden_act` nor a norm epsilon.
    r = resolve_snapshot("stepfun-ai__Step-3.7-Flash.json")
    assert "hidden_act" not in r.view_values
    assert "norm.eps" not in r.view_values
    mc = build_model_config({}, r)
    assert mc["ffn"]["activation"] is None
    assert mc["norm"]["eps"] is None


def test_s_max_is_a_registered_whole_dim_bound_from_the_published_context():
    assert dims.is_registered("s_max")
    r = resolve_snapshot("Qwen__Qwen3-8B.json")
    assert r.dim_values["s_max"] == snapshot_config("Qwen__Qwen3-8B.json")["max_position_embeddings"]
    # GLM-4 publishes its context length as `seq_length`; its spec binds the
    # symbol from the field this config writes.
    r = resolve_snapshot("THUDM__glm-4-9b.json")
    assert r.dim_values["s_max"] == snapshot_config("THUDM__glm-4-9b.json")["seq_length"]


def test_the_renderer_reads_the_view_not_the_config():
    r = resolve_snapshot("Qwen__Qwen3-8B.json")
    cfg = r.config
    assert cfg.get("rope_theta") is not None and cfg.get("hidden_act")
    blind = ResolvedSpec(r.spec, dict(r.dim_values), cfg, {})
    mc = build_model_config({}, blind)
    assert "rope_theta" not in mc["attention"]
    assert mc["ffn"]["activation"] is None
    seeing = build_model_config({}, r)
    assert seeing["attention"]["rope_theta"] == "1M"
    assert seeing["ffn"]["activation"] == cfg["hidden_act"]


def test_model_type_is_read_where_the_match_was_made():
    r = resolve_snapshot("Qwen__Qwen3-8B.json")
    assert r.model_type == "qwen3"
