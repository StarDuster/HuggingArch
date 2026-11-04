"""A `derived:` name is never a `dims:` key.

`n_expert_copies` (= `n_experts`) and `n_route_targets`
(= `n_experts + n_zero_experts`) are computed in every frame
(`Binding.frame`), so a spec that writes one states a second value for a fact
the walk already derives — which is how DeepSeek-V4.1-Flash's draft stages
came to say the same 128 four times. The loader refuses the key wherever a
`dims:` map is written: the model's, a component call's, a primitive's and a
block assignment's. The refusal names what to bind instead, and reaches the
spec writer's `validate` as an `errors[]` entry it can repair.

A symbol of `kind: expert_copies` is the expert axis, and the walk derives
it: a spec may not declare a second, writable one (DeepSeek-V4.1-Flash's
retired `n_dspark_expert_copies`).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from backend.arch import dims
from backend.arch import loader as loader_mod
from backend.arch.loader import SpecLoader, SpecLoadError
from tests.backend.arch.eval.test_frame_rederivation import (
    COMPONENTS, CONFIG, MODEL, PRIMITIVE_REBINDS)

DERIVED = ("n_expert_copies", "n_route_targets")


def _load(model: str, primitive: str = PRIMITIVE_REBINDS):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        (base / "components" / "library"
         / "frames_toy_primitive_rebinds.yaml").write_text(primitive)
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "frames_toy.yaml").write_text(model)
        return SpecLoader(base_dir=str(base)).match(CONFIG, model_id="frames_toy")


def _written(scope: str, name: str):
    """The synthetic spec with `name` written as a key at `scope`."""
    model, primitive = MODEL, PRIMITIVE_REBINDS
    if scope == "model":
        model = model.replace("  n_experts: config.num_experts\n",
                              f"  n_experts: config.num_experts\n  {name}: n_experts\n", 1)
    elif scope == "component call":
        model = model.replace(
            "dims: { n_experts: n_draft_experts, d_ffn_expert: d_draft_ffn }",
            f"dims: {{ n_experts: n_draft_experts, d_ffn_expert: d_draft_ffn, {name}: n_draft_experts }}", 1)
    elif scope == "primitive":
        primitive = primitive.replace(
            "dims: { n_experts: n_draft_experts }",
            f"dims: {{ n_experts: n_draft_experts, {name}: n_draft_experts }}", 1)
    elif scope == "block assignment":
        model = model.replace(
            "          dims: { n_experts: n_draft_experts }",
            f"          dims: {{ n_experts: n_draft_experts, {name}: n_draft_experts }}", 1)
    assert (model, primitive) != (MODEL, PRIMITIVE_REBINDS), scope
    return model, primitive


def test_the_synthetic_spec_loads_as_written():
    assert _load(MODEL) is not None


@pytest.mark.parametrize("name", DERIVED)
@pytest.mark.parametrize("scope", ["model", "component call", "primitive",
                                   "block assignment"])
def test_a_derived_name_is_refused_as_a_key(scope, name):
    model, primitive = _written(scope, name)
    with pytest.raises((SpecLoadError, ValueError)) as refused:
        if scope == "primitive":
            # A library component is parsed on its own, by the function the
            # loader uses (a library file that fails to parse is logged and
            # skipped by the loader, as for any other refusal).
            loader_mod._parse_component_document(
                yaml.safe_load(primitive), "frames_toy_primitive_rebinds.yaml")
        else:
            _load(model, primitive)
    message = str(refused.value)
    assert f"`{name}` is derived" in message, message
    # The repair: what to bind instead.
    assert "`n_experts`" in message
    if name == "n_route_targets":
        assert "`n_zero_experts`" in message


def test_an_expert_axis_a_spec_could_write_is_refused():
    with pytest.raises(ValueError, match="kind 'expert_copies' and is not `derived:`"):
        dims.local_symbols({"n_pool_copies": {
            "kind": "expert_copies", "shardable_by": ["moe_copies"],
            "group": "moe_routing", "meaning": "a second expert axis",
            "config": "pool_experts"}})


def test_the_writer_gets_the_refusal_as_an_error_to_repair():
    from backend.arch.validate import validate

    model, _ = _written("model", "n_expert_copies")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "models").mkdir()
        (Path(tmp) / "models" / "frames_toy.yaml").write_text(model)
        with mock.patch("backend.arch.validate._orchestrator._get_safetensors_data",
                        return_value={"total_params": 0, "model_storage_bytes": 0,
                                      "tensors": {}}), \
             mock.patch("backend.arch.validate._orchestrator._fetch_config",
                        return_value=dict(CONFIG)):
            result = validate("frames_toy", spec_name="frames_toy",
                              overlay_dir=tmp)
    assert result["valid"] is False
    joined = "; ".join(str(e) for e in result["errors"])
    assert "`n_expert_copies` is derived" in joined, joined
    assert "Bind `n_experts` instead" in joined, joined


def test_the_retired_router_width_name_is_not_a_dim():
    """`n_router_classes` said the same thing as `n_route_targets` on every
    corpus model and was deleted; a spec that still binds it is told the name
    is not registered, the error the writer repairs any unknown name by."""
    model = MODEL.replace("  n_experts: config.num_experts\n",
                          "  n_experts: config.num_experts\n  n_router_classes: n_experts\n", 1)
    with pytest.raises((SpecLoadError, ValueError),
                       match=r"n_router_classes.*neither.*dims\.yaml"):
        _load(model)
