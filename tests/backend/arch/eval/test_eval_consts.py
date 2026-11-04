"""Function constants: what a clamp clamps to, what beta a SiTU gate runs at.

An op says only THAT its math takes a real number beside its operands
(`OpSpec.consts`: `clamp` takes `clamp_limit`, `situ` takes `situ_beta` and
`situ_linear_beta`). Where a model publishes it is the spec's: its `dims:`
binds the symbol (kind `constant`) to a config field, in the config scope it
lives in, and a node or call site rebinds it with `dims:` — so a vision
tower's clamp reads the tower's bound, not the backbone's. The walk reads
each constant from the node's own frame and records it per node
(`Analysis.consts`); a constant the config does not publish is absent,
never defaulted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.dim_resolution import resolve_dimensions
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.ops import OP_REGISTRY
import backend.arch.dims as dims
from tests.snapshots import shared_loader, snapshot_config

_IMAGE = {"pixel_values": {"n_img": 1, "img_h": 224, "img_w": 224}}


def _deployment(model_id: str, config: dict | None = None) -> Deployment:
    resolved = shared_loader().match(
        config if config is not None else snapshot_config(model_id),
        model_id=model_id)
    return Deployment(Program.of(resolved)).at(input_tokens=128)


def _consts_by_op(analysis) -> dict:
    """`{(op, stack): {constants}}` over the walk, one entry per distinct
    binding — a key with two bindings would collapse to a set of two."""
    out: dict = {}
    for uid, consts in analysis.consts.values.items():
        node = analysis.program.by_uid(uid)
        out.setdefault((node.op, node.stack_name), set()).add(
            tuple(sorted(consts.items())))
    return out


def test_every_declared_constant_is_a_constant_symbol():
    """The op names the symbol; dims.yaml says what it is. A const that is
    not a registered `constant` would be read as an extent."""
    kinds = dims.kinds()
    for name, spec in OP_REGISTRY.items():
        for const in spec.consts:
            assert kinds.get(const) == "constant", (name, const)


@pytest.mark.parametrize("model_id, want", [
    ("deepseek-ai/DeepSeek-V4-Flash", {("clamp_limit", 10.0)}),
    ("zai-org/GLM-5.3-Flash", {("clamp_limit", 10.0)}),
    ("moonshotai/Kimi-K3", {("situ_beta", 4.0), ("situ_linear_beta", 25.0)}),
    ("Qwen/Qwen3-8B", set()),
])
def test_the_backbone_reads_the_spec_binding(model_id, want):
    analysis = _deployment(model_id).analyze()
    got = {pair for bindings in _consts_by_op(analysis).values()
           for binding in bindings for pair in binding}
    assert got == want


def test_a_vision_tower_reads_its_own_scope():
    """GLM-5.3 publishes the bound twice, 10.0 in each scope, so a tower that
    read the backbone's would go unnoticed. Move the tower's and see whose
    each clamp takes — fractional, too, since a bound is no extent."""
    config = snapshot_config("zai-org/GLM-5.3-Flash")
    config["vision_config"]["swiglu_limit"] = 7.5
    deployment = _deployment("zai-org/GLM-5.3-Flash", config).at(media=_IMAGE)
    backbone = _consts_by_op(deployment.analyze())
    assert backbone and all(b == {(("clamp_limit", 10.0),)}
                            for b in backbone.values())
    (vision,) = [e for e in deployment.encoders() if e.phase.stack == "vision"]
    tower = _consts_by_op(vision.analyze())
    assert tower and all(b == {(("clamp_limit", 7.5),)}
                         for b in tower.values())


def test_an_unpublished_constant_is_absent_not_defaulted():
    config = snapshot_config("zai-org/GLM-5.3-Flash")
    del config["vision_config"]["swiglu_limit"]
    deployment = _deployment("zai-org/GLM-5.3-Flash", config).at(media=_IMAGE)
    assert "vis_clamp_limit" not in deployment.program.resolved.dim_values
    (vision,) = [e for e in deployment.encoders() if e.phase.stack == "vision"]
    analysis = vision.analyze()
    clamps = [n for n in analysis.program.nodes() if n.op == "clamp"]
    assert clamps
    assert not any(n.uid in analysis.consts.values for n in clamps)


def test_a_constant_binding_keeps_the_number_and_may_be_unbound():
    config = {"swiglu_limit": 7.5, "hidden_size": 64}
    got = resolve_dimensions(
        {"d_model": "config.hidden_size",
         "clamp_limit": "config.swiglu_limit",
         "situ_beta": "config.activation_situ_beta"}, config)
    assert got["clamp_limit"] == 7.5
    assert "situ_beta" not in got
    # An extent stays held to a whole number.
    with pytest.raises(ValueError, match="whole number"):
        resolve_dimensions({"d_model": "config.swiglu_limit"}, config)
    # A field the config sets to something else is not "unpublished".
    with pytest.raises(ValueError, match="not a number"):
        resolve_dimensions({"clamp_limit": "config.swiglu_limit"},
                           {"swiglu_limit": "auto"})
