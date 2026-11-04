"""`stamp_match_model_id` self-registration + the overlay seed that applies it.

The loader resolves a checkpoint to its spec by model_id (exact lookup, no
model_type scoring). A stored spec_yaml carrying only `match.model_type` — every
generation record written before self-registration existed — must be pinned with
`match.model_id` on the way into any overlay, or review / analyze / refine see
"No spec matched".
"""
from __future__ import annotations

from pathlib import Path

import yaml

from backend.spec_worker.self_register import stamp_match_model_id
from backend.spec_worker.bundle import FormalSource, normalize_bundle


def _match(yaml_text: str):
    return yaml.safe_load(yaml_text)["match"]


def test_stamps_after_model_type():
    src = "match:\n  model_type: [glm4_moe]\n\ncomponents:\n  attention: gqa\n"
    out = stamp_match_model_id(src, "zai-org/GLM-4.5")
    assert _match(out) == {"model_type": ["glm4_moe"], "model_id": ["zai-org/GLM-4.5"]}
    # comment/format preserving elsewhere
    assert "components:" in out


def test_noop_when_model_id_present():
    src = 'match:\n  model_id: ["a/B"]\n  model_type: [x]\n'
    assert stamp_match_model_id(src, "other/id") == src


def test_stamps_when_match_has_no_model_type():
    # A match block keyed on something else still gets pinned (inserted first).
    src = "match:\n  architectures: [FooForCausalLM]\n"
    out = stamp_match_model_id(src, "foo/Bar")
    assert _match(out)["model_id"] == ["foo/Bar"]
    assert _match(out)["architectures"] == ["FooForCausalLM"]


def test_idempotent():
    src = "match:\n  model_type: [glm4_moe]\n"
    once = stamp_match_model_id(src, "zai-org/GLM-4.5")
    assert stamp_match_model_id(once, "zai-org/GLM-4.5") == once


def test_leaves_unparseable_untouched():
    src = "match: {this is: [not, valid, yaml"
    assert stamp_match_model_id(src, "x/y") == src


def test_no_match_block_untouched():
    src = "dims:\n  d_model: config.hidden_size\n"
    assert stamp_match_model_id(src, "x/y") == src


def test_overlay_seed_stamps_model_id(tmp_path, monkeypatch):
    """The overlay writer pins model_id so a model_type-only spec resolves out of
    the overlay — the exact path review/analyze/refine go through."""
    from backend.spec_worker import overlay as _ov

    monkeypatch.setattr(_ov, "user_specs_root", lambda: tmp_path / "user_specs")
    spec = "match:\n  model_type: [glm4_moe]\ncomponents:\n  attention: gqa\n"
    bundle = normalize_bundle(
        "zai-org/GLM-4.5", spec, [], FormalSource.from_deployment())
    root = _ov.materialize_user_spec("u1", "zai-org/GLM-4.5", bundle)
    written = (_ov.overlay_models_dir(Path(root)) / "zai-org__GLM-4.5.yaml").read_text()
    assert _match(written)["model_id"] == ["zai-org/GLM-4.5"]
