"""normalize_bundle applies the stored-spec rewrite; a blocker is SpecLoadError."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.arch.loader import SpecLoadError, SpecLoader, overlay_kwargs
from backend.arch.spec_migrate import REGENERATE_HINT, SCHEMA_ERROR, SpecMigrationBlocked
from backend.spec_worker import overlay as ov
from backend.spec_worker.bundle import FormalSource, NormalizedBundle, normalize_bundle


OLD_TIE = """
match:
  model_type: [qwen3]
model:
  - {name: embed_tokens, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - {name: lm_head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}, tie_weights: config.tie_word_embeddings}
"""


def _empty_source() -> FormalSource:
    return FormalSource(models={}, components={})


def _normalize(model_id: str, spec_yaml: str):
    return normalize_bundle(model_id, spec_yaml, [], _empty_source())


def test_normalize_rewrites_tie_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    bundle = _normalize("org/Q", OLD_TIE)
    root = ov.materialize_user_spec("u1", "org/Q", bundle)
    written = Path(root) / "models" / "org__Q.yaml"
    doc = yaml.safe_load(written.read_text())
    head = next(layer for layer in doc["model"] if layer["name"] == "lm_head")
    assert "tie_weights" not in head
    assert head["tied_weight"]["to"] == "embed_tokens"
    loader = SpecLoader(**overlay_kwargs(root))
    fails = [
        str(e) for (p, _t, _i, e) in loader._model_load_failures
        if str(p).startswith(root)
    ]
    assert fails == []


def test_overlay_write_keeps_the_bundle_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    raw = NormalizedBundle(
        model_id="org/Q",
        spec_yaml=OLD_TIE,
        serving_files=(),
        proposal_files=(),
        notes=(),
        formal_source=_empty_source(),
    )
    root = ov.materialize_user_spec("u1", "org/Q", raw)
    text = (Path(root) / "models" / "org__Q.yaml").read_text()
    assert "tie_weights:" in text
    assert "tied_weight:" not in text


def test_unmigratable_is_a_spec_error_and_does_not_wipe(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    good = ov.materialize_user_spec("u1", "org/Q", _normalize("org/Q", OLD_TIE))
    marker = Path(good) / "models" / "org__Q.yaml"
    assert marker.is_file()
    broken = """
role: attention
kv_cached: [k, v]
graph: {x: {op: identity, in: [x]}}
"""
    # a retired syntax the migrator cannot rewrite: regenerate, not declare
    with pytest.raises(SpecLoadError) as refused:
        _normalize("org/Q", broken)
    assert not isinstance(refused.value, SpecMigrationBlocked)
    assert str(refused.value) == f"{SCHEMA_ERROR} {REGENERATE_HINT}"
    assert marker.is_file()
    assert "tied_weight" in marker.read_text()
