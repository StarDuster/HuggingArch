"""Registering a custom model requires its modeling source.

A custom model's `forward.py` is its only modeling source, and spec
generation refuses to start without one. Registration therefore rejects an
empty forward up front — before any file lands in the cache — rather than
leaving a half-registered directory behind.
"""
from __future__ import annotations

import json

import pytest

import backend.custom_id as custom_id
from backend import custom_model


@pytest.fixture
def cache(monkeypatch, tmp_path):
    root = tmp_path / "custom_models"
    monkeypatch.setattr(custom_id, "CACHE_DIR", root)
    return root


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "toy"}))
    return str(path)


@pytest.mark.parametrize("forward_text", [None, "", "  \n"])
def test_a_model_without_forward_source_is_rejected_before_writing(
        cache, config, tmp_path, forward_text):
    forward_paths = None
    if forward_text is not None:
        fwd = tmp_path / "forward.py"
        fwd.write_text(forward_text)
        forward_paths = [str(fwd)]
    with pytest.raises(ValueError, match="no modeling source"):
        custom_model.register(config_path=config, forward_paths=forward_paths,
                              name="toy")
    assert not (cache / "toy").exists()


def test_an_hf_repo_without_modeling_files_is_rejected(cache, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps({"model_type": "toy"}))
    with pytest.raises(ValueError, match="no modeling source"):
        custom_model.register(hf_model_path=str(repo), name="toy")
    assert not (cache / "toy").exists()


def test_a_rejected_overwrite_keeps_the_existing_registration(cache, config, tmp_path):
    existing = cache / "toy"
    existing.mkdir(parents=True)
    (existing / "forward.py").write_text("class Toy: ...")
    with pytest.raises(ValueError):
        custom_model.register(config_path=config, name="toy", overwrite=True)
    assert (existing / "forward.py").read_text() == "class Toy: ..."


def test_a_model_with_forward_source_registers(cache, config, tmp_path):
    fwd = tmp_path / "modeling_toy.py"
    fwd.write_text("class ToyForCausalLM:\n    def forward(self, x):\n        return x\n")
    out = custom_model.register(config_path=config, forward_paths=[str(fwd)], name="toy")
    assert "class ToyForCausalLM" in (cache / "toy" / "forward.py").read_text()
    assert out["status"]["prompt_written"], out["status"]
