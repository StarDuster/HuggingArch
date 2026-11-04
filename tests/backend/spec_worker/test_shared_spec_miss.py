"""Offline self-host shared cache and its loader miss hook."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import yaml

from backend.spec_worker import overlay, worker
from backend.spec_worker import shared_spec_miss


MODEL = "cache/Fork"
SPEC = "match: {model_id: cache/Fork}\ncomponents: {attention: gqa_attention}\n"


def _offline_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("HA_SHARED_SPEC_CACHE_ROOT", str(tmp_path / "shared"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user-specs"))
    monkeypatch.setattr(
        "backend.supabase_env.jwt_verification_available", lambda: False)


def test_local_cache_uses_the_overlay_safe_rule():
    assert shared_spec_miss._safe is overlay._safe


def test_local_cache_survives_a_reader_restart(tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    shared_spec_miss.write_local_after_success(
        MODEL, SPEC, snapshot_json={"config": {}}, agent_cli="claude")
    leaf = tmp_path / "shared" / "cache__Fork"
    row = json.loads((leaf / "bundle.json").read_text())
    assert row["storage"] == "selfhost-local"
    assert leaf.stat().st_mode & 0o700 == 0o700

    first = shared_spec_miss.materialize_cached_spec(MODEL)
    assert first is not None
    second = shared_spec_miss.materialize_cached_spec(MODEL)
    assert second is not None
    model_file = "cache__Fork.yaml"
    assert ((tmp_path / "user-specs" / "_cached_specs" / "cache__Fork" / "models")
            .joinpath(model_file)).is_file()


def test_formal_model_expires_local_entry(tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    shared_spec_miss.write_local_after_success(MODEL, SPEC)
    monkeypatch.setattr(shared_spec_miss, "formal_model_exists", lambda _mid: True)
    assert shared_spec_miss.materialize_cached_spec(MODEL) is None
    assert not (tmp_path / "shared" / "cache__Fork").exists()


def test_local_cache_serves_through_a_fresh_loader(tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    import backend.arch.loader as loader_mod
    from backend.arch.loader import SpecLoader

    model_id = "cache/Synthetic"
    spec_yaml = (
        "match:\n"
        "  model_id: [cache/Synthetic]\n"
        "dims:\n"
        "  d_model: config.hidden_size\n"
        "  n_vocab: config.vocab_size\n"
        "blocks:\n"
        "  layer:\n"
        "    graph:\n"
        "      y: { op: linear, in: x, dims: { in: d_model, out: d_model } }\n"
        "    outputs: [y]\n"
        "model:\n"
        "  - { name: embed, op: embedding, boundary_role: input_embedding, dims: { n_vocab: n_vocab, d_model: d_model } }\n"
        "  - { name: y, op: linear, in: embed, boundary_role: output_head, dims: { in: d_model, out: d_model } }\n"
    )
    shared_spec_miss.write_local_after_success(model_id, spec_yaml)
    monkeypatch.setattr(
        loader_mod, "_spec_miss_hook", shared_spec_miss.materialize_cached_spec)
    resolved = SpecLoader().match(
        {"model_type": "unused", "hidden_size": 8, "vocab_size": 16},
        model_id=model_id)
    assert resolved is not None
    assert resolved.spec.name == "cache__Synthetic"


def test_db_row_has_priority_and_local_is_fallback(tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    shared_spec_miss.write_local_after_success(MODEL, SPEC)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    monkeypatch.setattr(shared_spec_miss, "bump_hit", lambda _mid: None)
    monkeypatch.setattr(
        shared_spec_miss, "fetch",
        lambda _mid: {"spec_yaml": "match: {model_id: cache/Fork-from-db}\n"})
    root = shared_spec_miss.materialize_cached_spec(MODEL)
    assert root is not None
    assert "cache/Fork-from-db" in (
        Path(root) / "models" / "cache__Fork.yaml").read_text()

    monkeypatch.delenv("SUPABASE_URL")
    monkeypatch.setattr(shared_spec_miss, "fetch", lambda _mid: None)
    root = shared_spec_miss.materialize_cached_spec(MODEL)
    assert root is not None
    assert yaml.safe_load((Path(root) / "models" / "cache__Fork.yaml").read_text()) == (
        yaml.safe_load(SPEC))


def test_api_registers_the_core_hook_once():
    from backend import service
    from backend import api
    import backend.arch.loader as loader

    assert api is not None
    assert loader._spec_miss_hook is shared_spec_miss.materialize_cached_spec
    assert service is not None
    assert loader._spec_miss_hook is shared_spec_miss.materialize_cached_spec


def test_offline_anonymous_run_writes_local_not_shared_tree(tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    root = Path(overlay.allocate_run_overlay(MODEL, kind="generate"))
    target = root / "models" / "cache__Fork.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SPEC)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())
    landed = []
    monkeypatch.setattr(
        worker, "land_anonymous_overlay",
        lambda *args, **kwargs: landed.append((args, kwargs)))

    result = worker._publish_to_cache(
        MODEL,
        io.StringIO(),
        started_at=time.time(),
        target_root=str(root),
        val_result={"errors": []},
    )

    assert result is None
    assert landed == []
    row = json.loads(
        (tmp_path / "shared" / "cache__Fork" / "bundle.json").read_text())
    assert yaml.safe_load(row["spec_yaml"]) == yaml.safe_load(SPEC)


def test_local_cache_failure_is_a_success_with_actionable_done_message(
        tmp_path, monkeypatch):
    _offline_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    root = Path(overlay.allocate_run_overlay(MODEL, kind="generate"))
    target = root / "models" / "cache__Fork.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SPEC)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())

    def _fail(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shared_spec_miss, "write_local_after_success", _fail)
    log = io.StringIO()
    result = worker._publish_to_cache(
        MODEL, log, started_at=time.time(), target_root=str(root),
        val_result={"errors": []})
    assert result is not None and result.kind == "cache_warning"
    assert "LOCAL-FAIL" in log.getvalue()
    assert "scratch run overlay" in result.reason
