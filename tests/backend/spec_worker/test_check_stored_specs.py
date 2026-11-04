"""The stored-spec check tool reports loader errors, not a parallel rule set."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from backend.spec_worker import overlay as ov
from backend.spec_worker.bundle import FormalSource, NormalizedBundle


def _tool():
    import sys
    path = Path(__file__).resolve().parents[3] / "tools" / "check_stored_specs.py"
    spec = importlib.util.spec_from_file_location("check_stored_specs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_check_overlay_uses_the_loader(tmp_path, monkeypatch):
    check_overlay = _tool().check_overlay

    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    bad = """
match: {model_type: [x]}
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head, tie_weights: config.tie_word_embeddings}
"""
    root = ov.materialize_user_spec("u", "org/M", NormalizedBundle(
        model_id="org/M",
        spec_yaml=bad,
        serving_files=(),
        proposal_files=(),
        notes=(),
        formal_source=FormalSource(models={}, components={}),
    ))
    status, msg = check_overlay(root, model_id="org/M")
    assert status == "LOAD_FAIL"
    assert "tie_weights" in msg


def test_check_bundle_migrate_preview_reports_the_rewrite(tmp_path, monkeypatch):
    check_bundle = _tool().check_bundle

    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    src = """
match: {model_type: [x]}
model:
  - {name: embed_tokens, op: embedding}
  - {name: lm_head, op: linear, boundary_role: output_head, tie_weights: config.tie_word_embeddings}
"""
    row = check_bundle("org/M", src, [], migrate_preview=True)
    assert row.load == "LOAD_FAIL"
    assert "tie_weights" in row.message
    assert row.migrate_blockers == ()
    assert any("tie_weights" in n or "tied_weight" in n for n in row.migrate_notes)
    assert "tie_weights" not in (row.migrate_message or "")


def test_migrate_preview_missing_base_is_a_row_not_a_crash(tmp_path, monkeypatch):
    check_bundle = _tool().check_bundle
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    src = """
match: {model_type: [x]}
base: missing_base
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
"""
    row = check_bundle("org/M", src, [], migrate_preview=True)
    assert row.migrate_load == "LOAD_FAIL"
    assert "missing_base" in (row.migrate_message or "")


def test_migrate_preview_cli_missing_base_prints_a_line(tmp_path, monkeypatch):
    tool = _tool()
    root = tmp_path / "user_specs" / "u" / "org__M"
    (root / "models").mkdir(parents=True)
    (root / "models" / "org__M.yaml").write_text(
        "match: {model_type: [x]}\nbase: missing_base\n", encoding="utf-8")
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    rc = tool.main([
        "--local-only", "--migrate-preview",
        "--user-specs-root", str(tmp_path / "user_specs"),
    ])
    assert rc == 1


def test_default_exits_nonzero_when_db_is_unavailable(monkeypatch):
    tool = _tool()
    monkeypatch.setattr("backend.supabase_env.configured", lambda: False)
    rc = tool.main([])
    assert rc == 1


def test_older_row_load_failure_is_nonzero(tmp_path, monkeypatch):
    tool = _tool()

    def fake_check(model_id, spec_yaml, comps, **kw):
        row = tool.CheckRow(source="record", model_id=model_id, load="LOAD_OK")
        if "missing_base" in (spec_yaml or ""):
            row.load = "LOAD_FAIL"
            row.message = "missing base"
        return row

    monkeypatch.setattr(tool, "check_bundle", fake_check)
    monkeypatch.setattr(tool, "iter_local_overlays", lambda *_a, **_k: [])
    dump = tmp_path / "dump.json"
    dump.write_text("""
{"spec_generations": [
  {"id": "new", "user_id": "u", "model_id": "org/M", "status": "candidate",
   "created_at": "2026-09-27T00:00:00", "spec_yaml": "ok: true\\n"},
  {"id": "old", "user_id": "u", "model_id": "org/M", "status": "candidate",
   "created_at": "2026-09-01T00:00:00", "spec_yaml": "base: missing_base\\n"}
]}
""", encoding="utf-8")
    rc = tool.main(["--db-json", str(dump)])
    assert rc == 1
