"""Publish-bound provenance regressions from real spec-generation failures."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.spec_worker import fixer, worker
from backend.spec_worker.bundle import BundleFile, FormalSource, normalize_bundle
from backend.spec_worker.overlay import _safe


_VALID = {"valid": True, "errors": []}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _publish(tmp_path: Path, model_id: str):
    return worker._publish_to_cache(
        model_id,
        io.StringIO(),
        started_at=0.0,
        user_id="user-1",
        agent_cli="claude",
        target_root=str(tmp_path),
        val_result=_VALID,
        seeded=False,
        code_revision="pinned",
    )


def test_publish_flattens_agent_base_and_inherited_draft(tmp_path, monkeypatch):
    model_id = "acme/Fork"
    target = f"match: {{model_id: [{model_id}]}}\nbase: fork_base\nchild: yes\n"
    base = "match: {model_type: [fork]}\ndrafts: [fork_attention]\nparent: yes\n"
    draft = "role: norm\ngraph: {out: {op: rmsnorm, in: x, dim: d_model}}\noutputs: [out]\n"
    _write(tmp_path / "models" / f"{_safe(model_id)}.yaml", target)
    _write(tmp_path / "models" / "fork_base.yaml", base)
    _write(tmp_path / "components" / "drafts" / "fork_attention.yaml", draft)
    records = []

    from backend.service import spec_generations
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(spec_generations, "record", lambda **kw: records.append(kw) or {"outcome": "recorded", "id": "r1"})
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success", lambda **kw: True)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())

    downgrade = _publish(tmp_path, model_id)

    assert downgrade is None
    assert len(records) == 1
    assert "base" not in records[0]["spec_yaml"]
    assert "parent: true" in records[0]["spec_yaml"]
    assert "child: yes" in records[0]["spec_yaml"]  # the child's own line, as written
    assert records[0]["new_component_yamls"] == [{
        "path": "backend/arch/components/drafts/fork_attention.yaml",
        "body": draft,
    }]


@pytest.mark.parametrize("variant,base_name", [
    ("MiMo-V2.6-Pro-RL", "mimo_v2_6_pro"),
    ("MiMo-V2.6-Flash-RL", "mimo_v2_6_flash"),
])
def test_mimo_r4_records_do_not_lose_their_private_base(
        tmp_path, monkeypatch, variant, base_name):
    """Regressions for the two 09-26 records: a three-line identity plus base."""
    model_id = f"XiaomiMiMo/{variant}"
    _write(
        tmp_path / "models" / f"{_safe(model_id)}.yaml",
        f"match: {{model_id: [{model_id}]}}\nbase: {base_name}\n",
    )
    _write(
        tmp_path / "models" / f"{base_name}.yaml",
        "match: {model_type: [mimo_v2_6]}\ncomponents: {attention: gqa_attention}\n",
    )
    records = []
    from backend.service import spec_generations
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(spec_generations, "record", lambda **kw: records.append(kw) or {"outcome": "recorded", "id": "r1"})
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success", lambda **kw: True)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())

    downgrade = _publish(tmp_path, model_id)

    assert downgrade is None
    assert "base" not in records[0]["spec_yaml"]
    assert "components" in records[0]["spec_yaml"]
    assert records[0]["new_component_yamls"] == []


def test_mimo_r3_regenerate_that_only_edits_base_is_no_output(tmp_path):
    """The 09-27 Pro regenerate must not validate its stale identity."""
    model_id = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    old = tmp_path / "models" / f"{_safe(model_id)}.yaml"
    fresh_base = tmp_path / "models" / "mimo_v2_6_pro.yaml"
    _write(old, f"match: {{model_id: [{model_id}]}}\nbase: mimo_v2_6_pro\n")
    _write(fresh_base, "match: {model_type: [mimo_v2_6]}\n")
    old_mtime = 1_000_000.0
    os.utime(old, (old_mtime, old_mtime))

    with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
         patch("backend.arch.validate.validate") as validate:
        downgrade, result = worker._verify_spec_output(
            model_id, io.StringIO(), started_at=old_mtime + 100,
            root=str(tmp_path), seeded=False)

    assert downgrade is not None
    assert downgrade.kind == "no_output"
    assert f"expected models/{_safe(model_id)}.yaml" in downgrade.reason
    assert "mimo_v2_6_pro.yaml" in downgrade.reason
    assert result is None
    validate.assert_not_called()


def test_publish_missing_draft_is_validation_failure_with_no_writes(tmp_path, monkeypatch):
    model_id = "acme/Fork"
    _write(tmp_path / "models" / f"{_safe(model_id)}.yaml",
           f"match: {{model_id: [{model_id}]}}\ndrafts: [missing]\n")
    records, caches = [], []
    from backend.service import spec_generations
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(spec_generations, "record", lambda **kw: records.append(kw) or {"outcome": "recorded", "id": "r1"})
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success",
                        lambda **kw: caches.append(kw) or True)

    downgrade = _publish(tmp_path, model_id)

    assert downgrade is not None
    assert downgrade.kind == "validation"
    assert "draft component 'missing'" in downgrade.reason
    assert records == []
    assert caches == []

    with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
         patch("backend.arch.validate.validate") as validate:
        verdict, result = worker._verify_spec_output(
            model_id, io.StringIO(), started_at=0.0,
            root=str(tmp_path), seeded=False)
    assert verdict is not None
    assert verdict.error_kind == "spec_bundle_incomplete"
    assert result is None
    validate.assert_not_called()


def test_publish_malformed_draft_is_validation_failure_with_no_writes(
        tmp_path, monkeypatch):
    model_id = "acme/Fork"
    _write(tmp_path / "models" / f"{_safe(model_id)}.yaml",
           f"match: {{model_id: [{model_id}]}}\ndrafts: [bad]\n")
    _write(tmp_path / "components" / "drafts" / "bad.yaml",
           "role: attention\ngraph: [not-a-map]\n")
    records, caches = [], []
    from backend.service import spec_generations
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(spec_generations, "record",
                        lambda **kw: records.append(kw) or {"outcome": "recorded", "id": "r1"})
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success",
                        lambda **kw: caches.append(kw) or True)

    downgrade = _publish(tmp_path, model_id)

    assert downgrade is not None
    assert downgrade.kind == "validation"
    assert downgrade.error_kind == "spec_bundle_incomplete"
    assert "draft component 'bad'" in downgrade.reason
    assert "not parseable" in downgrade.reason
    assert records == []
    assert caches == []


def test_fix_no_edit_round_trips_unknown_formal_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    intended = "backend/arch/components/library/removed_attention.yaml"
    proposal = BundleFile(
        intended,
        "role: norm\ngraph: {out: {op: rmsnorm, in: x, dim: d_model}}\noutputs: [out]\n",
        formal_body_sha256="0" * 64,
    )
    seed = normalize_bundle(
        "acme/Fork", "match: {model_id: acme/Fork}\n", [proposal],
        FormalSource(models={}, components={}))
    from backend.spec_worker import overlay
    root = overlay.seed_fix_overlay("fixer", "record", "acme/Fork", seed)

    fixed = fixer._harvest_fixed_bundle(root, seed)

    assert fixed is not None
    assert fixed.proposal_files == (proposal,)
    assert (Path(root) / "components" / "formal_proposals" / "library"
            / "removed_attention.yaml").read_text() == proposal.body


def test_fix_formal_proposal_edit_delete_and_omission_round_trip(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    intended = "backend/arch/components/library/removed_attention.yaml"
    original = ("role: norm\ngraph: {out: {op: rmsnorm, in: x, "
                "dim: d_model}}\noutputs: [out]\n")
    proposal = BundleFile(
        intended, original, formal_body_sha256="1" * 64)
    seed = normalize_bundle(
        "acme/Fork", "match: {model_id: acme/Fork}\n", [proposal],
        FormalSource(models={}, components={}))
    from backend.spec_worker import overlay
    root = overlay.seed_fix_overlay("fixer", "record", "acme/Fork", seed)
    lazy = (Path(root) / "components" / "formal_proposals" / "library"
            / "removed_attention.yaml")

    edited = original.replace("op: rmsnorm", "op: layernorm")
    lazy.write_text(edited, encoding="utf-8")
    fixed = fixer._harvest_fixed_bundle(root, seed)
    assert fixed is not None
    assert fixed.proposal_files == (BundleFile(
        intended, edited, formal_body_sha256="1" * 64),)

    lazy.unlink()
    withdrawn = fixer._harvest_fixed_bundle(root, seed)
    assert withdrawn is not None
    assert withdrawn.proposal_files == ()
    assert "formal proposal withdrawn: " + intended in withdrawn.notes

    # A new candidate without a current formal target is omitted, not silently
    # promoted into the serving bundle; its author-facing note must survive.
    new_lazy = (Path(root) / "components" / "formal_proposals" / "library"
                / "absent_attention.yaml")
    new_lazy.write_text(original, encoding="utf-8")
    omitted = fixer._harvest_fixed_bundle(root, seed)
    assert omitted is not None
    assert omitted.proposal_files == ()
    assert ("formal proposal omitted: no current library component named "
            "absent_attention") in omitted.notes


def test_refine_no_edit_publish_round_trips_formal_proposal(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    intended = "backend/arch/components/library/removed_attention.yaml"
    body = ("role: norm\ngraph: {out: {op: rmsnorm, in: x, "
            "dim: d_model}}\noutputs: [out]\n")
    proposal = BundleFile(
        intended, body, formal_body_sha256="2" * 64)
    seed = normalize_bundle(
        "acme/Fork", "match: {model_id: acme/Fork}\n", [proposal],
        FormalSource(models={}, components={}))
    from backend.spec_worker import overlay
    target_root = overlay.seed_run_overlay(
        "user-1", "acme/Fork", seed, kind="refine")
    records = []
    from backend.service import spec_generations
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(spec_generations, "record",
                        lambda **kw: records.append(kw) or {"outcome": "recorded", "id": "r1"})
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success",
                        lambda **kw: True)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())

    downgrade = worker._publish_to_cache(
        "acme/Fork", io.StringIO(), started_at=time.time() - 1,
        user_id="user-1", agent_cli="claude", target_root=target_root,
        val_result=_VALID, seeded=True, seed_bundle=seed,
        code_revision="pinned")

    assert downgrade is None
    assert len(records) == 1
    assert records[0]["new_component_yamls"] == [proposal.as_json()]


def test_selfhost_shared_tree_publish_collects_fresh_provenance(
        tmp_path, monkeypatch):
    """Auth-off runs still write the shared tree before Batch 3 overlays."""
    model_id = "selfhost/Fork"
    safe_id = _safe(model_id)
    arch = tmp_path / "backend" / "arch"
    started_at = time.time()
    formal = _write(arch / "models" / "qwen2.yaml",
                    "match: {model_type: [qwen2]}\n")
    target = _write(
        arch / "models" / f"{safe_id}.yaml",
        f"match: {{model_id: [{model_id}]}}\nbase: agent_base\nchild: yes\n")
    _write(arch / "models" / "agent_base.yaml",
            "match: {model_type: [agent]}\ndrafts: [brand_new]\nparent: yes\n")
    draft = ("role: norm\ngraph: {out: {op: rmsnorm, in: x, "
             "dim: d_model}}\noutputs: [out]\n")
    _write(arch / "components" / "drafts" / "brand_new.yaml", draft)
    old = started_at - 100
    os.utime(formal, (old, old))
    caches = []
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success",
                        lambda **kw: caches.append(kw) or True)
    monkeypatch.setattr("backend.analyzer.fetch_config", lambda *_: object())
    monkeypatch.setattr(worker, "_fresh_model_yaml",
                        lambda *_args, **_kw: target)
    shared_tree_views = worker._shared_tree_views
    monkeypatch.setattr(
        worker, "_shared_tree_views",
        lambda _repo_root, started, safe: shared_tree_views(
            tmp_path, started, safe))

    downgrade = worker._publish_to_cache(
        model_id, io.StringIO(), started_at=started_at,
        user_id=None, agent_cli="claude", target_root=None,
        val_result=_VALID, seeded=False, code_revision="pinned")

    assert downgrade is None
    assert len(caches) == 1
    assert "base" not in caches[0]["spec_yaml"]
    assert "parent: true" in caches[0]["spec_yaml"]
    assert "child: yes" in caches[0]["spec_yaml"]  # the child's own line, as written
    assert caches[0]["new_component_yamls"] == [{
        "path": "backend/arch/components/drafts/brand_new.yaml",
        "body": draft,
    }]


def test_selfhost_formal_namesake_conflict_is_validation_before_cache(
        tmp_path, monkeypatch):
    """A fresh draft cannot hide its formal library namesake."""
    model_id = "selfhost/Fork"
    safe_id = _safe(model_id)
    arch = tmp_path / "backend" / "arch"
    started_at = time.time()
    formal = _write(
        arch / "components" / "library" / "attn.yaml",
        "role: norm\ngraph: {out: {op: rmsnorm, in: x, "
        "dim: d_model}}\noutputs: [out]\n")
    target = _write(
        arch / "models" / f"{safe_id}.yaml",
        f"match: {{model_id: [{model_id}]}}\ndrafts: [attn]\n")
    _write(
        arch / "components" / "drafts" / "attn.yaml",
        "role: norm\ngraph: {out: {op: rmsnorm, in: x, "
        "dim: d_head}}\noutputs: [out]\n")
    old = started_at - 100
    os.utime(formal, (old, old))
    caches = []
    from backend.spec_worker import shared_spec_miss as shared_spec_cache
    monkeypatch.setattr(worker.supabase_env, "configured", lambda: True)
    monkeypatch.setattr(shared_spec_cache, "upsert_after_success",
                        lambda **kw: caches.append(kw) or True)
    monkeypatch.setattr(worker, "_fresh_model_yaml",
                        lambda *_args, **_kw: target)
    shared_tree_views = worker._shared_tree_views
    monkeypatch.setattr(
        worker, "_shared_tree_views",
        lambda _repo_root, started, safe: shared_tree_views(
            tmp_path, started, safe))

    downgrade = worker._publish_to_cache(
        model_id, io.StringIO(), started_at=started_at,
        user_id=None, agent_cli="claude", target_root=None,
        val_result=_VALID, seeded=False, code_revision="pinned")

    assert downgrade is not None
    assert downgrade.kind == "validation"
    assert downgrade.error_kind == "spec_bundle_incomplete"
    assert "conflicts with the different formal component" in downgrade.reason
    assert caches == []


def test_selfhost_malformed_draft_is_validation_before_validate(tmp_path, monkeypatch):
    model_id = "selfhost/Fork"
    safe_id = _safe(model_id)
    started_at = time.time()
    target = _write(
        tmp_path / "backend" / "arch" / "models" / f"{safe_id}.yaml",
        f"match: {{model_id: [{model_id}]}}\ndrafts: [bad]\n")
    _write(tmp_path / "backend" / "arch" / "components" / "drafts" / "bad.yaml",
           "role: attention\ngraph: [not-a-map]\n")
    shared_tree_views = worker._shared_tree_views
    monkeypatch.setattr(
        worker, "_shared_tree_views",
        lambda _repo_root, started, safe: shared_tree_views(
            tmp_path, started, safe))
    monkeypatch.setattr(worker, "_fresh_model_yaml",
                        lambda *_args, **_kw: target)

    with patch("backend.spec_worker._op_chain.find_fresh_gap", return_value=None), \
         patch("backend.arch.validate.validate") as validate:
        verdict, result = worker._verify_spec_output(
            model_id, io.StringIO(), started_at=started_at,
            root=None, seeded=False)

    assert verdict is not None
    assert verdict.kind == "validation"
    assert verdict.error_kind == "spec_bundle_incomplete"
    assert "draft component 'bad'" in verdict.reason
    assert result is None
    validate.assert_not_called()


def test_publish_fallback_validate_gates_the_publish(tmp_path, monkeypatch):
    model_id = "acme/Fork"
    _write(tmp_path / "models" / f"{_safe(model_id)}.yaml",
           f"match: {{model_id: [{model_id}]}}\n")
    seen = []
    monkeypatch.setattr(
        worker, "_invoke_validate",
        lambda *a, **k: seen.append(a) or {"valid": False, "errors": ["x"]})
    log = io.StringIO()
    worker._publish_to_cache(
        model_id, log, started_at=0.0, user_id="user-1",
        agent_cli="claude", target_root=str(tmp_path),
        val_result=None, seeded=False)
    assert seen
    assert "SKIP — spec failed validate" in log.getvalue()
