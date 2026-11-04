"""Spec PR construction starts from a normalized bundle."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.service import github
from backend.service.routes import account
from backend.spec_worker.bundle import FormalSource, SpecBundleError


def test_legacy_base_is_flattened_before_pr_files(monkeypatch):
    source = FormalSource.from_deployment()
    monkeypatch.setattr(github, "_verify_remote_reuse", lambda *_a, **_kw: None)
    files = [{
        "path": "backend/arch/models/fork_base.yaml",
        "body": "match: {model_type: fork}\ncomponents: {attention: gqa_attention}\n",
    }]

    pr = github.prepare_spec_pr(
        formal_source=source,
        record_id="record-1",
        model_id="acme/Fork",
        spec_yaml="match: {model_id: acme/Fork}\nbase: fork_base\n",
        new_component_yamls=files,
        snapshot_json={"config": {}},
    )

    assert set(pr.files) == {
        "backend/arch/models/acme__Fork.yaml",
        "backend/arch/snapshots/acme__Fork.json",
    }
    assert "base" not in pr.files["backend/arch/models/acme__Fork.yaml"]
    assert "flattened agent base chain: fork_base" in pr.body


def test_incomplete_bundle_stops_before_any_github_call(monkeypatch):
    def _forbidden(**_kw):
        raise AssertionError("GitHub must not be called for an invalid bundle")

    monkeypatch.setattr(github, "open_files_pr", _forbidden)
    with pytest.raises(SpecBundleError, match="draft component 'missing'"):
        github.check_spec_bundle(
            "acme/Fork", "match: {model_id: acme/Fork}\ndrafts: [missing]\n", [])
    with pytest.raises(SpecBundleError, match="draft component 'missing'"):
        github.prepare_spec_pr(
            formal_source=FormalSource.from_deployment(),
            record_id="record-1",
            model_id="acme/Fork",
            spec_yaml="match: {model_id: acme/Fork}\ndrafts: [missing]\n",
            new_component_yamls=[],
        )


def test_incomplete_bundle_route_is_409_not_502(monkeypatch):
    record = {
        "id": "record-1",
        "model_id": "acme/Fork",
        "user_id": "user-uuid-1",
        "status": "candidate",
        "spec_yaml": "match: {model_id: acme/Fork}\ndrafts: [missing]\n",
        "new_component_yamls": [],
        "snapshot_json": {},
    }
    monkeypatch.setattr(account._gens, "get", lambda _id: record)
    monkeypatch.setattr(account, "_pass_review", lambda *_a: {"id": "rev-1"})
    monkeypatch.setattr(account._github, "enabled", lambda: True)

    def _invalid(*_a, **_kw):
        raise SpecBundleError("draft component 'missing'")

    monkeypatch.setattr(account._github, "check_spec_bundle", _invalid)

    with pytest.raises(HTTPException) as raised:
        account.create_pr_my_generation(
            "record-1", user={"sub": "user-uuid-1"})

    assert raised.value.status_code == 409
    assert raised.value.detail["error_kind"] == "bundle_invalid"
    assert "draft component 'missing'" in raised.value.detail["errors"]


def test_malformed_draft_route_is_409_not_502(monkeypatch):
    record = {
        "id": "record-1",
        "model_id": "acme/Fork",
        "user_id": "user-uuid-1",
        "status": "candidate",
        "spec_yaml": "match: {model_id: acme/Fork}\ndrafts: [bad]\n",
        "new_component_yamls": [{
            "path": "backend/arch/components/drafts/bad.yaml",
            "body": "role: attention\ngraph: [not-a-map]\n",
        }],
        "snapshot_json": {},
    }
    monkeypatch.setattr(account._gens, "get", lambda _id: record)
    monkeypatch.setattr(account, "_pass_review", lambda *_a: {"id": "rev-1"})
    monkeypatch.setattr(account._github, "enabled", lambda: True)

    def _forbidden(*_a, **_kw):
        raise AssertionError("GitHub must not see a malformed saved draft")

    monkeypatch.setattr(account._github, "open_files_pr", _forbidden)
    monkeypatch.setattr(account, "launch_spec_pr", _forbidden)

    with pytest.raises(HTTPException) as raised:
        account.create_pr_my_generation(
            "record-1", user={"sub": "user-uuid-1"})

    assert raised.value.status_code == 409
    detail = raised.value.detail
    assert detail["error_kind"] == "bundle_invalid"
    assert any(
        "draft component 'bad'" in error
        and "not parseable" in error
        for error in detail["errors"])


def _remote_component(name, body, tier="library"):
    from backend.arch.component_signature import component_structure_signature
    from backend.spec_worker.bundle import FormalComponent

    path = f"backend/arch/components/{tier}/{name}.yaml"
    return FormalComponent(
        tier=tier,
        body=body,
        signature=component_structure_signature(body, path=path),
        path=path,
    )


def test_exact_remote_draft_is_reused_and_references_are_rewritten(monkeypatch):
    from pathlib import Path

    body = (Path(__file__).resolve().parents[3]
            / "backend/arch/components/library/gqa_attention.yaml").read_text()
    source = FormalSource(
        models={},
        components={"main_attention": _remote_component("main_attention", body)},
    )
    monkeypatch.setattr(github, "_verify_remote_reuse", lambda *_a, **_kw: None)

    pr = github.prepare_spec_pr(
        formal_source=source,
        record_id="record-2",
        model_id="acme/Fork",
        spec_yaml=(
            "match: {model_id: acme/Fork}\n"
            "drafts: [toy_attention]\n"
            "components: {attention: toy_attention}\n"
        ),
        new_component_yamls=[{
            "path": "backend/arch/components/drafts/toy_attention.yaml",
            "body": body,
        }],
    )

    assert "backend/arch/components/drafts/toy_attention.yaml" not in pr.files
    target = pr.files["backend/arch/models/acme__Fork.yaml"]
    assert "attention: main_attention" in target
    assert "toy_attention" not in target


def test_exact_remote_draft_keeps_optin_when_main_is_still_draft_tier(monkeypatch):
    from pathlib import Path

    body = (Path(__file__).resolve().parents[3]
            / "backend/arch/components/library/gqa_attention.yaml").read_text()
    source = FormalSource(
        models={},
        components={"main_attention": _remote_component(
            "main_attention", body, tier="drafts")},
    )
    monkeypatch.setattr(github, "_verify_remote_reuse", lambda *_a, **_kw: None)

    pr = github.prepare_spec_pr(
        formal_source=source,
        record_id="record-3",
        model_id="acme/Fork",
        spec_yaml=(
            "match: {model_id: acme/Fork}\n"
            "drafts: [toy_attention]\n"
            "components: {attention: toy_attention}\n"
        ),
        new_component_yamls=[{
            "path": "backend/arch/components/drafts/toy_attention.yaml",
            "body": body,
        }],
    )

    assert "backend/arch/components/drafts/toy_attention.yaml" not in pr.files
    target = pr.files["backend/arch/models/acme__Fork.yaml"]
    assert "attention: main_attention" in target
    assert "drafts: [main_attention]" in target


def test_same_named_remote_component_conflict_stops_before_open(monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    candidate = (root / "backend/arch/components/library/gqa_attention.yaml").read_text()
    other = (root / "backend/arch/components/library/swiglu_ffn.yaml").read_text()
    source = FormalSource(
        models={},
        components={"toy_attention": _remote_component("toy_attention", other)},
    )

    def _forbidden(*_a, **_kw):
        raise AssertionError("a same-name conflict must not open a PR")

    monkeypatch.setattr(github, "open_files_pr", _forbidden)
    with pytest.raises(SpecBundleError, match=r"toy_attention.*conflicts"):
        github.prepare_spec_pr(
            formal_source=source,
            record_id="record-4",
            model_id="acme/Fork",
            spec_yaml=(
                "match: {model_id: acme/Fork}\n"
                "drafts: [toy_attention]\n"
                "components: {attention: toy_attention}\n"
            ),
            new_component_yamls=[{
                "path": "backend/arch/components/drafts/toy_attention.yaml",
                "body": candidate,
            }],
        )


def test_unmigratable_spec_route_is_409_spec_invalid(monkeypatch):
    from backend.arch.loader import SpecLoadError

    record = {
        "id": "record-1",
        "model_id": "acme/Fork",
        "user_id": "user-uuid-1",
        "status": "candidate",
        "spec_yaml": "match: {model_id: acme/Fork}\n",
        "new_component_yamls": [],
        "snapshot_json": {},
    }
    monkeypatch.setattr(account._gens, "get", lambda _id: record)
    monkeypatch.setattr(account, "_pass_review", lambda *_a: {"id": "rev-1"})
    monkeypatch.setattr(account._github, "enabled", lambda: True)

    def _invalid(*_a, **_kw):
        raise SpecLoadError("cannot be migrated automatically: retired kv")

    monkeypatch.setattr(account._github, "check_spec_bundle", _invalid)

    with pytest.raises(HTTPException) as raised:
        account.create_pr_my_generation(
            "record-1", user={"sub": "user-uuid-1"})

    assert raised.value.status_code == 409
    assert raised.value.detail["error_kind"] == "spec_invalid"
    assert "cannot be migrated" in raised.value.detail["errors"][0]
