"""Remote-main deduplication and formal proposal admission."""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import yaml

from backend.arch.component_signature import component_structure_signature
from backend.service import github
from backend.spec_worker.bundle import (
    BundleFile,
    FormalComponent,
    FormalSource,
    SpecBundleError,
    normalize_bundle,
    rename_component_references,
)


ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / "backend" / "arch"


def remote_component(name: str, body: str, *, tier: str = "library") -> FormalComponent:
    path = f"backend/arch/components/{tier}/{name}.yaml"
    return FormalComponent(
        tier=tier,
        body=body,
        signature=component_structure_signature(body, path=path),
        path=path,
    )


def _tar_archive(files: dict[str, str]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for path, body in files.items():
            data = body.encode("utf-8")
            info = tarfile.TarInfo(f"repo-main/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def test_component_renames_cover_every_authoring_channel():
    text = """
drafts: [old]
components: {attention: old}
params: {unrelated: hidden}
model:
  - {name: block, template: old, params: {inner: old}}
blocks:
  root: {op: old, decode_variant: old}
per_layer:
  map:
    dense: {attention: old, ffn: old}
    moe: {template: old, dims: {unrelated: old}}
"""
    got = yaml.safe_load(rename_component_references(
        text, {"old": "new"}, label="test spec"))
    assert got["drafts"] == ["new"]
    assert got["components"]["attention"] == "new"
    assert got["params"]["unrelated"] == "hidden"
    assert got["model"][0]["template"] == "new"
    assert got["model"][0]["params"]["inner"] == "new"
    assert got["blocks"]["root"]["op"] == "new"
    assert got["blocks"]["root"]["decode_variant"] == "new"
    assert got["per_layer"]["map"]["dense"]["attention"] == "new"
    assert got["per_layer"]["map"]["dense"]["ffn"] == "new"
    assert got["per_layer"]["map"]["moe"]["template"] == "new"
    assert got["per_layer"]["map"]["moe"]["dims"]["unrelated"] == "old"


def test_draft_to_draft_calls_are_rewritten_too():
    text = """
role: block
params:
  attn: {kind: component, default: old}
graph:
  attention: {op: old, in: x}
per_layer:
  map:
    dense: {attention: old, ffn: old}
outputs: [attention]
"""
    got = yaml.safe_load(rename_component_references(
        text, {"old": "new"}, label="caller draft"))
    assert got["params"]["attn"]["default"] == "new"
    assert got["graph"]["attention"]["op"] == "new"
    assert got["per_layer"]["map"]["dense"] == {
        "attention": "new", "ffn": "new"}


def test_remote_reuse_load_and_fingerprint_failures_are_bundle_invalid(monkeypatch):
    class ExplodingLoader:
        def load_model_spec(self, _target):
            raise ValueError("missing draft")

    monkeypatch.setattr(github, "SpecLoader", lambda **_kw: ExplodingLoader())
    with pytest.raises(SpecBundleError, match="fix missing component references"):
        github._verify_remote_reuse(
            "acme/Fork",
            ("match: {model_id: acme/Fork}\n", []),
            "match: {model_id: acme/Fork}\n",
            [],
            FormalSource(models={}, components={}),
        )

    class SilentLoader:
        def load_model_spec(self, _target):
            return object()

    monkeypatch.setattr(github, "SpecLoader", lambda **_kw: SilentLoader())
    monkeypatch.setattr(
        github, "arch_fingerprint",
        lambda _arch: (_ for _ in ()).throw(RuntimeError("invalid graph")))
    with pytest.raises(SpecBundleError, match="fix invalid component references"):
        github._verify_remote_reuse(
            "acme/Fork",
            ("match: {model_id: acme/Fork}\n", []),
            "match: {model_id: acme/Fork}\n",
            [],
            FormalSource(models={}, components={}),
        )


def test_exact_draft_reuse_preserves_arch_fingerprint_against_main():
    body = (ARCH / "components/library/gqa_attention.yaml").read_text()
    raw = yaml.safe_load((ARCH / "models/qwen2.yaml").read_text())
    raw["match"] = {"model_id": ["acme/Fork"]}
    raw["components"]["attention"] = "toy_attention"
    raw["drafts"] = ["toy_attention"]
    source = FormalSource.from_deployment()
    bundle = normalize_bundle(
        "acme/Fork",
        yaml.safe_dump(raw, sort_keys=False),
        [BundleFile(
            path="backend/arch/components/drafts/toy_attention.yaml",
            body=body,
        )],
        source,
    )
    deduped, mapping = github._dedupe_remote_drafts(bundle, source)
    assert mapping == {"toy_attention": "gqa_attention"}
    assert deduped.serving_files == ()
    control = rename_component_references(
        bundle.spec_yaml, mapping, label="control")
    github._verify_remote_reuse(
        "acme/Fork",
        (deduped.spec_yaml, list(deduped.serving_files)),
        control,
        list(bundle.serving_files),
        source,
    )


def test_remote_view_comes_from_one_default_branch_archive(monkeypatch, httpx_mock):
    import httpx

    body = (ARCH / "components/library/gqa_attention.yaml").read_text()
    monkeypatch.setattr(github, "_repo", lambda: "owner/repo")
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo",
        json={"default_branch": "main"})
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/git/ref/heads/main",
        json={"object": {"sha": "main-sha"}})
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/tarball/main-sha",
        content=_tar_archive({
            "backend/arch/models/fork.yaml": "match: {model_id: acme/Fork}\n",
            "backend/arch/components/library/gqa_attention.yaml": body,
            "README.md": "ignored",
        }))
    with httpx.Client() as client:
        sha, source = github._remote_formal_source(client, "owner/repo")
    assert sha == "main-sha"
    assert source.models == {"fork": "match: {model_id: acme/Fork}\n"}
    assert source.components["gqa_attention"].path == (
        "backend/arch/components/library/gqa_attention.yaml")


class _Response:
    def __init__(self, value):
        self.value = value
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class _PaginatedClient:
    def __init__(self, candidate_path: str):
        self.candidate_path = candidate_path
        self.file_pages: list[int] = []

    def get(self, url: str, params=None):
        params = params or {}
        page = params.get("page")
        if url.endswith("/pulls"):
            if page == 1:
                return _Response([{
                    "number": 1,
                    "html_url": "https://example/pr/open-1",
                    "head": {"ref": "other/spec-1"},
                }])
            return _Response([])
        if url.endswith("/pulls/1/files"):
            self.file_pages.append(page)
            if page == 1:
                filler = [{"filename": f"file-{i}"} for i in range(99)]
                return _Response([*filler, {"filename": self.candidate_path}])
            return _Response([])
        if url.endswith(f"/contents/{self.candidate_path}"):
            return _Response(None)
        raise AssertionError(f"unexpected URL {url}")


def test_open_pr_file_conflict_is_found_on_the_second_files_page():
    path = "backend/arch/components/drafts/toy_attention.yaml"
    body = (ARCH / "components/library/gqa_attention.yaml").read_text()
    client = _PaginatedClient(path)
    with pytest.raises(SpecBundleError, match="https://example/pr/open-1"):
        github._reject_open_pr_draft_conflicts(
            client,
            "owner/repo",
            branch="spec/current",
            drafts=[BundleFile(path=path, body=body)],
        )
    assert client.file_pages == [1, 2]


def test_matching_formal_proposal_rides_and_mismatched_proposal_is_omitted(monkeypatch):
    body = (ARCH / "components/library/swiglu_ffn.yaml").read_text()
    intended = "backend/arch/components/library/swiglu_ffn.yaml"
    good = BundleFile(
        path=intended,
        body=body + "\n# edited proposal\n",
        formal_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    source = FormalSource(
        models={},
        components={"swiglu_ffn": remote_component("swiglu_ffn", body)},
    )
    monkeypatch.setattr(github, "_verify_remote_reuse", lambda *_a, **_kw: None)

    pr = github.prepare_spec_pr(
        formal_source=source,
        record_id="record-4",
        model_id="acme/Fork",
        spec_yaml="match: {model_id: acme/Fork}\n",
        new_component_yamls=[good.as_json()],
    )

    assert pr.files[intended] == good.body
    assert "shared-layer component change" in pr.body
    assert "exact main baseline" in pr.body

    stale = BundleFile(
        path=intended,
        body=good.body,
        formal_body_sha256="0" * 64,
    )
    pr = github.prepare_spec_pr(
        formal_source=source,
        record_id="record-4",
        model_id="acme/Fork",
        spec_yaml="match: {model_id: acme/Fork}\n",
        new_component_yamls=[stale.as_json()],
    )
    assert intended not in pr.files
    assert ("formal proposal omitted from PR: stale/missing/mismatched baseline"
            in pr.body)
