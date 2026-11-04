"""open_files_pr writes ONE commit on current main, not one commit per file."""
from __future__ import annotations

import pytest

from backend.service import github


@pytest.fixture(autouse=True)
def _gh_pat(monkeypatch):
    monkeypatch.setenv("HA_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("HA_GITHUB_REPO", "shenh10/HuggingArch")
    for var in ("HA_GITHUB_APP_ID", "HA_GITHUB_APP_PRIVATE_KEY",
                "HA_GITHUB_APP_INSTALLATION_ID"):
        monkeypatch.delenv(var, raising=False)
    github._inst_token["token"] = None
    github._inst_token["exp"] = 0.0


_BASE = "https://api.github.com/repos/shenh10/HuggingArch"


def _stub_git_write(httpx_mock, *, ref_status=201):
    from httpx import Response
    httpx_mock.add_response(method="GET", url=_BASE, json={"default_branch": "main"})
    httpx_mock.add_response(
        method="GET", url=f"{_BASE}/git/ref/heads/main",
        json={"object": {"sha": "aaa111"}})
    httpx_mock.add_response(
        method="GET", url=f"{_BASE}/git/commits/aaa111",
        json={"tree": {"sha": "tree0"}})
    httpx_mock.add_callback(
        lambda req: Response(201, json={"sha": "blob"}),
        method="POST", url=f"{_BASE}/git/blobs",
        is_reusable=True)
    httpx_mock.add_response(method="POST", url=f"{_BASE}/git/trees", json={"sha": "tree1"})
    httpx_mock.add_response(method="POST", url=f"{_BASE}/git/commits", json={"sha": "ccc"})
    httpx_mock.add_response(
        method="POST", url=f"{_BASE}/git/refs", status_code=ref_status, json={})
    if ref_status == 422:
        httpx_mock.add_response(
            method="PATCH", url=f"{_BASE}/git/refs/heads/spec/x", json={})
    httpx_mock.add_response(
        method="POST", url=f"{_BASE}/pulls",
        json={"html_url": "https://github.com/shenh10/HuggingArch/pull/99"})


def test_open_files_pr_one_commit_not_per_file(httpx_mock):
    _stub_git_write(httpx_mock)
    url = github.open_files_pr(
        branch="spec/x",
        files={"backend/arch/models/m.yaml": "a: 1\n",
               "backend/arch/snapshots/m.json": "{}"},
        title="Add spec: m",
        body="hi",
        commit_prefix="spec(m)",
    )
    assert url.endswith("/pull/99")
    reqs = httpx_mock.get_requests()
    assert not any(r.method == "PUT" and "/contents/" in str(r.url) for r in reqs)
    commits = [r for r in reqs if r.method == "POST" and str(r.url).endswith("/git/commits")]
    assert len(commits) == 1
    import json
    body = json.loads(commits[0].content)
    assert body["parents"] == ["aaa111"]
    assert body["message"].startswith("spec(m):")


def test_open_files_pr_force_moves_existing_branch(httpx_mock):
    _stub_git_write(httpx_mock, ref_status=422)
    github.open_files_pr(
        branch="spec/x",
        files={"a.yaml": "x"},
        title="t", body="b",
    )
    patches = [r for r in httpx_mock.get_requests()
               if r.method == "PATCH" and "/git/refs/heads/spec/x" in str(r.url)]
    assert len(patches) == 1
    import json
    assert json.loads(patches[0].content) == {"sha": "ccc", "force": True}
