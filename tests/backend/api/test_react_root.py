"""Top-tab URLs serve the React SPA. Vanilla HTML is not the production entry."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.api._app import get_web_dist_path, require_web_dist_if_configured, spa_index_path


def test_get_web_dist_path_is_repo_web_dist():
    path = Path(get_web_dist_path())
    assert path.name == "dist"
    assert path.parent.name == "web"


def _assert_top_tab(path: str) -> None:
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get(path)
    if spa_index_path():
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert 'id="root"' in r.text
        assert 'data-page="' not in r.text
    else:
        assert r.status_code == 503
        assert "Model Gallery" not in r.text
        assert 'data-page="' not in r.text


def test_root_is_react_spa_not_vanilla():
    _assert_top_tab("/")


def test_gallery_html_is_react_spa_not_vanilla():
    _assert_top_tab("/gallery.html")


def test_inference_html_is_react_spa_not_vanilla():
    _assert_top_tab("/inference.html")


def test_inspector_html_is_react_spa_not_vanilla():
    _assert_top_tab("/inspector.html")


def test_playground_html_is_react_spa_not_vanilla():
    # Regression: the Playground page shipped without its backend SPA route, so
    # /playground.html 404'd on the server even though the API and bundle were
    # present. Every top-tab page needs its `@app.get` in routes/meta.py.
    _assert_top_tab("/playground.html")


def test_fusion_html_is_react_spa_not_vanilla():
    _assert_top_tab("/fusion.html")


def test_kernel_corpus_html_is_react_spa_not_vanilla():
    _assert_top_tab("/kernel-corpus.html")


def test_privacy_is_react_spa_not_vanilla():
    _assert_top_tab("/privacy")


def test_compare_html_is_react_spa_not_vanilla():
    _assert_top_tab("/compare.html")


def test_account_profile_html_is_react_spa_not_vanilla():
    _assert_top_tab("/account/profile.html")


def test_account_workspace_html_is_react_spa_not_vanilla():
    _assert_top_tab("/account/workspace.html")


def test_account_bench_review_html_is_react_spa_not_vanilla():
    _assert_top_tab("/account/bench-review.html")


def test_account_benchmarks_html_redirects_to_workspace_benchmarks():
    # Benchmark management was folded into the workspace page's #benchmarks
    # panel; bookmarks and stale login ?next= targets must not land on a
    # bare 404.
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get("/account/my-benchmarks.html", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "/account/workspace.html#benchmarks"


def test_admin_is_react_spa_not_vanilla():
    _assert_top_tab("/admin/")


def test_auth_login_html_is_react_spa_not_vanilla():
    _assert_top_tab("/auth/login.html")


def test_auth_verify_html_is_react_spa_not_vanilla():
    _assert_top_tab("/auth/verify.html")


def test_auth_reset_html_is_react_spa_not_vanilla():
    _assert_top_tab("/auth/reset-password.html")


def test_require_web_dist_is_silent_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("HA_REQUIRE_WEB_DIST", raising=False)
    require_web_dist_if_configured(str(tmp_path / "missing"))


def test_require_web_dist_refuses_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HA_REQUIRE_WEB_DIST", "1")
    with pytest.raises(RuntimeError, match="HA_REQUIRE_WEB_DIST"):
        require_web_dist_if_configured(str(tmp_path / "missing"))


def test_require_web_dist_accepts_a_built_index(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>")
    monkeypatch.setenv("HA_REQUIRE_WEB_DIST", "1")
    require_web_dist_if_configured(str(dist))


def test_unknown_page_for_a_browser_is_the_spa_not_found_page():
    # A browser opening a URL no route serves gets the site's page (the SPA
    # draws "Page not found"), with status 404 — not raw JSON.
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get("/no-such-page", headers={"accept": "text/html,*/*;q=0.8"})
    assert r.status_code == 404
    if spa_index_path():
        assert "text/html" in r.headers.get("content-type", "")
        assert 'id="root"' in r.text


def test_unknown_path_for_an_api_caller_stays_json():
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get("/no-such-endpoint", headers={"accept": "*/*"})
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


def test_an_endpoints_own_404_stays_json_for_a_browser():
    # Only "no route" becomes the page; a route that answers 404 itself
    # keeps its JSON answer.
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get("/static/no-such-file.png", headers={"accept": "text/html"})
    assert r.status_code == 404
    assert "application/json" in r.headers.get("content-type", "")


@pytest.mark.parametrize("path, kind", [("/favicon.ico", "image/"), ("/robots.txt", "text/plain")])
def test_root_level_public_files_are_served(path, kind):
    from fastapi.testclient import TestClient
    from backend.api import app

    r = TestClient(app).get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(kind)
