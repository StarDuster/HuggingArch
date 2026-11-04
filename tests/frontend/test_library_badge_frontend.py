"""Headless test of the "in library" badge on the account workspace: a merged
record whose model the library serves (backend `library_spec`, no
`superseded_by`) is marked in library with the spec file in its tooltip, and
its own review verdict stays visible but greyed. A record with no library
spec keeps its plain verdict, and a failed run's log of a model the library
now serves is archived with the superseded records. The built `web/dist` is served from a throwaway
server and every API call is answered with canned JSON in the shapes the
backend returns. Skips cleanly when the app is not built or
playwright/chromium is absent.
"""
from __future__ import annotations

import json

import pytest

from tests.frontend.test_playground_frontend import (
    _DIST, _ensure_built, _serve_dist, AGENTS)

ROW_TITLE = "View this spec's YAML, reviews, and PR actions"


def _rec(rid, model_id, status, **kw):
    return {"id": rid, "model_id": model_id, "status": status, "agent_cli": "claude",
            "created_at": "2026-09-01T00:00:00Z", "pr_url": None,
            "superseded_by": None, "library_spec": None, **kw}


MERGED = _rec("rec-merged", "acme/InLibrary", "merged",
              pr_url="https://github.com/o/r/pull/2", latest_verdict="no_pass",
              library_spec="acme__InLibrary")
PRIVATE = _rec("rec-private", "acme/Private", "generated", latest_verdict="no_pass")
DETAIL = {**MERGED, "spec_yaml": "match: {model_id: acme/InLibrary}\n",
          "new_component_yamls": [], "snapshot_json": {}}
# A failed run of a model the library now serves: the backend returns it only
# with `?superseded=true`.
LIBRARY_LOG = {"id": "log-library", "model_id": "acme/Superseded", "success": False,
               "created_at": "2026-09-03T00:00:00Z", "run_key": "u|acme/Superseded",
               "library_spec": "acme__Superseded", "superseded_by": "acme__Superseded"}
REVIEWS = [{"id": "rev-1", "verdict": "no_pass", "reviewer_role": "user",
            "created_at": "2026-09-02T00:00:00Z", "report": "## Agent review\nNO_PASS",
            "bundle_current": True}]


@pytest.fixture(scope="module")
def page():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    httpd = _serve_dist()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # chromium not installed
            httpd.shutdown()
            pytest.skip(f"chromium unavailable: {exc}")

        def _json(route, payload):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

        def api(route):
            url = route.request.url
            rest = url.split("://", 1)[1].split("/", 1)[1]
            path, _, query = rest.partition("?")
            if path == "api/agents":
                return _json(route, AGENTS)
            if path == "api/auth/me":
                return _json(route, {"authEnabled": True, "isAdmin": False,
                                     "user": {"id": "u", "email": "u@example.com"},
                                     "profile": {"username": "u"}})
            if path == "api/account/generations":
                if "superseded=true" in query:
                    return _json(route, {"items": [], "orphan_logs": [LIBRARY_LOG]})
                return _json(route, {"items": [MERGED, PRIVATE], "orphan_logs": []})
            if path == f"api/account/generations/{MERGED['id']}":
                return _json(route, DETAIL)
            if path == f"api/account/generations/{MERGED['id']}/reviews":
                return _json(route, {"items": REVIEWS})
            return _json(route, {})

        context = browser.new_context(viewport={"width": 1400, "height": 1100})
        context.add_init_script(
            "window.SITE_CONFIG={authEnabled:true,supabase:{url:'',anon_key:''}};")
        pg = context.new_page()
        pg.route("**/api/**", api)
        pg.goto(f"http://127.0.0.1:{httpd.server_address[1]}/account/workspace.html#generations")
        pg.wait_for_selector(f"[title=\"{ROW_TITLE}\"]", timeout=20000)
        try:
            yield pg
        finally:
            browser.close()
            httpd.shutdown()


def _row(page, model_id):
    return page.locator(f"[title=\"{ROW_TITLE}\"]", has_text=model_id).first


def test_merged_library_record_is_marked_in_library(page):
    page.locator("summary", has_text="Archived (merged or closed)").click()
    row = _row(page, "acme/InLibrary")
    badge = row.locator("[data-in-library]")
    assert badge.inner_text().strip() == "in library"
    assert "acme__InLibrary.yaml" in badge.get_attribute("title")
    verdict = row.locator("[data-verdict]")
    assert verdict.inner_text().strip().endswith("NO_PASS")
    assert verdict.get_attribute("data-verdict-historical") is not None
    assert verdict.get_attribute("title") == (
        "Review of this record's own spec; the library spec is what the product uses")
    assert "text-red" not in verdict.get_attribute("class")


def test_record_without_library_spec_keeps_its_verdict(page):
    row = _row(page, "acme/Private")
    assert row.locator("[data-in-library]").count() == 0
    verdict = row.locator("[data-verdict]")
    assert verdict.get_attribute("data-verdict-historical") is None
    assert verdict.get_attribute("title") == "Latest review verdict"
    assert "text-red" in verdict.get_attribute("class")


def test_detail_drawer_marks_in_library_and_greys_reviews(page):
    if not _row(page, "acme/InLibrary").is_visible():
        page.locator("summary", has_text="Archived (merged or closed)").click()
    _row(page, "acme/InLibrary").click()
    page.wait_for_selector("aside [data-in-library]", timeout=10000)
    head = page.locator("aside [data-verdict]").first
    assert head.get_attribute("data-verdict-historical") is not None
    card = page.locator("aside span", has_text="NO_PASS").last
    page.wait_for_function(
        "() => [...document.querySelectorAll('aside span')].some("
        "s => s.textContent === 'NO_PASS' && s.title.startsWith('Review of this record'))",
        timeout=10000)
    assert "text-red" not in card.get_attribute("class")


def test_a_library_models_failed_run_log_is_archived_as_superseded(page):
    page.reload()  # an earlier test may have left the spec drawer open
    page.wait_for_selector(f"[title=\"{ROW_TITLE}\"]", timeout=20000)
    archive = page.locator("details", has=page.locator(
        "summary", has_text="Archived (superseded by library spec)"))
    archive.locator("summary").click()
    assert archive.get_by_text("acme/Superseded").count() >= 1
    assert archive.get_by_text("log only").count() == 1
