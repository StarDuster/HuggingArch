"""Headless test of a record's corpus admission on the account workspace
(ADMIT §4.2): the line under a PASS review, its grouped table, and the Open
PR button that follows the admission's status. The built `web/dist` is
served from a throwaway server and every API call is answered with canned
JSON in the shapes the backend returns. Skips cleanly when the app is not
built or playwright/chromium is absent.
"""
from __future__ import annotations

import json

import pytest

from tests.frontend.test_playground_frontend import (
    _DIST, _ensure_built, _serve_dist, AGENTS)

RID = "rec-00000001"
RECORD = {"id": RID, "model_id": "acme/Model", "status": "generated", "pr_url": None,
          "agent_cli": "claude", "created_at": "2026-10-01T00:00:00Z",
          "spec_yaml": "match: {model_id: acme/Model}\n", "new_component_yamls": [],
          "snapshot_json": {}, "superseded_by": None}


def _report(status, blocked_kind=None):
    return {
        "status": status, "blocked_kind": blocked_kind, "mode": "dry_run",
        "base": {"sha": "abcdef1234567890" + "0" * 24, "kind": "main"},
        "gates": {"G0": [], "G1": {"P9": ["drift"]} if blocked_kind == "base" else {},
                  "G2": ["C1 chain stacks"] if blocked_kind == "candidate_or_rule" else []},
        "items": [
            {"id": "P4", "title": "checkpoint reconciliation", "key_states": {
                "EXACT/acme__Model": {"state": "ready", "waits_on": []}}},
            {"id": "P2", "title": "SOL surface", "expected_failing_tests": ["tests/t.py::test_sol"],
             "key_states": {"acme__Model.json|single": {"state": "withheld", "waits_on": ["G4"]}}},
        ],
        "checks": [{"id": "C1", "title": "chain stacks carry hidden",
                    "state": "failed" if blocked_kind == "candidate_or_rule" else "passed",
                    "violations": ["visual.deepstack_merger_list"]
                    if blocked_kind == "candidate_or_rule" else []}],
        "decisions": [{"id": "D1", "state": "pending", "detail": ["visual: no entry"],
                       "suggestion": {"nodes": {"merger.linear_fc1": "split"}}},
                      {"id": "D5", "state": "info", "detail": ["not in supported_models.txt"]}],
        "base_notes": [], "errors": [],
    }


def _review(admission, *, current=True, unavailable=False):
    row = {"id": "rev-1", "verdict": "pass", "reviewer_role": "user",
           "created_at": "2026-10-01T00:00:00Z", "report": "## Agent review\nPASS",
           "bundle_current": current}
    if unavailable:
        row["admission_unavailable"] = True
    else:
        row["admission"] = admission
    return row


@pytest.fixture(scope="module")
def site():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    httpd = _serve_dist()
    state: dict = {"is_admin": False, "reviews": [], "posts": [], "pr_status": {"state": "idle"}}
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
            path = url.split("://", 1)[1].split("/", 1)[1].split("?", 1)[0]
            if route.request.method == "POST":
                state["posts"].append((path, route.request.post_data))
                if path.endswith("/create-pr"):
                    return _json(route, {"status": "started", "run_key": f"pr|u|{RID}"})
                return _json(route, {"status": "started"})
            if path == "api/agents":
                return _json(route, AGENTS)
            if path == "api/auth/me":
                return _json(route, {"authEnabled": True, "isAdmin": state["is_admin"],
                                     "user": {"id": "u", "email": "u@example.com"},
                                     "profile": {"username": "u"}})
            if path == "api/account/generations":
                return _json(route, {"items": [RECORD], "orphan_logs": []})
            if path == f"api/account/generations/{RID}":
                return _json(route, RECORD)
            if path == f"api/account/generations/{RID}/reviews":
                return _json(route, {"items": state["reviews"]})
            if path == f"api/account/generations/{RID}/pr-status":
                return _json(route, state["pr_status"])
            return _json(route, {})

        context = browser.new_context(viewport={"width": 1400, "height": 1100})
        context.add_init_script(
            "window.SITE_CONFIG={authEnabled:true,supabase:{url:'',anon_key:''}};")
        page = context.new_page()
        page.route("**/api/**", api)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def open_record(reviews, *, is_admin=False):
            state.update(reviews=reviews, is_admin=is_admin, posts=[],
                         pr_status={"state": "idle"})
            page.goto("about:blank")
            page.goto(f"{base}/account/workspace.html#generations")
            page.wait_for_selector("[title=\"View this spec's YAML, reviews, and PR actions\"]",
                                   timeout=20000)
            page.locator("[title=\"View this spec's YAML, reviews, and PR actions\"]").first.click()
            page.wait_for_selector("[data-admission-line]", timeout=10000)
            return page

        try:
            yield open_record, state
        finally:
            browser.close()
            httpd.shutdown()


def _open_pr_button(page):
    return page.locator("aside button", has_text="Open").first


def test_needs_review_shows_the_groups_and_offers_a_draft(site):
    open_record, state = site
    page = open_record([_review({"job": "done", "mode": "dry_run",
                                 "report": _report("needs_review")})])
    line = page.locator("[data-admission-line]")
    assert "needs a maintainer's decision" in line.inner_text()
    assert "base abcdef1" in line.inner_text()
    assert "2 awaiting a decision" in line.inner_text()
    line.locator("button", has_text="Details").click()
    table = page.locator("[data-admission-table]").inner_text()
    assert "Needs a maintainer's decision" in table and "D1" in table
    assert "tests/t.py::test_sol" in table
    assert "Ready to admit" in table and "EXACT/acme__Model" in table
    assert "not in supported_models.txt" in table
    button = _open_pr_button(page)
    assert button.inner_text() == "Open draft PR" and button.is_enabled()
    # opening is a job: the page polls pr-status and reports the result
    state["pr_status"] = {"state": "done", "success": True, "message": "Opened as a draft.",
                          "events": [{"type": "done", "success": True, "draft": True,
                                      "pr_url": "https://github.com/o/r/pull/1"}],
                          "next_cursor": 1}
    button.click()
    page.wait_for_selector("text=Draft PR ready", timeout=10000)
    assert any(p.endswith("/create-pr") and json.loads(b) == {"draft_anyway": False}
               for p, b in state["posts"])


def test_ready_offers_a_plain_pr(site):
    open_record, _ = site
    page = open_record([_review({"job": "done", "report": _report("ready")})])
    assert "ready to admit" in page.locator("[data-admission-line]").inner_text()
    assert _open_pr_button(page).inner_text() == "Open PR"


def test_blocked_by_base_drift_disables_open_pr_for_everyone(site):
    open_record, _ = site
    page = open_record([_review({"job": "done", "report": _report("blocked", "base")})],
                       is_admin=True)
    assert "blocked (main's baseline)" in page.locator("[data-admission-line]").inner_text()
    assert not _open_pr_button(page).is_enabled()
    assert "baseline needs a refresh" in page.locator("[data-pr-gate]").inner_text()
    assert page.locator("button", has_text="Open draft PR anyway").count() == 0


def test_blocked_by_checks_lets_only_an_admin_open_a_draft(site):
    open_record, state = site
    blocked = [_review({"job": "done", "report": _report("blocked", "candidate_or_rule")})]
    page = open_record(blocked)
    assert not _open_pr_button(page).is_enabled()
    assert "Agent Fix" in page.locator("[data-pr-gate]").inner_text()
    assert page.locator("button", has_text="Open draft PR anyway").count() == 0
    page = open_record(blocked, is_admin=True)
    anyway = page.locator("button", has_text="Open draft PR anyway")
    assert anyway.count() == 1
    anyway.click()
    page.click("[data-dialog=confirm] button[type=submit]")  # the site's own confirm
    for _ in range(50):
        if state["posts"]:
            break
        page.wait_for_timeout(100)
    assert any(p.endswith("/create-pr") and json.loads(b) == {"draft_anyway": True}
               for p, b in state["posts"])


@pytest.mark.parametrize("review,expect,enabled", [
    (_review({"job": "running", "report": None}), "measuring this spec against main", False),
    (_review({"job": "done", "report": _report("ready")}, current=False), "stale", False),
    (_review(None, unavailable=True), "cannot store admission reports", True),
    (_review({"job": "failed", "error": "disk full"}), "failed (disk full)", True),
    (_review({"job": "done", "report": {**_report("unavailable"),
                                        "gates": {"G0": ["main is unreachable"]}}}),
     "main is unreachable", True),
])
def test_the_other_states_say_what_they_are(site, review, expect, enabled):
    open_record, _ = site
    page = open_record([review])
    assert expect in page.locator("[data-admission-line]").inner_text()
    assert _open_pr_button(page).is_enabled() is enabled
