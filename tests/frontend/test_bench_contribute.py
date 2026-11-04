"""Contributing measurement batches: when the batches about to be nominated
(a set's private ones) hold rows faster than a peak they recorded, the
dialog offers to leave them out, and the choice rides the nomination as
`within_peak`. A run whose batches differ shows each status. Above each
table, "Select page" and "Select all" pick the page or every batch."""
from __future__ import annotations

import json

import pytest

from tests.frontend.test_playground_frontend import _DIST, _ensure_built, _serve_dist

def _batch(upload_id: str, status: str, rows: int, anomalous: int) -> dict:
    return {"upload_id": upload_id, "status": status, "row_count": rows, "anomalous": anomalous}


# Run a: an earlier batch already shared (700 over-peak rows went in with
# it), a later one still private (34 over peak) — what contribute nominates.
_SETS = [
    {"kind": "run", "id": "run-a", "run_id": "run-a", "label": "sweep a", "row_count": 70893,
     "status": "private", "anomalous": 734, "mfu_over": 674, "mbu_over": 60, "checked": 70893,
     "batches": [_batch("a1", "shared", 70000, 700), _batch("a2", "private", 893, 34)]},
    {"kind": "run", "id": "run-b", "run_id": "run-b", "label": "sweep b", "row_count": 100,
     "status": "private", "anomalous": 0, "mfu_over": 0, "mbu_over": 0, "checked": 100,
     "batches": [_batch("b1", "private", 100, 0)]},
]


# More batches than one page (50) holds.
_MANY = [{"kind": "run", "id": f"run-{i:03d}", "run_id": f"run-{i:03d}", "label": f"many {i:03d}",
          "row_count": 1, "status": "private", "anomalous": 0, "mfu_over": 0, "mbu_over": 0,
          "checked": 1, "batches": [_batch(f"m{i}", "private", 1, 0)]}
         for i in range(62)]


@pytest.fixture
def workspace(request):
    uploads = getattr(request, "param", _SETS)
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    httpd = _serve_dist()
    posts: list[str] = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # chromium not installed
            httpd.shutdown()
            pytest.skip(f"chromium unavailable: {exc}")

        def api(route):
            url = route.request.url
            path = url.split("://", 1)[1].split("/", 1)[1]
            if route.request.method == "POST":
                posts.append(path)
                body = {"nominated": True}
            elif path.startswith("api/auth/me"):
                body = {"authEnabled": True, "isAdmin": False,
                        "user": {"id": "u", "email": "u@example.com"}, "profile": {"username": "u"}}
            elif path.startswith("api/bench/list"):
                body = {"uploads": uploads}
            else:
                body = {}
            route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

        context = browser.new_context(viewport={"width": 1400, "height": 1000})
        context.add_init_script("window.SITE_CONFIG={authEnabled:true,supabase:{url:'',anon_key:''}};")
        page = context.new_page()
        page.route("**/api/**", api)
        page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/account/workspace.html#benchmarks")
        page.wait_for_selector(f"text={uploads[0]['label']}", timeout=20000)
        try:
            yield page, posts
        finally:
            browser.close()
            httpd.shutdown()


def _select(page, label):
    page.locator("tr", has_text=label).locator("input[type=checkbox]").check()


def test_over_peak_rows_can_be_left_out(workspace):
    page, posts = workspace
    _select(page, "sweep a")
    page.click("button:has-text('Contribute selected')")
    box = page.locator("[data-dialog=choice] input[type=checkbox]")
    box.wait_for(timeout=5000)
    # Only the private batch's over-peak rows are in question.
    assert "34 rows" in page.inner_text("[data-dialog=choice]")
    assert "734" not in page.inner_text("[data-dialog=choice]")
    assert box.is_checked()  # leaving them out is the offered default
    page.click("[data-dialog=choice] button[type=submit]")
    page.wait_for_selector("text=over-peak rows left out", timeout=5000)
    assert posts == ["api/bench/run/run-a/contribute?within_peak=true"]


def test_a_run_shows_the_status_of_each_batch(workspace):
    page, _posts = workspace
    status = page.locator("tr", has_text="sweep a").locator("td", has_text="×1").inner_text()
    assert "private" in status and "shared" in status and "×1" in status


def test_unticking_nominates_every_row(workspace):
    page, posts = workspace
    _select(page, "sweep a")
    page.click("button:has-text('Contribute selected')")
    page.locator("[data-dialog=choice] input[type=checkbox]").uncheck()
    page.click("[data-dialog=choice] button[type=submit]")
    page.wait_for_selector("text=pending admin review", timeout=5000)
    assert posts == ["api/bench/run/run-a/contribute"]


def test_a_clean_selection_asks_no_question_about_it(workspace):
    page, posts = workspace
    _select(page, "sweep b")
    page.click("button:has-text('Contribute selected')")
    page.wait_for_selector("[data-dialog=confirm]", timeout=5000)
    assert page.locator("[data-dialog=choice]").count() == 0
    page.click("[data-dialog=confirm] button[type=submit]")
    page.wait_for_selector("text=pending admin review", timeout=5000)
    assert posts == ["api/bench/run/run-b/contribute"]


@pytest.mark.parametrize("workspace", [_MANY], indirect=True)
def test_select_page_and_select_all_pick_the_two_scopes(workspace):
    page, _posts = workspace
    page.click("button:has-text('Select page (50)')")
    assert page.locator("text=50 selected").count() >= 1
    assert page.locator("button:has-text('Select page (50)')").is_disabled()
    page.click("button:has-text('Select all (62)')")
    assert page.locator("text=62 selected").count() >= 1
    assert page.locator("button:has-text('Select all (62)')").is_disabled()
    page.click("button:has-text('Clear')")
    assert page.locator("text=62 selected").count() == 0
