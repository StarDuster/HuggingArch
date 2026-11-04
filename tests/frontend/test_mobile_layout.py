"""Narrow-width layout checks for every React page.

Same harness as `test_pages_smoke.py` (built `web/dist`, stubbed backend,
headless chromium). No page may scroll sideways at a phone width (375 px) or
across the top nav's collapse boundary (1024 px = lg, where the tab row comes
back, and 1200 px, just under xl, where the API badge / display name come
back; the tagline waits for 2xl): wide tables, charts and graphs scroll inside their own
containers. The header must also fit when a long signed-in email is the
display name. The top nav folds into a menu button below lg; the menu opens,
navigates and closes.
"""

from __future__ import annotations

import pytest

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401  (fixture)
from tests.frontend.test_playground_frontend import (
    _BINDINGS_BY_COMPONENT, _COMPONENTS, _DIST, _bench_payload,
)

_PHONE = {"width": 375, "height": 812}
_STUBS = {"/api/playground/components": _COMPONENTS}
_WIDTHS = [375, 1024, 1200]
_LONG_EMAIL = "someone.with.a.rather.long.address@example-domain.com"


def _overflow(page) -> tuple[int, int]:
    return page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("path, ready", [
    ("/", "#model-id"),
    ("/gallery.html", "#g-status"),
    ("/inference.html", "#infer-scaling"),
    ("/inference.html#sub=sweep&m=deepseek-ai%2FDeepSeek-V3%7Cdeepseek-ai%2FDeepSeek-V3.2", "#ic-run"),
    ("/playground.html", "#pg-groups"),
    ("/fusion.html", "main"),
    ("/kernel-corpus.html", "#f-source"),
    ("/kernel-corpus.html#view=workload&model=deepseek-ai%2FDeepSeek-V3&batch=16&context=4096", "#f-batch"),
    ("/kernel-corpus.html#view=providers", "#providers-section"),
    ("/privacy", "h1"),
    ("/auth/login.html", "body"),
    ("/auth/verify.html", "body"),
    ("/auth/reset-password.html", "body"),
    ("/account/profile.html", "body"),
    ("/account/workspace.html", "body"),
    ("/admin/", "body"),
])
def test_page_has_no_horizontal_overflow(browser_base, path, ready, width):  # noqa: F811
    browser, base = browser_base
    # /privacy and /admin/ are extensionless SPA routes the stub router does
    # not pass through; serve them the app shell directly.
    shell = ("text/html", (_DIST / "index.html").read_text())
    s = Session(browser, base, {**_STUBS, "/privacy": shell, "/admin/": shell})
    s.page.set_viewport_size({"width": width, "height": 812})
    s.goto(path)
    s.page.wait_for_selector(ready, state="attached", timeout=15000)
    s.page.wait_for_timeout(1500)  # let the load-time fetches render
    scroll_w, view_w = _overflow(s.page)
    assert scroll_w <= view_w, f"{path}: page is {scroll_w}px wide in a {view_w}px viewport"
    assert not [e for e in s.errors if e.startswith("pageerror")], s.errors


@pytest.mark.parametrize("width", [375, 768, 1124])
def test_long_spec_model_ids_stay_readable(browser_base, width):  # noqa: F811
    """The model id is a row's identity — never sacrifice it to badges/dates.

    The account sidebar makes the generations card much narrower than the
    viewport. A row that turns horizontal solely on viewport width can collapse
    the model link to a sliver even on a desktop-sized window. Keep the id on
    its own clamped line until the card is genuinely wide enough for the table.
    """
    browser, base = browser_base
    user = {"id": "u1", "email": "mobile@example.com", "user_metadata": {}}
    model_id = "a-very-long-organization/An-Extremely-Long-Model-Name-For-Mobile-Layout"
    s = Session(browser, base, {
        **_STUBS,
        # The cookie flow's session probe: signed in, no username, not admin.
        "/api/auth/me": {"authEnabled": True, "user": user,
                         "profile": {"username": None}, "isAdmin": False},
        "/api/account/agent-credentials": {"items": [], "byok_enabled": True},
        "/api/account/agent-credentials/models": {"models": []},
        "/api/account/tokens": {"tokens": []},
        "/api/account/generations": {
            "items": [
                {"id": "g1", "model_id": model_id, "status": "generated",
                 "agent_cli": "claude", "created_at": "2026-09-28T01:00:00Z",
                 "release_date": "2026-09-27", "latest_verdict": "no_pass"},
                {"id": "g2", "model_id": model_id, "status": "generated",
                 "agent_cli": "claude", "created_at": "2026-09-27T01:00:00Z",
                 "release_date": "2026-09-27"},
            ],
            "orphan_logs": [],
        },
    })
    s.page.set_viewport_size({"width": width, "height": 900})
    s.goto("/account/workspace.html#generations")
    s.page.wait_for_selector("[data-model-name]", timeout=15000)
    s.page.wait_for_timeout(600)

    metrics = s.page.evaluate(
        """() => {
            const name = document.querySelector('[data-model-name]');
            const hf = document.querySelector('a[href^="https://huggingface.co/"]');
            const row = name.closest('[role="button"]');
            const runs = [...row.querySelectorAll("button")]
                .find((button) => button.textContent.includes("runs"));
            const n = name.getBoundingClientRect();
            const b = runs.getBoundingClientRect();
            const h = hf.getBoundingClientRect();
            return {
                nameWidth: n.width,
                nameHeight: n.height,
                nameBottom: n.bottom,
                hfWidth: h.width,
                runsTop: b.top,
                rowScrollWidth: row.scrollWidth,
                rowClientWidth: row.clientWidth,
            };
        }"""
    )
    assert metrics["nameWidth"] >= 150, metrics
    assert metrics["nameHeight"] <= 48, metrics
    assert metrics["hfWidth"] <= 32, metrics
    assert metrics["runsTop"] >= metrics["nameBottom"] - 1, metrics
    assert metrics["rowScrollWidth"] <= metrics["rowClientWidth"] + 1, metrics
    scroll_w, view_w = _overflow(s.page)
    assert scroll_w <= view_w, f"page is {scroll_w}px wide in a {view_w}px viewport"
    s.assert_clean()
    s.page.close()


@pytest.mark.parametrize("width", [768, 1024, 1100, 1200, 1280, 1440])
def test_header_fits_with_a_long_signed_in_email(browser_base, width):  # noqa: F811
    """Signed in with no username, the display name is the email — the
    longest thing the header ever shows."""
    browser, base = browser_base
    user = {"id": "u1", "email": _LONG_EMAIL, "user_metadata": {}}
    s = Session(browser, base, {
        **_STUBS,
        # Signed in via the cookie flow's session probe; no username, so the
        # header's display name is the email.
        "/api/auth/me": {"authEnabled": True, "user": user,
                         "profile": {"username": None}, "isAdmin": False},
    })
    s.page.set_viewport_size({"width": width, "height": 812})
    s.goto("/fusion.html")
    s.page.wait_for_selector("#auth-userbox:not(.hidden)", timeout=15000)
    s.page.wait_for_timeout(1000)
    row_w, row_client = s.page.evaluate(
        "(() => { const r = document.querySelector('#top-nav > div > div');"
        " return [r.scrollWidth, r.clientWidth]; })()")
    assert row_w <= row_client, f"header row is {row_w}px in {row_client}px at {width}"
    scroll_w, view_w = _overflow(s.page)
    assert scroll_w <= view_w, f"page is {scroll_w}px wide in a {view_w}px viewport"
    s.assert_clean()


def test_top_nav_menu_toggles_and_navigates_on_a_phone(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, _STUBS)
    s.page.set_viewport_size(_PHONE)
    s.goto("/inference.html")
    s.page.wait_for_selector("#nav-menu-btn", timeout=15000)
    assert not s.page.is_visible("a[data-tab='fusion']")  # desktop tab row hidden
    assert s.page.query_selector("#nav-menu") is None

    s.page.click("#nav-menu-btn")
    s.page.wait_for_selector("#nav-menu", state="visible")
    assert s.page.get_attribute("#nav-menu-btn", "aria-expanded") == "true"
    assert len(s.page.query_selector_all("#nav-menu a")) == 6
    assert s.page.get_attribute("a[data-menu-tab='inference']", "aria-current") == "page"

    s.page.click("#nav-menu-btn")  # second tap closes
    s.page.wait_for_selector("#nav-menu", state="detached")

    s.page.click("#nav-menu-btn")
    s.page.keyboard.press("Escape")  # Escape closes
    s.page.wait_for_selector("#nav-menu", state="detached")

    s.page.click("#nav-menu-btn")
    s.page.click("a[data-menu-tab='fusion']")  # navigating closes
    s.page.wait_for_url("**/fusion.html")
    s.page.wait_for_selector("#nav-menu", state="detached")
    assert _overflow(s.page)[0] <= _PHONE["width"]
    s.assert_clean()


def test_playground_with_results_fits_a_phone(browser_base):  # noqa: F811
    """With a compare on screen, the page must not scroll sideways, the
    graph-diff box is 55vh tall, and the summary starts within a screen of
    the top of the results."""
    browser, base = browser_base
    payload = _bench_payload({
        "entries": [
            {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3",
             "dims": {"d_kv_lora": 512}, "seeded": True},
            {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3.2",
             "dims": {"d_kv_lora": 512, "d_rope": 64}, "seeded": True},
        ],
    })
    s = Session(browser, base, {
        "/api/playground/components": _COMPONENTS,
        "/api/playground/module-bench": payload,
        "/api/playground/component-dims": {
            "success": True, "component": "mla_attention",
            "bindings": _BINDINGS_BY_COMPONENT["mla_attention"], "failed": {},
        },
    })
    s.page.set_viewport_size(_PHONE)
    s.goto("/playground.html")
    s.page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=15000)
    s.page.click("button.uppercase:has-text('MLA')")
    s.page.click("#pg-groups >> text=mla_attention")
    s.page.click("#pg-run-picks")
    s.page.wait_for_selector("#pg-graph-diff", timeout=20000)
    s.page.wait_for_timeout(800)
    scroll_w, view_w = _overflow(s.page)
    assert scroll_w <= view_w, f"playground with results is {scroll_w}px wide in a {view_w}px viewport"
    box = s.page.locator("[data-diff-box]")
    box.wait_for(state="visible")
    h = box.evaluate("e => e.getBoundingClientRect().height")
    vh = s.page.evaluate("window.innerHeight")
    assert abs(h - 0.55 * vh) <= 1, (h, vh)
    # The structure section (title + the bounded diff box) takes at most one
    # screen, so the summary follows it within a screen. Measured from the top
    # of the results, not of the page: the controls above them (precision
    # presets, wrapping picks) are not what this bound is about.
    top = "e => e.getBoundingClientRect().top + window.scrollY"
    results_y = s.page.locator("#pg-structure").evaluate(top)
    y = s.page.locator("#pg-summary").evaluate(top)
    assert y - results_y <= _PHONE["height"], (results_y, y)

    bar = s.page.locator("#pg-picks")
    assert bar.is_visible()
    assert s.page.locator("#pg-edit-picks").count() == 0
    assert bar.evaluate("e => getComputedStyle(e).position") == "sticky"
    # The bar wraps (8dd8ac2a2): two chips do not fit one 375px row, so it is
    # one row per line of chips, and no chip breaks its own text.
    assert bar.evaluate("e => getComputedStyle(e).flexWrap") == "wrap"
    items = bar.evaluate(
        "e => [...e.children].map(c => c.getBoundingClientRect())"
        ".filter(r => r.height > 0).map(r => [Math.round(r.top), r.height])")
    assert all(h <= 36 for _, h in items), f"a chip broke onto two lines: {items}"
    rows = len({t for t, _ in items})
    bar_h = bar.evaluate("e => e.getBoundingClientRect().height")
    assert bar_h <= 56 + 36 * (rows - 1), f"{rows} row(s) of chips in {bar_h}px"
    s.page.locator("#pg-summary").scroll_into_view_if_needed()
    s.page.wait_for_timeout(200)
    stuck = bar.bounding_box()
    nav = s.page.locator("#top-nav").bounding_box()
    assert stuck is not None and nav is not None
    assert stuck["y"] + stuck["height"] > 0
    assert stuck["y"] >= nav["y"]
    assert stuck["y"] <= nav["y"] + nav["height"] + 8
    assert _overflow(s.page)[0] <= _PHONE["width"]


def test_top_nav_keeps_the_tab_row_on_desktop(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, _STUBS)
    s.page.set_viewport_size({"width": 1440, "height": 900})
    s.goto("/fusion.html")
    s.page.wait_for_selector("a[data-tab='fusion']", state="visible", timeout=15000)
    assert not s.page.is_visible("#nav-menu-btn")
    s.assert_clean()
