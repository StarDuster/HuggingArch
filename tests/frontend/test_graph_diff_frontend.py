"""Headless test for the Playground's compute-graph diff view.

Responses are computed in-process with `module_bench` on offline snapshots
(the same calls as `tests/backend/views/test_diff.py`), so the page is driven
with the backend's real diff; the built `web/dist` is served from a throwaway
server and the module-bench POST answered with the chosen response. Skips
cleanly when the app is not built or playwright/chromium is absent.

Asserts the page draws what the backend aligned: the shared skeleton as
Structure-view boxes and each structure's own paths as one coloured node that
opens in place; a changed node's hover carrying its per-entry attributes and
the "because"; KV marks on the existing badges; a legend of what is present;
three entries; two bindings of one component; the backend's refusal; and a
wide diff scrolling inside its own box on a phone.
"""
from __future__ import annotations

import json

import pytest

from tests.frontend.test_playground_frontend import _DIST, _call, _ensure_built, _ensure_picker, _run_picks, _serve_dist

_GLA = "gla_qknorm_partial_rope_sigmoid_gated_attention"
_CASES = {
    "gqa_mla": ("deepseek-ai/DeepSeek-V3", [
        {"name": "gqa_qknorm_attention", "model_id": "Qwen/Qwen3-8B"},
        {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3"}]),
    "linear": ("Qwen/Qwen3-Coder-Next", [
        {"name": "gated_delta_net_attention", "model_id": "Qwen/Qwen3.5-35B-A3B"},
        {"name": "kimi_k3_kda_attention", "model_id": "moonshotai/Kimi-K3"},
        {"name": "nemotron_h_mamba_mixer",
         "model_id": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"}]),
    "same": ("deepseek-ai/DeepSeek-V3.2", [
        {"name": "mla_dsa_attention", "model_id": "deepseek-ai/DeepSeek-V3.2"},
        {"name": "mla_dsa_attention", "model_id": "zai-org/GLM-5"}]),
    "refused": ("deepseek-ai/DeepSeek-V3", [
        {"name": "mla_attention", "model_id": None},
        {"name": _GLA, "model_id": None}]),
}


def _bench(base, entries):
    from backend.playground import module_bench as mb
    from tests.snapshots import snapshot_config
    snapshot_config(base)
    for e in entries:
        if e["model_id"]:
            snapshot_config(e["model_id"])
    from backend.arch.eval.precision import AS_SHIPPED
    out = mb.module_bench(base_model_id=base, entries=entries, batch=1,
                          input_tokens=512, precision=AS_SHIPPED)
    out["success"] = True
    return json.loads(json.dumps(out, default=str))


@pytest.fixture(scope="module")
def responses():
    return {k: _bench(*v) for k, v in _CASES.items()}


def _class_named(resp, i, name):
    """Test-side reading aid: the class whose entry-`i` member is `name`."""
    nodes = {n["id"]: n for n in resp["results"][i]["block_graph"]["nodes"]}
    return next(c for c in resp["graph_diff"]["classes"]
                if c["members"][i] and nodes[c["members"][i]].get("tensor_name") == name)


@pytest.fixture(scope="module")
def site():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from backend.playground.module_bench import list_library_components

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    comps = {"success": True, "components": list_library_components()}
    httpd = _serve_dist()
    current: dict = {}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # chromium not installed
            httpd.shutdown()
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1400, "height": 1200})

        def _json(route, payload):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

        page.add_init_script("window.SITE_CONFIG={authEnabled:false,supabase:{url:'',anon_key:''}};")
        page.route("**/api/**", lambda r: _json(r, {}))
        page.route("**/supported-models", lambda r: _json(r, {"model_ids": []}))
        page.route("**/api/playground/components", lambda r: _json(r, comps))
        page.route("**/api/playground/component-dims**",
                   lambda r: _json(r, {"success": True, "bindings": [
                       {"model_id": "deepseek-ai/DeepSeek-V3", **_call("mla_attention"), "dims": {}}]}))
        page.route("**/api/playground/module-bench", lambda r: _json(r, current["resp"]))
        page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/playground.html")
        page.wait_for_selector("button.uppercase:has-text('MLA')", timeout=20000)
        page.click("button.uppercase:has-text('MLA')")

        def show(resp):
            """Answer the next bench with `resp`; re-pick to trigger one and
            wait until the page draws it."""
            current["resp"] = resp
            # The page skips a bench whose request it just sent; pick a
            # different chip each time so every call asks anew.
            chips = ['"mla_attention"', '"mla_attention_absorbed"']
            k = current.get("k", -1)
            _ensure_picker(page)
            if k >= 0:
                page.click(f"#pg-groups >> text={chips[k % 2]}")   # unpick
            page.click(f"#pg-groups >> text={chips[(k + 1) % 2]}")
            _run_picks(page)
            current["k"] = k + 1
            if resp["graph_diff"]["available"]:
                r = resp["results"][0]
                label = f"{r['component']} @ {r['model_id'].split('/')[-1]}" if r["model_id"] else r["entry"]
                page.wait_for_function(
                    "([label, n]) => { const l = document.querySelector('#pg-graph-diff [data-legend-item=\"entry:0\"]');"
                    " return !!l && l.innerText.trim().startsWith(label)"
                    " && document.querySelectorAll('#pg-graph-diff [data-legend-item^=\"entry:\"]').length === n; }",
                    arg=[label, len(resp["results"])], timeout=30000)
                page.wait_for_selector("#pg-graph-diff .dag-node-slot", timeout=30000)
                page.wait_for_timeout(300)
            else:
                page.wait_for_selector("[data-diff-refused]", timeout=30000)
            return page.locator("#pg-graph-diff")

        try:
            yield page, show
        finally:
            browser.close()
            httpd.shutdown()


def _slot(sec, node_id):
    return sec.locator(f".dag-node-slot[data-node='{node_id}']")


def _tip_text(page, loc):
    """Hover `loc` and read the shared `#ha-tip` popover's text; the tip has a
    200ms hide delay, so it is readable right after the hover lands."""
    loc.hover()
    text = page.locator("#ha-tip").inner_text()
    page.mouse.move(0, 0)
    return text


def _tip_shown(page, loc):
    """Whether hovering `loc` opens the shared tip at all."""
    loc.hover()
    page.wait_for_timeout(100)
    shown = page.locator("#ha-tip").is_visible()
    page.mouse.move(0, 0)
    return shown


def test_skeleton_boxes_and_path_nodes_open_in_place(site, responses):
    page, show = site
    resp = responses["gqa_mla"]
    gd = resp["graph_diff"]
    sec = show(resp)
    # the shared skeleton: its Structure-view boxes, every one paired but changed
    for b in gd["skeleton_boxes"]:
        if b["id"] == "input":
            continue
        box = _slot(sec, f"db:{b['id']}")
        assert box.count() == 1 and b["label"] in box.inner_text()
        assert "s-diff-changed" in box.locator(":scope > .dag-node-box").get_attribute("class")
    # one node per path, in its owner's colour, with its sums per owner
    for p in gd["paths"]:
        node = _slot(sec, f"dp:{p['id']}")
        assert node.count() == 1, p["id"]
        assert f"s-diff-u{p['owners'][0]}" in node.locator(":scope > .dag-node-box").get_attribute("class")
        assert f"{len(p['classes'])} node" in node.inner_text()
    # the path's hover is rendered markup (the 'only in' pill), not its source
    tip = _tip_text(page, _slot(sec, f"dp:{gd['paths'][0]['id']}").locator(":scope > .dag-node-box"))
    assert tip.startswith("only in ") and "<span" not in tip
    # GQA's Q projection and MLA's low-rank Q are alternatives into one port:
    # one note names the group, the paths stay ordinary nodes
    q_gqa = next(p for p in gd["paths"] if _class_named(resp, 0, "q_proj")["id"] in p["classes"])
    q_mla = next(p for p in gd["paths"] if _class_named(resp, 1, "q_a_proj")["id"] in p["classes"])
    assert q_gqa["group"] == q_mla["group"]
    assert "2 alternatives" in _slot(sec, f"dg:{q_gqa['group']}").inner_text()
    # open MLA's Q path: its classes appear; close it: they are gone again
    inner = [f"dc:{c}" for c in q_mla["classes"]]
    assert _slot(sec, inner[0]).count() == 0
    _slot(sec, f"dp:{q_mla['id']}").click(force=True)
    _slot(sec, inner[0]).wait_for(timeout=20000)
    assert all(_slot(sec, c).count() == 1 for c in inner)
    sec.locator(f"[data-compound='dp:{q_mla['id']}'] .dag-elk-collapse-tab").click(force=True)
    _slot(sec, inner[0]).wait_for(state="detached", timeout=20000)
    _slot(sec, f"dp:{q_mla['id']}").wait_for(timeout=20000)


def test_changed_nodes_say_what_differs_in_their_rows(site, responses):
    page, show = site
    resp = responses["gqa_mla"]
    sec = show(resp)
    o_proj = _class_named(resp, 0, "o_proj")
    box = _slot(sec, f"db:{o_proj['boxes'][0]}")
    # the shared box answers no hover of its own (no blank popup)
    assert not _tip_shown(page, box.locator(":scope > .dag-node-box"))
    box.click(force=True)
    node = _slot(sec, f"dc:{o_proj['id']}")
    node.wait_for(timeout=20000)
    card = node.locator(":scope > .dag-node-box")
    assert "s-diff-changed" in card.get_attribute("class")
    rows = card.locator("[data-compare-entry]")
    assert rows.count() == 2
    assert "[1,512,4096]" in rows.nth(0).inner_text() and "[1,512,16384]" in rows.nth(1).inner_text()
    # the rows are the answer; the changed card carries no hover dump
    assert not _tip_shown(page, card)


def test_kv_marks_and_the_legend_come_from_what_is_drawn(site, responses):
    _page, show = site
    resp = responses["gqa_mla"]
    gd = resp["graph_diff"]
    sec = show(resp)
    v_path = next(p for p in gd["paths"] if _class_named(resp, 0, "v_proj")["id"] in p["classes"])
    assert _slot(sec, f"dp:{v_path['id']}").locator("[data-kv='write']").inner_text() == "KV write 2.00 KiB/tok"
    latent = next(p for p in gd["paths"] if _class_named(resp, 1, "kv_a_proj_with_mqa")["id"] in p["classes"])
    badges = _slot(sec, f"dp:{latent['id']}").locator("[data-kv]")
    assert {badges.nth(k).get_attribute("data-kv") for k in range(badges.count())} == {"write", "read"}
    legend = sec.locator("[data-diff-legend]")
    items = {legend.locator("[data-legend-item]").nth(k).get_attribute("data-legend-item")
             for k in range(legend.locator("[data-legend-item]").count())}
    assert items == {"entry:0", "entry:1", "skeleton", "only", "changed", "group", "edge:entry"}
    assert "gqa_qknorm_attention @ Qwen3-8B" in legend.inner_text()
    kv_legend = sec.locator("[data-legend-item='kv:write'], [data-legend-item='kv:read']")
    assert kv_legend.count() == 2
    assert sec.locator("[data-edge-tone='0']").count() > 0 and sec.locator("[data-edge-tone='1']").count() > 0
    # two bindings of one component: every node paired, so no path and no
    # per-entry legend line for one
    sec = show(responses["same"])
    assert sec.locator(".dag-node-slot[data-node^='dp:']").count() == 0
    assert sec.locator("[data-legend-item='only']").count() == 0
    assert sec.locator("[data-legend-item='entry:1']").count() == 1


def test_the_box_keeps_its_height_and_open_paths_across_responses(site, responses):
    """The box is 600 px from the placeholder on, so nothing below it moves
    when the layout lands or a later response arrives; a later response
    re-lays the graph out in place, so a path the reader opened stays open."""
    page, show = site
    resp = responses["gqa_mla"]
    sec = show(resp)
    box = sec.locator("[data-diff-box]")
    height = "e => e.getBoundingClientRect().height"
    assert box.evaluate(height) == 600
    slot = sec.locator(".dag-node-slot[data-openable='1'][data-node^='dp:']").first
    node = slot.get_attribute("data-node")
    slot.click()
    sec.locator(f"[data-compound='{node}']").wait_for(timeout=10000)
    # a new response for the same selection (the contract changed)
    _ensure_picker(page)
    with page.expect_response("**/api/playground/module-bench"):
        page.locator("input[type=number][min='1']").first.fill("2")
    page.wait_for_timeout(800)
    assert sec.locator(f"[data-compound='{node}']").count() == 1
    assert box.evaluate(height) == 600
    # a drawing wider than the box says so at the box's right edge
    drawing = sec.locator("[data-diff-scroll] .dag-elk-nodes")
    wider = drawing.evaluate(
        "n => { const e = n.parentElement; return e.scrollLeft + e.clientWidth < e.scrollWidth - 1; }")
    assert sec.locator("[data-diff-more-right]").count() == (1 if wider else 0)
    with page.expect_response("**/api/playground/module-bench"):
        page.locator("input[type=number][min='1']").first.fill("1")


def test_three_entries(site, responses):
    _page, show = site
    resp = responses["linear"]
    gd = resp["graph_diff"]
    sec = show(resp)
    assert sec.locator("[data-legend-item='entry:2']").count() == 1
    mamba = [p for p in gd["paths"] if p["owners"] == [2]]
    assert mamba and all("s-diff-u2" in _slot(sec, f"dp:{p['id']}").locator(":scope > .dag-node-box")
                         .get_attribute("class") for p in mamba)
    assert sec.locator(".dag-node-slot[data-node^='dg:']").count() == len(gd["groups"])
    core = next(c for c in gd["classes"] if all(c["members"]) and c["diff"].get("op"))
    _slot(sec, f"db:{core['boxes'][0]}").click(force=True)
    node = _slot(sec, f"dc:{core['id']}")
    node.wait_for(timeout=20000)
    assert node.locator("[data-compare-entry]").count() == 3
    assert node.locator("[data-kv='state']").count() == 3


def test_the_backends_refusal_is_shown(site, responses):
    page, show = site
    resp = responses["refused"]
    assert resp["graph_diff"]["available"] is False
    sec = show(resp)
    msg = page.inner_text("[data-diff-refused]")
    assert "No compute-graph diff" in msg and _GLA in msg
    if len(resp["results"]) == 1:
        # the entry that did bench still draws its graph
        assert sec.locator(".dag-node-slot").count() > 0
        return
    # Overlay refused → each entry's own graph is stacked, named so they
    # can be told apart (the same label the summary table and overlay
    # legend would have used).
    labels = [t.strip() for t in sec.locator("[data-graph-label]").all_inner_texts()]
    expected = []
    for r in resp["results"]:
        if r.get("model_id"):
            expected.append(f"{r['component']} @ {r['model_id'].split('/')[-1]}")
        else:
            expected.append(r.get("entry") or r["component"])
    assert labels == expected
    assert sec.locator("[data-graph-label]").count() == len(resp["results"])


def test_a_wide_diff_scrolls_inside_its_own_box_on_a_phone(site, responses):
    page, show = site
    sec = show(responses["linear"])
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(800)
    scroll_w, view_w = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    assert scroll_w <= view_w, (scroll_w, view_w)
    inner = sec.locator(".dag-svg-container").first
    assert inner.evaluate("e => e.scrollWidth > e.clientWidth && getComputedStyle(e).overflowX === 'auto'")
    page.set_viewport_size({"width": 1400, "height": 1200})
