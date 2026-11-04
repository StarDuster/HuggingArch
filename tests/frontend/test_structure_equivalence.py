"""The structure view, fully opened, IS the detailed graph.

For one component per kind of box the structure view draws — shared inputs
and empty role paths (MLA, DSA), a recurrence with a parallel branch (KDA),
folded per-head parameters (Mamba2), routed experts with a latent path and
shared experts (Kimi-K3 MoE), a fused QKV projection — open every box in the
Structure view, then open every compound in the Detailed view, and compare
what is drawn: the same leaf nodes and the same edges, endpoint for endpoint.
The structure view's `input` box stands for the component's input node.

Reads the drawn DOM (`data-node` on node slots, `data-edge` on edge paths),
so it checks what a reader sees, not the data behind it. Skips when the app
is not built or chromium is absent.
"""
from __future__ import annotations

import json

import pytest

from tests.frontend.test_playground_frontend import _DIST, _ensure_built, _serve_dist
from tests.frontend.test_structure_frontend import _dims_route, _picks, _response

COMPONENTS = [
    "mla_dsa_attention",
    "mla_attention",
    "kimi_k3_kda_attention",
    "nemotron_h_mamba_mixer",
    "kimi_k3_latent_moe",
    "gqa_fused_qkv_attention",
]


def _slot(node):
    return f".dag-node-slot[data-node={json.dumps(node)}]"


def _open_all(sec, anchor):
    """Open every closed box of the drawn view. `anchor` is a node this view
    always draws (its input), so "no closed box" is only read off a drawing
    that exists: the ELK layout is asynchronous and a view whose layout has
    not landed yet has no slots at all."""
    for _ in range(80):
        sec.locator(_slot(anchor)).wait_for(timeout=30000)
        closed = sec.locator(".dag-node-slot[data-openable='1']").evaluate_all(
            "els => els.map(e => e.dataset.node)")
        if not closed:
            return
        sec.locator(_slot(closed[0])).dispatch_event("click")  # the element itself, not whatever is drawn over it
        # The opened box is drawn as a compound once the new layout lands.
        sec.locator(f"[data-compound={json.dumps(closed[0])}]").wait_for(timeout=30000)
    raise AssertionError("boxes kept appearing closed")


def _drawn(sec):
    nodes = set(sec.locator(".dag-node-slot").evaluate_all("els => els.map(e => e.dataset.node)"))
    edges = set(sec.locator("path[data-edge]").evaluate_all("els => els.map(e => e.dataset.edge)"))
    return nodes, edges


@pytest.mark.parametrize("name", COMPONENTS)
def test_opened_structure_view_draws_the_detailed_graph(name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from backend.playground import module_bench as mb

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    from backend.arch.block_synthesis import binding_of
    from tests.snapshots import resolve_snapshot
    model = mb._component_to_models()[name][0]
    # The page opens a component at one of the model's calls of it.
    resolved = resolve_snapshot(model)
    call = next(site.call for site in mb.CallSite.calls(resolved)
                if binding_of(resolved.spec, site.component, name))
    resp = _response([name], base_model=model, call=call)
    # The Detailed view draws the component's own input node; Structure draws
    # the `l1:root:input` box in its place.
    detailed_input = next(n["id"] for n in resp["results"][0]["block_graph"]["nodes"]
                          if n.get("op") == "input")
    comps = {"success": True, "components": mb.list_library_components()}
    role = next(c["role"] for c in comps["components"] if c["name"] == name)

    httpd = _serve_dist()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1600, "height": 3000})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload, default=str))

            page.add_init_script("window.SITE_CONFIG={authEnabled:false,supabase:{url:'',anon_key:''}};")
            page.route("**/api/**", lambda r: _json(r, {}))
            page.route("**/supported-models", lambda r: _json(r, {"model_ids": []}))
            page.route("**/api/playground/components", lambda r: _json(r, comps))
            dims_route = _dims_route(model)
            page.route("**/api/playground/component-dims**",
                       lambda r: _json(r, dims_route(r.request.url)))
            page.route("**/api/playground/module-bench", lambda r: _json(r, resp))

            page.goto(f"{base}/playground.html?" + _picks([name], call=call, model=model, role=role))
            page.wait_for_selector("#pg-structure", timeout=30000)
            sec = page.locator("#pg-structure")

            _open_all(sec, "l1:root:input")
            s_nodes, s_edges = _drawn(sec)
            sec.locator("button", has_text="Detailed").click()
            _open_all(sec, detailed_input)
            d_nodes, d_edges = _drawn(sec)
            browser.close()
    finally:
        httpd.shutdown()

    boxes = {n for n in s_nodes if n.startswith("l1:")}
    inputs = d_nodes - s_nodes
    assert boxes == {"l1:root:input"} and len(inputs) == 1, (boxes, inputs)
    stand_in = {"l1:root:input": next(iter(inputs))}
    s_nodes = {stand_in.get(n, n) for n in s_nodes}
    s_edges = {"->".join(stand_in.get(x, x) for x in e.split("->")) for e in s_edges}
    assert s_nodes == d_nodes
    assert s_edges == d_edges
