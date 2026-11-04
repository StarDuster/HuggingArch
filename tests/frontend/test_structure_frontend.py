"""Headless test for the Playground structure view (L1).

The response is built in-process from offline snapshots — `bench_component`
on DeepSeek-V3's config for plain and absorbed MLA, plus the backend's own
`graph_diff` — so the page is driven with real graphs and real L1 boxes, and
the test needs neither the network nor a running backend. Skips cleanly when
the app is not built or playwright/chromium is absent.

Asserts the page draws what the backend grouped: a role-aligned comparison
whose rows are the declared operand roles, "in Shared by K, V" for absorbed MLA's V
(it has no box of its own), folded under the compute-graph diff, and a
single-component Structure/Detailed switch whose boxes expand in place.
"""
from __future__ import annotations

import json
import re

import pytest

from tests.frontend.test_playground_frontend import _DIST, _ensure_built, _serve_dist


def _response(names, base_model="deepseek-ai/DeepSeek-V3", call=None):
    """`call` names the model's call each entry is evaluated at (`CallSite`),
    for a component the model calls more than once; None takes its only call."""
    from backend.arch.loader import SpecLoader
    from backend.playground import module_bench as mb
    from backend.arch.eval.precision import AS_SHIPPED
    from tests.snapshots import resolve_snapshot
    loader = SpecLoader()
    base = resolve_snapshot(base_model, loader)

    def site(n):
        if call is None:
            return None
        return mb.CallSite.of(mb.inject_component(base, n, loader), n, call=call)
    results = [mb.bench_component(base, n, batch=1, input_tokens=512, dim_overrides={}, loader=loader,
                                  precision=AS_SHIPPED, site=site(n))
               for n in names]
    for n, r in zip(names, results, strict=True):
        r["entry"] = n
    resp = {"success": True, "base_model_id": base_model,
            "contract": {"batch": 1, "input_tokens": 512, "dim_overrides": {},
                         "phases": mb.phase_contract(1, 512)},
            "results": results, "errors": []}
    if len(results) > 1:
        resp["graph_diff"] = mb.graph_diff(base, results, [], loader)
    return resp


def _dims_route(base_model):
    """`/api/playground/component-dims` as the backend answers it, for the
    one model the page is driven with: the bindings table's rows for that
    model (`component_bindings`, filtered to it), or one call's dims."""
    from urllib.parse import parse_qs, urlsplit
    from backend.arch.loader import SpecLoader
    from backend.arch.block_synthesis import binding_of
    from backend.playground import module_bench as mb
    from tests.snapshots import resolve_snapshot
    loader = SpecLoader()
    resolved = resolve_snapshot(base_model, loader)

    def answer(url):
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        component = q["component"]
        if "model_id" in q:
            return {"success": True, "component": component, "model_id": q["model_id"],
                    "call": q.get("call"), "dims": mb.component_dims_from_model(
                        component, q["model_id"], q.get("call"))}
        sites = [site for site in mb.CallSite.calls(resolved)
                 if binding_of(resolved.spec, site.component, component)]
        rows = (mb._model_bindings(resolved, base_model, component, loader, sites)
                if sites else [])
        return {"success": True, "component": component, "bindings": rows, "failed": {}}
    return answer


def _picks(names, *, call, model="deepseek-ai/DeepSeek-V3", role=None):
    """The Playground query opening `names` at the model's call `call` (the
    one `_response` evaluated them at)."""
    from urllib.parse import urlencode
    query = [("pick", f"{n}@{model}#{call}") for n in names]
    if role:
        query.append(("role", role))
    return urlencode(query)


def test_no_anchor_component_omits_l1_and_the_structure_switch():
    """Playground no-anchor is N/A: no entry l1, no Structure switch (plan §4)."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from backend.playground.module_bench import list_library_components

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    names = [c["name"] for c in list_library_components()]
    ffn = next((n for n in ("swiglu_ffn", "silu_ffn", "dense_ffn", "mlp") if n in names), None)
    if ffn is None:
        ffn = next((n for n in names if "ffn" in n and "moe" not in n), None)
    if ffn is None:
        pytest.skip("no dense FFN library component")
    # DeepSeek-V3 calls its SwiGLU in the dense layers, the MoE layers'
    # experts and shared experts, and the MTP head: the dense layers' call.
    solo = _response([ffn], call="0/mlp")
    assert "l1" not in solo["results"][0], solo["results"][0].keys()

    comps = {"success": True, "components": list_library_components()}
    dims_route = _dims_route("deepseek-ai/DeepSeek-V3")
    httpd = _serve_dist()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1200})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload, default=str))

            page.add_init_script("window.SITE_CONFIG={authEnabled:false,supabase:{url:'',anon_key:''}};")
            page.route("**/api/**", lambda r: _json(r, {}))
            page.route("**/supported-models", lambda r: _json(r, {"model_ids": []}))
            page.route("**/api/playground/components", lambda r: _json(r, comps))
            page.route("**/api/playground/component-dims**",
                       lambda r: _json(r, dims_route(r.request.url)))
            page.route("**/api/playground/module-bench", lambda r: _json(r, solo))
            # Opened on the call `solo` was evaluated at; a pick in the URL runs.
            role = next(c["role"] for c in comps["components"] if c["name"] == ffn)
            page.goto(f"{base}/playground.html?" + _picks([ffn], call="0/mlp", role=role))
            page.wait_for_selector("#pg-structure", timeout=30000)
            sec = page.locator("#pg-structure")
            assert sec.locator("button", has_text="Structure").count() == 0
            browser.close()
    finally:
        httpd.shutdown()


def test_structure_view_compares_by_role_and_expands_in_place():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from backend.playground.module_bench import list_library_components

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    # The dense layers' attention call; absorbed MLA takes its place there.
    pair = _response(["mla_attention", "mla_attention_absorbed"], call="0/self_attn")
    solo = _response(["mla_attention_absorbed"], call="0/self_attn")
    assert all(r["l1"]["available"] for r in pair["results"])
    law_box = next(b["label"] for b in solo["results"][0]["l1"]["boxes"]
                   if b.get("flops_is_law") and b.get("flops_symbolic")
                   and b["label"] == "Shared by K, V")
    comps = {"success": True, "components": list_library_components()}
    dims_route = _dims_route("deepseek-ai/DeepSeek-V3")
    httpd = _serve_dist()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    current = {"resp": pair}
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1200})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload, default=str))

            page.add_init_script("window.SITE_CONFIG={authEnabled:false,supabase:{url:'',anon_key:''}};")
            page.route("**/api/**", lambda r: _json(r, {}))
            page.route("**/supported-models", lambda r: _json(r, {"model_ids": []}))
            page.route("**/api/playground/components", lambda r: _json(r, comps))
            page.route("**/api/playground/component-dims**",
                       lambda r: _json(r, dims_route(r.request.url)))
            page.route("**/api/playground/module-bench", lambda r: _json(r, current["resp"]))

            page.goto(f"{base}/playground.html?" + _picks(
                ["mla_attention", "mla_attention_absorbed"], call="0/self_attn"))
            page.wait_for_selector("#pg-graph-diff .dag-node-slot", timeout=30000)
            page.click("#pg-graph-diff summary:has-text('per-box figures')")
            page.wait_for_selector("text=Structure comparison", timeout=30000)
            table = page.locator("#pg-graph-diff").inner_text()
            for row in ("Q path", "Shared by K, V", "Attention core", "Output"):
                assert row in table, row
            parts = page.locator("#pg-graph-diff table tbody tr td:first-child").all_inner_texts()
            assert "Total" not in parts
            # Absorbed V reads the shared latent through a head-broadcast view
            # of its own (b89c2d115, "Absorbed MLA joins its cached latent once"):
            # the V row is a box in both columns, and neither holds a weight
            # (unabsorbed MLA's kv_b_proj sits in the shared box).
            v_row = page.locator("#pg-graph-diff table tbody tr",
                                 has=page.locator("td:first-child", has_text=re.compile(r"^\s*V path\s*$")))
            assert v_row.count() == 1
            v_cells = v_row.locator("td").all_inner_texts()
            assert [c.strip().startswith("0 params") for c in v_cells[1:]] == [True, True], v_cells
            # same weights, held in different boxes: the diff says so, and
            # the change against the first column is the diff's own delta
            assert "weights regrouped" in table
            assert page.locator("#pg-graph-diff [data-delta]").count() > 0

            # one component: Structure is the default, boxes open in place
            current["resp"] = solo
            page.goto(f"{base}/playground.html?" + _picks(
                ["mla_attention_absorbed"], call="0/self_attn"))
            page.wait_for_selector("#pg-structure", timeout=30000)
            sec = page.locator("#pg-structure")
            sec.locator(".dag-node-slot", has_text="Shared by K, V").first.wait_for(timeout=20000)
            # The formula stays on the box as its subtitle; hovering does not
            # open a second copy in the shared tip.
            assert page.locator("#ha-tip").is_hidden()
            sec.locator(".dag-node-slot", has_text=law_box).first.hover()
            page.wait_for_timeout(250)
            assert page.locator("#ha-tip").is_hidden()
            # a box carries its members' KV marks: the latent is written in
            # the shared box, the rope key on the K path, and the absorbed
            # core reads both (1024 B + 128 B per cached token)
            def kv(box):
                slot = sec.locator(".dag-node-slot", has_text=box).first
                return {b.get_attribute("data-kv"): b.inner_text()
                        for b in slot.locator("[data-kv]").all()}
            sec.locator(".dag-node-slot", has_text="Attention core").first \
                .locator("[data-kv='read']").wait_for(timeout=20000)
            assert kv("Shared by K, V") == {"write": "KV write 1.00 KiB/tok"}
            assert kv("K path") == {"write": "KV write 128 B/tok"}
            assert kv("Attention core") == {"read": "KV read 1.13 KiB/tok × s_kv"}
            # closed: the op is named in the box's contents line, not drawn as a node
            own = sec.locator(".dag-node-slot .dag-node-label", has_text=re.compile(r"^\s*kv_a_proj_with_mqa\s*$"))
            assert own.count() == 0
            assert sec.locator(".dag-node-slot", has_text="Shared by K, V").first.inner_text().count("kv_a_proj_with_mqa") == 1
            sec.locator(".dag-node-slot", has_text="Shared by K, V").first.click(force=True)
            own.first.wait_for(timeout=20000)
            sec.locator(".dag-elk-collapse-tab", has_text=law_box).first.hover()
            page.wait_for_timeout(250)
            assert page.locator("#ha-tip").is_hidden()
            sec.locator("button", has_text="Detailed").click()
            sec.locator(".dag-node-slot", has_text="q_absorbed_nope").first.wait_for(timeout=20000)
            assert sec.locator("[data-edge-kind='cache']").count() > 0
            browser.close()
    finally:
        httpd.shutdown()


def test_each_recurrent_state_is_drawn_on_its_carrier():
    """Gated DeltaNet keeps two states, and the detailed graph badges each on
    the kernel that streams it: the delta rule's recurrent matrix and the
    causal conv's window. Qwen3-Coder-Next: the recurrent state is fp32
    (32·128·128·4 B = 2 MiB — the reference semantics, transformers/sglang
    keep it fp32) and the conv window stays on the deployment dtype (bf16:
    8192·4·2 B = 64 KiB — the HF cache keeps `conv_kernel_size` columns)
    per request, which are the layer's whole state."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from backend.playground.module_bench import list_library_components

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built")
    name = "gated_delta_net_fused_attention"
    solo = _response([name], base_model="Qwen/Qwen3-Coder-Next")
    assert solo["results"][0]["kv"]["storage_bytes_at_L"] == 32*128*128*4 + 8192*4*2
    comps = {"success": True, "components": list_library_components()}
    dims_route = _dims_route("Qwen/Qwen3-Coder-Next")
    httpd = _serve_dist()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 1200})

            def _json(route, payload):
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(payload, default=str))

            page.add_init_script("window.SITE_CONFIG={authEnabled:false,supabase:{url:'',anon_key:''}};")
            page.route("**/api/**", lambda r: _json(r, {}))
            page.route("**/supported-models", lambda r: _json(r, {"model_ids": []}))
            page.route("**/api/playground/components", lambda r: _json(r, comps))
            page.route("**/api/playground/component-dims**",
                       lambda r: _json(r, dims_route(r.request.url)))
            page.route("**/api/playground/module-bench", lambda r: _json(r, solo))

            page.goto(f"{base}/playground.html?" + _picks(
                [name], call="1/layer.self_attn", model="Qwen/Qwen3-Coder-Next"))
            page.wait_for_selector("text=Architecture ·", timeout=30000)
            sec = page.locator("section").filter(has=page.locator("h2", has_text="Architecture ·")).first
            sec.locator("button", has_text="Detailed").click()

            def state(label):
                slot = sec.locator(".dag-node-slot").filter(
                    has=page.locator(".dag-node-label", has_text=re.compile(rf"^\s*{label}\s*$"))).first
                badge = slot.locator("[data-kv='state']")
                badge.first.wait_for(timeout=20000)
                assert badge.count() == 1, label
                return badge.first.inner_text()

            assert state("delta_rule") == "state R/W 2.00 MiB/req"
            assert state("conv1d") == "state R/W 64.00 KiB/req"
            browser.close()
    finally:
        httpd.shutdown()
