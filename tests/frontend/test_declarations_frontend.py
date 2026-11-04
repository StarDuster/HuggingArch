"""Derived groups shown per declaration on the three views that list blocks.

Kimi-K3's decoder reads its AttnRes history (`read: all`), so the backend
splits three of its declarations into one group per history length (24 groups
in all). The pages show them per declaration: the per-op table has one row per
op of a declaration with the groups' rows under a toggle, the serving graph
one folded box per declaration with the groups inside, the Inspector's model
composition one row per declaration. Every figure a parent shows is the
backend's sum over its children, and the tests hold each to the sum of the
children the page shows under it (the composition row carries its histogram;
its sums are held to the model's totals by `tests/backend/views`).

Same harness as `test_pages_smoke.py` (built `web/dist`, canned payloads).
The Kimi-K3 fixtures are real responses, trimmed; `gen_fixtures.py` beside
the H3UI report regenerates them. DeepSeek-V3, which has no derived group,
groups its flat kernel rows by owning module, collapsed to one summary row
per module.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_KIMI = ("/inference.html#model=moonshotai%2FKimi-K3&gpu=H100-SXM"
         "&framework=vllm")
_DSV3 = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest")
_DSV3_GRAPH = {"graph": {"nodes": [
    {"id": "embed", "op": "embedding", "label": "embed_tokens"},
    {"id": "layers", "op": "block_stack", "label": "layers × 61", "children": [
        {"id": "input", "op": "input", "label": "input"},
        {"id": "input-ln", "op": "rmsnorm", "label": "input_layernorm"},
        {"id": "attn", "op": "mla_attention", "label": "self_attn"},
        {"id": "post-ln", "op": "fused_add_rmsnorm", "label": "post_attention_layernorm"},
        {"id": "mlp", "op": "swiglu_ffn", "label": "mlp"},
        {"id": "out", "op": "add", "label": "out"},
    ]},
    {"id": "norm", "op": "rmsnorm", "label": "norm"},
    {"id": "head", "op": "linear", "label": "lm_head"},
]}}


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _estimate_page(browser, base, link, overrides=None):
    s = Session(browser, base, overrides=overrides or {})
    s.goto(link)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-phase-breakdown", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    return s


def _rows(s):
    return s.page.eval_on_selector_all(
        "#infer-phase-breakdown tbody tr",
        """els => els.filter(e => !e.hasAttribute('data-total-row')).map(e => ({
            parent: e.hasAttribute('data-declaration-row'),
            child: e.hasAttribute('data-declaration-child'),
            module: e.hasAttribute('data-module-row'),
            leaf: e.hasAttribute('data-leaf-row'),
            children: e.getAttribute('data-children'),
            visible: e.getAttribute('data-visible'),
            flops: e.getAttribute('data-flops'),
            bytes: e.getAttribute('data-bytes'),
            tsol: e.getAttribute('data-t-sol'),
        }))""")


def test_the_per_op_table_lists_ops_per_declaration(browser_base):  # noqa: F811
    browser, base = browser_base
    est = _fixture("estimate_kimi_k3_h100.json")
    s = _estimate_page(browser, base, _KIMI, {"/inference/estimate": est})
    rows = est["per_op_timing_decode"]              # the table opens on decode
    parents = est["per_op_timing_decode_by_declaration"]
    folded = {i for p in parents for i in p["children"]}
    shown = _rows(s)
    assert len(shown) == len(rows) - len(folded) + len(parents)
    assert sum(r["parent"] for r in shown) == len(parents)
    assert not any(r["child"] for r in shown)
    header = s.page.inner_text("#infer-phase-breakdown")
    assert f"{len(rows) - len(folded) + len(parents)} rows per declaration" in header
    assert f"{len(rows)} leaves" in header
    # Each parent is the backend's sum of the rows it stands for.
    for p, r in zip(parents, [r for r in shown if r["parent"]], strict=True):
        assert r["children"] == ",".join(map(str, p["children"]))
        kids = [rows[i] for i in p["children"]]
        assert int(r["flops"]) == p["flops_per_rank"] == sum(k["flops_per_rank"] for k in kids)
        assert int(r["bytes"]) == sum(k["access_bytes_per_rank"] for k in kids)
        assert float(r["tsol"]) == p["t_sol_s"] == sum(k["t_sol_s"] for k in kids)

    # Opened, a parent lists its groups by history length, and their figures
    # add up to its own.
    s.page.click("#infer-phase-breakdown [data-declaration-toggle]")
    s.page.wait_for_selector("#infer-phase-breakdown [data-declaration-child]", timeout=5000)
    shown = _rows(s)
    first = next(i for i, r in enumerate(shown) if r["parent"])
    kids = []
    for r in shown[first + 1:]:
        if not r["child"]:
            break
        kids.append(r)
    p = parents[0]
    assert [int(k["visible"]) for k in kids] == [
        rows[i]["derived_group"]["visible"] for i in p["children"]]
    assert sum(int(k["flops"]) for k in kids) == int(shown[first]["flops"])
    assert sum(int(k["bytes"]) for k in kids) == int(shown[first]["bytes"])
    child_text = s.page.inner_text("#infer-phase-breakdown [data-declaration-child]")
    assert "Sar=1" in child_text and "layer" in child_text
    s.assert_clean()


def test_a_model_without_derived_groups_paints_the_walk_fold(browser_base):  # noqa: F811
    browser, base = browser_base
    est = _fixture("estimate_deepseek_v3_b200.json")
    s = _estimate_page(browser, base, _DSV3, {"/inference/estimate": est})
    s.page.wait_for_function(
        "() => document.querySelectorAll('#infer-phase-breakdown [data-fold-row]').length >= 1",
        timeout=5000,
    )
    names = s.page.eval_on_selector_all(
        "#infer-phase-breakdown [data-fold-row]",
        "els => els.map(e => e.dataset.name)",
    )
    assert "self_attn" in names
    leaves = s.page.eval_on_selector_all(
        "#infer-phase-breakdown [data-leaf-row]",
        "els => els.map(e => e.innerText)",
    )
    assert any("q_b_proj" in t for t in leaves)
    assert "linear" in " ".join(leaves)
    header = s.page.inner_text("#infer-phase-breakdown")
    assert "leaves" in header
    s.assert_clean()


def _drawn(s) -> set:
    return set(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-edge]", "els => els.map(e => e.getAttribute('data-edge'))"))


def test_the_graph_draws_one_box_per_declaration(browser_base):  # noqa: F811
    browser, base = browser_base
    payload = _fixture("compute_graph_kimi_k3_decode.json")
    s = _estimate_page(browser, base, _KIMI, {
        "/inference/estimate": _fixture("estimate_kimi_k3_h100.json"),
        "/inference/compute-graph": payload})
    s.page.click("#infer-graph-phase-decode")
    s.page.click("#infer-graph-refresh")
    s.page.wait_for_selector("#infer-compute-graph [data-node]", timeout=15000)
    nodes = payload["graph"]["nodes"]
    folds = [n for n in nodes if "derived_groups" in n]
    assert [f["derived_groups"]["declaration"]["base_template"] for f in folds] == [
        "block_mla_moe", "block_kda_moe_start", "block_kda_moe"]
    for f in folds:
        assert f["count"] == sum(c["count"] for c in f["children"])
        assert f["count"] == sum(h["layers"] for h in f["derived_groups"]["histogram"])
    drawn_nodes = set(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-node]", "els => els.map(e => e.getAttribute('data-node'))"))
    assert {f["id"] for f in folds} <= drawn_nodes
    assert not any(n.startswith("__block__block_kda_moe[") for n in drawn_nodes)
    assert _drawn(s) == {f"{e['src']}->{e['dst']}" for e in payload["graph"]["edges"]}

    kda = folds[-1]
    before = _drawn(s)
    s.page.click(f'#infer-compute-graph [data-node="{kda["id"]}"]')
    s.wait_for(lambda: _drawn(s) != before, timeout_ms=5000)
    want = {f"{e['src']}->{e['dst']}" for e in kda["child_edges"]}
    assert len(want) == len(kda["children"]) - 1
    assert not (want - _drawn(s))
    assert "Sar 1–8" in kda["label"]
    s.assert_clean()


def test_the_inspector_shows_one_row_per_declaration(browser_base):  # noqa: F811
    browser, base = browser_base
    analysis = _fixture("analyze_kimi_k3.json")
    s = Session(browser, base, overrides={"/analyze": analysis})
    s.goto("/inspector.html#model=moonshotai%2FKimi-K3")
    s.page.wait_for_selector("#model-composition [data-composition-row]",
                             state="attached", timeout=20000)
    rows = s.page.eval_on_selector_all(
        "#model-composition [data-composition-row]",
        """els => els.map(e => ({
            kind: e.getAttribute('data-composition-row'),
            layers: +e.getAttribute('data-layers'),
            params: +e.getAttribute('data-params'),
            groups: e.getAttribute('data-groups'),
        }))""")
    composition = analysis["composition"]["rows"]
    assert [(r["kind"], r["layers"], r["params"]) for r in rows] == [
        (r["kind"], r["layers"], r["params"]["total"]) for r in composition]
    # The three split declarations, in the order of their first layer
    # (1, 3, 12), one row each with the history lengths under it.
    declared = [r for r in rows if r["groups"] is not None]
    assert [r["layers"] for r in declared] == [61, 24, 7]
    assert [int(r["groups"]) for r in declared] == [
        len(r["histogram"]) for r in composition if "histogram" in r]
    # Block rows: the dense first layer, the three declarations, the vision
    # tower — not one per history length.
    assert sum(r["kind"] == "block" for r in rows) == 1 + 3 + 1
    dash = analysis["dashboard"]
    assert sum(r["params"] for r in rows) == dash["total_params"]
    assert s.page.text_content("#composition-total-params") == dash["total_params_formatted"]
    assert s.page.query_selector("#tab-layers") is None
    s.assert_clean()
