"""The inference page's compute graph, drawn by the shared renderer.

Same harness as `test_pages_smoke.py`: the built `web/dist` in headless
chromium, every backend call answered from canned payloads. The graph
payloads are real `/inference/compute-graph` responses (vllm, one
rank, decode): Qwen3-8B whole, DeepSeek-V3 trimmed to the embedding and the
dense block.

What is pinned: the page keeps its DOM ids; it draws exactly the backend's
edges — the root edges when every compound is closed, and every compound's
`child_edges` once it is opened. A compound the backend gives an L1 view
(`l1`, `views/abstraction.py`) opens onto its boxes first; once every box is
opened its members and their edges are drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.frontend.test_pages_smoke import Refusal, Session, browser_base  # noqa: F401

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest")


def _graph(name: str) -> dict:
    return json.loads((_FIXTURES / f"compute_graph_{name}_decode.json").read_text())


def _open_graph(browser, base, payload, link=_LINK):
    s = Session(browser, base, overrides={"/inference/compute-graph": payload})
    s.goto(link)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-graph-refresh", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.click("#infer-graph-phase-decode")
    s.page.click("#infer-graph-refresh")
    s.page.wait_for_selector("#infer-compute-graph [data-node]", timeout=15000)
    return s


def _drawn(s) -> set:
    return set(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-edge]", "els => els.map(e => e.getAttribute('data-edge'))"))


def _expand(s, node_id: str):
    before = _drawn(s)
    s.page.click(f'#infer-compute-graph [data-node="{node_id}"]')
    s.wait_for(lambda: _drawn(s) != before, timeout_ms=5000)
    s.page.wait_for_timeout(100)


def _open(s, compound: dict):
    """Open a compound, then every L1 box it opened onto."""
    _expand(s, compound["id"])
    for box in (compound.get("l1") or {}).get("boxes") or []:
        sel = f'#infer-compute-graph [data-node="l1:{compound["id"]}:{box["id"]}"]'
        if s.page.query_selector(sel):
            _expand(s, f'l1:{compound["id"]}:{box["id"]}')


def _compounds(nodes):
    for n in nodes:
        if n.get("expandable") and n.get("children"):
            yield n


@pytest.mark.parametrize("name", ["qwen3_8b", "deepseek_v3"])
def test_the_page_draws_the_backend_edges(browser_base, name):  # noqa: F811
    """G8. Closed, the graph is the root edges; each compound, once opened
    (with its L1 boxes, its children still closed), draws its own
    `child_edges`."""
    browser, base = browser_base
    payload = _graph(name)
    s = _open_graph(browser, base, payload)
    graph = payload["graph"]
    root = {f"{e['src']}->{e['dst']}" for e in graph["edges"]}
    assert _drawn(s) == root

    opened = []

    def walk(nodes):
        for n in _compounds(nodes):
            _open(s, n)
            opened.append(n["id"])
            want = {f"{e['src']}->{e['dst']}" for e in n.get("child_edges") or []}
            missing = want - _drawn(s)
            assert not missing, (n["id"], sorted(missing))
            walk(n["children"])

    walk(graph["nodes"])
    assert len(opened) >= 10
    s.assert_clean()


def test_the_graph_keeps_its_dom_ids(browser_base):  # noqa: F811
    """The page's controls and the dark graph panel the shared renderer
    draws on."""
    browser, base = browser_base
    s = _open_graph(browser, base, _graph("qwen3_8b"))
    for sel in ("#infer-graph-phase-prefill", "#infer-graph-phase-decode",
                "#infer-graph-refresh", "#infer-graph-summary",
                "#infer-compute-graph"):
        assert s.page.query_selector(sel) is not None, sel
    classes = s.page.get_attribute("#infer-compute-graph", "class") or ""
    assert "dag-container" in classes
    q = s.requests["/inference/compute-graph"][-1]["query"]
    assert q["phase"] == "decode"
    # EP=8, PP=1: every GPU of the plan runs this one graph, so there is
    # no rank to pick and none is asked for.
    assert s.page.query_selector("#infer-graph-stage") is None
    assert "pp_rank" not in q and "ep_rank" not in q
    s.assert_clean()


def test_a_pipelined_plan_draws_one_stage(browser_base):  # noqa: F811
    """PP > 1: the graph is one stage's GPU, stage 0 until another is
    picked — never the whole model the plan does not run on one GPU."""
    browser, base = browser_base
    s = _open_graph(browser, base, _graph("qwen3_8b"),
                    link=_LINK.replace("&ep=8&dp_attn=8", "&pp=2"))
    options = s.page.eval_on_selector_all(
        "#infer-graph-stage option", "els => els.map(e => e.value)")
    assert options == ["0", "1"]
    assert s.requests["/inference/compute-graph"][-1]["query"]["pp_rank"] == "0"
    s.page.select_option("#infer-graph-stage", "1")
    assert s.wait_for(lambda: s.requests["/inference/compute-graph"][-1]
                      ["query"].get("pp_rank") == "1")
    s.assert_clean()


_INDIVISIBLE = {
    "detail": "embed_tokens: token extent 1023 does not divide the sequence world degree 2",
    "reason": "sequence_not_divisible", "extent": 1023, "world": "cp", "degree": 2,
}
_CP_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&cp=2"
            "&framework=vllm&selection=latest")


def _refused_graph_text(browser, base, body) -> str:
    s = Session(browser, base, overrides={"/inference/compute-graph": Refusal(400, body)})
    s.goto(_CP_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-graph-refresh", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.click("#infer-graph-refresh")
    s.page.wait_for_function(
        "() => /Failed|Cannot draw graph/.test(document.querySelector('#infer-compute-graph')?.textContent || '')",
        timeout=10000)
    assert s.requests["/inference/compute-graph"][-1]["query"]["cp"] == "2"
    assert not s.unstubbed
    return s.page.text_content("#infer-compute-graph") or ""


def test_an_indivisible_length_says_why_there_is_no_graph(browser_base):  # noqa: F811
    """The endpoint names the refusal with `reason`; the page turns that
    code — not the message text — into its sentence, with the numbers the
    body carries."""
    browser, base = browser_base
    text = _refused_graph_text(browser, base, _INDIVISIBLE)
    assert "Cannot draw graph" in text and "1023" in text and "cp=2" in text, text


def test_other_cp_prefill_failures_are_shown_as_they_came(browser_base):  # noqa: F811
    """A CP prefill that fails for any other reason is not dressed up as a
    length problem — even when its message happens to read like one."""
    browser, base = browser_base
    text = _refused_graph_text(browser, base, {"detail": _INDIVISIBLE["detail"]})
    assert text.startswith("Failed: embed_tokens"), text
    assert "Cannot draw graph" not in text, text


def test_the_fused_subgraph_says_why_there_is_none(browser_base):  # noqa: F811
    """The fusion popover's subgraph reads the same graph endpoint and says
    the same thing on the same structured refusal."""
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/compute-graph": Refusal(400, _INDIVISIBLE)})
    s.goto(_CP_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.wait_for_selector("[data-fusion-pop]", timeout=15000)
    s.page.hover("[data-fusion-pop]")
    s.page.wait_for_function(
        "() => /Failed|Cannot draw graph/.test(document.querySelector('#fusion-popover-dag')?.textContent || '')",
        timeout=10000)
    text = s.page.text_content("#fusion-popover-dag") or ""
    assert "Cannot draw graph" in text and "1023" in text, text
    q = s.requests["/inference/compute-graph"][-1]["query"]
    assert q["avg_input_tokens"] and q["batch_size"], q
    assert not s.unstubbed


def test_the_fused_subgraph_fits_inside_the_popover(browser_base):  # noqa: F811
    """The popover draws the absorbed ops as a compact DAG: short labels,
    no namespaced ids spilling out of the rounded cards, no 240px empty
    canvas from the full-page renderer floor."""
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/compute-graph": _graph("deepseek_v3")})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("[data-fusion-pop]", timeout=15000)
    s.page.locator("[data-fusion-pop]").first.hover()
    s.page.wait_for_selector("#fusion-popover-dag [data-node]", timeout=15000)
    info = s.page.evaluate("""() => {
      const box = document.querySelector('#fusion-popover-dag');
      if (!box) return { missing: true };
      const br = box.getBoundingClientRect();
      const nodes = [...box.querySelectorAll('[data-node]')];
      const labels = nodes.map((n) =>
        (n.querySelector('.dag-node-label')?.textContent || n.textContent || '').trim());
      const spill = nodes.filter((n) => {
        const r = n.getBoundingClientRect();
        return r.right > br.right + 2 || r.left < br.left - 2
            || r.bottom > br.bottom + 2 || r.top < br.top - 2;
      }).map((n) => n.getAttribute('data-node'));
      return {
        n: nodes.length,
        labels,
        overflowX: box.scrollWidth > box.clientWidth + 8,
        height: box.clientHeight,
        spill,
      };
    }""")
    assert info.get("n", 0) >= 1, info
    assert not info.get("overflowX"), info
    assert not info.get("spill"), info
    assert info.get("height", 999) < 280, info
    for lab in info.get("labels") or []:
        assert "|" not in lab and "__block__" not in lab, info
    s.assert_clean()


# --- KV cache marks ----------------------------------------------------------

def _kv_badges(s, node_id: str) -> dict:
    return dict(s.page.eval_on_selector_all(
        f'#infer-compute-graph [data-node="{node_id}"] [data-kv]',
        "els => els.map(e => [e.getAttribute('data-kv'), e.textContent])"))


def _edge_kinds(s) -> dict:
    return dict(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-edge]",
        "els => els.map(e => [e.getAttribute('data-edge'), e.getAttribute('data-edge-kind')])"))


def _attention(payload):
    block = next(n for n in payload["graph"]["nodes"] if n["op"] == "block_stack")
    attn = next(n for n in block["children"] if n.get("style") == "attn")
    return block, attn


def _child(attn, op):
    return next(n for n in attn["children"] if n["op"] == op)["id"]


def test_qwen3_decode_shows_what_each_node_writes_and_reads(browser_base):  # noqa: F811
    """GQA: the closed block shows the layer's 4096 B per token; opened, K is
    written by the fused rope, V by its head split, and the fused attention
    reads both over s_kv. The cache edges run writer → reader in the cache
    style, distinct from an activation edge."""
    browser, base = browser_base
    payload = _graph("qwen3_8b")
    s = _open_graph(browser, base, payload)
    block, attn = _attention(payload)
    assert _kv_badges(s, block["id"]) == {
        "write": "KV write 4.00 KiB/tok", "read": "KV read 4.00 KiB/tok × s_kv"}
    _expand(s, block["id"])
    _open(s, attn)
    rope = _child(attn, "fused_qk_rope")
    v = _child(attn, "head_split")
    flash = _child(attn, "flash_attention")
    assert _kv_badges(s, rope) == {"write": "KV write 2.00 KiB/tok"}
    assert _kv_badges(s, v) == {"write": "KV write 2.00 KiB/tok"}
    assert _kv_badges(s, flash) == {"read": "KV read 4.00 KiB/tok × s_kv"}
    kinds = _edge_kinds(s)
    assert kinds[f"{rope}->{flash}"] == "cache"
    assert kinds[f"{v}->{flash}"] == "cache"
    assert set(kinds.values()) == {"activation", "cache"}
    title = s.page.get_attribute(f'#infer-compute-graph [data-node="{flash}"] .dag-kv-line', "title")
    assert "k_rope" in title and "v_split_heads" in title and "/step" in title
    legend = set(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-legend-item]", "els => els.map(e => e.getAttribute('data-legend-item'))"))
    assert {"kv:write", "kv:read", "edge:cache", "style:fused"} <= legend
    s.assert_clean()


def test_deepseek_v3_decode_reads_the_latent(browser_base):  # noqa: F811
    """Absorbed MLA: the latent (1024 B) and the rope key (128 B) are written
    by the norm and the fused rope; the fused latent attention reads 1152 B
    per cached token."""
    browser, base = browser_base
    payload = _graph("deepseek_v3")
    s = _open_graph(browser, base, payload)
    block, attn = _attention(payload)
    _expand(s, block["id"])
    _open(s, attn)
    norm = next(n["id"] for n in attn["children"]
                if n.get("kv") and n["kv"]["write_bytes_per_token"] == 1024)
    rope = _child(attn, "fused_qk_rope")
    flash = _child(attn, "flash_attention")
    assert _kv_badges(s, norm) == {"write": "KV write 1.00 KiB/tok"}
    assert _kv_badges(s, rope) == {"write": "KV write 128 B/tok"}
    assert _kv_badges(s, flash) == {"read": "KV read 1.13 KiB/tok × s_kv"}
    assert "cache" in set(_edge_kinds(s).values())
    s.assert_clean()


def test_a_mask_edge_and_a_cache_edge_draw_differently(browser_base):  # noqa: F811
    """The two non-activation kinds get different strokes."""
    browser, base = browser_base
    payload = _graph("qwen3_8b")
    block = next(n for n in payload["graph"]["nodes"] if n["op"] == "block_stack")
    # A mask edge beside the cache ones: relabel one activation edge.
    edge = payload["graph"]["edges"][0]
    edge["kind"] = "mask"
    s = _open_graph(browser, base, payload)
    _expand(s, block["id"])
    _open(s, _attention(payload)[1])
    strokes = dict(s.page.eval_on_selector_all(
        "#infer-compute-graph [data-edge]",
        "els => els.map(e => [e.getAttribute('data-edge-kind'), "
        "e.getAttribute('stroke') + '|' + e.getAttribute('stroke-dasharray')])"))
    assert strokes["mask"] != strokes["cache"] != strokes["activation"]
    s.assert_clean()


def _deepseek_v3_prefill_with_mtp() -> dict:
    """A real vllm prefill graph of DeepSeek-V3, built in-process, trimmed to
    the MoE block and the MTP head: fused MLA (a merged q/kv_a projection
    and the q/k rope as one kernel, both read through named ports), routed
    experts whose dispatch reads the block input beside the router, and a
    block of two formals (the MTP head reads `norm` and `input_ids`)."""
    import inspect

    from backend.api.routes import inference as routes
    kw = {k: (None if p.default is inspect.Parameter.empty else p.default)
          for k, p in inspect.signature(routes._compute_graph).parameters.items()}
    kw.update(model_id="deepseek-ai/DeepSeek-V3", framework="vllm", pp=1,
              tp_attention=1, tp_dense_ffn=1, ep_moe=1, tp_moe_expert=1, cp=1,
              dp_attention=1, dp_replicas=1, sequence_parallel=False,
              cp_scheme="ulysses", phase="prefill", avg_input_tokens=1024,
              avg_output_tokens=0, prefix_cached_tokens=0,
              speculative_draft_tokens=0, batch_size=1, framework_flags="")
    payload = routes._compute_graph(**kw)
    graph = payload["graph"]
    dense = next(n["id"] for n in graph["nodes"] if n.get("op") == "block_stack"
                 and not n.get("is_moe") and not n.get("aux_kind"))
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != dense]
    kept = {n["id"] for n in graph["nodes"]}
    graph["edges"] = [e for e in graph["edges"] if e["src"] in kept and e["dst"] in kept]
    return payload


def test_an_outer_edge_lands_on_what_reads_it(browser_base):  # noqa: F811
    """Opened, a compound draws no edge onto itself: each outer edge lands on
    the node inside that reads it. A block of two formals binds each outer
    input to its own input node (`formal`); an L1 box lists what its members
    read from outside the component, so the edge reaches the member even when
    every box is fed by another."""
    browser, base = browser_base
    payload = _deepseek_v3_prefill_with_mtp()
    graph = payload["graph"]
    s = _open_graph(browser, base, payload)
    mtp = next(n for n in graph["nodes"] if n.get("aux_kind") == "mtp")
    formal = {c["formal"]: c["id"] for c in mtp["children"] if "formal" in c}
    assert mtp["inputs"] == ["norm", "input_ids"]

    opened = set()
    reads = []   # (outer producer, member) of each opened component
    nested = []  # an opened component's edges out of a closed member's insides

    def walk(nodes):
        for n in _compounds(nodes):
            inside = {c["id"] for c in n["children"]}
            # a read of a node nested in a closed sibling lands on the sibling
            # (the scope's edge names it `src`, the nested node `via`)
            holder = {e["via"]: e["src"] for e in n.get("child_edges") or []
                      if e.get("via") and e.get("src")}
            if n.get("l1"):
                nested.extend(f'{e["src"]}->{e["dst"]}' for e in n.get("child_edges") or []
                              if e.get("via") and e["via"] != e["src"])
                for c in n["children"]:
                    ports = c.get("input_ports") or []
                    for k, ref in enumerate(c.get("inputs") or []):
                        port = ports[k] if k < len(ports) else ""
                        bare = ref[:-len(port) - 1] if port and ref.endswith("." + port) else ref
                        bare = bare if bare in inside else holder.get(bare, bare)
                        if bare not in inside:
                            reads.append((bare, c["id"]))
            _open(s, n)
            opened.add(n["id"])
            opened.update(f'l1:{n["id"]}:{b["id"]}' for b in (n.get("l1") or {}).get("boxes") or [])
            walk([c for c in n["children"] if c.get("l1")])   # the components

    walk([n for n in graph["nodes"] if n.get("is_moe") and not n.get("aux_kind")])
    _expand(s, mtp["id"])
    opened.add(mtp["id"])
    drawn = _drawn(s)
    assert f"norm->{formal[0]}" in drawn and f"input_ids->{formal[1]}" in drawn
    onto = sorted(e for e in drawn if e.split("->", 1)[1] in opened)
    assert not onto, onto
    # the MoE dispatch reads the block input beside the router, whose box
    # also feeds it: the edge still reaches it
    missing = sorted(f"{a}->{b}" for a, b in reads
                     if a not in opened and f"{a}->{b}" not in drawn)
    assert not missing, missing
    # the router's topk_scale, nested in the closed gate, feeds both the
    # dispatch and the combine: each read is drawn from the gate
    assert nested and not sorted(set(nested) - drawn), sorted(set(nested) - drawn)
    s.assert_clean()


def test_the_section_says_it_expands(browser_base):  # noqa: F811
    """Closed, the section shows an expand marker and says it opens; opened,
    the empty panel says what to click rather than standing blank."""
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/compute-graph": _graph("qwen3_8b")})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    summary = s.page.locator("#infer-graph-details > summary")
    summary.wait_for(timeout=15000)
    assert "▶" in summary.inner_text() and "click to expand" in summary.inner_text()
    summary.click()
    status = s.page.locator("#infer-graph-status")
    status.wait_for(timeout=5000)
    assert "Draw graph" in status.inner_text(), status.inner_text()
    assert s.count("/inference/compute-graph") == 0  # drawn on request only
    s.assert_clean()


def test_a_new_phase_never_leaves_the_old_graph_standing(browser_base):  # noqa: F811
    """Once a graph is drawn, switching phase clears it at once, says it is
    drawing, and redraws for the new phase."""
    browser, base = browser_base
    s = _open_graph(browser, base, _graph("qwen3_8b"))
    n = s.count("/inference/compute-graph")
    s.hold = {"/inference/compute-graph"}
    s.page.click("#infer-graph-phase-prefill")
    assert s.wait_for(lambda: s.count("/inference/compute-graph") == n + 1)
    assert s.requests["/inference/compute-graph"][-1]["query"]["phase"] == "prefill"
    s.page.wait_for_selector("#infer-compute-graph [data-node]", state="detached", timeout=5000)
    assert "Drawing the prefill graph" in s.page.inner_text("#infer-graph-status")
    assert s.page.is_disabled("#infer-graph-refresh")
    s.hold = set()
    for route in s.held.pop("/inference/compute-graph"):
        s.fulfill(route, "/inference/compute-graph")
    s.page.wait_for_selector("#infer-compute-graph [data-node]", timeout=15000)
    s.assert_clean()
