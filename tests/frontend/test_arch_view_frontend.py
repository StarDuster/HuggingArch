"""The Inspector's Architecture Diagram, laid out as the graph it is.

Same harness as `test_pages_smoke.py`: the built `web/dist` in headless
chromium, every backend call answered from canned payloads. The payload is
a real `/analyze` answer for DeepSeek-V4.1-Flash with its block graphs
dropped (`architecture_view` whole).

What is pinned: every box the view names is drawn, every edge the backend
gives (`architecture_view.edges`) is drawn between those two boxes and no
other, no box spells its inputs out as text instead, and a card whose
layers read each other's streams draws a row per stream.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analyze_deepseek_v41_arch.json"


def test_the_diagram_draws_the_backend_edges(browser_base):  # noqa: F811
    browser, base = browser_base
    analysis = json.loads(_FIXTURE.read_text())
    view = analysis["architecture_view"]
    s = Session(browser, base, overrides={
        "/analyze": analysis,
        "/supported-models": {"model_ids": [analysis["model_id"]]},
    })
    s.goto("/inspector.html#model=deepseek-ai%2FDeepSeek-V4.1-Flash")
    s.page.wait_for_selector("#tab-diagram", timeout=20000)
    s.page.click("#tab-diagram")
    s.page.wait_for_selector("#arch-view-container [data-arch-edge]", state="attached", timeout=20000)

    boxes = set(s.page.eval_on_selector_all(
        "#arch-view-container [data-arch-node]",
        "els => els.map(e => e.getAttribute('data-arch-node'))"))
    want = ({m["id"] for m in view["model_inputs"]}
            | {c["id"] for c in view["layers"]}
            | {sec["id"] for sec in view["extra_sections"]})
    assert boxes == want

    drawn = s.page.eval_on_selector_all(
        "#arch-view-container [data-arch-edge]",
        "els => els.map(e => [e.getAttribute('data-from'), e.getAttribute('data-to'),"
        " e.getAttribute('data-arch-edge')])")
    assert sorted(map(tuple, drawn)) == sorted(
        (e["from"], e["to"], e["kind"]) for e in view["edges"])
    assert "from:" not in s.page.inner_text("#arch-view-container")
    # The CSA cards map their layers' stream reads: a row per stream.
    csa = max((c for c in view["layers"] if c.get("stream_map")),
              key=lambda c: len(c["layers"]))
    rows = s.page.eval_on_selector_all(
        f'[data-arch-node="{csa["id"]}"] [data-stream-row]',
        "els => els.map(e => e.getAttribute('data-stream-row'))")
    assert rows == [row["name"] for row in csa["stream_map"]]
    # Its variant rows name the components that tell them apart, each
    # with its declared description.
    slots = s.page.eval_on_selector_all(
        f'[data-arch-node="{csa["id"]}"] [data-variant-slot]',
        "els => els.map(e => e.textContent)")
    want = [f'{x["slot"]} = {x["component"]}{x["description"]}'
            for v in csa["variants"] for x in v["slots"]]
    assert slots == want
    s.assert_clean()
