"""The Inference page's paged-KV page control says what the backend declares.

Same harness as `test_pages_smoke.py` (built `web/dist` in headless chromium,
canned payloads). The page size is part of every cache-reading kernel's
measurement key, so these check the three things that would make the control
dishonest: a default the page invented rather than read from the framework
binding, an option list the client carries instead of the backends' served
union, and a request that names a page the user never chose.
"""
from __future__ import annotations

import copy

from tests.frontend.test_pages_smoke import Session, _fixture, browser_base  # noqa: F401

_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest")

#: The same link with no framework named, so the control must fall back to the
#: binding `default_framework` points at.
_LINK_NO_FW = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
               "&selection=latest")


def _registry(**fields) -> dict:
    reg = copy.deepcopy(_fixture("fusion_registry.json"))
    reg.update(fields)
    return reg


def _control_ready(s) -> None:
    s.page.wait_for_selector("#infer-kv-block", state="attached", timeout=15000)


def _default_label(s) -> str:
    """The text of the unchosen option — what the user reads as the default."""
    return s.page.evaluate(
        "document.querySelector('#infer-kv-block option[value=\"\"]').textContent")


def _offered(s) -> list[str]:
    """Every page the control offers, the unchosen option aside."""
    return s.page.evaluate(
        "Array.from(document.querySelectorAll('#infer-kv-block option'))"
        ".map(o => o.value).filter(v => v !== '')")


def _last_estimate(s) -> dict:
    return s.requests["/inference/estimate"][-1]["body"]


def test_the_unchosen_page_shows_the_bindings_default_and_sends_no_page(browser_base):  # noqa: F811
    """vLLM declares 16, so the control reads 16 — and because nobody chose it,
    the request omits `kv_block_size` entirely rather than sending the number
    back or sending null."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _control_ready(s)
    assert s.page.input_value("#infer-kv-block") == ""
    assert "16" in _default_label(s), _default_label(s)
    body = _last_estimate(s)
    assert "kv_block_size" not in body, body
    s.assert_clean()


def test_the_default_follows_the_framework(browser_base):  # noqa: F811
    """sglang's binding declares no page of its own, so switching to it must
    stop claiming 16 — the estimate asks for no particular page there, and a
    substituted 16 would be a page nobody chose."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _control_ready(s)
    assert "16" in _default_label(s)
    s.page.select_option("#infer-framework", "sglang")
    assert s.wait_for(lambda: "16" not in _default_label(s)), _default_label(s)
    assert "declares none" in _default_label(s), _default_label(s)
    # Back to the framework that does declare one, and the number returns.
    s.page.select_option("#infer-framework", "vllm")
    assert s.wait_for(lambda: "16" in _default_label(s)), _default_label(s)
    s.assert_clean()


def test_an_unnamed_framework_reads_the_default_bindings_page(browser_base):  # noqa: F811
    """A link naming no framework runs under `default_framework`, so the
    control shows THAT binding's page rather than guessing a name."""
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/fusion/registry": _registry(default_framework="sglang"),
    })
    s.goto(_LINK_NO_FW)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _control_ready(s)
    assert s.page.input_value("#infer-framework") == ""
    # sglang is the server's default here, and it declares no page.
    assert s.wait_for(lambda: "declares none" in _default_label(s)), _default_label(s)
    s.assert_clean()


def test_a_chosen_page_is_sent_on_the_estimate(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _control_ready(s)
    before = s.count("/inference/estimate")
    s.page.select_option("#infer-kv-block", "64")
    s.page.click("[data-action='run-estimate']")
    assert s.wait_for(lambda: s.count("/inference/estimate") > before)
    assert _last_estimate(s)["kv_block_size"] == 64, _last_estimate(s)
    s.assert_clean()


def test_the_offered_pages_are_the_backends_served_union(browser_base):  # noqa: F811
    """The list is whatever the backend says it serves. Feeding a different
    union has to move the dropdown: a client-side copy of the canned union
    would keep answering the canned pages here and this test would catch it."""
    browser, base = browser_base
    served = [str(n) for n in _fixture("fusion_registry.json")["served_kv_block_sizes"]]
    assert served, "the canned registry serves no page"
    s = Session(browser, base)
    s.goto(_LINK)
    _control_ready(s)
    assert s.wait_for(lambda: _offered(s) == served), _offered(s)
    s.assert_clean()

    other = [32, 512]
    assert [str(n) for n in other] != served
    odd = Session(browser, base, overrides={
        "/fusion/registry": _registry(served_kv_block_sizes=other),
    })
    odd.goto(_LINK)
    _control_ready(odd)
    assert odd.wait_for(lambda: _offered(odd) == ["32", "512"]), _offered(odd)
    odd.assert_clean()


_SWEEPS = ("/inference/estimate-sweep", "/inference/estimate-parallel-sweep")


def test_the_sweep_reads_rows_at_the_page_the_plan_deploys(browser_base):  # noqa: F811
    """The Scaling panel inherits the choice instead of carrying a second
    control: the page is in the measurement key, so a sweep at another page
    would look rows up under a key this deployment never runs. Both of its
    sweeps (Context x Batch and the parallel one) carry it."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(_LINK)
    for path in _SWEEPS:
        assert s.wait_for(lambda path=path: s.count(path) >= 1), path
    _control_ready(s)
    # Unchosen: the sweeps omit it too, so each framework's own default stands.
    for path in _SWEEPS:
        assert all("kv_block_size" not in (r["body"] or {})
                   for r in s.requests.get(path, [])), (path, s.requests[path])
    s.page.select_option("#infer-kv-block", "128")
    for path in _SWEEPS:
        assert s.wait_for(lambda path=path: any(
            (r["body"] or {}).get("kv_block_size") == 128
            for r in s.requests.get(path, []))), (path, s.requests[path])
    s.assert_clean()


def test_the_page_choice_survives_a_shared_link(browser_base):  # noqa: F811
    """The control is a deployment knob, so it travels in the hash like the
    framework does — a shared link reproduces the page it was estimated at."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _control_ready(s)
    s.page.select_option("#infer-kv-block", "32")
    assert s.wait_for(lambda: "kv_block=32" in s.page.evaluate("window.location.hash"))

    shared = Session(browser, base)
    shared.goto(f"{_LINK}&kv_block=32")
    assert shared.wait_for(lambda: shared.count("/inference/estimate") >= 1)
    _control_ready(shared)
    assert shared.page.input_value("#infer-kv-block") == "32"
    assert _last_estimate(shared)["kv_block_size"] == 32, _last_estimate(shared)
    shared.assert_clean()
