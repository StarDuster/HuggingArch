"""Kernel Bench asks the server for one page of one view — the source, the
batch, the filters, the sort and the page ride the query — and shows what it
answers: the rows, the total and the pager."""
import json
from urllib.parse import parse_qs, urlparse

from tests.frontend.test_pages_smoke import Session, _CORPUS_ROW, browser_base  # noqa: F401

_RUN = "20261001T000000Z-batch"
_TOTAL = 2500


def _answer(q: dict) -> dict:
    page, size = int(q.get("page", 0)), int(q.get("page_size", 50))
    rows = [{**_CORPUS_ROW, "id": f"r{i}", "origin": "mine", "run_id": _RUN,
             "shape_json": {"B": i + 1}, "dtype_env_json": {"activation": "bf16"},
             "mfu": 0.5, "mbu": 0.4, "roof_status": "ok"}
            for i in range(page * size, min((page + 1) * size, _TOTAL))]
    return {"total": _TOTAL, "items": rows, "facets": {"model": [_CORPUS_ROW["model"]]}}


def test_a_batch_view_is_one_server_page_at_a_time(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, {"/api/bench/browse": _answer})
    s.goto(f"/kernel-corpus.html#view=rows&source=mine&run={_RUN}&check=over")
    s.page.wait_for_selector("text=2,500 rows", timeout=15000)
    asked = [r["query"] for r in s.requests["/api/bench/browse"] if r["query"].get("view") == "rows"]
    assert asked[-1] == {"view": "rows", "source": "mine", "run": _RUN, "check": "over",
                         "page": "0", "page_size": "50"}
    assert s.page.locator("#table-section tbody tr").count() == 50
    s.page.click("#table-pager button:has-text('Next')")
    assert s.wait_for(lambda: any(r["query"].get("page") == "1" for r in s.requests["/api/bench/browse"]))
    s.page.click("button[data-sort='mfu']")
    assert s.wait_for(lambda: any(r["query"].get("sort") == "mfu" and r["query"].get("dir") == "desc"
                                  and r["query"].get("page") == "0"
                                  for r in s.requests["/api/bench/browse"]))
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("run") == [_RUN] and h.get("check") == ["over"]
    s.assert_clean()


def test_a_pinned_column_stays_at_the_left_edge_while_the_table_scrolls(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, {"/api/bench/browse": _answer})
    s.page.set_viewport_size({"width": 700, "height": 900})
    s.goto(f"/kernel-corpus.html#view=rows&source=mine&run={_RUN}")
    s.page.wait_for_selector("text=2,500 rows", timeout=15000)
    table = s.page.locator("#table-section table").first
    key = table.locator("thead tr:last-child th button[data-pin]").nth(1).get_attribute("data-pin")
    pin = table.locator(f"button[data-pin='{key}']")
    pin.click(force=True)
    assert pin.get_attribute("aria-pressed") == "true"
    th = pin.locator("xpath=ancestor::th[1]")
    frame = table.locator("xpath=..")
    assert frame.evaluate("f => f.scrollWidth > f.clientWidth + 100")
    frame.evaluate("f => { f.scrollLeft = f.scrollWidth; }")
    s.page.wait_for_timeout(100)
    left = lambda el: el.evaluate("e => e.getBoundingClientRect().left")
    assert abs(left(th) - left(frame)) < 2
    cell = table.locator("tbody tr").first.locator("td[data-pinned]")
    assert cell.count() == 1 and abs(left(cell) - left(frame)) < 2
    pin.click(force=True)
    assert table.locator("[data-pinned]").count() == 0
    s.assert_clean()


def test_a_refused_view_says_why(browser_base):  # noqa: F811
    from tests.frontend.test_pages_smoke import Refusal
    browser, base = browser_base
    s = Session(browser, base, {"/api/bench/browse": Refusal(
        503, {"detail": "Kernel Bench store unavailable: the import is running"})})
    s.goto("/kernel-corpus.html#view=envelope")
    s.page.wait_for_selector("#browse-error", timeout=15000)
    assert "the import is running" in s.page.inner_text("#browse-error")


def test_rows_list_the_scope_and_an_envelope_opens_its_rows(browser_base):  # noqa: F811
    """The rows view takes the envelope view's scope and filters — one page,
    no model or phase needed (the shared corpus by default); an envelope's
    candidates open that op shape's rows."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=rows")
    assert s.wait_for(lambda: any(r["query"].get("view") == "rows" and r["query"].get("source") == "shared"
                                  and not r["query"].get("model")
                                  for r in s.requests.get("/api/bench/browse", [])))
    s.goto("/kernel-corpus.html#view=envelope&signature=rmsnorm")
    s.page.reload()   # only the hash changed: the page reads it on load
    s.page.wait_for_selector("#table-section button[data-open-rows]", timeout=15000)
    s.page.locator("#table-section button[data-open-rows]").first.click()
    assert s.wait_for(lambda: any(r["query"].get("view") == "rows" and r["query"].get("key")
                                  for r in s.requests["/api/bench/browse"]))
    key = json.loads(next(r["query"]["key"] for r in s.requests["/api/bench/browse"] if r["query"].get("key")))
    assert key["signature"] == "rmsnorm" and key["gpu"] == "B200"
    assert s.page.locator("#f-key-clear").inner_text().startswith("rmsnorm")
    # The charts below follow the drill-down too.
    assert s.wait_for(lambda: any(r["query"].get("view") == "curves" and r["query"].get("key")
                                  for r in s.requests["/api/bench/browse"]))
    s.assert_clean()


def test_a_workload_names_its_context_free_ops_and_the_unplaced_rows(browser_base):  # noqa: F811
    def answer(q: dict) -> dict:
        out = {"facets": {},
               "batches": [8], "unplaced": 120}
        if q.get("batch"):
            out["contexts"] = [4096, 0]
        if q.get("context"):
            out.update(candidates=10, total=0, total_latency_us=0, items=[])
        return out
    browser, base = browser_base
    s = Session(browser, base, {"/api/bench/browse": answer})
    s.goto("/kernel-corpus.html#view=workload&model=m&phase=decode")
    assert s.wait_for(lambda: s.page.input_value("#f-context") == "4096")
    assert s.wait_for(lambda: s.page.locator("#f-context option").all_inner_texts()[-1] == "no context axis")
    assert "120 measurements" in s.page.inner_text("#workload-unplaced")
    s.assert_clean()


def test_a_row_timed_before_the_drain_carries_the_caveat(browser_base):  # noqa: F811
    """A measurement naming no clock (`timing_method` null) was timed before
    the harness billed a launch's own L2 write-back: its latency is marked,
    and nothing else about it changes. A row naming its clock is not."""
    browser, base = browser_base
    rows = [{**_CORPUS_ROW, "id": "old", "latency_us": 7.0, "timing_method": None,
             "shape_json": {"B": 1}, "dtype_env_json": {"activation": "bf16"}},
            {**_CORPUS_ROW, "id": "new", "latency_us": 9.0, "timing_method": "l2_drain",
             "shape_json": {"B": 2}, "dtype_env_json": {"activation": "bf16"}}]
    s = Session(browser, base, {"/api/bench/browse": lambda q: {
        "total": 2, "items": rows, "facets": {"model": [_CORPUS_ROW["model"]]}}})
    s.goto(f"/kernel-corpus.html#view=rows&source=mine&run={_RUN}")
    s.page.wait_for_selector("text=2 rows", timeout=15000)
    marked = s.page.locator("#table-section tbody [title^='Measured before the L2 drain']")
    assert marked.count() == 1
    assert marked.inner_text().startswith("7.0")
    assert s.page.locator("#table-section tbody tr").count() == 2
    s.assert_clean()
