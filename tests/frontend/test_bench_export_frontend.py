"""The Inference page's "Export bench jobs" button (`web/src/pages/BenchExport.tsx`).

Two things are worth holding on the page side. The request must describe the
step ON SCREEN — the mesh the user chose whole (not a flattened TP degree),
the estimate's own workload, the framework and precision it ran under —
because a job keyed at another deployment measures another plan. And the
drop list must be shown: a plan silently missing its unmeasurable ops reads
exactly like full coverage.

Same harness as `test_pages_smoke.py` (built `web/dist`, canned payloads).
"""
from __future__ import annotations

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401

#: The deployment the link states: DeepSeek-V3 on B200 at EP=8 × DP-attn=8,
#: vLLM, KV in FP8 — every one of them has to reach the builder.
_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest&prec=kv_fp8")

_PLAN = {
    "jsonl": '{"op": "linear", "signature": "gemm"}\n',
    "job_count": 1,
    "dropped": ["view_split",
                "mul [mlp.experts.act]: per-rank shape has an axis below one "
                "row (M=0) — an average, not a geometry"],
    "filename": "deepseek-ai_DeepSeek-V3_decode_jobs.jsonl",
}


def _page(browser, base, plan=None):
    s = Session(browser, base,
                overrides={"/inference/bench/plan-jobs": plan or _PLAN})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-phase-breakdown", state="attached", timeout=15000)
    return s


def _export(s, phase: str = "decode"):
    s.page.click(f"#infer-bench-export-{phase}")
    assert s.wait_for(lambda: s.count("/inference/bench/plan-jobs") >= 1)
    return s.requests["/inference/bench/plan-jobs"][0]["body"]


def test_the_export_asks_for_the_deployment_on_screen(browser_base):  # noqa: F811
    browser, base = browser_base
    s = _page(browser, base)
    body = _export(s)
    assert body["model_id"] == "deepseek-ai/DeepSeek-V3"
    assert body["gpu_sku"] == "B200"
    assert body["phase"] == "decode"
    # The mesh whole — every axis the page edits, plus the CP scheme. A
    # flattened `tp` would plan shapes of a deployment nobody asked for.
    assert body["parallel"]["ep_moe"] == 8
    assert body["parallel"]["dp_attention"] == 8
    for axis in ("pp", "tp_attention", "tp_dense_ffn", "tp_moe_expert",
                 "cp", "dp_replicas"):
        assert axis in body["parallel"], axis
    assert body["parallel"]["cp_scheme"] == "pcp"
    assert body["parallel"]["ep_load_factor"] == 1
    # The framework whose kernels the table timed, and the precision profile
    # the estimate ran under: a job under another dtype environment keys to
    # measurements that are not of this plan.
    assert body["framework"] == "vllm"
    assert body["precision"] == {"preset": "kv_fp8"}
    s.assert_clean()


def test_the_export_asks_at_the_step_the_table_shows(browser_base):  # noqa: F811
    """The workload is the ESTIMATE's, as the phase table reports it — the
    decode batch is the backend's KV ceiling, not the form's."""
    browser, base = browser_base
    s = _page(browser, base)
    body = _export(s)
    assert body["avg_input_tokens"] == 1024
    assert body["avg_output_tokens"] == 1024
    assert body["prefix_cached_tokens"] == 0
    assert body["speculative_draft_tokens"] == 0
    assert body["batch_size"] == 507  # phases.decode.batch in the fixture
    s.assert_clean()


def test_each_phase_exports_its_own_step(browser_base):  # noqa: F811
    browser, base = browser_base
    s = _page(browser, base)
    s.page.click("#infer-phase-tab-prefill")
    body = _export(s, "prefill")
    assert body["phase"] == "prefill"
    assert body["batch_size"] == 1  # phases.prefill.batch in the fixture
    s.assert_clean()


def test_one_export_per_step_not_per_table(browser_base):  # noqa: F811
    """The control belongs to the STEP. A per-op table is not one — the
    speculative-draft breakdown is part of the decode step, and the decode
    export already carries its ops — so a second button here would offer the
    same file twice under the same id."""
    browser, base = browser_base
    s = _page(browser, base)
    assert len(s.page.query_selector_all("[data-bench-export]")) == 1
    s.assert_clean()


def test_the_ops_with_no_signature_are_named_on_the_page(browser_base):  # noqa: F811
    """`dropped` is part of the answer, so it is on screen with the ops in
    it — the page can never show a measured total while one is unmeasured."""
    browser, base = browser_base
    s = _page(browser, base)
    _export(s)
    note = s.page.inner_text("#infer-bench-export-note-decode")
    assert "1 job" in note
    assert "view_split" in note
    # The op, not the whole planner note, with the reason in the tooltip.
    assert "mul" in note
    assert "per-rank shape" not in note
    assert "per-rank shape" in s.page.get_attribute(
        "[data-bench-export-dropped]", "title")
    s.assert_clean()


def test_a_plan_that_drops_nothing_says_nothing(browser_base):  # noqa: F811
    browser, base = browser_base
    s = _page(browser, base, plan={**_PLAN, "dropped": []})
    _export(s)
    note = s.page.inner_text("#infer-bench-export-note-decode")
    assert "not exported" not in note
    assert s.page.query_selector("[data-bench-export-dropped]") is None
    s.assert_clean()
