"""Regression tests for the Inference-compare per-model config plumbing.

A clobber used to drop a per-model row's saved config on URL restore, so the
request the frontend sent to /inference/estimate-sweep didn't match the row's
cfg (big-MoE rows then OOM'd in compare but not on the single inference page):
_icApplyStashedCfgs read live `_icState.slots[i].cfg` in a loop while
applyInferenceControlsSnapshot's change events re-snapshotted not-yet-applied
rows at their defaults → every row after the first lost its parallel
(ep_moe→1).

Compare prices every model as shipped: the link below was saved before
precision profiles and still carries precision fields, which must not reach
the backend (it refuses them by name).

These are frontend (browser) bugs, so the test drives the real JS in headless
chromium against the running dev server (it serves the working-copy frontend
live). Only estimate-sweep is intercepted, to capture the request bodies. We
then assert every captured request carries that row's ep_moe and no precision
field.

Skips cleanly where playwright/chromium isn't installed or the dev server
(127.0.0.1:2053) isn't reachable — the frontend is only served by the app.
"""
from __future__ import annotations

import json
import socket

import pytest

_HOST, _PORT = "127.0.0.1", 2053
_BASE = f"https://{_HOST}:{_PORT}"

# The saved override under test: ep_moe=8 (shards a big MoE).
_OVERRIDE_EP = 8

# Precision fields an old link carries; Compare drops them.
_RETIRED_PRECISION_FIELDS = {
    "act_dtype_bytes", "act_dtype_per_role", "kv_dtype", "kv_quant_granularity",
    "kv_scale_dtype", "weight_dtype_bytes", "weight_quant_granularity",
    "weight_scale_dtype", "weight_dtype_per_role", "compute_dtype_bytes",
}

_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V4-Pro",
]

# A DISTINCT non-default GPU per row. None is the form default (H100-SXM), so a
# row reverting to default OR bleeding another row's value is both visible — and
# the GPU select restores via the async options-load + pendingValue path, which
# this exercises (the rest of the cfg is identical across rows).
_GPUS = ["B200", "A100-80G-SXM", "H200"]


def _ctrl_cfg(gpu_sku="B200"):
    """A full inference-controls snapshot where EVERY field carries a distinct
    non-default value — so a clobber back to defaults is visible on any field,
    not just the one that originally regressed. (`tie_tp` off so the three TP
    knobs can differ.) It also carries precision fields from before
    profiles, as an old link does."""
    return {
        "gpu_sku": gpu_sku, "kv_dtype": "fp8",
        "kv_quant_granularity": "per-tile", "kv_scale_dtype": "ue8m0",
        "weight_dtype_bytes": 2,
        "weight_quant_granularity": "per-tensor", "weight_scale_dtype": "fp32",
        "compute_dtype_bytes": None, "act_dtype_bytes": 1,
        "gpu_memory_fraction": 0.85, "tie_tp": False,
        "parallel": {
            "tp_attention": 2, "tp_dense_ffn": 1, "tp_moe_expert": 1,
            "ep_moe": _OVERRIDE_EP, "cp": 1, "pp": 1,
            "dp_attention": 4, "dp_replicas": 1,
        },
        "act_dtype_per_role": {}, "weight_dtype_per_role": {},
        "prefix_cached_tokens": 256,
        "prefill_chunk_size": 4096,
        "speculative_mode": "none", "speculative_draft_tokens": 0,
        "acceptance_rate": 1, "include_aux_weights": True,
    }


def _compare_url() -> str:
    import base64
    from urllib.parse import quote
    state = {"template": _ctrl_cfg(),
             "cfgs": [_ctrl_cfg(g) for g in _GPUS]}
    s = base64.urlsafe_b64encode(json.dumps(state).encode()).decode().rstrip("=")
    m = "%7C".join(quote(x, safe="") for x in _MODELS)
    return f"{_BASE}/compare.html#sub=inference&m={m}&s={s}"


def _server_up() -> bool:
    try:
        with socket.create_connection((_HOST, _PORT), timeout=2):
            return True
    except OSError:
        return False


def _stub_routes(page, captured: list):
    def estimate_sweep(route):
        try:
            captured.append(route.request.post_data_json)
        except Exception:
            pass
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"results": []}))

    page.route("**/inference/estimate-sweep", estimate_sweep)


def test_per_model_config_reaches_backend():
    sync_api = pytest.importorskip("playwright.sync_api")
    if not _server_up():
        pytest.skip(f"dev server {_BASE} not reachable (frontend is app-served)")

    captured: list = []
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            page = browser.new_context(ignore_https_errors=True).new_page()
            _stub_routes(page, captured)
            page.goto(_compare_url(), wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)            # hydrate + first-paint prefill
            page.click("#ic-run", timeout=15000)
            page.wait_for_timeout(4000)            # let all sweeps fire
        finally:
            browser.close()

    assert captured, "no /inference/estimate-sweep requests were captured"
    by_model: dict = {}
    for body in captured:
        by_model.setdefault(body.get("model_id"), []).append(body)

    # All three rows must have fired (none dropped).
    assert set(by_model) == set(_MODELS), f"missing rows: {set(_MODELS) - set(by_model)}"

    # EVERY scalar + parallel field must arrive as configured — not just the
    # one that originally regressed — and no precision field arrives at all.
    # speculative_mode is materialised into speculative_draft_tokens /
    # acceptance_rate, so it's not a 1:1 payload field; left at the default
    # 'none' here.
    cfg = _ctrl_cfg()
    expected_gpu = dict(zip(_MODELS, _GPUS, strict=True))  # each row has its OWN gpu
    scalar_fields = [
        "gpu_memory_fraction", "prefix_cached_tokens", "prefill_chunk_size",
        "include_aux_weights",
    ]
    parallel_fields = list(cfg["parallel"])
    for model, bodies in by_model.items():
        for b in bodies:
            assert b.get("gpu_sku") == expected_gpu[model], (
                f"{model}: gpu_sku={b.get('gpu_sku')!r}, expected "
                f"{expected_gpu[model]!r} (GPU restored to a default / wrong row)")
            assert not _RETIRED_PRECISION_FIELDS & set(b), (
                f"{model}: precision fields reached the backend: "
                f"{sorted(_RETIRED_PRECISION_FIELDS & set(b))}")
            assert "precision" not in b, f"{model}: Compare row carries a precision"
            for f in scalar_fields:
                assert b.get(f) == cfg[f], (
                    f"{model}: {f}={b.get(f)!r}, expected {cfg[f]!r} "
                    "(per-model config clobbered back to a default on restore)")
            bp = b.get("parallel") or {}
            for f in parallel_fields:
                assert bp.get(f) == cfg["parallel"][f], (
                    f"{model}: parallel.{f}={bp.get(f)!r}, "
                    f"expected {cfg['parallel'][f]!r}")


if __name__ == "__main__":
    test_per_model_config_reaches_backend()
    print("ok")
