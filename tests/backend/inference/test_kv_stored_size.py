"""A cache position's stored size is one fact, and every consumer reads it.

A quantised cache stores, beside each position's data, the dynamic scales its
format writes per token-head / channel group / tile. Those scales are part of
the position: they occupy HBM and the decode kernel reads them. They used to be
added back by some consumers and not others, so one deployment had two
per-token figures (the budget's, with scales; the KV formula badge's and the
Inspector's, without) and two per-request figures (the batch ceiling's, and the
per-class breakdown's that was meant to explain it).

What is checked, for every cache in a set of deployments that includes formats
with dynamic scales — shipped (DeepSeek-V3.2, V4.1-Flash) and chosen by a
precision profile (`fp8_per_token_head`) — beside ones with none or only static
scales:

* the per-class storage rows add up to the request's stored bytes, at lengths
  inside and past every window;
* per-token growth × L + floor is the stored bytes past every window (the slope
  is the slope of the same law);
* the KV formula badge (`growth_formula`) states the per-token figure the budget
  divides by, and the per-class request breakdown adds up to the per-request
  figure the batch ceiling divides by;
* every formula tree in the estimate: a node's children add up to it.
"""
from __future__ import annotations

from functools import lru_cache

import pytest

from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.projection import project_model
from backend.arch.precision_profile import PrecisionProfile
from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config

#: Every cache slot kind the corpus quantises, stored per token-head.
_PER_TOKEN_HEAD = {"canonical": {"cache": [{
    "select": {"slots": ["kv", "latent", "rope_key", "index_key",
                         "compressed_kv"]},
    "format": "fp8_per_token_head"}]}}
_KV_FP8 = {"preset": "kv_fp8"}

#: Long enough to be past every window in the corpus; a multiple of every
#: compression ratio, so the slope between the two is exact.
_LONG = (1 << 17, 1 << 18)

FAST = [("Qwen/Qwen3-8B", _PER_TOKEN_HEAD)]
SWEEP = [
    ("deepseek-ai/DeepSeek-V3.2", None),
    ("deepseek-ai/DeepSeek-V4.1-Flash", None),
    ("zai-org/GLM-5.3-Flash", None),
    ("Qwen/Qwen3-Coder-Next", None),
    ("deepseek-ai/DeepSeek-V3.1", _KV_FP8),
    ("openai/gpt-oss-20b", _PER_TOKEN_HEAD),
]


def _profile(request):
    return None if request is None else PrecisionProfile.from_request(request)


def _id(case):
    model, request = case
    return f"{model}|{'shipped' if request is None else next(iter(request))}"


@lru_cache(maxsize=None)
def _view(model, key):
    request = dict(FAST + SWEEP)[model] if key else None
    report = project_model(resolve_snapshot(model), batch=1, input_tokens=1,
                           precision=_profile(request))
    return cache_view_from_analysis(report.analysis)


def _estimate(model, request, length):
    stem = model.replace("/", "__")
    return estimate_capacity(
        config=snapshot_config(stem), gpu_sku="H100-SXM",
        resolved=resolve_snapshot(stem, shared_loader()),
        selection_mode="theoretical_only",
        avg_input_tokens=length, avg_output_tokens=1,
        parallel=ParallelConfig(tp_attention=1), precision=request)


def _children_add_up(tree, where):
    subs = tree.get("subterms") or []
    if subs and tree.get("result_bytes") is not None:
        total = sum(st["result_bytes"] for st in subs)
        assert total == pytest.approx(tree["result_bytes"], rel=1e-9), (
            f"{where}: {tree.get('label')} is {tree['result_bytes']}, "
            f"its children add up to {total}")
    for i, st in enumerate(subs):
        _children_add_up(st, f"{where}/{i}")


def _check(model, request):
    view = _view(model, request is not None)
    for length in (4096, *_LONG):
        rows = sum(r["request_bytes"] for r in view.storage_rows(length))
        assert rows == pytest.approx(view.storage_bytes_at(length), rel=1e-12)
    growth, floor = view.growth_and_floor()
    stored = [view.storage_bytes_at(length) for length in _LONG]
    assert (stored[1] - stored[0]) / (_LONG[1] - _LONG[0]) == pytest.approx(
        growth, rel=1e-12)
    assert growth * _LONG[0] + floor == pytest.approx(stored[0], rel=1e-12)
    assert view.growth_formula["result_bytes"] == growth

    # The deployment: one GPU, so per rank is the whole model.
    est = _estimate(model, request, _LONG[0] - 1)
    kv = est["kv_cache"]
    assert kv["per_token_bytes"] == pytest.approx(growth, rel=1e-12)
    assert kv["growth_formula"]["result_bytes"] == kv["per_token_bytes"]
    assert kv["per_request_kv_bytes"] == pytest.approx(stored[0], rel=1e-12)
    if kv.get("per_class_request"):
        assert sum(c["request_bytes"] for c in kv["per_class_request"]) == (
            pytest.approx(kv["per_request_kv_bytes"], rel=1e-12))
    assert est["formulas"]["per_request_kv"]["result_bytes"] == (
        kv["per_request_kv_bytes"])
    _children_add_up(kv["growth_formula"], "growth_formula")
    for name, tree in est["formulas"].items():
        if isinstance(tree, dict):
            _children_add_up(tree, name)


@pytest.mark.parametrize("case", FAST, ids=_id)
def test_one_stored_size_per_position(case):
    _check(*case)


@pytest.mark.regression
@pytest.mark.parametrize("case", SWEEP, ids=_id)
def test_one_stored_size_per_position_across_formats(case):
    _check(*case)
