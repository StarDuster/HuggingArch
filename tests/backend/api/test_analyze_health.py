"""Helpers for tools/analyze_health.py — no live /analyze sweep."""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_health import (
    INSPECTOR_DEFAULTS,
    cluster_size,
    collect_models,
    estimate_body,
    jobs_for_model,
    parse_match_model_ids,
    scan_bad_values,
    _par,
)


def test_parse_match_model_ids_list_and_scalar():
    assert parse_match_model_ids("match:\n  model_id: [a/b, c/d]\n") == ["a/b", "c/d"]
    assert parse_match_model_ids("  model_id: [deepseek-ai/DeepSeek-V3]\n") == [
        "deepseek-ai/DeepSeek-V3",
    ]


def test_scan_bad_values_nan_inf_negative_empty():
    hits = scan_bad_values({
        "dashboard": {"total_flops": float("nan"), "total_params": -1},
        "layer_groups": [],
        "t_sol_s": float("inf"),
    })
    joined = " ".join(hits)
    assert "NaN" in joined
    assert "Inf" in joined
    assert "negative" in joined
    assert "empty list" in joined
    assert not scan_bad_values({"total_params": 1, "layer_groups": [{"n": 1}]})
    assert math.isnan(float("nan"))


def test_cluster_size_matches_frontend():
    assert cluster_size(_par()) == 1
    assert cluster_size(_par(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)) == 8
    assert cluster_size(_par(ep_moe=8, dp_attention=8)) == 8
    assert cluster_size(_par(cp=8)) == 8


def test_inspector_defaults_match_frontend():
    assert INSPECTOR_DEFAULTS == {
        "batch_size": 1, "seq_len": 2048,
    }
    body = estimate_body("m", _par())
    assert body["gpu_sku"] == "B200"
    assert body["avg_input_tokens"] == 1024
    assert body["num_gpus"] == 1


def test_jobs_for_dense_vs_moe():
    dense = jobs_for_model({
        "model_id": "Qwen/Qwen3-8B", "moe": False, "fp8": False,
        "int4": False, "has_attention": True,
    })
    moe = jobs_for_model({
        "model_id": "deepseek-ai/DeepSeek-V3", "moe": True, "fp8": True,
        "int4": False, "has_attention": True,
    })
    dense_cfgs = {j["config"] for j in dense if j["kind"] == "estimate"}
    moe_cfgs = {j["config"] for j in moe if j["kind"] == "estimate"}
    assert "single_as_shipped" in dense_cfgs
    assert "tp8_kv_fp8" in dense_cfgs
    assert "cp8_as_shipped" in dense_cfgs
    assert "ep8_as_shipped" not in dense_cfgs
    assert "ep8_as_shipped" in moe_cfgs
    # Every estimate body is one the API takes: precision, never the
    # retired dtype fields.
    from backend.api._models import InferenceEstimateRequest
    for job in (*dense, *moe):
        if job["kind"] == "estimate":
            InferenceEstimateRequest(**job["json"])
    assert any(j["kind"] == "analyze" and j["json"]["seq_len"] == 2048 for j in dense)


def test_collect_models_sees_shared_tree():
    models = collect_models()
    ids = {m["model_id"] for m in models}
    assert "deepseek-ai/DeepSeek-V3" in ids
    assert any(m.get("moe") for m in models)
