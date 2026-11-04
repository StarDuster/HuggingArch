"""The deployment a model opens at on a card (`default_deployment`): one GPU
for a dense model; for a MoE, DP = EP from the card's fabric unit, doubled
through the layout's accepted degrees until the weights fit."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _opening(model_id: str, gpu_sku: str) -> dict:
    from backend.api._models import InferenceEstimateRequest
    from backend.api.estimate_env import resolve_estimate_environment
    from backend.arch.loader import get_default_loader
    from backend.inference.default_deployment import default_deployment
    env = resolve_estimate_environment(
        InferenceEstimateRequest(model_id=model_id, gpu_sku=gpu_sku))
    resolved = get_default_loader().match(env["config"], model_id=model_id)
    return default_deployment(env, resolved)


@pytest.mark.parametrize("model_id,gpu_sku,start,degree", [
    # GB200's tray is 4 GPUs; DeepSeek-V3's FP8 weights need 8 of them.
    ("deepseek-ai/DeepSeek-V3", "GB200", 4, 8),
    ("deepseek-ai/DeepSeek-V3", "H100-SXM", 16, 16),
])
def test_a_moe_opens_at_the_first_degree_that_fits(model_id, gpu_sku, start, degree):
    out = _opening(model_id, gpu_sku)
    assert out["moe"] and out["fits"]
    assert out["start_degree"] == start
    assert (out["parallel"]["dp_attention"], out["parallel"]["ep_moe"]) == (degree, degree)


def test_a_dense_model_opens_on_one_gpu():
    out = _opening("meta-llama/Llama-3.2-1B", "H100-SXM")
    assert out == {"parallel": {}, "moe": False, "start_degree": 1, "fits": True}
