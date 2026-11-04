"""A recurrent state's axes are the ones its HF forward allocates.

Every corpus GDN / KDA model has d_k = d_v, so neither the element count nor
any golden tells `[H, d_k, d_v]` from `[H, d_v, d_k]`. These cases take the
two apart: an asymmetric head (d_k != d_v) for the GDN forwards, and the
symbolic walk, whose extents keep their dim names, for KDA, where the two
forwards in the corpus allocate opposite layouts.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import expr_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_loader, snapshot_config


def _states(model_id, config, symbolic=False):
    resolved = shared_loader().match(config, model_id=model_id)
    deployment = Deployment(Program.of(resolved)).at(batch=1, input_tokens=8)
    if symbolic:
        # The walk over the workload's counts: each axis's identity half is
        # its law in the model's names.
        deployment = replace(deployment,
                             workload=deployment.workload.symbolic()[0])
    analysis = deployment.analyze()
    out = {}
    for states in analysis.cache_state.values.values():
        for name, axes in states.items():
            out.setdefault(name, set()).add(
                tuple(str(expr_of(a)) if symbolic else int(a) for a in axes))
    return out


@pytest.mark.parametrize("model_id, scope", [
    # modeling_qwen3_5_moe.py:298 `torch.zeros(batch_size, num_heads,
    # k_head_dim, v_head_dim)`; the conv cache keeps `conv_kernel_size`
    # columns per channel (transformers cache_utils.py:1033-1034).
    ("Qwen/Qwen3.5-35B-A3B", "text_config"),
    # modeling_qwen3_next.py:426 the same allocation.
    ("Qwen/Qwen3-Coder-Next", None),
])
def test_a_gated_delta_net_state_is_heads_by_key_by_value(model_id, scope):
    config = copy.deepcopy(snapshot_config(model_id))
    text = config[scope] if scope else config
    text["linear_key_head_dim"] = 64
    heads, d_v = text["linear_num_value_heads"], text["linear_value_head_dim"]
    assert d_v != 64
    conv_channels = (2 * text["linear_num_key_heads"] * 64 + heads * d_v)
    states = _states(model_id, config)
    assert states["recurrent_state"] == {(heads, 64, d_v)}
    assert states["conv_state"] == {(conv_channels, text["linear_conv_kernel_dim"])}


def test_kimi_kda_keeps_its_state_value_first():
    """Kimi's forward calls fla's KDA kernels with
    `transpose_state_layout=True` (modeling_kimi_linear.py:625, :643): the
    state is [H, V, K]."""
    states = _states("moonshotai/Kimi-K3", snapshot_config("moonshotai/Kimi-K3"),
                     symbolic=True)
    assert states["recurrent_state"] == {
        ("n_lin_v_heads", "d_lin_v_head", "d_lin_k_head")}


def test_glm_kda_keeps_its_state_key_first():
    """GLM-5.3's own KDA recurrence allocates
    `torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)`
    (modeling_glm5_next.py:460): the state is [H, K, V]."""
    states = _states("zai-org/GLM-5.3-Flash",
                     snapshot_config("zai-org/GLM-5.3-Flash"), symbolic=True)
    assert states["recurrent_state"] == {
        ("n_lin_v_heads", "d_lin_k_head", "d_lin_v_head")}
