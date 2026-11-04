"""conv2d provider — torch F.conv2d baseline.

The ViT downsamplers (Step-3.7) are full k×k strided convs over a flattened
square patch grid. Only a portable torch baseline is registered: PyTorch
dispatches conv2d to cuDNN on CUDA and to its own kernels on CPU — the same op
a real stack runs (no vendor custom conv in the serving path), so no
`_require_cuda` gate (matches the torch GEMM baseline, which the CPU xcheck
suite relies on). Same reshape contract as the reference: `[B, s_conv_grid=H·W,
C_in]` → `[B, C_in, H, W]` → padded (k//2) strided conv → flatten back.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import BenchCase, KernelSignature, Provider, register
from .attention import _torch_dtype


def _torch_conv2d_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch.nn.functional as F
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)                          # [B, s_conv_grid, C_in]
    w = inputs["weight"].to(dt)                     # [C_out, C_in, k, k]
    bias = inputs["bias"].to(dt)
    B, s, C_in = x.shape
    h = math.isqrt(s)                          # s is a perfect square
    grid = x.transpose(1, 2).reshape(B, C_in, h, h).contiguous()
    stride = int(inputs["stride"])
    pad = int(w.shape[-1]) // 2

    def run():
        return F.conv2d(grid, w, bias=bias, stride=stride, padding=pad)

    def to_canonical(out):
        Bo, Co, Ho, Wo = out.shape
        return out.reshape(Bo, Co, Ho * Wo).transpose(1, 2)

    return BenchCase(run=run, to_canonical=to_canonical)


register(Provider(
    library="torch", signature="conv2d", kernel_impl="torch_conv2d",
    import_name="torch",
    # Every operand cast to the cell's dtype; plain F.conv2d.
    serves=KernelSignature(signature="conv2d",
                           weight=("bf16", "fp16", "fp32"),
                           activation=("bf16", "fp16", "fp32")),
    build=_torch_conv2d_build,
))
