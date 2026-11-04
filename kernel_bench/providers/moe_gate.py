"""MoE router providers — aux-loss-free grouped top-k (DeepSeek-V3 noaux_tc).

The `moe_gate` signature: sigmoid(gate) + correction bias → per-group top-2
sum → pick `topk_group` groups → mask → top-k experts → gather weights from
the unbiased scores → renormalize + routed scaling. One kernel in vLLM
(`_moe_C.grouped_topk`) and sglang (`moe_fused_gate`); torch is the always-
available baseline (the eager `biased_grouped_topk_impl`).

Kernels emit unsorted top-k ids. `to_canonical` scatters weights onto a
dense [tokens, E] map so the numeric cross-check is order-invariant.

Import-guarded: library imports are lazy inside `build`.
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register
from .attention import _require_cuda, _torch_dtype


# The router logits vLLM's CUDA routers dispatch on are torch's float types
# — float, half and bfloat16 (v0.32.0 csrc/moe/topk_softmax_kernels.cu:845-858
# softmax, :886-899 sigmoid; topk_softplus_sqrt_kernels.cu:999-1013) — and
# sglang's Triton `moe_fused_gate` asserts the same three (v0.5.22
# kernels/ops/moe/moe_fused_gate.py:377-381), so both declare `TORCH_FLOAT`
# rather than a second spelling of it. The two launches that take fp32 logits
# only say so where they are registered.


def _grouping(inputs: Dict[str, Any]):
    return (inputs["num_expert_group"], inputs["topk_group"], inputs["topk"])


def _moe_gate_dense_out(out, n_experts):
    from ..refs import _moe_gate_dense
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return _moe_gate_dense(out[0], out[1], n_experts)
    raise TypeError(
        f"moe_gate kernel returned {type(out).__name__}, expected (weights, ids)")


# --- torch baseline (always available): the eager noaux_tc router ----------

def _torch_moe_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    from ..refs import _moe_gate_dense
    n_group, topk_group, topk = _grouping(inputs)
    gating, bias = inputs["gating"], inputs["bias"]
    n_experts = int(gating.shape[-1])

    def _run():
        scores = gating.sigmoid()
        n_tok, E = scores.shape
        sfc = scores + bias.unsqueeze(0)
        group_scores = sfc.view(n_tok, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        gidx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        gmask = torch.zeros_like(group_scores).scatter_(1, gidx, 1)
        smask = gmask.unsqueeze(-1).expand(n_tok, n_group, E // n_group).reshape(n_tok, -1)
        masked = sfc.masked_fill(~smask.bool(), float("-inf"))
        _, ids = torch.topk(masked, k=topk, dim=-1, sorted=False)
        weights = scores.gather(1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, ids
    return BenchCase(run=_run,
                     to_canonical=lambda o: _moe_gate_dense(o[0], o[1], n_experts))


register(Provider(
    library="torch", signature="moe_gate", kernel_impl="torch_grouped_topk",
    import_name="torch",
    serves=KernelSignature(signature="moe_gate", activation=TORCH_FLOAT),
    build=_torch_moe_gate_build,
))


def _torch_softmax_act_build(inv, env, inputs) -> BenchCase:
    import torch
    x = inputs["x"]
    return BenchCase(
        run=lambda: torch.nn.functional.softmax(x, dim=-1),
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="softmax_act", kernel_impl="torch_softmax",
    import_name="torch",
    serves=KernelSignature(signature="softmax_act", activation=TORCH_FLOAT),
    build=_torch_softmax_act_build,
))


def _torch_topk_build(inv, env, inputs) -> BenchCase:
    x, k = inputs["x"], int(inputs["k"])
    return BenchCase(
        run=lambda: x.topk(k, dim=-1).indices,
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="topk", kernel_impl="torch_topk",
    import_name="torch",
    # The launch writes int64 indices whatever it reads, but the corpus keys
    # that index edge at a float label (DeepSeek-V4: `bf16`). Until eval
    # states the port's dtype (docs/known_limitations.md §4.3) the output is any
    # float label the build can be keyed at, not one model's label.
    serves=KernelSignature(signature="topk", activation=TORCH_FLOAT,
                           output=TORCH_FLOAT),
    build=_torch_topk_build,
))


def _torch_gather_build(inv, env, inputs) -> BenchCase:
    x, idx = inputs["x"], inputs["idx"]
    return BenchCase(
        run=lambda: x.gather(-1, idx),
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="gather", kernel_impl="torch_gather",
    import_name="torch",
    serves=KernelSignature(signature="gather", activation=TORCH_FLOAT),
    build=_torch_gather_build,
))


# --- vLLM: torch.ops._moe_C.grouped_topk ------------------------------------

def _gate_int(inv: Invocation, name: str) -> int:
    """A gate parameter the invocation states, as an int. Absent or
    unparseable is an error: a top-k of 0 is a cell that looks measured and
    measured nothing."""
    raw = inv.kernel_params.get(name)
    if raw is None:
        raw = inv.dims.get(name)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{inv.signature}: gate parameter {name!r} is {raw!r}, not an int"
        ) from None


def _vllm_moe_gate_applies(inv: Invocation, env: Env) -> bool:
    E = _gate_int(inv, "n_experts")
    ng = _gate_int(inv, "n_expert_groups")
    topk = _gate_int(inv, "n_experts_active")
    return (inv.signature == "moe_gate" and ng > 0 and topk > 0
            and ng <= 32 and topk <= 32 and E % ng == 0)


def _vllm_fused_grouped_topk():
    """vLLM's own entry, `fused_grouped_topk` — what serving calls: at
    v0.32.0 `GroupedTopKRouter` picks `grouped_topk`
    (router/grouped_topk_router.py:337), which hands a biased CUDA gate to
    `fused_grouped_topk` (:99), which runs `ops.grouped_topk` (:44). The
    function moved in v0.15.0 (the router refactor) from
    `fused_moe/fused_moe.py` to `fused_moe/router/grouped_topk_router.py`
    with its signature unchanged, so it is ONE kernel entry relocated: the
    audited release imports the new path, a box on an earlier release (0.11)
    the old one. `ops.grouped_topk` beneath it did change signature (0.11
    takes scores + scores_with_bias and no scoring function), which is why
    the provider calls the entry and not the op."""
    from .linear_ssm import _relocated
    return _relocated(
        "fused_grouped_topk",
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
        "vllm.model_executor.layers.fused_moe.fused_moe")


def _vllm_moe_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    fused_grouped_topk = _vllm_fused_grouped_topk()
    n_group, topk_group, topk = _grouping(inputs)
    scores, bias = inputs["gating"], inputs["bias"]      # raw logits, sigmoid scoring
    n_experts = int(scores.shape[-1])

    def _run():
        return fused_grouped_topk(
            scores, scores, topk, True, bias, num_expert_group=n_group,
            topk_group=topk_group, scoring_func="sigmoid",
            routed_scaling_factor=1.0)
    return BenchCase(
        run=_run,
        to_canonical=lambda o: _moe_gate_dense_out(o, n_experts),
    )


register(Provider(
    library="vllm", signature="moe_gate", kernel_impl="vllm_moe_c_grouped_topk",
    import_name="vllm",
    applies_to=_vllm_moe_gate_applies, build=_vllm_moe_gate_build,
))


# --- sglang: sgl_kernel.moe_fused_gate --------------------------------------

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _sgl_moe_gate_applies(inv: Invocation, env: Env) -> bool:
    E = _gate_int(inv, "n_experts")
    ng = _gate_int(inv, "n_expert_groups")
    # sglang kernel: experts_per_group <= 32 and num_experts a power of two.
    return (inv.signature == "moe_gate" and ng > 0 and E % ng == 0
            and (E // ng) <= 32 and _is_pow2(E))


def _sgl_moe_gate_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sgl_kernel import moe_fused_gate
    n_group, topk_group, topk = _grouping(inputs)
    gate, bias = inputs["gating"], inputs["bias"]        # kernel applies sigmoid internally
    n_experts = int(gate.shape[-1])

    def _run():
        return moe_fused_gate(gate, bias, n_group, topk_group, topk)
    return BenchCase(
        run=_run,
        to_canonical=lambda o: _moe_gate_dense_out(o, n_experts),
    )


register(Provider(
    library="sglang", signature="moe_gate", kernel_impl="sgl_moe_fused_gate",
    import_name="sgl_kernel",
    # The AOT kernel this calls was retired in sglang-kernel 0.4.5 (sglang
    # #26771, commit 1da7d3a50b; 0.4.4 still exports it): `sgl_kernel/moe.py`
    # keeps only a note that it moved. At the audited sglang v0.5.22 a CUDA
    # DeepSeek-style gate runs FlashInfer's `fused_topk_deepseek`
    # (srt/layers/moe/topk.py:1838) or, failing its constraints, the Triton
    # `sglang.kernels.ops.moe.moe_fused_gate` (:1894); `sgl_kernel`'s is
    # called only on XPU (:1331). Those are different kernels, so they are
    # not swapped in under this name: on a release without it this kernel is
    # simply absent.
    version_specifier="<0.4.5",
    applies_to=_sgl_moe_gate_applies, build=_sgl_moe_gate_build,
))


# ---------------------------------------------------------------------------
# Ungrouped top-k routers — the fused spans `moe_topk_softmax`,
# `moe_topk_sigmoid`, `moe_topk_softmax_bias`, `moe_topk_sqrtsoftplus` and
# `moe_topk_hash_sqrtsoftplus`. Each framework has one entry for all of them,
# and the providers call it the way the serving router does (vLLM v0.32.0,
# sglang v0.5.22):
#
#   vLLM   fused_moe/router/fused_topk_router.py:80 `fused_topk` (softmax →
#          `ops.topk_softmax`), and fused_topk_bias_router.py:146
#          `fused_topk_bias` (sigmoid → `ops.topk_sigmoid`; softmax with the
#          bias → `ops.topk_softmax`; sqrtsoftplus → the Triton `dsv4_topk`
#          when `can_use_dsv4_topk` holds — fp32 logits, 256/384 experts,
#          top-6, renormalised — else `ops.topk_hash_softplus_sqrt`, which
#          also serves the hash table). `hidden_states` is read only for its
#          row count, so the logits stand in for it.
#   sglang kernels/ops/moe/moe_fused_gate.py:337 `moe_fused_gate`, the
#          unified Triton router `fused_topk` (srt/layers/moe/topk.py:1061)
#          and `biased_topk_jit_kernel_impl` (:1471) call on CUDA for
#          softmax / sigmoid / sqrtsoftplus (its sigmoid path takes the radix
#          fast path itself when `moe_route_radix.covered`); the hash router
#          is kernels/ops/moe/dsv4.py:114 `hash_topk` (JIT), as
#          srt/layers/moe/hash_topk.py:236 calls it.
#
# Both return (weights, ids) per token — fp32 weights and int32 ids
# (vLLM fused_topk_router.py:93-98, sglang moe_fused_gate.py:451-452) — while
# the corpus keys the router's output at the logits' dtype; `output` is left
# undeclared so those cells stay this launch's; `to_canonical` scatters them onto
# [tokens, n_experts] as the oracle does. Renormalisation and the routed
# scale come from the inputs (`refs._moe_topk_inputs`).
# ---------------------------------------------------------------------------

def _topk_dense(out, n_experts):
    from ..refs import _moe_gate_dense
    return _moe_gate_dense(out[0], out[1], n_experts)


def _topk_args(inputs):
    return (inputs["gating"], int(inputs["topk"]), bool(inputs["renormalize"]),
            float(inputs["routed_scaling_factor"]))


def _vllm_topk_softmax_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )
    g, k, renorm, _scale = _topk_args(inputs)
    return BenchCase(
        run=lambda: fused_topk(g, g, k, renorm, scoring_func="softmax")[:2],
        to_canonical=lambda o: _topk_dense(o, int(g.shape[-1])))


def _vllm_topk_bias_build(scoring: str):
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )
        g, k, renorm, scale = _topk_args(inputs)
        bias = inputs.get("bias")
        table = inputs.get("table")
        ids = inputs.get("input_ids")
        return BenchCase(
            run=lambda: fused_topk_bias(
                hidden_states=g, gating_output=g, scoring_func=scoring,
                e_score_correction_bias=bias, topk=k, renormalize=renorm,
                input_tokens=ids, hash_indices_table=table,
                routed_scaling_factor=scale)[:2],
            to_canonical=lambda o: _topk_dense(o, int(g.shape[-1])))
    return build


def _sgl_moe_fused_gate_build(scoring: str):
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        g, k, renorm, scale = _topk_args(inputs)
        bias = inputs.get("bias")
        return BenchCase(
            run=lambda: moe_fused_gate(g, bias, k, scoring_func=scoring,
                                       renormalize=renorm,
                                       routed_scaling_factor=scale),
            to_canonical=lambda o: _topk_dense(o, int(g.shape[-1])))
    return build


def _sgl_hash_topk_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.moe.dsv4 import hash_topk
    g, _k, _renorm, scale = _topk_args(inputs)
    ids, table = inputs["input_ids"], inputs["table"]
    return BenchCase(
        run=lambda: hash_topk(router_logits=g, input_ids=ids, tid2eid=table,
                              num_fused_shared_experts=0,
                              routed_scaling_factor=scale,
                              scoring_func="sqrtsoftplus"),
        to_canonical=lambda o: _topk_dense(o, int(g.shape[-1])))


def _torch_router_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The span's naive ops, eager — the oracle's own routine."""
    from ..refs import get_ref
    reference = get_ref(inv.signature).reference
    return BenchCase(run=lambda: reference(inputs), to_canonical=lambda o: o)


def _torch_softmax_router_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The softmax router's naive ops, at the logits' own width.

    The oracle (`refs.moe_topk_router`) scores in fp32: it casts the whole
    `[tokens, n_experts]` logits and takes their softmax, so run as the
    floor it materialised an fp32 copy and an fp32 probability matrix as
    wide as the logits. The serving launch keeps those in registers (vLLM
    `ops.topk_softmax`, csrc/moe/topk_softmax_kernels.cu:79-134 reads the
    logits at their dtype and reduces in fp32 registers; its only fp32
    write is the `[tokens, top-k]` weights, fused_topk_router.py:92-94), so
    the launch gate rightly refused the floor's latency as fp32's: 1,490
    H100 rows of `torch_moe_topk_softmax`, and its canary disabled the
    path (Qwen3.5 / gpt-oss, 2026-10-26).

    Softmax is monotone in the logits, so the experts are the top-k of the
    logits themselves; the weights are computed in fp32 from those k
    logits, and only the unrenormalised form needs the row's normaliser —
    summed in fp32 from bf16 exponentials. Nothing as wide as the logits is
    wider than them. The same spelling `attention._torch_eager_gqa` uses
    for a score matrix."""
    import torch
    g, k, renorm, scale = _topk_args(inputs)

    def run():
        top, ids = torch.topk(g, k, dim=-1)
        top = top.float()
        if renorm:
            weights = torch.softmax(top, dim=-1)
        else:
            peak = g.amax(dim=-1, keepdim=True)
            total = torch.exp(g - peak).sum(dim=-1, keepdim=True,
                                            dtype=torch.float32)
            weights = torch.exp(top - peak.float()) / total
        return weights * scale, ids
    return BenchCase(run=run,
                     to_canonical=lambda o: _topk_dense(o, int(g.shape[-1])))


# (signature, vLLM build, sglang build, torch floor build, the logits
# sglang's launch takes).
for _sig, _vllm, _sgl, _floor, _sgl_logits in (
    ("moe_topk_softmax", _vllm_topk_softmax_build,
     _sgl_moe_fused_gate_build("softmax"), _torch_softmax_router_build,
     TORCH_FLOAT),
    ("moe_topk_sigmoid", _vllm_topk_bias_build("sigmoid"),
     _sgl_moe_fused_gate_build("sigmoid"), _torch_router_build,
     TORCH_FLOAT),
    # sglang does not fuse this span (the binding leaves it): its softmax
    # router takes no bias and LongCat runs the eager path.
    ("moe_topk_softmax_bias", _vllm_topk_bias_build("softmax"), None,
     _torch_router_build, ()),
    ("moe_topk_sqrtsoftplus", _vllm_topk_bias_build("sqrtsoftplus"),
     _sgl_moe_fused_gate_build("sqrtsoftplus"), _torch_router_build,
     TORCH_FLOAT),
    ("moe_topk_hash_sqrtsoftplus", _vllm_topk_bias_build("sqrtsoftplus"),
     # sglang's JIT `hash_topk` takes fp32 logits only (v0.5.22
     # kernels/jit/csrc/deepseek_v4/hash_topk.cuh:116-119).
     _sgl_hash_topk_build, _torch_router_build, ("fp32",)),
):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=f"vllm_{_sig}",
        import_name="vllm",
        serves=KernelSignature(signature=_sig, activation=TORCH_FLOAT),
        build=_vllm, fused_op=_sig,
    ))
    if _sgl is not None:
        register(Provider(
            library="sglang", signature=_sig, kernel_impl=f"sglang_{_sig}",
            import_name="sglang",
            serves=KernelSignature(signature=_sig, activation=_sgl_logits),
            build=_sgl, fused_op=_sig,
        ))
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch",
        serves=KernelSignature(signature=_sig, activation=TORCH_FLOAT),
        build=_floor, fused_op=_sig,
    ))


# ---------------------------------------------------------------------------
# DeepInfer's MegaGate (`moe_mega_gate_sqrtsoftplus`): the router GEMM and
# the sqrt-softplus top-k in one launch (providers/deepinfer.py). The floor
# runs the plan's chain eagerly with the GEMM at the cell's width — the
# `[tokens, n_experts]` logits it scores in fp32 are narrower than the
# `[tokens, d_model]` row it reads.
# ---------------------------------------------------------------------------

def _torch_mega_gate_build(inv: Invocation, env: Env,
                           inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import _route_dense, _sqrtsoftplus_scores
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    w = inputs["weight"].to(dt)
    bias = inputs["bias"].float()
    k, renorm = int(inputs["topk"]), bool(inputs["renormalize"])
    scale = float(inputs["routed_scaling_factor"])

    def run():
        score = _sqrtsoftplus_scores(x @ w.t())
        return _route_dense(score, score + bias, k, renorm, scale)

    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="moe_mega_gate_sqrtsoftplus",
    kernel_impl="torch_moe_mega_gate_sqrtsoftplus", import_name="torch",
    serves=KernelSignature(signature="moe_mega_gate_sqrtsoftplus",
                           activation=TORCH_FLOAT, weight=TORCH_FLOAT),
    build=_torch_mega_gate_build, fused_op="moe_mega_gate_sqrtsoftplus",
))
