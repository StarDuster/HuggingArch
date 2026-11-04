"""The deployment a model opens at on a card: what the Inference page seeds
its form with before the first estimate.

A dense model opens on one GPU. A MoE model opens at DP = EP, starting from
the card's fabric unit (`defaults.moe_dp_ep_degree`: an SXM node, a GB200
tray) and doubling through the degrees the layout accepts
(`parallel_sweep.legal_parallel_degrees`) until its weights fit
(`_CapacityPreparation.fits`, the estimate's own answer). When no degree up
to the sweep's cap fits, it opens at the first accepted one and the
estimate says why it does not run.

Asked of the backend, before an estimate, so the page never shows one
deployment and then another: the seed used to be a per-card table the page
applied after an estimate had told it the model was MoE, and that table
knew nothing of whether the model fit (Kimi-K3 opened at 8 B200s, which
hold less than its weights).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .capacity import CapacityKnobs, _capacity_preparation
from .defaults import moe_dp_ep_degree
from .parallel_config import ParallelConfig
from .parallel_sweep import legal_parallel_degrees, sweep_caps

#: The request's own axes `resolve_estimate_environment` carries beside the
#: capacity knobs; the opening deployment reads none of them but the config
#: and the card.
_ENVELOPE = frozenset({"config", "gpu_sku", "avg_output_tokens",
                       "prefill_batch_size", "decode_batch_size",
                       "prefill_chunk_size"})


def default_deployment(env: Mapping[str, Any], resolved: Any) -> Dict[str, Any]:
    """`{parallel, fits, moe, start_degree}` for the model (`resolved`, its
    matched spec) and card of an estimate environment
    (`resolve_estimate_environment`); the environment's own `parallel` is
    ignored — this answers what it should be."""
    config, gpu_sku = env["config"], env["gpu_sku"]
    caps = sweep_caps(config, resolved=resolved)
    start = moe_dp_ep_degree(gpu_sku)
    if not caps.moe_enabled:
        return {"parallel": {}, "moe": False, "start_degree": 1,
                "fits": _fits(env, ParallelConfig())}
    accepted = [a for a in legal_parallel_degrees(resolved, "dp_ep", caps)["accepted"]
                if a["x"] >= start]
    if not accepted:
        raise ValueError(f"no DP-EP degree of at least {start} is a layout "
                         f"of this model")
    for a in accepted:
        parallel = ParallelConfig.from_dict(a["parallel"])
        if _fits(env, parallel):
            return {"parallel": a["parallel"], "moe": True,
                    "start_degree": start, "fits": True}
    return {"parallel": accepted[0]["parallel"], "moe": True,
            "start_degree": start, "fits": False}


def fit_identity(env: Mapping[str, Any]) -> Dict[str, Any]:
    """What the answer depends on beside the model: the card and every
    capacity knob but the deployment itself (precision, HBM fraction, draft,
    aux weights, offload … all move what a rank must hold)."""
    knobs = CapacityKnobs.from_kwargs(
        {**env, "parallel": ParallelConfig(), "num_gpus": None},
        envelope=_ENVELOPE).as_flat()
    knobs.pop("parallel")
    knobs.pop("num_gpus")
    return {"gpu_sku": env["gpu_sku"], **knobs}


def _fits(env: Mapping[str, Any], parallel: ParallelConfig) -> bool:
    """Whether a rank of `parallel` holds the model's weights and its cache
    floor on the card — the preparation's answer, as the estimate gives it."""
    knobs = CapacityKnobs.from_kwargs(
        {**env, "parallel": parallel, "num_gpus": None}, envelope=_ENVELOPE)
    return bool(_capacity_preparation(
        config=env["config"], gpu_sku=env["gpu_sku"], knobs=knobs).fits)
