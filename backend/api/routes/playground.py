"""Playground routes.

Semantics B (isolated module comparison) — public, no auth, same as `/analyze`.
The heavy eval runs in a compute worker (`backend.api.offload`). Contract errors
(a base model with no spec, an unbound dim) come back as `{"success": False,
"message": ...}`; per-component gaps are already folded into the payload's
`errors[]` by `module_bench`.
"""

from __future__ import annotations

from typing import Optional

from .._app import app
from ..offload import call_key, ensure_spec_generation, map_off_loop, offload, spec_generation, within_deadline
from .._models import ModuleBenchRequest, ComposeModelRequest, ModuleLatencySweepRequest
from backend.playground import latency_sweep
from backend.playground.module_bench import (
    component_bindings,
    component_dims_from_model,
    list_library_components,
    module_bench,
)
from backend.playground.compose import (
    compose_groups,
    model_cache_at,
    model_skeleton,
)


@app.get("/api/playground/components")
async def playground_components():
    """Library components grouped by role, for the module picker."""
    return {"success": True,
            "components": await offload(list_library_components)}


@app.get("/api/playground/model-skeleton")
async def playground_model_skeleton(model_id: str):
    """The model editor skeleton: every layer group with its layers, slots
    and candidates, and the dims its evaluation depends on."""
    try:
        skel = await offload(model_skeleton, model_id)
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}
    return {"success": True, **skel}


@app.post("/api/playground/compose-model")
async def playground_compose_model(req: ComposeModelRequest):
    """Whole-model cost of a base with per-group edits (`groups`) —
    canonical, byte-for-byte the same as opening the model in the Inspector."""
    try:
        result = await offload(
            compose_groups,
            base_model_id=req.base_model_id,
            groups=[g.model_dump() for g in req.groups],
            dim_overrides=dict(req.dim_overrides),
            batch=req.batch_size,
            input_tokens=req.seq_len,
            precision=req.precision_profile(),
            kv_cache_lens=list(req.kv_cache_lens),
            gpu_sku=req.gpu_sku,
        )
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}
    return result


@app.post("/api/playground/module-bench")
async def playground_module_bench(req: ModuleBenchRequest):
    try:
        result = await offload(
            module_bench,
            base_model_id=req.base_model_id,
            components=list(req.components),
            batch=req.batch_size,
            input_tokens=req.seq_len,
            d_model=req.d_model,
            dim_overrides=dict(req.dim_overrides),
            per_component_dims={k: dict(v) for k, v in req.per_component_dims.items()},
            entries=[{"name": e.name, "model_id": e.model_id,
                      "call": e.call,
                      "dims": dict(e.dims), "seeded": e.seeded,
                      "label": e.label}
                     for e in req.entries],
            precision=req.precision_profile(),
            kv_cache_lens=list(req.kv_cache_lens),
            gpu_sku=req.gpu_sku,
            model_cache=model_cache_at,
        )
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}
    return {"success": True, **result}


@app.post("/api/playground/module-latency-sweep")
async def playground_module_latency_sweep(req: ModuleLatencySweepRequest):
    """Each compare entry's prefill and decode time at every length of
    `lens` — the bench's own timing, one compute job per (entry, phase)."""
    overrides = dict(req.dim_overrides)
    if req.d_model is not None:
        overrides.setdefault("d_model", req.d_model)
    sweep = latency_sweep.LatencySweep(
        entries=tuple(
            [{"name": e.name, "model_id": e.model_id, "call": e.call,
              "dims": dict(e.dims), "seeded": e.seeded, "label": e.label}
             for e in req.entries]
            or [{"name": n, "model_id": None,
                 "dims": dict(req.per_component_dims.get(n) or {})}
                for n in req.components]),
        batch=req.batch_size,
        lens=tuple(req.lens),
        gpu_sku=req.gpu_sku,
        precision=req.precision_profile(),
        base_model_id=req.base_model_id,
        dim_overrides=tuple(sorted(overrides.items())),
    )

    async def run():
        generation = spec_generation()
        results = await map_off_loop(latency_sweep.cell, sweep.cells())
        ensure_spec_generation(generation)
        return sweep.assemble(results)

    try:
        payload = await within_deadline(
            "latency:" + call_key(playground_module_latency_sweep, (sweep,), {}), run)
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}
    return {"success": True, **payload}


@app.get("/api/playground/component-dims")
async def playground_component_dims(component: str, model_id: Optional[str] = None,
                                    call: Optional[str] = None):
    """With `model_id`: seed one compare entry — the internal dims `component`
    uses as shipped model `model_id` parameterizes it at its call `call`
    (needed when the model calls it more than once). Without: the full
    bindings table — every call of `component` in every model that binds it
    and the dims each resolves to (the single-pick view's parameterizations)."""
    if model_id is None:
        try:
            table = await offload(component_bindings, component)
        except (ValueError, KeyError) as exc:
            return {"success": False, "message": str(exc)}
        return {"success": True, "component": component, **table}
    try:
        dims = await offload(component_dims_from_model, component,
                                       model_id, call)
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}
    return {"success": True, "component": component, "model_id": model_id,
            "call": call, "dims": dims}
