"""`/precision/*` — the precision profile vocabulary, its canonical form,
and what a profile changes on one model.

Canonicalisation lives here and in `arch.eval.precision` only: a client
keeps the canonical JSON this returns and sends it back as
`{"canonical": ...}`; it never normalises a profile itself.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from backend.arch.eval.precision import (PrecisionError, PrecisionProfile,
                                         presets, vocabulary)
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment

from .._app import app
from .._models import PrecisionEffectiveRequest
from ..offload import offload_response
from .inference import _resolved_or_404


@app.get("/precision/presets")
async def precision_presets():
    """The named profiles a precision panel fills in with one click."""
    return {"presets": presets()}


@app.get("/precision/vocabulary")
async def precision_vocabulary():
    """Every role, stack, cache slot kind, weight scheme, activation format,
    cache format and layout a rule may name."""
    return vocabulary()


@app.post("/precision/canonicalize")
async def precision_canonicalize(body: Dict[str, Any]):
    """A profile body (`{weights, activations, compute, cache, layouts,
    label}`), canonical: presets expanded, omitted dimensions filled, the
    rule lattice closed, sorted. Refusals come back as `errors[]`, each
    with what to change."""
    try:
        profile = PrecisionProfile.canonicalize(body)
    except PrecisionError as exc:
        return {"canonical": None, "profile_key": None, "label": None,
                "errors": exc.errors}
    return {**profile.to_json(), "errors": []}


@app.post("/precision/effective")
async def precision_effective(req: PrecisionEffectiveRequest):
    """`Analysis.precision_effective()` of `model_id` under `precision`:
    per weight role, edge, matmul role and cache slot kind, the shipped and
    the effective precision, and the rules that select nothing here."""
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    return await offload_response(_effective, req.model_id,
                                  req.precision_profile())


def _effective(model_id: str, profile: PrecisionProfile) -> Dict[str, Any]:
    resolved = _resolved_or_404(model_id)
    analysis = Deployment(Program.of(resolved), precision=profile).at(
        batch=1, input_tokens=1).analyze()
    return {"model_id": model_id, **analysis.precision_effective()}
