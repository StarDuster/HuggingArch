"""Account module-variant routes — "my modules".

A variant is a named, user-saved parameterization of a library component:
the Playground saves an edited column (`component` + the column's resolved
own dims, tuned under `base_model_id`) here, and later uses it as a compare
entry — dims come from this row, never guessed. Login required; every query
pins the caller’s user_id (service role bypasses RLS, so ownership is
enforced in code, mirroring the other account routes).
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.api._app import app
from backend.playground.module_bench import _is_contract_dim, list_library_components
from backend.service import auth as _auth
from backend.service import module_variants as _mv


class ModuleVariantCreate(BaseModel):
    component: str
    name: str = Field(min_length=1, max_length=80)
    dims: Dict[str, int]
    base_model_id: Optional[str] = None

    @field_validator("component")
    @classmethod
    def _known_component(cls, v: str) -> str:
        names = {c["name"] for c in list_library_components()}
        if v not in names:
            raise ValueError(f"unknown library component: {v!r}")
        return v

    @field_validator("dims")
    @classmethod
    def _sane_dims(cls, v: Dict[str, int]) -> Dict[str, int]:
        if not v or len(v) > 64:
            raise ValueError("dims must be 1..64 entries")
        for k, n in v.items():
            if _is_contract_dim(k):
                raise ValueError(f"{k!r} is a shared-contract dim, never a variant dim")
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise ValueError(f"dims[{k!r}] must be a positive int")
        return v


class ModuleVariantPatch(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.get("/api/account/modules")
def list_module_variants(component: Optional[str] = None,
                         user: dict = Depends(_auth.require_user)):
    rows = _mv.list_for_user(_auth.uid(user), component=component)
    if rows is None:
        raise HTTPException(status_code=502, detail="could not load the variants")
    return {"success": True, "variants": rows}


@app.post("/api/account/modules")
def create_module_variant(body: ModuleVariantCreate,
                          user: dict = Depends(_auth.require_user)):
    row = _mv.create(_auth.uid(user), component=body.component, name=body.name,
                     dims=dict(body.dims), base_model_id=body.base_model_id)
    if row is None:
        raise HTTPException(status_code=502, detail="could not save the variant")
    return {"success": True, "variant": row}


@app.patch("/api/account/modules/{variant_id}")
def update_module_variant(variant_id: str, body: ModuleVariantPatch,
                          user: dict = Depends(_auth.require_user)):
    import uuid as _uuid
    try:
        _uuid.UUID(variant_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="variant_id must be a UUID") from None
    try:
        updated = _mv.update(_auth.uid(user), variant_id, name=body.name)
    except _mv._NameConflict:
        raise HTTPException(status_code=409, detail="a variant with that name already exists") from None
    if updated is None:
        raise HTTPException(status_code=502, detail="could not update the variant")
    if not updated:
        raise HTTPException(status_code=404, detail="variant not found")
    return {"success": True}


@app.delete("/api/account/modules/{variant_id}")
def delete_module_variant(variant_id: str, user: dict = Depends(_auth.require_user)):
    import uuid as _uuid
    try:
        _uuid.UUID(variant_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="variant_id must be a UUID") from None
    deleted = _mv.delete(_auth.uid(user), variant_id)
    if deleted is None:
        raise HTTPException(status_code=502, detail="could not delete the variant")
    if not deleted:
        raise HTTPException(status_code=404, detail="variant not found")
    return {"success": True}
