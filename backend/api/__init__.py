"""FastAPI server for HuggingArch.

Public surface (consumed by uvicorn + tests + custom_model):
  • `app` — the FastAPI application instance.

Internal layout:
  _app      app instance + middleware + static-file mounts
  _models   Pydantic request/response models + MODALITY_REGISTRY
  _helpers  forward-source payload + supported-model list loader
  routes/   per-domain APIRouter modules

The opt-in `backend.service` package is installed last; it registers the
auth + admin + analytics endpoints (Supabase-backed) and mounts the
service-only static dirs. Drop that import — and the directory — to
ship a no-service build.
"""
from ._app import app

# Side-effect registration in the OSS core. The optional service layer below
# adds routes but must not replace (or duplicate) this miss hook.
from backend.arch.loader import set_spec_miss_hook
from backend.spec_worker.shared_spec_miss import materialize_cached_spec

set_spec_miss_hook(materialize_cached_spec)

# Side-effect import: each route module registers its endpoints against `app`.
from . import routes  # noqa: F401

# Opt-in service layer (auth + admin + analytics). Endpoints degrade
# gracefully when SUPABASE_* env vars are unset, so this import is
# safe in a single-user / no-supabase deployment.
from backend.service import ha_api  # noqa: F401

__all__ = ["app"]
