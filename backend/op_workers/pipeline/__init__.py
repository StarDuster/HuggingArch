"""Op-pipeline coordinator — autonomous propose → review → auto-merge.

Watches `logs/op_proposals/` for gap statements the spec_worker drops and
drives them through the same pipeline that `POST /propose-op` with
`chain_review=True` triggers, but without a human in the loop.

Runtime form: registered as a FastAPI startup/shutdown hook on the shared
`app` (see `backend/api/_app.py`). Lives and dies with the API process —
no orphan worker, but a uvicorn reload mid-pipeline could cut a promote.
Mitigation: shutdown drains in-flight pipelines up to
`COORDINATOR_DRAIN_TIMEOUT` seconds before letting reload proceed.

The watcher is on by default, in local self-host and Supabase-backed service
mode alike; `OP_PIPELINE_ENABLED=0` turns it off. `OP_PIPELINE_DISABLED=1`
remains supported for compatibility.

Spec-gen chains propose→review in the same job (`chain.run_propose_review`)
when the agent writes a gap.json. Ready envelopes continue spec-gen with
`HA_PENDING_OP_PATCH` in the agent subprocess only. The generation record
stores the proposal package (not spliced `ops_registry.py`); opening the PR
splices into canonical files. New dim symbols are in-scope when
the proposal's `dims.yaml` fragment declares them. The generate job never
opens a PR and never writes the serving checkout.
"""
from .coordinator import pipeline_enabled, shutdown, startup

__all__ = ["pipeline_enabled", "shutdown", "startup"]
