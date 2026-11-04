"""Op-proposal endpoints — wraps `backend.op_workers.proposal.stream_agent`.

  • POST /propose-op   — kick off an op-proposal run for a gap statement.
                         Streams SSE.
  • POST /stop-propose-op — kill a running op-proposal worker.
  • GET  /list-op-gaps — list .gap.json files dropped by spec workers.

Why a separate route module (not piggyback on /generate-spec): the
concurrency key is op_name, not model_id; the prep function takes a
gap path; the artifact target is a different directory tree. Sharing
one route would force a discriminator union and make the failure modes
muddier.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import shutil
from contextlib import aclosing

from fastapi import Depends
from fastapi.responses import StreamingResponse

from backend.op_workers import proposal as op_proposal_worker
from backend.op_workers.proposal.runtime import GAP_DIR

from .. import _sse as sse
from .._access import require_admin_if_configured
from .._app import app
from .._models import ProposeOpRequest


async def stream_with_prep(op_name: str, prep_fn, *, chain_review: bool):
    """Async SSE wrapper around prep + `op_proposal_worker.stream_agent`.
    Emits a heartbeat comment every 15 s of agent silence and releases
    the per-op lock when the stream ends or the client disconnects.

    When `chain_review` is True AND the proposer finishes with success,
    the generator suppresses the proposer's `done` event and continues
    into `review_pipeline_stream` so the client sees one continuous
    SSE stream: prep → propose → review → (optional auto-merge) → done.
    """
    proposer_succeeded = False
    try:
        yield f"data: {_json.dumps({'type': 'log', 'text': 'Starting op proposal...'})}\n\n"
        try:
            prep_result = await asyncio.to_thread(prep_fn)
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'message': f'Prep failed: {e}'})}\n\n"
            return

        sync_iter = iter(op_proposal_worker.stream_agent(op_name, prep_result))
        async with aclosing(sse.pump(sync_iter)) as frames:
            async for chunk in frames:
                # Inspect for the proposer's terminal `done` event when chaining.
                # If success, suppress so review can emit the real terminal done;
                # if failure, pass through and stop.
                if chain_review and isinstance(chunk, str):
                    payload = sse.payload(chunk)
                    if payload and payload.get("type") == "done":
                        if payload.get("success", False):
                            proposer_succeeded = True
                            yield f"data: {_json.dumps({'type': 'log', 'text': 'Proposer done; chaining into reviewer...'})}\n\n"
                            continue
                        yield chunk
                        return
                yield chunk
    finally:
        op_proposal_worker.release(op_name)

    if chain_review and proposer_succeeded:
        # Lazy import to avoid a routes-package import cycle (this module
        # is imported by routes/__init__.py before op_reviews is).
        from .op_reviews import review_pipeline_stream
        async for chunk in review_pipeline_stream(op_name, auto_merge=True):
            yield chunk


def _resolve_gap_path(gap_path: str) -> str:
    """Resolve repo-relative or absolute gap path; reject anything
    outside the gap directory. The op proposal worker reads this file
    and prompts the agent with its contents, so an attacker-controlled
    path would let an arbitrary file leak into the prompt."""
    if os.path.isabs(gap_path):
        resolved = os.path.realpath(gap_path)
    else:
        repo_root = os.path.dirname(GAP_DIR.rstrip("/"))  # logs/op_proposals → logs
        repo_root = os.path.dirname(repo_root)            # logs → repo root
        resolved = os.path.realpath(os.path.join(repo_root, gap_path))
    gap_dir_real = os.path.realpath(GAP_DIR)
    if not resolved.startswith(gap_dir_real + os.sep):
        raise ValueError(
            f"gap_path must live under {gap_dir_real}, got {resolved}"
        )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"gap file not found: {resolved}")
    return resolved


@app.get(
    "/list-op-gaps",
    dependencies=[Depends(require_admin_if_configured)],
)
async def list_op_gaps():
    """List `.gap.json` files dropped by the spec worker that don't yet
    have a corresponding proposal artifact directory.

    Useful when triaging which gaps to spawn proposal workers on.
    """
    from backend.op_workers.proposal.runtime import PROPOSALS_DIR

    gaps = []
    if os.path.isdir(GAP_DIR):
        for fname in sorted(os.listdir(GAP_DIR)):
            if not fname.endswith(".gap.json"):
                continue
            path = os.path.join(GAP_DIR, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    body = _json.load(f)
            except (OSError, _json.JSONDecodeError):
                continue
            op_name = body.get("op_name") or fname.removesuffix(".gap.json")
            artifact_dir = os.path.join(PROPOSALS_DIR, op_name)
            gaps.append({
                "op_name": op_name,
                "gap_path": os.path.relpath(
                    path,
                    os.path.dirname(os.path.dirname(GAP_DIR)),  # repo root
                ),
                "context_model_id": body.get("context_model_id", ""),
                "has_proposal": os.path.isdir(artifact_dir),
            })
    return {"gaps": gaps}


@app.post(
    "/propose-op",
    dependencies=[Depends(require_admin_if_configured)],
)
async def propose_op(req: ProposeOpRequest):
    """Trigger the configured agent CLI to author an op proposal."""
    try:
        agent_binary = op_proposal_worker.get_agent_binary()
    except ValueError as e:
        return {"success": False, "message": f"Invalid coding agent: {e}"}
    if not shutil.which(agent_binary):
        return {"success": False, "message": "Coding agent CLI not installed"}

    if not req.op_name or "/" in req.op_name or req.op_name.startswith("."):
        return {"success": False, "message": "op_name must be a simple identifier"}

    # Resolve gap input: either a path on disk, or an inline dict.
    if req.gap is not None:
        gap = req.gap
    elif req.gap_path:
        try:
            resolved = _resolve_gap_path(req.gap_path)
        except (ValueError, FileNotFoundError) as e:
            return {"success": False, "message": str(e)}
        try:
            with open(resolved, encoding="utf-8") as f:
                gap = _json.load(f)
        except (OSError, _json.JSONDecodeError) as e:
            return {"success": False, "message": f"failed to read gap: {e}"}
    else:
        return {
            "success": False,
            "message": "either gap or gap_path must be provided",
        }

    status = op_proposal_worker.acquire(req.op_name)
    if status == "duplicate":
        return {
            "success": False, "status": "duplicate",
            "message": f"Already proposing op {req.op_name}",
        }
    if status == "overloaded":
        return {
            "success": False, "status": "overloaded",
            "message": "Too many concurrent agent tasks, please retry later",
        }

    op_name = req.op_name
    prep_fn = lambda: prep_with_gap(op_name, gap)
    return StreamingResponse(
        stream_with_prep(op_name, prep_fn, chain_review=req.chain_review),
        media_type="text/event-stream",
        headers=sse.headers(),
    )


def prep_with_gap(op_name: str, gap: dict) -> dict:
    """Build prompt + prefetch logs given the resolved gap dict."""
    from backend.op_workers.proposal._prompts.tasks import prep_from_gap
    return prep_from_gap(op_name, gap)


@app.post(
    "/stop-propose-op",
    dependencies=[Depends(require_admin_if_configured)],
)
async def stop_propose_op(req: dict):
    """Stop a running op-proposal worker."""
    op_name = req.get("op_name", "")
    if not op_name:
        return {"success": False, "message": "op_name required"}
    killed = op_proposal_worker.stop(op_name)
    return {
        "success": killed,
        "message": "Stopped" if killed else "No running task found",
    }
