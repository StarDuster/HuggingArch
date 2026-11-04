"""Op-review endpoints — wraps `backend.op_workers.review.stream_agent`
plus the post-agent pipeline (verdict parse → rule gate → optional
auto-merge via `backend.arch.promote_op_proposal.run_promote`).

  • POST /review-op   — run reviewer on an existing artifact dir,
                         stream SSE, optionally auto-merge.
  • POST /stop-review-op — kill a running review.
  • GET  /list-op-proposals — triage view: artifacts + review state.

Why split from /propose-op: the reviewer can be invoked standalone for
re-review or debugging. /propose-op chains into the same pipeline
when `chain_review` is True — see `review_pipeline_stream`.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import threading
from contextlib import aclosing
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import Depends
from fastapi.responses import StreamingResponse

from backend.op_workers import review as op_review_worker
from backend.arch.promote_op_proposal import (
    run_promote,
)
from backend.op_workers.review.runtime import PROPOSALS_DIR

from .. import _sse as sse
from .._access import require_admin_if_configured
from .._app import app
from .._models import ReviewOpRequest

# The review lock is released before the post-agent stages (so a rerun of
# /review-op can start), which left promote — graft code, run pytest, archive
# — unguarded: two reviews of one op could promote it at once. One lock per
# op serializes promote itself.
_promote_locks: dict = {}
_promote_locks_guard = threading.Lock()


def _promote_lock(op_name: str) -> threading.Lock:
    with _promote_locks_guard:
        return _promote_locks.setdefault(op_name, threading.Lock())


def _promote_exclusive(op_name: str):
    """run_promote under the op's lock, taken and released in the promoting
    thread itself — so the lock lives exactly as long as the promote, even
    when the SSE stream that asked for it is cancelled. None when another
    promote of the op holds it."""
    lock = _promote_lock(op_name)
    if not lock.acquire(blocking=False):
        return None
    try:
        return run_promote(op_name)
    finally:
        lock.release()


def _sse(data: dict) -> str:
    return f"data: {_json.dumps(data)}\n\n"


async def _stream_review_agent(
    op_name: str,
) -> AsyncGenerator[tuple[str, Optional[dict]], None]:
    """Yield (chunk, parsed_payload) tuples from the reviewer agent.

    parsed_payload is None for heartbeats / non-JSON chunks. Caller is
    responsible for deciding whether to yield each chunk to the client
    (e.g. to suppress the agent's `done` and continue to post-agent
    stages).
    """
    prep_result = await asyncio.to_thread(
        op_review_worker.prepare_review_prompt, op_name
    )
    sync_iter = iter(op_review_worker.stream_agent(op_name, prep_result))
    async with aclosing(sse.pump(sync_iter)) as frames:
        async for chunk in frames:
            yield (chunk, sse.payload(chunk))


async def _post_agent_pipeline(
    op_name: str, auto_merge: bool,
) -> AsyncGenerator[str, None]:
    """Read review.json, run rule gate, optionally call run_promote.

    Emits its own SSE events (logs + final `done`). The caller is
    responsible for ensuring the reviewer agent finished successfully
    before invoking this.
    """
    review_path = Path(PROPOSALS_DIR) / op_name / "review.json"
    if not review_path.is_file():
        yield _sse({"type": "log",
                    "text": "reviewer did not write review.json"})
        yield _sse({"type": "done", "success": False,
                    "message": "missing review.json"})
        return
    try:
        review = _json.loads(review_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as e:
        yield _sse({"type": "log", "text": f"review.json malformed: {e}"})
        yield _sse({"type": "done", "success": False,
                    "message": "malformed review.json"})
        return

    verdict = review.get("verdict", "")
    red_flags = review.get("red_flags") or []
    yield _sse({"type": "log",
                "text": f"Reviewer verdict: {verdict or '<missing>'}"})
    if red_flags:
        yield _sse({"type": "log",
                    "text": f"Red flags: {', '.join(red_flags)}"})

    if verdict != "auto_merge":
        yield _sse({"type": "done", "success": True,
                    "message": verdict or "no_verdict"})
        return

    if not auto_merge:
        yield _sse({"type": "log",
                    "text": "auto_merge disabled by request; stopping"})
        yield _sse({"type": "done", "success": True,
                    "message": "auto_merge_disabled"})
        return

    eligible, reasons = op_review_worker.check_auto_merge_eligibility(
        Path(PROPOSALS_DIR) / op_name
    )
    if not eligible:
        joined = "; ".join(reasons)
        yield _sse({"type": "log",
                    "text": f"Auto-merge rules: fail ({joined})"})
        yield _sse({"type": "done", "success": True,
                    "message": f"rules_blocked: {joined}"})
        return

    yield _sse({"type": "log", "text": "Auto-merge rules: pass"})
    yield _sse({"type": "log",
                "text": "[promote] grafting + running pytest..."})

    result = await asyncio.to_thread(_promote_exclusive, op_name)
    if result is None:
        yield _sse({"type": "log",
                    "text": "[promote] another promote of this op is running"})
        yield _sse({"type": "done", "success": False,
                    "message": "promote_in_progress"})
        return

    if not result.ok:
        yield _sse({"type": "log",
                    "text": f"[promote] FAILED at {result.stage}: {result.error}"})
        yield _sse({"type": "done", "success": False,
                    "message": f"promote_failed: {result.error}"})
        return

    if result.stage == "test":
        yield _sse({"type": "log",
                    "text": "[pytest] failed; grafts reverted"})
        yield _sse({"type": "done", "success": True,
                    "message": "pytest_failed_reverted"})
        return

    yield _sse({"type": "log", "text": f"[archive] {result.message}"})
    yield _sse({"type": "done", "success": True, "message": "auto_merged"})


async def review_pipeline_stream(
    op_name: str, *, auto_merge: bool,
) -> AsyncGenerator[str, None]:
    """Reviewer agent SSE + post-agent pipeline. Used by /review-op
    AND by /propose-op when chain_review=True.

    Acquires its own review-side lock. The proposer's lock is a
    SEPARATE lock (op_workers.proposal maintains it); callers from
    chain-review should have already released that one before invoking
    this generator.
    """
    status = op_review_worker.acquire(op_name)
    if status == "overloaded":
        yield _sse({"type": "log",
                    "text": "Review queue full; skipping review stage"})
        yield _sse({"type": "done", "success": True,
                    "message": "review_overloaded"})
        return
    if status == "duplicate":
        yield _sse({"type": "log",
                    "text": "Another review already in flight for this op"})
        yield _sse({"type": "done", "success": True,
                    "message": "review_duplicate"})
        return

    agent_succeeded = False
    try:
        async for chunk, payload in _stream_review_agent(op_name):
            if payload and payload.get("type") == "done":
                if payload.get("success", False):
                    agent_succeeded = True
                    yield _sse({"type": "log",
                                "text": "Reviewer agent done; reading verdict..."})
                    continue
                yield chunk
                return
            yield chunk
    finally:
        # Release review-side lock before post-agent stages so a parallel
        # rerun of /review-op for THIS op can proceed if needed (the
        # post-agent stages run synchronously inside this generator, no
        # subprocess to clean up).
        op_review_worker.release(op_name)

    if not agent_succeeded:
        yield _sse({"type": "done", "success": False,
                    "message": "reviewer agent ended without success"})
        return

    async for chunk in _post_agent_pipeline(op_name, auto_merge):
        yield chunk


@app.post(
    "/review-op",
    dependencies=[Depends(require_admin_if_configured)],
)
async def review_op(req: ReviewOpRequest):
    """Run the reviewer worker against an existing artifact directory."""
    try:
        agent_binary = op_review_worker.get_agent_binary()
    except ValueError as e:
        return {"success": False, "message": f"Invalid coding agent: {e}"}
    import shutil
    if not shutil.which(agent_binary):
        return {"success": False, "message": "Coding agent CLI not installed"}

    if not req.op_name or "/" in req.op_name or req.op_name.startswith("."):
        return {"success": False, "message": "op_name must be a simple identifier"}

    artifact_dir = Path(PROPOSALS_DIR) / req.op_name
    if not artifact_dir.is_dir():
        return {
            "success": False,
            "message": f"no artifact dir at {artifact_dir.relative_to(Path(PROPOSALS_DIR).parent.parent.parent)}",
        }

    return StreamingResponse(
        review_pipeline_stream(req.op_name, auto_merge=req.auto_merge),
        media_type="text/event-stream",
        headers=sse.headers(),
    )


@app.post(
    "/stop-review-op",
    dependencies=[Depends(require_admin_if_configured)],
)
async def stop_review_op(req: dict):
    op_name = req.get("op_name", "")
    if not op_name:
        return {"success": False, "message": "op_name required"}
    killed = op_review_worker.stop(op_name)
    return {
        "success": killed,
        "message": "Stopped" if killed else "No running review found",
    }


@app.get(
    "/list-op-proposals",
    dependencies=[Depends(require_admin_if_configured)],
)
async def list_op_proposals():
    """List artifact directories under PROPOSALS_DIR with their review
    state. Drives a triage queue (proposed → reviewed → promoted)."""
    items: list[dict] = []
    if not os.path.isdir(PROPOSALS_DIR):
        return {"proposals": items}

    for name in sorted(os.listdir(PROPOSALS_DIR)):
        if name.startswith(("_", ".")):
            continue  # _promoted is the archive, skip
        path = Path(PROPOSALS_DIR) / name
        if not path.is_dir():
            continue

        review_path = path / "review.json"
        verdict = None
        red_flag_count = 0
        if review_path.is_file():
            try:
                review = _json.loads(review_path.read_text(encoding="utf-8"))
                verdict = review.get("verdict")
                red_flag_count = len(review.get("red_flags") or [])
            except _json.JSONDecodeError:
                verdict = "<malformed>"

        items.append({
            "op_name": name,
            "review_state": (
                "reviewed" if verdict else "awaiting_review"
            ),
            "verdict": verdict,
            "red_flag_count": red_flag_count,
        })

    # Surface promoted ops too so the UI can show "this one already shipped"
    promoted_dir = Path(PROPOSALS_DIR) / "_promoted"
    if promoted_dir.is_dir():
        for name in sorted(os.listdir(promoted_dir)):
            p = promoted_dir / name
            if not p.is_dir():
                continue
            items.append({
                "op_name": name,
                "review_state": "promoted",
                "verdict": "auto_merge",
                "red_flag_count": 0,
            })

    return {"proposals": items, "checked_at": datetime.utcnow().isoformat() + "Z"}


__all__ = ["review_pipeline_stream"]
