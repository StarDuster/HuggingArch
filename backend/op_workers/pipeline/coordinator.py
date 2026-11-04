"""Background watcher + pipeline driver.

The actual propose → review → auto-merge code lives in the route layer
(`backend/api/routes/op_proposals.py::stream_with_prep`); the coordinator
just drains the same async generator into a trace log instead of an SSE
client. Reusing the route generator means there's a single code path for
"propose then review" — no risk of a divergent second implementation
silently drifting from what `POST /propose-op?chain_review=true` does.

Polling vs inotify: the gap directory is low-frequency (one file per spec
worker run that surfaces a missing primitive). Polling every
`COORDINATOR_POLL_INTERVAL` seconds adds no dependency and no Linux-only
edge cases. Latency cost is negligible next to a multi-minute pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from datetime import datetime
from typing import Optional

from backend.op_workers import proposal as op_proposal_worker
from backend.op_workers.proposal.runtime import GAP_DIR, PROPOSALS_DIR


logger = logging.getLogger(__name__)


POLL_INTERVAL = float(os.environ.get("COORDINATOR_POLL_INTERVAL", "10"))
DRAIN_TIMEOUT = float(os.environ.get("COORDINATOR_DRAIN_TIMEOUT", "60"))


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
TRACE_DIR = os.path.join(_PROJECT_ROOT, "logs", "op_pipeline")
os.makedirs(TRACE_DIR, exist_ok=True)


_watcher_task: Optional[asyncio.Task] = None
_inflight: set[asyncio.Task] = set()
# In-memory dedupe within one process — set is cleared on reload, which
# is the right behavior: if a previous pipeline crashed before creating
# the artifact dir, we want a reload-triggered retry. If it succeeded as
# far as writing the artifact dir, `_gap_already_processed` short-circuits
# regardless of whether this set remembers it.
_seen_gaps: set[str] = set()


def _optional_env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("coordinator: ignoring invalid %s=%r", name, value)
    return None


def pipeline_enabled() -> bool:
    """Resolve autonomous-pipeline policy without importing service auth.

    Explicit ``OP_PIPELINE_ENABLED`` wins. The legacy
    ``OP_PIPELINE_DISABLED`` flag remains supported. With neither set, the
    pipeline is on, in local self-host and Supabase-backed service alike.
    """
    enabled = _optional_env_bool("OP_PIPELINE_ENABLED")
    if enabled is not None:
        return enabled
    disabled = _optional_env_bool("OP_PIPELINE_DISABLED")
    if disabled is not None:
        return not disabled
    return True


def _gap_already_processed(op_name: str) -> bool:
    """Artifact dir exists ⇒ a previous run got at least as far as the
    proposer writing files. Don't re-enter the pipeline; rerun manually
    via `POST /review-op` if review or promote needs another pass."""
    return os.path.isdir(os.path.join(PROPOSALS_DIR, op_name))


def _scan_gaps() -> list[tuple[str, str, dict]]:
    """Return `(op_name, gap_path, gap_body)` for unprocessed gaps."""
    out: list[tuple[str, str, dict]] = []
    if not os.path.isdir(GAP_DIR):
        return out
    for fname in sorted(os.listdir(GAP_DIR)):
        if not fname.endswith(".gap.json"):
            continue
        gap_path = os.path.join(GAP_DIR, fname)
        if gap_path in _seen_gaps:
            continue
        try:
            with open(gap_path, encoding="utf-8") as f:
                body = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("coordinator: cannot read %s: %s", gap_path, e)
            _seen_gaps.add(gap_path)
            continue
        op_name = body.get("op_name") or fname.removesuffix(".gap.json")
        if _gap_already_processed(op_name):
            _seen_gaps.add(gap_path)
            continue
        out.append((op_name, gap_path, body))
    return out


async def _run_pipeline(op_name: str, gap_path: str, gap: dict) -> None:
    """Drive one op end-to-end. SSE chunks land in the per-run trace log;
    no client is listening but the trace preserves the same audit info a
    `POST /propose-op` caller would have seen."""
    # Lazy import — keeps the routes package out of module import time
    # and mirrors the same dodge used in op_proposals.py for op_reviews.
    from backend.api.routes.op_proposals import prep_with_gap, stream_with_prep

    status = op_proposal_worker.acquire(op_name)
    if status != "ok":
        logger.info(
            "coordinator: skip op=%s (proposer acquire=%s — manual run "
            "or duplicate already in flight)",
            op_name, status,
        )
        return

    # acquire() succeeded — `stream_with_prep`'s finally is responsible
    # for releasing the proposer lock once we start draining it.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = os.path.join(TRACE_DIR, f"{op_name}__{ts}.log")
    logger.info(
        "coordinator: pipeline op=%s gap=%s trace=%s",
        op_name, os.path.relpath(gap_path, _PROJECT_ROOT),
        os.path.relpath(trace_path, _PROJECT_ROOT),
    )

    def _prep() -> dict:
        return prep_with_gap(op_name, gap)

    # Ownership of the proposer-lock release: the coordinator holds it from
    # acquire() until stream_with_prep's generator starts; from the first
    # `async for` iteration onward stream_with_prep's own finally owns it (and
    # will release even on the chain_review path + on cancel/crash via aclose).
    # So the coordinator must release ONLY in the pre-handoff window — trace-file
    # open()/header-write failing after acquire but before streaming begins —
    # otherwise the op stays 'duplicate' forever + a semaphore permit leaks.
    # It must NOT release unconditionally: stream_with_prep releases op_name
    # mid-run (before the review chain), so a late unconditional release could
    # return a slot a DIFFERENT run re-acquired under the same op_name key.
    handed_off = False
    try:
        with open(trace_path, "w", encoding="utf-8") as trace:
            trace.write(
                f"=== coordinator pipeline op={op_name} @ {ts} ===\n"
                f"gap: {gap_path}\n\n"
            )
            handed_off = True  # no await between here and the first __anext__
            async for chunk in stream_with_prep(
                op_name, _prep, chain_review=True
            ):
                trace.write(chunk)
                trace.flush()
        logger.info("coordinator: pipeline op=%s finished", op_name)
    except asyncio.CancelledError:
        logger.warning("coordinator: pipeline op=%s cancelled mid-run", op_name)
        raise
    except Exception:
        logger.exception("coordinator: pipeline op=%s crashed", op_name)
    finally:
        if not handed_off:
            op_proposal_worker.release(op_name)


async def _watcher_loop() -> None:
    logger.info(
        "coordinator: watcher up (poll=%ss, drain=%ss, gap_dir=%s)",
        POLL_INTERVAL, DRAIN_TIMEOUT, GAP_DIR,
    )
    while True:
        try:
            gaps = await asyncio.to_thread(_scan_gaps)
            for op_name, gap_path, body in gaps:
                _seen_gaps.add(gap_path)
                task = asyncio.create_task(
                    _run_pipeline(op_name, gap_path, body),
                    name=f"op_pipeline:{op_name}",
                )
                _inflight.add(task)
                task.add_done_callback(_inflight.discard)
        except Exception:
            logger.exception("coordinator: scan failed (will retry)")
        await asyncio.sleep(POLL_INTERVAL)


async def startup() -> None:
    """FastAPI startup hook — registered from `backend/api/_app.py`."""
    global _watcher_task
    if not pipeline_enabled():
        logger.info(
            "coordinator: disabled (explicit pipeline flag)"
        )
        return
    if _watcher_task is not None:
        return
    _watcher_task = asyncio.create_task(
        _watcher_loop(), name="op_pipeline:watcher",
    )


async def shutdown() -> None:
    """FastAPI shutdown hook. Cancel the watcher so no new pipelines start,
    then give in-flight pipelines `DRAIN_TIMEOUT` seconds to finish before
    cancelling them. The agent subprocesses themselves are unmanaged after
    cancel — they'll exit on their own or be reaped by `release()`."""
    global _watcher_task
    if _watcher_task is not None:
        _watcher_task.cancel()
        try:
            await _watcher_task
        except asyncio.CancelledError:
            pass
        _watcher_task = None

    if not _inflight:
        return
    logger.info(
        "coordinator: draining %d in-flight pipeline(s), timeout=%ss",
        len(_inflight), DRAIN_TIMEOUT,
    )
    _, pending = await asyncio.wait(_inflight, timeout=DRAIN_TIMEOUT)
    if pending:
        logger.warning(
            "coordinator: %d pipeline(s) still running after drain; "
            "cancelling (agent subprocess may keep running until done)",
            len(pending),
        )
        for t in pending:
            t.cancel()
