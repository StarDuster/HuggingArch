"""Starting an agent job on a stored generation record: review, and fix.

The orchestration of a record job — refuse it on the record's status, take
the run lock, build the run generator (prep → stream, with the record's
identity stamped onto `prep_result`), hand it to `start_job` — has nothing
in it that needs a signed-in user or an HTTP request. It used to live in
`backend/service/routes/spec.py`, which made these jobs reachable only from
the service route layer: an offline run, whose record the host-local backend
keeps (`records`), had no way to drive them.

So it lives here, next to the runners it drives (`reviewer`, `fixer`), and
both callers use it:

  * `backend/service/routes/spec.py` wraps it with what only a service run
    has — the per-user hourly quota, a 503 for an overloaded server, and the
    dry-run admission that follows a PASS (passed in as `on_pass`).
  * a local driver calls it directly, with the record the local backend
    holds and the reserved `__anonymous__` owner as the actor.

Nothing here is re-exported from `backend.spec_worker`'s `__init__`: the
module reaches its collaborators through that package (`_sw.prepare_review`,
`_sw.stream_review`, …), which is the package's own public surface, so
importing it never re-enters a half-built package init.
"""
from __future__ import annotations

import enum
from typing import Callable, Optional

from backend import spec_worker as _sw
from backend.arch.loader import SpecLoadError
from backend.spec_worker.bundle import (
    BlockedBundle,
    FormalSource,
    NormalizedBundle,
    RefusedRecord,
    SpecBundleError,
    normalize_bundle,
    normalize_error_payload,
    open_for_fix,
    review_fingerprint,
)
from backend.spec_worker.outcome import PENDING, GenerationStatus


class RecordJob(enum.Enum):
    """An agent job run on a stored generation record rather than on a
    model: keyed `{job}|{user}|{record_id}`, so a user's and an admin's job
    on the same record never collide, and neither collides with a generate
    run (keyed `{user}|{model}`). The job's name is also the `kind` the
    run's log row is stamped with."""

    REVIEW = "review"
    FIX = "fix"
    #: the dry-run admission that follows a PASS (ADMIT §4.1)
    ADMISSION = "admission"
    #: Open PR / update-pr: one at a time per record (ADMIT §4.3)
    PR = "pr"
    #: the deterministic re-check of a pending record (`RecordRecheck`)
    RECHECK = "recheck"

    def run_key(self, user_id: str, record_id: str) -> str:
        return f"{self.value}|{user_id}|{record_id}"

    @classmethod
    def of_log(cls, log: dict) -> Optional["RecordJob"]:
        """The job that wrote a run log: its `kind` column, or its run key's
        first field when the kind was never stamped. None for a generate /
        refine log."""
        by_name = {job.value: job for job in cls}
        kind = str(log.get("kind") or "").lower()
        head = str(log.get("run_key") or "").split("|", 1)[0]
        return by_name.get(kind) or by_name.get(head)

    @classmethod
    def record_of_log(cls, log: dict) -> Optional[str]:
        """The record id a job's run key names, or None."""
        job = cls.of_log(log)
        parts = str(log.get("run_key") or "").split("|")
        if job is not None and len(parts) >= 3 and parts[0] == job.value:
            return parts[-1]
        return None


def review_run_key(reviewer_id: str, record_id: str) -> str:
    """The review job key — namespaced + per (reviewer, record) so a user's and
    an admin's reviews of the same record never collide, and neither collides
    with a generate run (keyed `{user}|{model}`)."""
    return RecordJob.REVIEW.run_key(reviewer_id, record_id)


def fix_run_key(fixer_id: str, record_id: str) -> str:
    """The fix job key — per (fixer, record), namespaced so it never collides
    with a review (`review|…`) or generate (`{user}|{model}`) run."""
    return RecordJob.FIX.run_key(fixer_id, record_id)


def record_bundle(model_id: str, spec_yaml: str, files) -> NormalizedBundle:
    """A stored record's files as one normalized bundle."""
    return normalize_bundle(
        model_id, spec_yaml, files or [], FormalSource.from_deployment())


def fix_seed_bundle(record: dict) -> NormalizedBundle | BlockedBundle | RefusedRecord:
    """What Agent Fix makes of a stored record (`open_for_fix`): its
    normalized bundle, the record as saved with its repairable refusals, or
    the unrepairable refusal that keeps Fix out."""
    return open_for_fix(
        record["model_id"], record.get("spec_yaml") or "",
        record.get("new_component_yamls") or [],
        FormalSource.from_deployment())


def pending_refusal(record: dict, action: str) -> dict:
    """The answer to `action` on a pending record: its status, and what to
    do instead. Decided on the record's status, never on report text."""
    status = record.get("status")
    if status == GenerationStatus.BLOCKED_BY_ENGINE:
        why = ("its spec is complete and only waits on the analysis engine. "
               "Re-check it when the engine changes.")
    else:
        why = "a re-check found errors in it. Run Agent Fix on them first."
    return {"status": status, "message": f"{action} is not available: {why}"}


def _slot(run_key: str, busy_message: str) -> Optional[dict]:
    """Take the run slot for `run_key`; the refusal to answer with, or None
    once the job holds it. `overloaded` is the server's, not the caller's:
    the service route turns it into a 503."""
    status = _sw.acquire(run_key)
    if status == "duplicate":
        return {"status": "already_running", "run_key": run_key}
    if status == "overloaded":
        return {"status": "overloaded", "run_key": run_key,
                "message": busy_message}
    return None


def launch_review(record: dict, *, reviewer_id: str, reviewer_role: str = "user",
                  agent_cli=None, agent_env=None,
                  on_pass: Optional[Callable[[dict], None]] = None) -> dict:
    """Start an async agent review of a generation record. Returns
    {status, run_key}. Mirrors a generate run's acquire + start_job, with a
    review run_fn (prepare_review → stream_review).

    `on_pass`, when given, is called with the stored review row once a PASS
    is recorded — the service route starts the record's admission there. It
    never changes the verdict: the review row is already written.
    """
    record_id = record["id"]
    model_id = record["model_id"]
    spec_yaml = record.get("spec_yaml") or ""
    comps = record.get("new_component_yamls") or []
    run_key = review_run_key(reviewer_id, record_id)
    if record.get("status") in PENDING:
        return pending_refusal(record, "Review")

    refused = _slot(run_key, "Server busy — try the review again shortly.")
    if refused:
        return refused

    def _run_fn():
        yield {"type": "log", "text": "Preparing review..."}
        try:
            bundle = record_bundle(model_id, spec_yaml, comps)
            prep_result = _sw.prepare_review(
                model_id, record_id=record_id, reviewer_id=reviewer_id,
                bundle=bundle)
        except (SpecBundleError, SpecLoadError) as exc:
            # The record cannot be reviewed as it stands. No review row is
            # written: a load refusal is a fact about the saved text under
            # this checkout, not a reviewer's verdict, and it would read as
            # the record's latest review. `fixable` tells the account page
            # to offer Agent Fix, which opens the record on the refusals.
            _sw.release(run_key)
            failure = normalize_error_payload(exc, model_id=model_id)
            yield {"type": "done", "success": False,
                   "message": failure["message"],
                   "error_kind": failure["error_kind"],
                   "errors": failure["errors"],
                   "fixable": failure["fixable"]}
            return
        except _sw.ForwardSourceUnavailable as exc:
            _sw.release(run_key)
            yield exc.done_event()
            return
        except Exception as e:
            # prep failed before stream_review's release path — free the slot.
            _sw.release(run_key)
            yield {"type": "done", "success": False,
                   "message": f"Review prep failed: {e}"}
            return
        prep_result["reviewer_id"] = reviewer_id
        prep_result["reviewer_role"] = reviewer_role
        prep_result["record_id"] = record_id
        # What this review judges: the record's files and the snapshot config
        # its validate reads. Open PR checks a PASS against the same hash.
        prep_result["bundle_fingerprint"] = review_fingerprint(
            spec_yaml, comps,
            (prep_result.get("snapshot_data") or {}).get("config"))
        if agent_cli:
            prep_result["agent_cli"] = agent_cli
        if agent_env:
            prep_result["agent_env"] = agent_env
        for frame in _sw.stream_review(model_id, prep_result, run_key=run_key):
            evt = _sw.frame_event(frame)
            if evt is not None:
                yield evt
        row = prep_result.get("review_row") or {}
        if on_pass is not None and row.get("verdict") == "pass" and row.get("id"):
            on_pass(row)

    _sw.start_job(
        run_key, _run_fn, model_id=model_id, user_id=reviewer_id,
        kind=RecordJob.REVIEW.value,
    )
    return {"status": "started", "run_key": run_key}


def launch_fix(record: dict, *, fixer_id: str, fixer_role: str = "user",
               review_report: str = "", agent_cli=None, agent_env=None) -> dict:
    """Start an async agent FIX of a generation record: the agent repairs the
    defects named in `review_report` (the latest NO_PASS review; empty when
    the caller opened Fix on a record's load refusals), editing only this
    record's own draft, then the fixed bundle is written back to the record.
    A record a repairable refusal stops is seeded as it was saved, with the
    refusals as the first work order (`fix_seed_bundle`). Mirrors
    `launch_review`, but WRITABLE (prepare_fix → stream_fix). Re-verification
    is the caller's job (it chains the existing review afterward)."""
    record_id = record["id"]
    model_id = record["model_id"]
    run_key = fix_run_key(fixer_id, record_id)
    if record.get("status") == GenerationStatus.BLOCKED_BY_ENGINE:
        return pending_refusal(record, "Agent Fix")
    # A record a re-check found errors in opens on them: they are the work
    # order, as a load refusal's are.
    work_order = (record.get("repairable_errors") or []
                  if record.get("status") == GenerationStatus.NEEDS_REPAIR
                  else [])

    refused = _slot(run_key, "Server busy — try the fix again shortly.")
    if refused:
        return refused

    # A user may only overwrite their OWN record; scope the write-back to them.
    # An admin may fix any record, so the write-back is unscoped.
    owner_scope = fixer_id if fixer_role == "user" else None

    def _run_fn():
        yield {"type": "log", "text": "Preparing fix..."}
        try:
            bundle = fix_seed_bundle(record)
            if isinstance(bundle, RefusedRecord):
                raise bundle.error
            if isinstance(bundle, BlockedBundle):
                yield {"type": "log", "text": (
                    "The saved spec does not load; fixing it as saved, with "
                    f"{len(bundle.errors)} refusal(s) as the work order.")}
            prep_result = _sw.prepare_fix(
                model_id, record_id=record_id, fixer_id=fixer_id,
                bundle=bundle, review_report=review_report,
                work_order=work_order)
        except (SpecBundleError, SpecLoadError) as exc:
            _sw.release(run_key)
            failure = normalize_error_payload(exc, model_id=model_id)
            yield {"type": "done", "success": False,
                   "message": failure["message"],
                   "error_kind": failure["error_kind"],
                   "errors": failure["errors"],
                   "fixable": failure["fixable"]}
            return
        except _sw.ForwardSourceUnavailable as exc:
            _sw.release(run_key)
            yield exc.done_event()
            return
        except Exception as e:
            _sw.release(run_key)
            yield {"type": "done", "success": False,
                   "message": f"Fix prep failed: {e}"}
            return
        prep_result["record_id"] = record_id
        # A pending record's fix lands through the pending promotion,
        # conditioned on the record as it was read here.
        prep_result["record_status"] = record.get("status")
        prep_result["record_revision"] = record.get("revision")
        prep_result["fixer_id"] = fixer_id
        prep_result["owner_scope"] = owner_scope
        if agent_cli:
            prep_result["agent_cli"] = agent_cli
        if agent_env:
            prep_result["agent_env"] = agent_env
        for frame in _sw.stream_fix(model_id, prep_result, run_key=run_key):
            evt = _sw.frame_event(frame)
            if evt is not None:
                yield evt

    _sw.start_job(
        run_key, _run_fn, model_id=model_id, user_id=fixer_id,
        kind=RecordJob.FIX.value,
    )
    return {"status": "started", "run_key": run_key}


__all__ = [
    "RecordJob",
    "fix_run_key",
    "fix_seed_bundle",
    "launch_fix",
    "launch_review",
    "pending_refusal",
    "record_bundle",
    "review_run_key",
]
