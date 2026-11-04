"""Agent FIX runner — repair the SPECIFIC defects a NO_PASS review named, editing
only this record's own draft, then update the record in place.

Writable twin of `reviewer.py`: same job/stream/env plumbing (`stream_agent_core`,
the run overlay, the staged snapshot), but the agent EDITS the seeded draft under
the writable allowlist (`models/**` + `components/drafts/**` only — never shared
repo components), the deterministic `validate()` gate + repair loop are enabled
(exactly as generate/refine), and on a clean pass the fixed bundle is harvested
and written back to the record through `records.update_spec` — whichever backend
holds it (Supabase, or the host-local store of an offline run).

Re-verification (does the fix flip NO_PASS→PASS) is NOT done here — the caller
chains the existing read-only review afterward, so each job stays single-purpose.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Generator, Optional, Sequence

from backend._agent_runner import stream_agent_core
from backend.agents.base import host_agent_credentials
from backend.arch.loader import SpecLoadError

from . import records
from .runtime import (
    _LOG_DIR, _locks_lock, _model_procs, bind_job_log, get_agent_cli, owns, release,
)
from ._ground_truth import assemble_task_prompt
from ._prompts.tasks import _build_prompt_context, _stage_source
from ._prompts.fix import build_fix_task_body
from .limits import (
    FIX_AGENT_SEC, FIX_TIMEOUT_KNOB, ROUND_BUDGET_SEC, ROUND_TIMEOUT_KNOB,
)
from .overlay import seed_fix_overlay, _safe
from .bundle import BlockedBundle, NormalizedBundle, SpecBundleError
from .review_evidence import ReviewEvidence, SnapshotFacts
from .outcome import PENDING, RecordOutcome
from .worker import (
    Downgrade,
    _bundle_failure,
    _code_revision,
    _invoke_validate,
    _load_failure,
    _normalize_overlay_bundle,
    _max_repair_rounds,
    _run_agent_model,
    _run_snapshot_json,
    _repair_prompt_fn,
    _stage_snapshot,
    _upload_run_log,
    _verify_spec_output,
    note_downgrade,
)

logger = logging.getLogger(__name__)


def prepare_fix(model_id: str, *, record_id: str, fixer_id: str,
                bundle: NormalizedBundle | BlockedBundle,
                review_report: str = "",
                work_order: Sequence[str] = ()) -> dict:
    """Build the fix prep dict (mirror `prepare_review`, but WRITABLE). Seeds the
    record's draft into a record-scoped writable overlay, runs the deterministic
    validator against it, and assembles the fix prompt whose work order is the
    NO_PASS review report after its evidence is checked against this run's
    snapshot: only verified findings are work orders, unverified ones are
    leads to confirm, contradicted ones are withheld. A `BlockedBundle` is
    seeded as it was saved, and its refusals come first in the work order;
    with no review report they are the whole of it. `work_order` is the
    validation errors a re-check stored on a `needs_repair` record: they
    order the fix the same way. Runs at prep time (network OK); only the
    agent subprocess is sandboxed."""
    target_root = seed_fix_overlay(
        fixer_id, record_id, model_id, bundle)
    system, ground_truth, meta = _build_prompt_context(model_id)
    source_files = _stage_source(target_root, model_id, meta)
    revision = _code_revision()
    # The overlay validate reads config + metadata from the staged snapshot,
    # exactly as the agent's own validate will. Unstaged, it fell back to a
    # live config fetch, which fails for a model_type the installed
    # transformers does not know (Qwen3-VL-MoE), and reported a load failure
    # the spec did not have.
    _stage_snapshot(model_id, meta.get("_snapshot_data"), target_root, None)
    validate_result = None
    try:
        validate_result = _invoke_validate(model_id, None, target_root)
    except Exception as e:  # deterministic validator is best-effort evidence
        logger.warning(f"fix: validate CLI failed for {model_id}: {e}")
    facts = SnapshotFacts.from_run(meta.get("_snapshot_data"), source_files)
    load_errors = bundle.errors if isinstance(bundle, BlockedBundle) else ()
    ordered = (*load_errors, *work_order)
    review = (ReviewEvidence.audit(review_report, facts)
              if review_report.strip() or not ordered else None)
    task_body = build_fix_task_body(
        model_id, bundle.spec_yaml, bundle.record_files(), validate_result,
        review, facts=facts, target_root=target_root,
        source_files=source_files, load_errors=load_errors,
        validation_errors=tuple(work_order))
    return {
        "prompt": assemble_task_prompt(system, ground_truth, task_body),
        "snapshot_data": meta.get("_snapshot_data"),
        "target_root": target_root,
        # The overlay is seeded, so a correct no-edit run judges the existing
        # spec rather than reporting "nothing produced" (see _verify_spec_output).
        "seeded_spec": True,
        "seed_bundle": bundle,
        "code_revision": revision,
    }


def _harvest_fixed_bundle(
        target_root: str,
        seed_bundle: NormalizedBundle | BlockedBundle) -> Optional[NormalizedBundle]:
    """Read the fixed identity and normalize it with this run's overlays."""
    return _normalize_overlay_bundle(
        seed_bundle.model_id, target_root, seed_bundle=seed_bundle)


def _persist_fix(*, record_id: str, model_id: str, target_root: str,
                 seed_bundle: NormalizedBundle | BlockedBundle,
                 owner_scope: Optional[str],
                 pending: Optional[dict] = None,
                 ) -> tuple[bool, Optional[Downgrade], Optional[dict]]:
    """Harvest the fixed bundle and write it back to the record in place. A
    pending record (`pending`: its read `revision` and the fix run's
    `run_started_at`) is promoted with the fixed bundle instead — the same
    conditional promotion a passing re-check makes — and the third value is
    that write's outcome. `records` decides which backend holds the record,
    so a fix lands on a local run's record as it does on a hosted one."""
    try:
        harvested = _harvest_fixed_bundle(target_root, seed_bundle)
        if harvested is None:
            return False, None, None
    except SpecBundleError as exc:
        return False, _bundle_failure(exc), None
    except SpecLoadError as exc:
        return False, _load_failure(exc), None
    try:
        if pending is not None:
            written = records.promote_pending(
                record_id, expected_revision=pending["revision"],
                spec_yaml=harvested.spec_yaml,
                new_component_yamls=harvested.record_files(),
                agent_cli=pending.get("agent_cli"),
                agent_model=pending.get("agent_model"),
                run_started_at=pending.get("run_started_at"))
            return (written.get("outcome") == RecordOutcome.RECORDED.value,
                    None, written)
        recorded = records.update_spec(
            record_id, spec_yaml=harvested.spec_yaml,
            new_component_yamls=harvested.record_files(),
            user_id=owner_scope,
            snapshot_json=_run_snapshot_json(model_id, target_root))
        return recorded, None, None
    except Exception:  # a storage outage must not surface to the fix run
        logger.warning("spec_generations write-back failed", exc_info=True)
        return False, None, None


def stream_fix(model_id: str, prep_result: dict, *,
               run_key: Optional[str] = None) -> Generator[str, None, None]:
    """Run the fix agent (writable) and stream normalized SSE events. On a clean
    validate the fixed bundle is written back to the record. Mirror of
    `worker.stream_agent`, scoped to this record's draft. `prep_result` carries
    `record_id` + `fixer_id`/`owner_scope` + BYOK config (`agent_cli`/`agent_env`),
    stamped by the route's run_fn."""
    run_key = run_key or model_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = _safe(model_id)
    log_path = os.path.join(_LOG_DIR, f"fix_{safe_id}_{ts}.log")

    agent_cli = prep_result.get("agent_cli") or get_agent_cli()
    agent_env = prep_result.get("agent_env")
    target_root = prep_result.get("target_root")
    if target_root:
        if agent_env is not None:
            agent_env = {**agent_env, "HA_SPEC_OVERLAY_DIR": target_root}
        else:
            agent_env = {**host_agent_credentials(agent_cli),
                         "HA_SPEC_OVERLAY_DIR": target_root}

    _stage_snapshot(model_id, prep_result.get("snapshot_data"), target_root, log_path)
    revision = prep_result.get("code_revision") or _code_revision()
    prep_result["code_revision"] = revision
    prep_result.setdefault(
        "run_started_at", datetime.now(timezone.utc).isoformat())
    bind_job_log(
        run_key, log_path,
        user_id=prep_result.get("user_id") or prep_result.get("fixer_id"),
        model_id=model_id)

    record_id = prep_result.get("record_id")
    owner_scope = prep_result.get("owner_scope")

    def _on_success(_key, log_file, started_at, pr) -> Optional[str]:
        # Deterministic gate + repair loop, identical to generate/refine: a fix
        # that leaves validate() with errors is downgraded and re-attempted.
        # An engine-blocked verdict ends the fix with its gaps on the `done`
        # event and the run's log row; the record is left as it was.
        pr.pop("terminal", None)
        downgrade, _vr = _verify_spec_output(
            model_id, log_file, started_at, root=target_root, seeded=True,
            code_revision=pr.get("code_revision") or prep_result.get("code_revision"))
        if downgrade:
            note_downgrade(pr, downgrade)  # `downgrade` read by _repair_prompt_fn
            return downgrade.reason
        pending = ({"revision": prep_result.get("record_revision"),
                    "agent_cli": agent_cli,
                    "agent_model": _run_agent_model(prep_result, agent_cli),
                    "run_started_at": prep_result.get("run_started_at")}
                   if prep_result.get("record_status") in PENDING else None)
        recorded, bundle_downgrade, written = _persist_fix(
            record_id=record_id, model_id=model_id, target_root=target_root,
            seed_bundle=prep_result["seed_bundle"], owner_scope=owner_scope,
            pending=pending)
        if written is not None:
            pr.setdefault("terminal", {})["record"] = written
        if bundle_downgrade is not None:
            note_downgrade(pr, bundle_downgrade)
            return bundle_downgrade.reason
        if written is not None and not recorded:
            pr.pop("downgrade", None)
            log_file.write(f"[fix] pending promotion: {written}\n")
            log_file.flush()
            return ("fix validated, but the record "
                    + ("changed during the fix; reload it and run the fix again"
                       if written.get("outcome") == RecordOutcome.CONFLICT.value
                       else "was dropped: a newer valid spec of this model stands"
                       if written.get("outcome") == RecordOutcome.SUPERSEDED.value
                       else f"could not be written back ({written.get('reason')})"))
        if not recorded:
            # Not a validation failure — drop any stale validation Downgrade from
            # an earlier repair round so the repair hook doesn't re-issue an
            # expired work order (it would burn a round, then still end failed).
            pr.pop("downgrade", None)
            # The spec validated but the fixed bundle did NOT durably reach the
            # record (harvest produced nothing, or the write-back failed / matched
            # no owned row). Report FAILURE — a "done" here would tell the user
            # the fix landed when it did not. No repair: the agent cannot fix a
            # storage/write-back problem (not a `validation` downgrade), so the
            # repair hook returns None and the job ends failed.
            log_file.write("[fix] write-back FAILED — reporting failure\n")
            log_file.flush()
            return ("fix validated but could not be written back to the record "
                    "(harvest or storage failure — see log)")
        return None

    # `_max_repair_rounds()` is evaluated as a kwarg here, before the core's own
    # finally can run; a malformed knob would otherwise leak the caller-held lock
    # (start_job never releases). Mirror stream_agent: release on any exit. release
    # is idempotent, so the core's own release_fn on the normal path is a no-op.
    final_success = None
    try:
        for frame in stream_agent_core(
            key=run_key,
            prep_result=prep_result,
            agent_cli=agent_cli,
            log_path=log_path,
            log_header=f"=== fix {model_id} @ {ts} sha={revision} ===",
            launch_banner="Fixing spec...",
            procs_dict=_model_procs,
            owns=owns,
            locks_lock=_locks_lock,
            release_fn=release,
            on_success=_on_success,
            env=agent_env,
            hard_cap_sec=FIX_AGENT_SEC,
            round_budget_sec=ROUND_BUDGET_SEC,
            timeout_hint=FIX_TIMEOUT_KNOB,
            round_timeout_hint=ROUND_TIMEOUT_KNOB,
            repair_prompt_fn=_repair_prompt_fn(model_id),
            max_repair_rounds=_max_repair_rounds(),
        ):
            if final_success is None and "done" in frame and frame.startswith("data: "):
                try:
                    evt = json.loads(frame[len("data: "):].strip())
                except (ValueError, json.JSONDecodeError):
                    evt = None
                if evt and evt.get("type") == "done":
                    final_success = bool(evt.get("success"))
            yield frame
    finally:
        release(run_key)
        stamp = {**prep_result, "user_id": (
            prep_result.get("user_id")
            or prep_result.get("fixer_id")
            or prep_result.get("owner_scope"))}
        if final_success is None:
            final_success = False
        _upload_run_log(model_id, stamp, run_key, log_path, final_success)
