"""Agent REVIEW runner — judge an already-generated spec against the model's
real architecture and emit a machine-parseable PASS / NO_PASS verdict + report.

Read-only counterpart to `worker.stream_agent`: same job/stream/env plumbing
(`stream_agent_core`, the run overlay, the staged snapshot), but the agent writes
nothing, there is no publish/repair, and the verdict is scanned out of the
streamed text. The deterministic `validate()` result is folded into both the
prompt and the stored report and is SUPREME — a validate failure forces NO_PASS.
The report's evidence is then checked against the run's snapshot
(`review_evidence`): the result is stored with the report, and a PASS whose
Table F is not verified, or that carries a contradicted claim, is recorded as
an error.

The review row is persisted (pass/no_pass/error alike) via a lazy, service-guarded
import of `backend.service.spec_generation_reviews`, exactly the core↔service
pattern `worker._upload_run_log` uses for `spec_generation_logs`.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Generator, Optional, Tuple

from backend import supabase_env
from backend.fingerprint import engine_version
from backend._agent_runner import scrub_paths, stream_agent_core
from backend.agents.base import host_agent_credentials
from .runtime import (
    _LOG_DIR, _locks_lock, _model_procs, bind_job_log, get_agent_cli, owns, release,
)
from ._ground_truth import assemble_task_prompt
from ._prompts.tasks import _build_prompt_context, _stage_source
from ._prompts.review import build_review_task_body, _render_validate_summary
from .bundle import BundleFile, NormalizedBundle
from .review_evidence import ReviewEvidence, SnapshotFacts
from .limits import REVIEW_AGENT_SEC, REVIEW_TIMEOUT_KNOB, review_budget_sec
from .overlay import seed_review_overlay
from .worker import (
    RunRevision,
    _code_revision,
    _invoke_validate,
    _stage_snapshot,
    _upload_run_log,
)

logger = logging.getLogger(__name__)



_SENTINEL_RE = re.compile(
    r"===REVIEW REPORT START===\s*(.*?)\bVERDICT:\s*(PASS|NO_PASS)\b.*?===REVIEW REPORT END===",
    re.DOTALL | re.IGNORECASE,
)
_BARE_VERDICT_RE = re.compile(r"\bVERDICT:\s*(PASS|NO_PASS)\b", re.IGNORECASE)


def _extract(text: str) -> Tuple[str, str]:
    """(verdict, body) from the agent's streamed text. Takes the LAST complete
    sentinel block (the agent's real conclusion — earlier hits are drafts, or
    the prompt's format example echoed into a tool log), then the LAST bare
    `VERDICT:` line, else ('error', full text). verdict ∈ {'pass','no_pass',
    'error'}. Scanning the FIRST match would let an echoed example record a
    structurally-bad spec as PASS, which is worse than failing to parse."""
    text = text or ""
    blocks = list(_SENTINEL_RE.finditer(text))
    if blocks:
        m = blocks[-1]
        return m.group(2).lower(), (m.group(1).strip() or text.strip())
    bares = list(_BARE_VERDICT_RE.finditer(text))
    if bares:
        return bares[-1].group(1).lower(), text.strip()
    return "error", text.strip()


def _final_verdict(agent_verdict: str, validate_result: Optional[dict],
                   evidence: ReviewEvidence) -> str:
    """Fold the deterministic gate and the evidence check into the agent's
    verdict. PASS iff validate() ran and returned valid:true AND the agent
    said PASS AND its report passes the evidence check. A failed gate forces
    no_pass; a gate that could not run (None), or a PASS whose evidence does
    not check out, can never yield PASS → error."""
    v = validate_result if isinstance(validate_result, dict) else None
    if v is not None and v.get("valid") is False:
        return "no_pass"
    if agent_verdict == "pass" and not (v is not None and v.get("valid") is True):
        return "error"
    if agent_verdict == "pass" and evidence.pass_blockers():
        return "error"
    return agent_verdict


def _verdict_note(verdict: str, agent_verdict: str,
                  validate_result: Optional[dict], evidence: ReviewEvidence) -> str:
    """The closing lines of a review's log: the verdict recorded, and why it
    is not the one the agent wrote when the two differ. The transcript above
    ends on the agent's own `VERDICT:`, which the validator and the evidence
    check can overrule (`_final_verdict`)."""
    head = f"[verdict] recorded {verdict.upper()}"
    if verdict == agent_verdict:
        return head + "\n"
    lines = [f"{head}; the agent wrote {agent_verdict.upper()}"]
    v = validate_result if isinstance(validate_result, dict) else None
    if v is None:
        lines.append("the deterministic validator did not run, so no PASS can be recorded")
    elif v.get("valid") is False:
        lines.append("the deterministic validator reports errors")
    blockers = evidence.pass_blockers() if agent_verdict == "pass" else []
    lines.extend(f"- {b}" for b in blockers)
    return "\n".join(lines) + "\n"


def _timed_out_body(budget_sec: int, partial: str) -> str:
    """The report body of a review stopped at its time budget: says so, and
    keeps whatever the agent had written as unfinished notes."""
    body = (f"The review agent was stopped at its time budget "
            f"({budget_sec // 60} min) before it wrote its report. This is "
            "not a verdict on the spec; run the review again.")
    if partial.strip():
        body += ("\n\n### Unfinished agent notes (not a report)\n\n"
                 + partial.strip())
    return body


def _review_budget(source_files: list, bundle: NormalizedBundle) -> int:
    """`review_budget_sec` for this record: the staged forward source's size
    and the component files the record brings."""
    source_bytes = 0
    for path in source_files or []:
        with contextlib.suppress(OSError):
            source_bytes += os.path.getsize(path)
    n_components = sum(
        1 for f in bundle.record_files()
        if BundleFile(path=f["path"], body=f["body"]).draft_name)
    return review_budget_sec(source_bytes, n_components)


def _compose_report(validate_result: Optional[dict], body: str,
                    evidence: ReviewEvidence, agent_verdict: str,
                    revision: Optional[RunRevision] = None) -> str:
    """Prepend the deterministic validator block to the agent's findings and
    append the evidence check, so the stored report carries all three. The
    validator block names the code revision the verdict ran against."""
    det = _render_validate_summary(validate_result)
    if revision is not None:
        det = f"{revision.stamp()}\n{det}"
    return (f"## Deterministic validator\n```\n{det}\n```\n\n## Agent review\n{body}"
            f"\n\n{evidence.render(agent_verdict)}")


def prepare_review(model_id: str, *, record_id: str, reviewer_id: str,
                   bundle: NormalizedBundle) -> dict:
    """Build the review prep dict (mirror `prepare_generate_prompt`). Seeds the
    spec into a record-scoped overlay, runs the deterministic validator against
    it, and assembles the read-only review prompt. Runs at prep time (network
    OK); only the agent subprocess is sandboxed."""
    target_root = seed_review_overlay(
        reviewer_id, record_id, model_id, bundle)
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
        logger.warning(f"review: validate CLI failed for {model_id}: {e}")
    facts = SnapshotFacts.from_run(meta.get("_snapshot_data"), source_files)
    task_body = build_review_task_body(
        model_id, bundle.spec_yaml, bundle.record_files(), validate_result,
        facts=facts, target_root=target_root, source_files=source_files)
    return {
        "prompt": assemble_task_prompt(system, ground_truth, task_body),
        "snapshot_data": meta.get("_snapshot_data"),
        "target_root": target_root,
        "validate_result": validate_result,
        "evidence_facts": facts,
        "code_revision": revision,
        "review_budget_sec": _review_budget(source_files, bundle),
    }


def _persist_review(*, record_id, model_id, reviewer_id, reviewer_role,
                    verdict, report, agent_cli, target_root,
                    bundle_fingerprint=None, run_revision=None) -> Optional[str]:
    """Store the review row (pass/no_pass/error) and return its id, or None
    when it was not stored. Lazy, service-guarded import so core never
    imports service at module top (test_service_layout)."""
    if not supabase_env.configured():
        return None
    try:
        from backend.service import spec_generation_reviews as _reviews
        return _reviews.record(
            record_id=record_id, model_id=model_id, reviewer_id=reviewer_id,
            reviewer_role=reviewer_role, verdict=verdict,
            report=scrub_paths(report, target_root), agent_cli=agent_cli,
            bundle_fingerprint=bundle_fingerprint, run_revision=run_revision,
            engine_version=engine_version(),
        )
    except Exception:  # storage outage must not surface to the review run
        logger.warning("spec_generation_reviews persist failed", exc_info=True)
        return None


def stream_review(model_id: str, prep_result: dict, *,
                  run_key: Optional[str] = None) -> Generator[str, None, None]:
    """Run the review agent and stream normalized SSE events, then extract the
    verdict from the accumulated text and persist the review row. Mirror of
    `worker.stream_agent`, read-only. `prep_result` carries reviewer identity
    (`reviewer_id`/`reviewer_role`/`record_id`) + BYOK config (`agent_cli`/
    `agent_env`), stamped by the route's run_fn."""
    run_key = run_key or model_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = model_id.replace("/", "__")
    log_path = os.path.join(_LOG_DIR, f"review_{safe_id}_{ts}.log")

    agent_cli = prep_result.get("agent_cli") or get_agent_cli()
    agent_env = prep_result.get("agent_env")
    target_root = prep_result.get("target_root")
    # Point the agent's in-loop validate/structure at the seeded overlay. On the
    # BYOK path agent_env is a scrubbed dict; on the admin host path it is None,
    # so build a minimal scrubbed env from the host creds + the overlay dir.
    if target_root:
        if agent_env is not None:
            agent_env = {**agent_env, "HA_SPEC_OVERLAY_DIR": target_root}
        else:
            agent_env = {**host_agent_credentials(agent_cli),
                         "HA_SPEC_OVERLAY_DIR": target_root}

    _stage_snapshot(model_id, prep_result.get("snapshot_data"), target_root, log_path)
    revision = prep_result.get("code_revision") or _code_revision()
    prep_result["code_revision"] = revision
    bind_job_log(
        run_key, log_path,
        user_id=prep_result.get("user_id") or prep_result.get("reviewer_id"),
        model_id=model_id)

    # Only the agent's own prose is its report; tool echoes and runner notes
    # (`$ ls`, "Coding agent connected") are the run log, uploaded separately.
    buffer: list[str] = []
    validate_result = prep_result.get("validate_result")
    final_success = None
    timed_out = False
    budget_sec = int(prep_result.get("review_budget_sec") or REVIEW_AGENT_SEC)
    try:
        for frame in stream_agent_core(
            key=run_key,
            prep_result=prep_result,
            agent_cli=agent_cli,
            log_path=log_path,
            log_header=f"=== review {model_id} @ {ts} sha={revision} ===",
            launch_banner="Reviewing spec...",
            procs_dict=_model_procs,
            owns=owns,
            locks_lock=_locks_lock,
            release_fn=release,
            on_success=None,
            env=agent_env,
            hard_cap_sec=budget_sec,
            timeout_hint=REVIEW_TIMEOUT_KNOB,
            repair_prompt_fn=None,
            max_repair_rounds=0,
        ):
            if frame.startswith("data: "):
                try:
                    evt = json.loads(frame[len("data: "):].strip())
                except (ValueError, KeyError):
                    evt = None
                if (evt and evt.get("type") == "log" and evt.get("text")
                        and evt.get("author") == "agent"):
                    buffer.append(str(evt["text"]))
                if evt and evt.get("type") == "done":
                    final_success = bool(evt.get("success"))
                    timed_out = bool(evt.get("timed_out"))
            yield frame
    finally:
        # The serving checkout may have taken commits since prep. The spec
        # and snapshot this run judged are fixed in its overlay, so the
        # verdict stands; it is validated against the checkout as it is now
        # and stamped with both revisions (`RunRevision`).
        judged_at = RunRevision.since(revision)
        if judged_at.moved:
            logger.info("review %s: %s", model_id, judged_at.stamp())
        try:
            validate_result = _invoke_validate(model_id, None, target_root)
        except Exception as exc:
            logger.warning("review: final validate CLI failed for %s: %s",
                           model_id, exc)
        agent_verdict, body = _extract("\n\n".join(buffer))
        if timed_out:
            # A review cut off before its report has no verdict to record:
            # an `error` that says so, not the validator's no_pass standing
            # in for findings nobody wrote.
            agent_verdict, body = "error", _timed_out_body(
                budget_sec, "\n\n".join(buffer))
        # No facts (a prep that did not build them) checks nothing, so no
        # claim verifies and a PASS cannot be recorded.
        evidence = ReviewEvidence.audit(
            body, prep_result.get("evidence_facts") or SnapshotFacts.empty())
        verdict = ("error" if timed_out
                   else _final_verdict(agent_verdict, validate_result, evidence))
        review_id = _persist_review(
            record_id=prep_result.get("record_id"),
            model_id=model_id,
            reviewer_id=prep_result.get("reviewer_id"),
            reviewer_role=prep_result.get("reviewer_role") or "user",
            verdict=verdict,
            report=_compose_report(
                validate_result, body, evidence, agent_verdict, judged_at),
            agent_cli=agent_cli,
            target_root=target_root,
            bundle_fingerprint=prep_result.get("bundle_fingerprint"),
            run_revision=judged_at.stamp(),
        )
        # The stored row, for the caller's follow-up: a PASS starts the
        # record's admission (ADMIT §4.1), which writes onto this row. The
        # verdict is the reviewer's alone; nothing downstream changes it.
        prep_result["review_row"] = {"id": review_id, "verdict": verdict}
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(_verdict_note(verdict, agent_verdict,
                                             validate_result, evidence))
        except OSError:
            logger.warning("review: could not note the verdict in %s", log_path)
        stamp = {**prep_result, "user_id": (
            prep_result.get("user_id") or prep_result.get("reviewer_id"))}
        if final_success is None:
            final_success = False
        _upload_run_log(model_id, stamp, run_key, log_path, final_success)
