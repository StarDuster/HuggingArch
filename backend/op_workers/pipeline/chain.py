"""Sync propose → review for one gap, without splicing the serving tree.

Spec-gen holds a semaphore slot for the whole job and chains this in-process.
The GitHub PR (or local CLI promote) is the caller's landing step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend.arch.promote_op_proposal import read_artifacts, spliced_files
from backend.op_workers import proposal as op_proposal
from backend.op_workers import review as op_review
from backend.op_workers.proposal._prompts.tasks import prep_from_gap
from backend.op_workers.proposal.runtime import PROPOSALS_DIR
from backend.op_workers.review._rules import check_auto_merge_eligibility


OnFrame = Optional[Callable[[str], None]]


@dataclass
class ChainResult:
    """Outcome of `run_propose_review`. `ready` means the envelope passed
    and `spliced` holds repo-relative file texts for a PR (or CLI apply)."""
    stage: str
    message: str
    op_name: str
    gap: dict
    ready: bool = False
    review: Optional[dict] = None
    reasons: tuple[str, ...] = ()
    patch_path: Optional[str] = None
    spliced: dict[str, str] = field(default_factory=dict)
    proposal_files: dict[str, str] = field(default_factory=dict)


def _done_payload(frame: str) -> Optional[dict]:
    if not isinstance(frame, str) or not frame.startswith("data: "):
        return None
    try:
        payload = json.loads(frame[6:].strip())
    except json.JSONDecodeError:
        return None
    if payload.get("type") == "done":
        return payload
    return None


def _artifact_files(op_name: str) -> dict[str, str]:
    op_dir = Path(PROPOSALS_DIR) / op_name
    if not op_dir.is_dir():
        return {}
    rel_root = Path("backend/arch/ops_proposals") / op_name
    files: dict[str, str] = {}
    for path in op_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            files[(rel_root / path.relative_to(op_dir)).as_posix()] = (
                path.read_text(encoding="utf-8")
            )
        except UnicodeDecodeError:
            continue
    return files


def _run_locked(locker, op_name: str, consume_slot: bool, body, on_frame: OnFrame) -> Optional[dict]:
    """Acquire, drain a sync SSE generator, return the terminal done payload."""
    status = locker.acquire(op_name, consume_slot=consume_slot)
    if status != "ok":
        return {"type": "done", "success": False, "message": status}
    held = True
    last = None
    try:
        for frame in body():
            if on_frame is not None:
                on_frame(frame)
            payload = _done_payload(frame)
            if payload is not None:
                last = payload
        # stream_agent_core releases in its finally.
        held = False
    finally:
        if held:
            locker.release(op_name)
    return last


def run_propose_review(
    op_name: str,
    gap: dict,
    *,
    consume_slot: bool = False,
    on_frame: OnFrame = None,
) -> ChainResult:
    """Propose then review. Never writes `ops_registry.py` on this tree."""
    last = _run_locked(
        op_proposal, op_name, consume_slot,
        lambda: op_proposal.stream_agent(op_name, prep_from_gap(op_name, gap)),
        on_frame,
    )
    if not last or not last.get("success"):
        return ChainResult(
            stage="propose_failed",
            message=(last or {}).get("message") or "proposer ended without success",
            op_name=op_name, gap=gap,
            proposal_files=_artifact_files(op_name),
        )

    last = _run_locked(
        op_review, op_name, consume_slot,
        lambda: op_review.stream_agent(
            op_name, op_review.prepare_review_prompt(op_name),
        ),
        on_frame,
    )
    if not last or not last.get("success"):
        return ChainResult(
            stage="review_failed",
            message=(last or {}).get("message") or "reviewer ended without success",
            op_name=op_name, gap=gap,
            proposal_files=_artifact_files(op_name),
        )

    review_path = Path(PROPOSALS_DIR) / op_name / "review.json"
    review = None
    if review_path.is_file():
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return ChainResult(
                stage="review_failed", message=f"malformed review.json: {e}",
                op_name=op_name, gap=gap,
                proposal_files=_artifact_files(op_name),
            )
    verdict = (review or {}).get("verdict") or ""
    if verdict != "auto_merge":
        return ChainResult(
            stage="rules_blocked",
            message=verdict or "no_verdict",
            op_name=op_name, gap=gap, review=review,
            proposal_files=_artifact_files(op_name),
        )

    eligible, reasons = check_auto_merge_eligibility(Path(PROPOSALS_DIR) / op_name)
    if not eligible:
        return ChainResult(
            stage="rules_blocked",
            message="rules_blocked: " + "; ".join(reasons),
            op_name=op_name, gap=gap, review=review,
            reasons=tuple(reasons),
            proposal_files=_artifact_files(op_name),
        )

    op_dir = Path(PROPOSALS_DIR) / op_name
    artifacts = read_artifacts(op_dir)
    return ChainResult(
        stage="ready", message="ready",
        op_name=op_name, gap=gap, review=review, ready=True,
        patch_path=str(op_dir / "patch.py"),
        spliced=spliced_files(artifacts),
        proposal_files=_artifact_files(op_name),
    )
