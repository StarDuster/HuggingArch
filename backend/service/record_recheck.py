"""Re-check a pending generation record against the current engine.

A `blocked_by_engine` record holds a spec the engine could not analyse when
it was written; the engine moves on. A re-check runs the same deterministic
`validate` the verdict gate runs — no agent — on the record's saved bundle and
snapshot, and moves the record on by the result (`RecheckOutcome`):

    passed        validate passes          → the record is promoted
    still_blocked engine refusals only     → its gap list is refreshed
    needs_repair  errors the writer fixes  → status needs_repair, with them
    unloadable    the bundle does not load → status needs_repair, with that
    conflict      the record changed       → nothing written
    failed        validate / storage broke → nothing written, retry

Every write is `transition_pending` conditioned on the revision read at the
start, so a generation, Fix, reject or delete that lands in between wins and
the re-check reports `conflict`. The job runs under `RecordJob.RECHECK`'s run
key in a `start_job` thread, which releases the key epoch-safely.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from backend.arch.loader import SpecLoadError
from backend.arch.validate._analysable import ENGINE_KIND, engine_gap_entries
from backend.fingerprint import engine_version
from backend.service import spec_generations as _gens
from backend.spec_worker import overlay as _overlay
from backend.spec_worker.bundle import (
    FormalSource,
    SpecBundleError,
    normalize_bundle,
    normalize_error_payload,
)
from backend.spec_worker.outcome import (
    PENDING,
    GenerationStatus,
    RecheckOutcome,
    RecordOutcome,
)
from backend.spec_worker.worker import _invoke_validate

logger = logging.getLogger(__name__)

_MESSAGE = {
    RecheckOutcome.PASSED: "The spec now passes validation; it is your "
                           "working candidate. Run a review to open a PR.",
    RecheckOutcome.STILL_BLOCKED: "The engine still lacks a capability this "
                                  "spec needs.",
    RecheckOutcome.NEEDS_REPAIR: "Validation now finds errors in the spec; "
                                 "Agent Fix can repair them.",
    RecheckOutcome.UNLOADABLE: "The saved spec no longer loads; Agent Fix "
                               "can repair it.",
    RecheckOutcome.CONFLICT: "The record changed during the re-check; "
                             "reload and try again.",
    RecheckOutcome.FAILED: "The re-check could not finish; try again.",
}


class RecordRecheck:
    """One re-check of one pending record, by `actor_id` (its owner or an
    admin): `run()` is the job's event generator."""

    def __init__(self, record_id: str, *, actor_id: str):
        self.record_id = record_id
        self.actor_id = actor_id

    def run(self) -> Iterator[dict]:
        yield {"type": "log",
               "text": "Re-checking the saved spec against the current engine..."}
        outcome, fields = self._judge()
        yield {"type": "done",
               "success": outcome not in (RecheckOutcome.CONFLICT,
                                          RecheckOutcome.FAILED),
               "message": fields.pop("message", None) or _MESSAGE[outcome],
               "recheck": outcome.value, **fields}

    def _judge(self) -> tuple[RecheckOutcome, dict]:
        try:
            rec = _gens.get(self.record_id, strict=True)
        except _gens.SpecGenerationsUnavailable as exc:
            # Could not ask: not the same as a record that moved on.
            return RecheckOutcome.FAILED, {
                "message": f"{_MESSAGE[RecheckOutcome.FAILED]} ({exc})"}
        if not rec or rec.get("status") not in PENDING:
            return RecheckOutcome.CONFLICT, {}
        model_id = rec["model_id"]
        revision = rec["revision"]
        try:
            bundle = normalize_bundle(
                model_id, rec.get("spec_yaml") or "",
                rec.get("new_component_yamls") or [],
                FormalSource.from_deployment())
        except (SpecBundleError, SpecLoadError) as exc:
            errors = normalize_error_payload(exc, model_id=model_id)["errors"]
            return self._write(RecheckOutcome.UNLOADABLE, revision,
                               to=GenerationStatus.NEEDS_REPAIR,
                               repairable_errors=errors)
        try:
            root = _overlay.seed_recheck_overlay(
                self.actor_id, self.record_id, model_id, bundle,
                rec.get("snapshot_json"))
            result = _invoke_validate(model_id, None, root)
        except Exception as exc:  # infrastructure: nothing is judged
            logger.warning("recheck validate failed for %s", self.record_id,
                           exc_info=True)
            return RecheckOutcome.FAILED, {
                "message": f"{_MESSAGE[RecheckOutcome.FAILED]} "
                           f"({exc.__class__.__name__}: {str(exc)[:200]})"}
        errors = [str(e) for e in result.get("errors") or []]
        if not errors:
            return self._write(RecheckOutcome.PASSED, revision, to=None)
        if result.get("blocked_by_engine"):
            gaps = engine_gap_entries(result)
            return self._write(RecheckOutcome.STILL_BLOCKED, revision,
                               to=GenerationStatus.BLOCKED_BY_ENGINE,
                               engine_gaps=gaps, engine_version=engine_version())
        engine = {r.get("detail") for r in result.get("analysis_refusals") or []
                  if r.get("kind") == ENGINE_KIND}
        repairable = [e for e in errors if e not in engine]
        return self._write(RecheckOutcome.NEEDS_REPAIR, revision,
                           to=GenerationStatus.NEEDS_REPAIR,
                           repairable_errors=repairable)

    def _write(self, outcome: RecheckOutcome, revision: int, *,
               to: Optional[GenerationStatus],
               engine_gaps: Optional[list] = None,
               repairable_errors: Optional[list] = None,
               engine_version: Optional[str] = None,
               ) -> tuple[RecheckOutcome, dict]:
        """Write the verdict: promote (`to` None) or refresh as `to`."""
        if to is None:
            written = _gens.promote_pending(
                self.record_id, expected_revision=revision)
        else:
            written = _gens.transition_pending(
                self.record_id, expected_revision=revision, to=to,
                engine_gaps=engine_gaps, repairable_errors=repairable_errors,
                engine_version=engine_version)
        kind = written.get("outcome")
        if kind == RecordOutcome.CONFLICT.value:
            return RecheckOutcome.CONFLICT, {"record": written}
        if kind == RecordOutcome.FAILED.value:
            return RecheckOutcome.FAILED, {"record": written}
        fields: dict = {"record": written}
        if engine_gaps is not None:
            fields["engine_gaps"] = engine_gaps
        if repairable_errors is not None:
            fields["errors"] = repairable_errors
        if kind == RecordOutcome.SUPERSEDED.value:
            fields["message"] = ("The spec now passes validation, but a newer "
                                 "valid spec of this model already stands; "
                                 "this record was dropped.")
        return outcome, fields
