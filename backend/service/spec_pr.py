"""A record's admission and its PR, as background jobs (ADMIT §4.1, §4.3, §4.4).

`DryRunAdmission` follows a PASS review: it lays the record's PR files over
main (or over the deployment's own commit when main's lock differs from the
deployed one) and stores the admitter's report on that review row. Nothing
is written anywhere else; the record page shows which rows are ready to
admit and which wait on a person's decision.

`SpecPRJob` is Open PR and update-pr. It recomputes everything against the
main commit it reads, runs the admitter in write mode, and opens (or
updates) the PR with the record's files plus the rows the admitter wrote —
or refuses, when the admission says the PR must not open. Before it writes
GitHub it checks the PASS it started from is still the record's latest
review for the bundle the record holds; after, it checks again and turns
the PR into a draft with a comment when either moved. It moves an existing
PR branch only when a stored report names the commit the bot last pushed
there; without one it cannot tell the bot's branch from a maintainer's and
leaves it alone.

Both are generators of job events (`spec_worker.start_job`).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Generator, Optional

import httpx

from backend.service import admission as _adm
from backend.service import github as _github
from backend.service import spec_generation_logs as _genlogs
from backend.service import spec_generation_reviews as _reviews
from backend.service import spec_generations as _gens
from backend.spec_worker.bundle import SpecBundleError, record_fingerprint
from backend.spec_worker.outcome import RunErrorKind
from backend.spec_worker.self_register import stamp_match_model_id
from backend.spec_worker.worker import _code_revision
from backend.arch.loader import SpecLoadError

logger = logging.getLogger(__name__)

#: How often Open PR recomputes when main moves under it before giving up.
MAIN_RETRIES = 2

Event = dict


def _done(success: bool, message: str, **extra: Any) -> Event:
    return {"type": "done", "success": success, "message": message, **extra}


def _log(text: str) -> Event:
    return {"type": "log", "text": text}


#: The first line of a job log that holds an admission envelope the review
#: row could not take (migration 0038 not applied, a failed PATCH).
REPORT_LOG_HEAD = "admission envelope (not stored on its review row):"


class BranchUnproven(_adm.TakenOver):
    """The PR branch exists and no stored report names the commit the bot
    last pushed to it, so nothing proves the branch is still the bot's."""


def persist(review_id: str, envelope: dict, *, record: dict,
            run_key: Optional[str]) -> bool:
    """Store `envelope` on its review row. When the row cannot take it, a
    finished envelope is written as the job's log instead (the record page
    lists that log; `report_from_log` reads it back) and the answer is
    False, which the caller passes on: a PR whose report is not on its
    review row cannot be updated later (`SpecPRJob.branch_state`)."""
    if _reviews.set_admission(review_id, envelope):
        return True
    if envelope.get("job") != "running":
        _genlogs.record(
            user_id=record.get("user_id") or "", model_id=record["model_id"],
            log=f"{REPORT_LOG_HEAD}\n{json.dumps(envelope, indent=1, sort_keys=True)}\n",
            run_key=run_key)
    return False


def report_from_log(log: str) -> Optional[dict]:
    """The envelope `persist` wrote into a job log; None for any other log."""
    head, _, body = (log or "").partition("\n")
    return json.loads(body) if head == REPORT_LOG_HEAD else None


def _record_files(record: dict) -> tuple[str, list, dict]:
    return (stamp_match_model_id(record.get("spec_yaml") or "", record["model_id"]),
            record.get("new_component_yamls") or [],
            record.get("snapshot_json") or {})


def _prepare(record: dict, source, *, note: str = "",
             contributor: Optional[str] = None) -> _github.SpecPR:
    spec_yaml, comps, snapshot = _record_files(record)
    return _github.prepare_spec_pr(
        record_id=record["id"], model_id=record["model_id"], spec_yaml=spec_yaml,
        new_component_yamls=comps, snapshot_json=snapshot, note=note,
        contributor=contributor, formal_source=source)


class DryRunAdmission:
    """The admission that follows a PASS (§4.1): a dry run, stored on the
    review row it follows."""

    def __init__(self, record: dict, review_id: str, *, run_key: Optional[str] = None) -> None:
        self.record = record
        self.review_id = review_id
        self.run_key = run_key
        self.orchestrator = _code_revision()

    def save(self, envelope: dict) -> bool:
        return persist(self.review_id, envelope, record=self.record, run_key=self.run_key)

    def base(self) -> tuple[Optional[_github.RemoteTree], str, Optional[str]]:
        """(tree, kind, reason): main's tip when its `uv.lock` is the
        deployed one, else the deployment's own commit; no tree, with the
        reason, when neither can be named."""
        if not _github.enabled():
            return None, "main", "GitHub is not configured; the admission needs main's tree"
        main = _github.fetch_remote_tree()
        if _adm.lock_sha256(main.archive) == _adm.deployed_lock_sha256():
            return main, "main", None
        if self.orchestrator == "unknown":
            return None, "deployment", (
                "main's uv.lock differs from the deployed one and the deployment's "
                "revision is unknown (no HA_GIT_SHA)")
        return _github.fetch_remote_tree(self.orchestrator), "deployment", None

    def run(self) -> Generator[Event, None, None]:
        self.save(_adm.envelope("running", mode=_adm.DRY_RUN))
        yield _log("Reading the base commit...")
        try:
            tree, kind, reason = self.base()
            if tree is None:
                report = _adm.unavailable_report(reason or "no base", mode=_adm.DRY_RUN,
                                                 base_kind=kind, orchestrator=self.orchestrator)
            else:
                yield _log(f"Measuring against {kind} {tree.sha[:12]}...")
                report = self.admit(tree, kind)
        except (httpx.HTTPError, _github.GitHubNotConfigured) as exc:
            report = _adm.unavailable_report(f"GitHub could not be read: {exc}",
                                             mode=_adm.DRY_RUN, orchestrator=self.orchestrator)
        except Exception as exc:  # the envelope must leave `running`
            logger.exception("admission of %s failed", self.record.get("id"))
            self.save(_adm.envelope("failed", mode=_adm.DRY_RUN, error=str(exc)))
            yield _done(False, f"Admission failed: {exc}")
            return
        stored = self.save(_adm.envelope("done", mode=_adm.DRY_RUN, report=report))
        yield _done(True, f"Admission: {report.get('status')}"
                    + ("" if stored else " (the review row could not take the report; "
                       "it is in this job's log)"),
                    admission_status=report.get("status"), report_stored=stored)

    def admit(self, tree: _github.RemoteTree, kind: str) -> dict:
        try:
            pr = _prepare(self.record, tree.source)
        except (SpecBundleError, SpecLoadError) as exc:
            return _adm.unavailable_report(
                f"the record's files do not lay out against {tree.sha[:12]}: {exc}",
                mode=_adm.DRY_RUN, base_sha=tree.sha, base_kind=kind,
                orchestrator=self.orchestrator)
        try:
            return _adm.AdmissionTree(
                tree.archive, base_sha=tree.sha, base_kind=kind,
                orchestrator=self.orchestrator,
            ).run(self.record["model_id"], pr.files, mode=_adm.DRY_RUN).report
        except _adm.AdmissionUnavailable as exc:
            return _adm.unavailable_report(str(exc), mode=_adm.DRY_RUN, base_sha=tree.sha,
                                           base_kind=kind, orchestrator=self.orchestrator)


class SpecPRJob:
    """Open PR (`update=False`) or update-pr (`update=True`) for a record
    whose latest review `review` is a PASS of the bundle it holds."""

    def __init__(self, record: dict, review: dict, *, owner_id: str, update: bool,
                 admin_override: bool = False, contributor: Optional[str] = None,
                 note: str = "", run_key: Optional[str] = None) -> None:
        self.record = record
        self.run_key = run_key
        self.review = review
        self.owner_id = owner_id
        self.update = update
        self.admin_override = admin_override
        self.contributor = contributor
        self.note = note
        self.orchestrator = _code_revision()
        self.branch = _github.spec_pr_branch(record["model_id"], record["id"])

    # ── the review the PR stands on ──────────────────────────────────────
    def review_moved(self) -> Optional[str]:
        """Why the PASS this run started from no longer stands, or None:
        a newer review (by created_at, then id), or the record's bundle
        changed since it was judged."""
        rows = _reviews.list_for_record(self.record["id"])
        newer = _reviews.newer_than(rows, self.review["id"])
        if newer:
            return f"a newer review ({newer[0].get('verdict')}) was written"
        current = _gens.get(self.record["id"])
        if current is None:
            return "the record is gone"
        if record_fingerprint(current) != self.review.get("bundle_fingerprint"):
            return "the record's spec changed since its PASS"
        return None

    def save(self, report: dict, carried: Optional[dict] = None) -> bool:
        extra = {"carried": carried} if carried else {}
        return persist(self.review["id"],
                       _adm.envelope("done", mode=_adm.PR, report=report, **extra),
                       record=self.record, run_key=self.run_key)

    # ── the branch as the bot left it ────────────────────────────────────
    def branch_state(self) -> tuple[Optional[_adm.Carry], Optional[str]]:
        """(decisions to carry, the head the branch must still be at).

        No branch: the run creates it, and refuses one that appears
        meanwhile. A branch is moved only when a stored PR report names the
        commit the bot last pushed: if that commit still heads it, nothing
        is carried; if a maintainer wrote this model's decisions on top,
        they are carried; anything else is `TakenOver`. A branch with no
        such report — a PR from before admissions, a report the review row
        could not take, a store that could not be read — is
        `BranchUnproven`: nothing shows the bot owns it, so it is not
        moved."""
        head = _github.branch_head(self.branch)
        if head is None:
            return None, None
        recorded = _reviews.latest_pr_report(_reviews.list_for_record(self.record["id"]))
        bot_head = ((recorded or {}).get("pr") or {}).get("head_sha")
        if bot_head is None:
            raise BranchUnproven(
                f"{self.branch} exists at {head[:12]}, and no stored admission report "
                "names the commit the bot last pushed to it")
        if head == bot_head:
            return None, head
        # Which of these are decisions this model owns is the admitter's
        # call (`tools.corpus_admit carry`); anything else stops the run.
        changed = _github.changed_paths(bot_head, head)
        return _adm.Carry(old={p: _github.file_at(bot_head, p) for p in changed},
                          new={p: _github.file_at(head, p) for p in changed}), head

    # ── the run ──────────────────────────────────────────────────────────
    def run(self) -> Generator[Event, None, None]:
        model_id = self.record["model_id"]
        drafts = _github.input_drafts(self.record.get("new_component_yamls"))
        for _attempt in range(1 + MAIN_RETRIES):
            yield _log("Reading main...")
            try:
                remote = _github.preflight_remote_tree(branch=self.branch, drafts=drafts)
                pr = _prepare(self.record, remote.source, note=self.note,
                              contributor=self.contributor)
                carry, expected_head = self.branch_state()
            except (SpecBundleError, SpecLoadError) as exc:
                yield _done(False, str(exc), error_kind=RunErrorKind.BUNDLE_INVALID.value)
                return
            except BranchUnproven as exc:
                yield _done(False, "Cannot prove the PR branch is still the bot's "
                                   f"({exc}); it is left as it is. A maintainer continues "
                                   "with the local admitter.", code="branch_unproven")
                return
            except _adm.TakenOver as exc:
                yield _done(False, "A maintainer has taken this PR over; update-pr is off "
                                   f"for it ({exc}). They continue with the local admitter.",
                            code="taken_over")
                return
            yield _log(f"Admitting against main {remote.sha[:12]}...")
            try:
                outcome = _adm.AdmissionTree(
                    remote.archive, base_sha=remote.sha, orchestrator=self.orchestrator,
                ).run(model_id, pr.files, mode=_adm.PR, carry=carry)
            except _adm.TakenOver as exc:
                yield _done(False, "A maintainer has taken this PR over; update-pr is off "
                                   f"for it ({exc}).", code="taken_over")
                return
            except _adm.AdmissionUnavailable as exc:
                report = _adm.unavailable_report(str(exc), mode=_adm.PR, base_sha=remote.sha,
                                                 orchestrator=self.orchestrator)
                self.save(report)
                gate = _adm.Gate.of(report)
                yield _done(False, gate.message, code=gate.code, admission_status="unavailable")
                return
            report = outcome.report
            gate = _adm.Gate.of(report, admin_override=self.admin_override)
            if not gate.opens:
                self.save(report)
                yield _done(False, gate.message, code=gate.code,
                            admission_status=report.get("status"))
                return
            moved = self.review_moved()
            if moved:
                self.save(report)
                yield _done(False, f"The review changed while the PR was prepared ({moved}); "
                                   "open the PR again.", code="review_changed")
                return
            files = {**pr.files, **outcome.written, **outcome.carried}
            try:
                published = _github.publish_files_pr(
                    branch=self.branch, files=files, title=pr.title,
                    body=f"{pr.body}\n\n{_adm.pr_section(report, outcome.carried_keys)}",
                    commit_prefix=pr.commit_prefix, contributor=self.contributor,
                    base_sha=remote.sha, draft=gate.draft, expected_head=expected_head,
                    must_create=expected_head is None)
            except _github.MainMoved as exc:
                # The rows were computed against a main that is gone.
                yield _log(f"{exc}; recomputing against the new main.")
                continue
            except _github.BranchTakenOver as exc:
                self.save(report)
                yield _done(False, f"Someone pushed to the PR branch while it was updated "
                                   f"({exc}); update-pr stopped.", code="taken_over")
                return
            report["pr"] = self.reconcile(published)
            stored = self.save(report, outcome.carried_keys)
            _gens.set_status(self.record["id"], self.record.get("status")
                             if self.update and self.record.get("status") else "contributed",
                             user_id=self.owner_id, pr_url=published.url)
            message = report["pr"].get("reconciled") or gate.message
            if not stored:
                message += (" The PR's admission report could not be stored on its review "
                            "(it is in this job's log), so update-pr cannot prove the branch "
                            "is the bot's: further changes go through a maintainer.")
            yield _done(True, message,
                        pr_url=published.url, admission_status=report.get("status"),
                        draft=report["pr"]["draft"], code=gate.code, report_stored=stored)
            return
        yield _done(False, "main is moving; try again shortly.", code="main_moving")

    def reconcile(self, published: _github.PullRequest) -> dict:
        """After the PR is written: the review and the branch head must
        still be the ones it was built on. Otherwise the PR becomes a
        draft, with a comment saying why, and the record needs a new Open
        PR / update-pr (ADMIT I2, I4)."""
        problems = []
        moved = self.review_moved()
        if moved:
            problems.append(moved)
        head = _github.branch_head(self.branch)
        if head != published.head_sha:
            problems.append(f"the branch head is {head}, not this run's commit "
                            f"{published.head_sha}")
        out = {"url": published.url, "number": published.number,
               "head_sha": published.head_sha, "base_sha": published.base_sha,
               "draft": published.draft}
        if problems:
            reason = ("This PR no longer stands on what it was built from: "
                      + "; ".join(problems) + ". It is a draft until the author "
                      "runs update-pr after a fresh PASS.")
            try:
                _github.reconcile_pr(published, reason)
            except httpx.HTTPError as exc:
                reason += f" (GitHub refused turning it into a draft: {exc})"
            out.update(draft=True, reconciled=reason)
        return out
