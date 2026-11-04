"""Admission on the service (ADMIT §4): run the admitter on a commit's full
tree with a record's PR files laid over it, and turn its report into what
the record page shows and what Open PR may do.

The deployed image carries neither `tests/` nor `tools/` nor the dev
dependencies, so the admitter cannot run in the deployment's own checkout.
It runs in a tree extracted from the commit's GitHub archive, as
`python -m tools.corpus_admit` under the deployment's interpreter: the
tree's own code measures the tree. The deployment only names the commit,
lays the files over it and reads the report back. Nothing a record holds
leaves the machine before Open PR; a report carries hashes, numbers and key
names, never a golden's body.

A report is stored on the review row it follows (`spec_generation_reviews.
admission`, migration 0038) inside a job envelope:
`{"job": "running" | "done" | "failed", "mode": ..., "report": ..., "error": ...}`.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Optional

from backend.arch.promote_op_proposal import REPO_ROOT

logger = logging.getLogger(__name__)

DRY_RUN, PR = "dry_run", "pr"
SCHEMA = 1
#: An admission measures every pin and re-measures the gate surfaces of
#: the corpus once per (base, environment); a cold run takes about five
#: minutes. Open PR may run it three times (main moving), and the whole job
#: must end inside the record lock's TTL (spec_worker.limits, an hour and
#: more), so one run gets twenty.
TIMEOUT_SEC = 20 * 60
_REPORT_TAIL = 4000


class Status(str, enum.Enum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"
    ALREADY_ADMITTED = "already_admitted"
    UNAVAILABLE = "unavailable"


class AdmissionUnavailable(RuntimeError):
    """The admitter did not produce a report."""


class TakenOver(RuntimeError):
    """The PR branch holds changes the admission does not own."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_dir() -> Path:
    """Where the admitter keeps the gate-surface baseline per (base,
    environment). On the image this is the persistent volume."""
    return Path(os.environ.get("HA_ADMISSION_CACHE_DIR")
                or Path.home() / ".cache" / "huggingarch" / "corpus_admit")


def unavailable_report(reason: str, *, mode: str, base_sha: Optional[str] = None,
                       base_kind: str = "main", orchestrator: Optional[str] = None) -> dict:
    """A report for a run that could not happen: the base cannot be named,
    GitHub is unreachable, the bundle does not lay out. Same schema as the
    admitter's, with G0 holding the reason."""
    return {
        "schema": SCHEMA, "status": Status.UNAVAILABLE.value, "blocked_kind": None,
        "preview": False, "mode": mode,
        "base": {"sha": base_sha, "kind": base_kind},
        "measurement_tree": base_sha, "orchestrator": orchestrator,
        "env": {}, "candidate": {},
        "gates": {"G0": [reason], "G1": {}, "G2": []},
        "items": [], "checks": [], "decisions": [], "base_notes": [],
        "would_write": [], "written": [], "errors": [reason],
        "generated_at": now(),
    }


def envelope(job: str, *, mode: str, report: Optional[dict] = None,
             error: Optional[str] = None, **extra: Any) -> dict:
    out = {"job": job, "mode": mode, "report": report, "updated_at": now()}
    if error:
        out["error"] = error
    out.update(extra)
    return out


# ── the tree and the run ──────────────────────────────────────────────────

def _safe_relative(path: str) -> PurePosixPath:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe path {path!r}")
    return pure


def extract_archive(archive: bytes, dest: Path) -> Path:
    """Unpack a GitHub tarball into `dest` and return the repository root
    (the archive's one top-level directory). Members that would land
    outside `dest`, links and devices are refused by the `data` filter."""
    with tarfile.open(fileobj=BytesIO(archive), mode="r:*") as tar:
        tar.extractall(dest, filter="data")
    tops = [p for p in dest.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise AdmissionUnavailable(f"archive has {len(tops)} top-level directories, not one")
    return tops[0]


def archive_file(archive: bytes, rel: str) -> Optional[bytes]:
    """One file of a GitHub tarball (under its top-level directory), or None."""
    with tarfile.open(fileobj=BytesIO(archive), mode="r:*") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and len(parts) > 1 and PurePosixPath(*parts[1:]).as_posix() == rel:
                extracted = tar.extractfile(member)
                return extracted.read() if extracted is not None else None
    return None


def deployed_lock_sha256() -> Optional[str]:
    path = REPO_ROOT / "uv.lock"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def lock_sha256(archive: bytes) -> Optional[str]:
    data = archive_file(archive, "uv.lock")
    return hashlib.sha256(data).hexdigest() if data is not None else None


@dataclass(frozen=True)
class Carry:
    """Every path a PR branch changed since the bot's last commit, as that
    commit held it (`old`) and as the head holds it (`new`), path -> text;
    None = absent. The admitter's `carry` takes the ones that are this
    model's decisions onto the new base and refuses the rest."""

    old: Mapping[str, Optional[str]]
    new: Mapping[str, Optional[str]]


@dataclass
class Outcome:
    """One admission run: the report, and the text of every file it wrote
    or carried into the tree (repo-relative path -> text)."""

    report: dict
    written: dict = field(default_factory=dict)
    carried: dict = field(default_factory=dict)
    #: decision file -> the entries carried from the PR branch
    carried_keys: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.report.get("status") or Status.UNAVAILABLE.value)


class AdmissionTree:
    """A commit's full tree, from its archive, with a record's files laid
    over it. `run` builds it in a temporary directory, runs the admitter
    there and reads back the report and the files it wrote."""

    #: how a command is run in the tree (`subprocess.run`'s signature)
    runner = staticmethod(subprocess.run)

    def __init__(self, archive: bytes, *, base_sha: str, base_kind: str = "main",
                 orchestrator: Optional[str] = None,
                 python: Optional[str] = None) -> None:
        self.archive = archive
        self.base_sha = base_sha
        self.base_kind = base_kind
        self.orchestrator = orchestrator
        self.python = python or sys.executable

    def run(self, model_id: str, files: Mapping[str, str], *, mode: str,
            carry: Optional[Carry] = None) -> Outcome:
        with TemporaryDirectory(prefix="admission-") as td:
            root = extract_archive(self.archive, Path(td) / "tree")
            for path, text in files.items():
                target = root / _safe_relative(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            carried = self._carry(root, Path(td), model_id, carry) if carry else {}
            report_path = Path(td) / "report.json"
            proc = self.execute(self.command(root, model_id, mode, report_path), root)
            if not report_path.exists():
                raise AdmissionUnavailable(
                    f"the admitter exited {proc.returncode} without a report: "
                    f"{(proc.stderr or proc.stdout)[-_REPORT_TAIL:]}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["generated_at"] = now()
            written = {rel: (root / _safe_relative(rel)).read_text(encoding="utf-8")
                       for rel in report.get("written") or []}
            carried_text = {rel: (root / rel).read_text(encoding="utf-8") for rel in carried}
            # The report stays as the admitter hashed it (`report_sha256`
            # covers everything but `generated_at` and `pr`); what was
            # carried travels beside it.
            return Outcome(report=report, written=written, carried=carried_text,
                           carried_keys=carried)

    def execute(self, cmd: list, root: Path) -> subprocess.CompletedProcess:
        try:
            return self.runner(cmd, cwd=root, env=self.environment(), capture_output=True,
                               text=True, timeout=TIMEOUT_SEC, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AdmissionUnavailable(
                f"the admitter ran longer than {TIMEOUT_SEC // 60} minutes") from exc

    def command(self, root: Path, model_id: str, mode: str, report: Path) -> list:
        lock = root / "uv.lock"
        cmd = [self.python, "-m", "tools.corpus_admit", model_id,
               "--base-sha", self.base_sha, "--base-kind", self.base_kind,
               "--env-root", str(REPO_ROOT), "--report", str(report),
               "--cache-dir", str(cache_dir())]
        if lock.exists():
            cmd += ["--base-lock-sha256", hashlib.sha256(lock.read_bytes()).hexdigest()]
        if self.orchestrator:
            cmd += ["--orchestrator", self.orchestrator]
        if mode == DRY_RUN:
            cmd.append("--dry-run")
        return cmd

    @staticmethod
    def environment() -> dict:
        """The service's environment without what would point the admitter
        at code or specs outside the tree: the tree's `backend` and `tools`
        come first on the path because the run starts in it."""
        drop = {"PYTHONPATH", "HA_SPEC_OVERLAY_DIR", "CORPUS_ADMIT_CHILD"}
        return {k: v for k, v in os.environ.items() if k not in drop}

    def _carry(self, root: Path, scratch: Path, model_id: str, carry: Carry) -> dict:
        sides = {}
        for side, texts in (("old", carry.old), ("new", carry.new)):
            base = scratch / f"carry_{side}"
            for rel, text in texts.items():
                if text is not None:
                    target = base / _safe_relative(rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
            base.mkdir(parents=True, exist_ok=True)
            sides[side] = base
        proc = self.execute(
            [self.python, "-m", "tools.corpus_admit", "carry", model_id,
             "--old", str(sides["old"]), "--new", str(sides["new"])], root)
        try:
            result = json.loads(proc.stdout)
        except ValueError:
            raise AdmissionUnavailable(
                f"carry exited {proc.returncode}: {(proc.stderr or proc.stdout)[-_REPORT_TAIL:]}"
            ) from None
        if result.get("problems"):
            raise TakenOver("; ".join(result["problems"]))
        return result.get("carried") or {}


# ── what a report lets Open PR do ─────────────────────────────────────────

@dataclass(frozen=True)
class Gate:
    """Open PR's answer to a report (§2.3): whether a PR opens, as a draft
    or not, and what to tell the person who asked."""

    opens: bool
    draft: bool
    code: str
    message: str

    @classmethod
    def of(cls, report: Mapping[str, Any], *, admin_override: bool = False) -> "Gate":
        status = report.get("status")
        kind = report.get("blocked_kind")
        gates = report.get("gates") or {}
        if status == Status.UNAVAILABLE.value:
            reason = "; ".join(str(g) for g in gates.get("G0") or []) or "the run failed"
            return cls(False, False, "unavailable",
                       f"Admission could not run ({reason}). Try again.")
        if status == Status.BLOCKED.value and kind == "base":
            pins = ", ".join(sorted(gates.get("G1") or {})) or "a pin"
            return cls(False, False, "blocked_base",
                       f"main's corpus baseline needs a refresh ({pins} no longer "
                       "matches main's own measurement). A maintainer refreshes it on "
                       "main; nobody can open this PR until then.")
        if status == Status.BLOCKED.value:
            if admin_override:
                return cls(True, True, "draft_anyway",
                           "Opened as a draft despite failed corpus checks; put the "
                           "rule fix in the same PR.")
            failed = ", ".join(c.get("id", "?") for c in report.get("checks") or []
                               if c.get("state") == "failed") or "a corpus check"
            return cls(False, False, "blocked",
                       f"The corpus checks fail for this model ({failed}). Either the "
                       "spec has a defect (run Agent Fix) or the check's rule has a gap "
                       "(a maintainer fixes it on main); the admission cannot tell which.")
        if status == Status.ALREADY_ADMITTED.value:
            return cls(False, False, "already_admitted",
                       "main already holds this model in every pin; there is nothing "
                       "to contribute.")
        if status == Status.NEEDS_REVIEW.value:
            return cls(True, True, "needs_review",
                       "Opened as a draft: some rows wait on a maintainer's decision.")
        if status == Status.READY.value:
            return cls(True, False, "ready", "Opened with every row admitted.")
        return cls(False, False, "unavailable", f"Unknown admission status {status!r}.")


# ── the PR body's admission section ───────────────────────────────────────

def _keys_in(item: Mapping[str, Any], states: tuple) -> list:
    return [k for k, v in (item.get("key_states") or {}).items() if v.get("state") in states]


def _clip(text: str, limit: int = 3000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def pr_section(report: Mapping[str, Any], carried: Optional[Mapping[str, Any]] = None) -> str:
    """The admission's part of the PR body: one machine-readable line, then
    what was written, what waits on a decision (with the suggested value and
    the tests expected to fail until it is written), base problems that are
    not the candidate's, and information."""
    head = {
        "base_sha": (report.get("base") or {}).get("sha"),
        "bundle_sha256": (report.get("candidate") or {}).get("bundle_sha256"),
        "env": report.get("env"),
        "status": report.get("status"),
        "report_sha256": report.get("report_sha256"),
    }
    lines = [f"<!-- admission {json.dumps(head, sort_keys=True)} -->", "",
             f"### Corpus admission: `{report.get('status')}`"
             + (f" ({report.get('blocked_kind')})" if report.get("blocked_kind") else ""),
             f"Measured against main `{(head['base_sha'] or '')[:12]}`.", ""]
    written = [(item, _keys_in(item, ("written", "ready"))) for item in report.get("items") or []]
    written = [(item, keys) for item, keys in written if keys]
    lines.append("**Written in this PR**")
    lines += [f"- {item['id']} {item.get('title', '')}: {', '.join(f'`{k}`' for k in keys)}"
              for item, keys in written] or ["- nothing"]
    lines.append("")
    pending = [d for d in report.get("decisions") or []
               if d.get("state") in ("pending", "inconsistent")]
    withheld = [(item, _keys_in(item, ("withheld",))) for item in report.get("items") or []]
    withheld = [(item, keys) for item, keys in withheld if keys]
    failed = [c for c in report.get("checks") or [] if c.get("state") == "failed"]
    lines.append("**Needs a maintainer's decision**")
    if not (pending or withheld or failed):
        lines.append("- nothing")
    for check in failed:
        lines.append(f"- check {check['id']} {check.get('title', '')} failed: "
                     + "; ".join(str(v) for v in check.get("violations") or []))
    for decision in pending:
        lines.append(f"- {decision['id']} ({decision['state']}): "
                     + "; ".join(str(d) for d in decision.get("detail") or []))
        if decision.get("suggestion"):
            lines += ["  <details><summary>suggested value</summary>", "",
                      "  ```json",
                      _clip(json.dumps(decision["suggestion"], indent=1, sort_keys=True)),
                      "  ```", "  </details>"]
    for item, keys in withheld:
        waits = sorted({w for k in keys for w in item["key_states"][k].get("waits_on") or []})
        tests = item.get("expected_failing_tests") or []
        lines.append(f"- {item['id']} {item.get('title', '')}: {len(keys)} key(s) wait on "
                     f"{', '.join(waits)}; failing until then: "
                     + (", ".join(f"`{t}`" for t in tests) or "none named"))
    lines.append("")
    lines.append("**Base problems (not this model's)**")
    lines += [f"- {note['id']}: " + "; ".join(str(d) for d in note.get("detail") or [])
              for note in report.get("base_notes") or []] or ["- none"]
    lines.append("")
    info = [d for d in report.get("decisions") or [] if d.get("state") == "info"]
    lines.append("**Information**")
    lines += [f"- {d['id']}: " + "; ".join(str(x) for x in d.get("detail") or [])
              for d in info] or ["- none"]
    if carried:
        lines += ["", "**Decisions carried from the previous PR head**"]
        lines += [f"- `{path}`: {', '.join(keys)}" for path, keys in carried.items()]
    return "\n".join(lines) + "\n"
