"""Open PR / update-pr through the corpus admission, against a fake GitHub
(ADMIT §4.3, §4.4, §6.2 "service").

The admitter itself is replaced at the subprocess boundary by a fake that
reads the extracted tree and writes its report and rows into it, so the
archive extraction, the file overlay, the read-back of written rows and the
carry of a maintainer's decisions all run for real. Supabase is an
in-memory store. Every race §4.3 names is staged through the fake GitHub's
hooks.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from backend.service import admission as adm
from backend.service import github
from backend.service import spec_generation_reviews as reviews
from backend.service import spec_generations as gens
from backend.service import spec_pr
from backend.spec_worker.bundle import record_fingerprint

from tools.corpus_pins.carry import ARTIFACTS
from tools.corpus_pins.pin import canonical

from .fake_github import FakeGitHub

MODEL = "acme/Model"
STEM = "acme__Model"
PIN = "tests/golden/spec_match_fingerprints.json"
TP = "tests/golden/pins/tp_split_decisions.json"
MAIN_FILES = {
    "uv.lock": "lock\n",
    "README.md": "readme\n",
    PIN: json.dumps({"other": "1"}, indent=2, sort_keys=True) + "\n",
    TP: json.dumps({"other": {}}, indent=2, sort_keys=True) + "\n",
}
RECORD = {
    "id": "rec-00000001", "user_id": "user-1", "model_id": MODEL, "status": "generated",
    "spec_yaml": f"match: {{model_id: {MODEL}}}\n", "new_component_yamls": [],
    "snapshot_json": {"config": {"model_type": "acme"}},
}
PR_FILES = {
    f"backend/arch/models/{STEM}.yaml": RECORD["spec_yaml"],
    f"backend/arch/snapshots/{STEM}.json": json.dumps(RECORD["snapshot_json"]),
}
BRANCH = github.spec_pr_branch(MODEL, RECORD["id"])


def _report(status: str, *, base_sha: str, mode: str, blocked_kind: Optional[str] = None,
            written: tuple = ()) -> dict:
    key_state = "written" if written else "withheld"
    return {
        "schema": 1, "status": status, "blocked_kind": blocked_kind, "preview": False,
        "mode": mode, "base": {"sha": base_sha, "kind": "main"},
        "env": {"python": "3.13"}, "candidate": {"bundle_sha256": "b" * 64},
        "gates": {"G0": [], "G1": {"P9": ["drift"]} if blocked_kind == "base" else {},
                  "G2": ["C1"] if blocked_kind == "candidate_or_rule" else []},
        "items": [{"id": "P1", "title": "spec-match fingerprint", "scope": "required",
                   "keys": [STEM], "key_states": {STEM: {
                       "state": key_state, "eligible": bool(written),
                       "waits_on": [] if written else ["G4"], "blocked_by": []}},
                   "expected_failing_tests": [] if written else ["tests/x.py::t"]}],
        "checks": [{"id": "C1", "title": "chain stacks", "state":
                    "failed" if blocked_kind == "candidate_or_rule" else "passed",
                    "violations": []}],
        "decisions": [{"id": "D1", "state": "pending" if status == "needs_review" else "consistent",
                       "detail": [], "suggestion": {"nodes": {}}},
                      {"id": "D5", "state": "info", "detail": ["not listed"], "suggestion": {}}],
        "base_notes": [], "would_write": list(written), "written": list(written),
        "errors": [],
    }


def _hashed(report: dict) -> dict:
    """The admitter's own hash: over everything but `generated_at` and `pr`."""
    body = {k: v for k, v in report.items() if k not in ("generated_at", "pr", "report_sha256")}
    return {**body, "report_sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}


class FakeAdmitter:
    """`python -m tools.corpus_admit` at the subprocess boundary: it reads
    the tree it runs in and writes its rows and report there."""

    def __init__(self, status: str = "ready", blocked_kind: Optional[str] = None) -> None:
        self.status = status
        self.blocked_kind = blocked_kind
        self.runs: list = []
        self.carries: list = []
        self.during = None  # a callable run while the admitter "measures"

    def __call__(self, cmd, cwd=None, **_kw):
        tree = Path(cwd)
        if cmd[3] == "carry":
            # the admitter's carry: decision files move, any other path refuses
            sides = {side: Path(cmd[cmd.index(f"--{side}") + 1]) for side in ("old", "new")}
            paths = sorted({p.relative_to(root).as_posix() for root in sides.values()
                            for p in root.rglob("*") if p.is_file()})
            self.carries.append(paths)
            problems = [f"{p}: not a decision file" for p in paths if p not in ARTIFACTS]
            carried = {} if problems else {p: [STEM] for p in paths}
            for rel in carried:
                (tree / rel).write_text((sides["new"] / rel).read_text())
            return subprocess.CompletedProcess(cmd, 1 if problems else 0, json.dumps(
                {"carried": carried, "problems": problems}), "")
        base = cmd[cmd.index("--base-sha") + 1]
        mode = adm.DRY_RUN if "--dry-run" in cmd else adm.PR
        self.runs.append({"base": base, "mode": mode,
                          "tree_files": sorted(p.relative_to(tree).as_posix()
                                               for p in tree.rglob("*") if p.is_file()),
                          "tp": (tree / TP).read_text(),
                          "lock_arg": cmd[cmd.index("--base-lock-sha256") + 1]})
        if self.during is not None:
            during, self.during = self.during, None
            during()
        written = ()
        if mode == adm.PR and self.status in ("ready", "needs_review"):
            doc = json.loads((tree / PIN).read_text())
            doc[STEM] = "fingerprint"
            (tree / PIN).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            written = (PIN,)
        report = _hashed(_report(self.status, base_sha=base, mode=mode,
                                 blocked_kind=self.blocked_kind, written=written))
        Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")


class Store:
    """The two Supabase tables the job reads and writes."""

    def __init__(self, record: dict) -> None:
        self.record = dict(record)
        self.reviews = [{"id": "rev-1", "verdict": "pass", "created_at": "2026-10-01T00:00:00+00:00",
                         "bundle_fingerprint": record_fingerprint(record), "admission": None}]
        self.status_calls: list = []
        #: set to make the review row refuse the admission column (0038 missing)
        self.admission_column = True
        self.job_logs: list = []

    def add_review(self, verdict: str, *, rid: str = "rev-2",
                   created_at: str = "2026-10-01T00:00:00+00:00") -> None:
        self.reviews.insert(0, {"id": rid, "verdict": verdict, "created_at": created_at,
                                "bundle_fingerprint": record_fingerprint(self.record),
                                "admission": None})

    def set_admission(self, review_id, envelope):
        if not self.admission_column:
            return False
        for row in self.reviews:
            if row["id"] == review_id:
                row["admission"] = json.loads(json.dumps(envelope))
        return True

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(reviews, "list_for_record",
                            lambda _rid: json.loads(json.dumps(self.reviews)))
        monkeypatch.setattr(reviews, "set_admission", self.set_admission)
        monkeypatch.setattr(spec_pr._genlogs, "record",
                            lambda **kw: self.job_logs.append(kw) or True)
        monkeypatch.setattr(gens, "get", lambda _rid: dict(self.record))
        monkeypatch.setattr(gens, "set_status", lambda rid, status, **kw:
                            self.status_calls.append((rid, status, kw)) or True)


@pytest.fixture
def world(monkeypatch, httpx_mock):
    monkeypatch.setenv("HA_GITHUB_TOKEN", "tok")
    monkeypatch.setenv("HA_GITHUB_REPO", "shenh10/HuggingArch")
    for var in ("HA_GITHUB_APP_ID", "HA_GITHUB_APP_PRIVATE_KEY", "HA_GITHUB_APP_INSTALLATION_ID"):
        monkeypatch.delenv(var, raising=False)
    gh = FakeGitHub(MAIN_FILES)
    gh.install(httpx_mock)
    store = Store(RECORD)
    store.install(monkeypatch)
    admitter = FakeAdmitter()
    monkeypatch.setattr(adm.AdmissionTree, "runner", admitter)
    monkeypatch.setattr(spec_pr, "_prepare", lambda record, source, **kw: github.SpecPR(
        branch=BRANCH, files=dict(PR_FILES), title=f"Add spec: {MODEL}",
        body="Contributed spec.", commit_prefix=f"spec({MODEL})"))
    monkeypatch.setattr(spec_pr, "_code_revision", lambda: "d" * 40)
    return gh, store, admitter


def _run(store: Store, *, update: bool = False, admin_override: bool = False) -> list:
    job = spec_pr.SpecPRJob(store.record, store.reviews[-1], owner_id="user-1",
                            update=update, admin_override=admin_override)
    return list(job.run())


def _done(events: list) -> dict:
    (done,) = [e for e in events if e["type"] == "done"]
    return done


def _stored_report(store: Store, rid: str = "rev-1") -> dict:
    row = next(r for r in store.reviews if r["id"] == rid)
    return row["admission"]["report"]


def test_ready_opens_a_pr_with_the_record_files_and_the_admitted_rows(world):
    gh, store, admitter = world
    main = gh.branches["main"]
    done = _done(_run(store))
    assert done["success"] and done["admission_status"] == "ready" and not done["draft"]
    head = gh.branches[BRANCH]
    assert gh.commits[head]["parents"] == [main]
    files = gh.files(head)
    for path, text in PR_FILES.items():
        assert files[path] == text
    assert json.loads(files[PIN]) == {"other": "1", STEM: "fingerprint"}
    pull = gh.pull(BRANCH)
    assert pull["draft"] is False
    assert "<!-- admission " in pull["body"] and f'"base_sha": "{main}"' in pull["body"]
    # the admitter ran in write mode on main's tree with the PR files laid over it
    (run,) = admitter.runs
    assert run["base"] == main and run["mode"] == adm.PR
    assert set(PR_FILES) <= set(run["tree_files"]) and "uv.lock" in run["tree_files"]
    report = _stored_report(store)
    assert report["mode"] == "pr" and report["pr"]["head_sha"] == head
    assert report["pr"]["url"] == pull["html_url"] and report["base"]["sha"] == main
    assert store.status_calls == [(RECORD["id"], "contributed",
                                   {"user_id": "user-1", "pr_url": pull["html_url"]})]


def test_needs_review_opens_a_draft_with_the_eligible_rows(world):
    gh, store, admitter = world
    admitter.status = "needs_review"
    done = _done(_run(store))
    assert done["success"] and done["draft"]
    assert gh.pull(BRANCH)["draft"] is True
    assert STEM in json.loads(gh.files(gh.branches[BRANCH])[PIN])
    assert "Needs a maintainer's decision" in gh.pull(BRANCH)["body"]


@pytest.mark.parametrize("admin_override", [False, True])
def test_blocked_by_base_drift_refuses_everyone(world, admin_override):
    gh, store, admitter = world
    admitter.status, admitter.blocked_kind = "blocked", "base"
    done = _done(_run(store, admin_override=admin_override))
    assert not done["success"] and done["code"] == "blocked_base"
    assert BRANCH not in gh.branches and not gh.pulls
    assert _stored_report(store)["status"] == "blocked"


def test_blocked_by_checks_refuses_the_owner_and_lets_an_admin_open_a_draft(world):
    gh, store, admitter = world
    admitter.status, admitter.blocked_kind = "blocked", "candidate_or_rule"
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "blocked"
    assert "Agent Fix" in done["message"] and "rule" in done["message"]
    assert not gh.pulls
    done = _done(_run(store, admin_override=True))
    assert done["success"] and done["code"] == "draft_anyway"
    assert gh.pull(BRANCH)["draft"] is True
    # blocked writes no rows: the draft carries the record's files only
    assert STEM not in json.loads(gh.files(gh.branches[BRANCH])[PIN])


def test_unavailable_refuses_and_says_retry(world, monkeypatch):
    gh, store, _ = world
    monkeypatch.setattr(adm.AdmissionTree, "runner", staticmethod(
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "Traceback: boom")))
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "unavailable"
    assert "Try again" in done["message"]
    assert not gh.pulls
    assert _stored_report(store)["status"] == "unavailable"


def test_main_moving_between_reading_it_and_writing_recomputes_on_the_new_main(world):
    gh, store, admitter = world
    first = gh.branches["main"]
    admitter.during = lambda: gh.push("main", {"README.md": "moved\n"})
    done = _done(_run(store))
    assert done["success"]
    second = gh.branches["main"]
    assert [r["base"] for r in admitter.runs] == [first, second]
    head = gh.branches[BRANCH]
    assert gh.commits[head]["parents"] == [second]
    assert _stored_report(store)["base"]["sha"] == second
    assert f'"base_sha": "{second}"' in gh.pull(BRANCH)["body"]


def test_main_moving_after_the_reread_leaves_a_self_consistent_pr_on_its_base(world):
    gh, store, _admitter = world
    base = gh.branches["main"]
    gh.hooks["before_branch_move"] = lambda g: g.push("main", {"README.md": "late\n"})
    done = _done(_run(store))
    assert done["success"]
    head = gh.branches[BRANCH]
    assert gh.commits[head]["parents"] == [base]
    assert _stored_report(store)["base"]["sha"] == base
    assert gh.branches["main"] != base  # behind main, as the design accepts


def test_main_that_keeps_moving_is_refused_after_the_retries(world):
    gh, store, admitter = world

    def move():
        gh.push("main", {"README.md": f"run {len(admitter.runs)}\n"})
        admitter.during = move
    admitter.during = move
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "main_moving"
    assert len(admitter.runs) == 1 + spec_pr.MAIN_RETRIES
    assert BRANCH not in gh.branches


def test_a_no_pass_written_during_the_admission_stops_before_github(world):
    gh, store, admitter = world
    admitter.during = lambda: store.add_review("no_pass")
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "review_changed"
    assert BRANCH not in gh.branches and not gh.pulls


def test_a_no_pass_at_the_same_instant_still_counts_as_newer(world):
    _gh, store, admitter = world
    # same created_at as the PASS: the id breaks the tie (ADMIT I1)
    admitter.during = lambda: store.add_review("no_pass", rid="rev-9")
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "review_changed"


def test_a_no_pass_written_while_github_is_written_turns_the_pr_into_a_draft(world):
    gh, store, _admitter = world
    gh.hooks["after_pull"] = lambda g: store.add_review(
        "no_pass", created_at="2026-10-01T00:05:00+00:00")
    done = _done(_run(store))
    pull = gh.pull(BRANCH)
    assert pull["draft"] is True
    assert gh.comments and "newer review" in gh.comments[0]["body"]
    report = _stored_report(store)
    assert report["pr"]["draft"] and "newer review" in report["pr"]["reconciled"]
    assert done["success"] and done["draft"]


def test_a_push_after_the_branch_move_is_caught_by_the_head_check(world):
    gh, store, _admitter = world
    gh.hooks["after_pull"] = lambda g: g.push(BRANCH, {"README.md": "maintainer\n"})
    _run(store)
    assert gh.pull(BRANCH)["draft"] is True
    report = _stored_report(store)
    assert "branch head" in report["pr"]["reconciled"]
    assert report["pr"]["head_sha"] != gh.branches[BRANCH]


def _open_then_maintainer(gh, store, changes):
    assert _done(_run(store))["success"]
    bot_head = gh.branches[BRANCH]
    store.record["pr_url"] = gh.pull(BRANCH)["html_url"]
    gh.push(BRANCH, changes, message="maintainer")
    return bot_head


def test_update_pr_carries_the_maintainers_decisions_to_the_new_main(world):
    gh, store, admitter = world
    decided = json.dumps({"other": {}, STEM: {"(backbone)": {"verdict": "framework_split"}}},
                         indent=2, sort_keys=True) + "\n"
    _open_then_maintainer(gh, store, {TP: decided})
    gh.push("main", {"README.md": "main moved on\n"})
    done = _done(_run(store, update=True))
    assert done["success"]
    run = admitter.runs[-1]
    assert run["base"] == gh.branches["main"] and run["tp"] == decided
    files = gh.files(gh.branches[BRANCH])
    assert files[TP] == decided and STEM in json.loads(files[PIN])
    assert gh.commits[gh.branches[BRANCH]]["parents"] == [gh.branches["main"]]
    envelope = next(r for r in store.reviews if r["id"] == "rev-1")["admission"]
    assert envelope["carried"] == {TP: [STEM]} and "carried" not in envelope["report"]
    assert "Decisions carried from the previous PR head" in gh.pull(BRANCH)["body"]
    # the stored report is the admitter's, hash and all
    stored = {k: v for k, v in envelope["report"].items() if k not in ("generated_at", "pr")}
    assert _hashed(stored) == stored
    assert len(gh.pulls) == 1  # the same PR


def test_update_pr_without_maintainer_changes_just_recomputes(world):
    gh, store, _admitter = world
    assert _done(_run(store))["success"]
    store.record["pr_url"] = gh.pull(BRANCH)["html_url"]
    gh.push("main", {"README.md": "next\n"})
    done = _done(_run(store, update=True))
    assert done["success"]
    assert gh.commits[gh.branches[BRANCH]]["parents"] == [gh.branches["main"]]


def test_update_pr_refuses_a_branch_a_maintainer_took_over(world):
    gh, store, admitter = world
    bot_head = _open_then_maintainer(
        gh, store, {f"backend/arch/models/{STEM}.yaml": "edited by hand\n"})
    maintainer = gh.branches[BRANCH]
    runs = len(admitter.runs)
    done = _done(_run(store, update=True))
    assert not done["success"] and done["code"] == "taken_over"
    assert f"backend/arch/models/{STEM}.yaml: not a decision file" in done["message"]
    assert gh.branches[BRANCH] == maintainer != bot_head
    assert len(admitter.runs) == runs


def test_update_pr_stops_when_someone_pushes_between_the_check_and_the_move(world):
    gh, store, admitter = world
    _open_then_maintainer(gh, store, {})
    admitter.during = lambda: gh.push(BRANCH, {"README.md": "late push\n"})
    late = None

    done = _done(_run(store, update=True))
    late = gh.branches[BRANCH]
    assert not done["success"] and done["code"] == "taken_over"
    assert gh.files(late)["README.md"] == "late push\n"  # their commit stays


def test_two_update_prs_of_one_record_do_not_run_together(monkeypatch):
    from backend import spec_worker
    from backend.service.routes import spec as routes
    from backend.spec_worker import runtime
    monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: False)
    started = []
    monkeypatch.setattr(spec_worker, "start_job", lambda key, fn, **kw: started.append(key))
    record = dict(RECORD, pr_url="https://github.com/x/y/pull/1")
    try:
        first = routes.launch_spec_pr(record, {"id": "rev-1"}, update=True)
        second = routes.launch_spec_pr(record, {"id": "rev-1"}, update=True)
    finally:
        spec_worker.release(routes.pr_run_key(record))
    assert first["status"] == "started" and second["status"] == "already_running"
    assert first["run_key"] == second["run_key"] == f"pr|user-1|{RECORD['id']}"
    assert started == [first["run_key"]]


def test_the_tree_gets_main_lock_hash_and_no_pythonpath(world, monkeypatch):
    _gh, store, admitter = world
    import hashlib
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    captured = {}
    real = admitter.__call__

    def spy(cmd, cwd=None, env=None, **kw):
        captured["env"] = env
        captured["cmd"] = cmd
        return real(cmd, cwd=cwd, **kw)
    monkeypatch.setattr(adm.AdmissionTree, "runner", staticmethod(spy))
    _run(store)
    assert admitter.runs[0]["lock_arg"] == hashlib.sha256(b"lock\n").hexdigest()
    assert "PYTHONPATH" not in captured["env"]
    assert captured["cmd"][captured["cmd"].index("--env-root") + 1] == str(adm.REPO_ROOT)


def test_update_pr_refuses_a_branch_whose_bot_head_was_never_stored(world):
    """0038 not applied (or the PATCH failed): the PR opens, its report goes
    to the job log, and a maintainer's D1 on the branch is never overwritten."""
    gh, store, admitter = world
    store.admission_column = False
    done = _done(_run(store))
    assert done["success"] and done["report_stored"] is False
    assert "cannot prove" in done["message"]
    (logged,) = store.job_logs
    envelope = spec_pr.report_from_log(logged["log"])
    assert envelope["report"]["pr"]["head_sha"] == gh.branches[BRANCH]
    assert envelope["report"]["status"] == "ready" and envelope["mode"] == "pr"
    store.record["pr_url"] = gh.pull(BRANCH)["html_url"]
    maintainer = gh.push(BRANCH, {TP: json.dumps({"other": {}, STEM: {}}) + "\n"})
    store.admission_column = True
    runs = len(admitter.runs)
    done = _done(_run(store, update=True))
    assert not done["success"] and done["code"] == "branch_unproven"
    assert gh.branches[BRANCH] == maintainer and len(admitter.runs) == runs


def test_a_legacy_branch_with_a_decision_on_it_is_refused(world):
    gh, store, admitter = world
    gh.branches[BRANCH] = gh.branches["main"]
    legacy = gh.push(BRANCH, {TP: json.dumps({"other": {}, STEM: {"": {}}}) + "\n"},
                     message="old bot commit + maintainer D1")
    gh.pulls.append({"number": 1, "head": BRANCH, "state": "open", "title": "t", "body": "b",
                     "draft": False, "node_id": "PR_1",
                     "html_url": "https://github.com/shenh10/HuggingArch/pull/1"})
    store.record["pr_url"] = "https://github.com/shenh10/HuggingArch/pull/1"
    done = _done(_run(store, update=True))
    assert not done["success"] and done["code"] == "branch_unproven"
    assert gh.branches[BRANCH] == legacy and admitter.runs == []


def test_a_review_store_that_cannot_be_read_does_not_move_the_branch(world, monkeypatch):
    gh, store, _admitter = world
    assert _done(_run(store))["success"]
    store.record["pr_url"] = gh.pull(BRANCH)["html_url"]
    head = gh.branches[BRANCH]
    monkeypatch.setattr(reviews, "list_for_record", lambda _rid: [])  # a failed read reads as []
    done = _done(_run(store, update=True))
    assert done["code"] == "branch_unproven" and gh.branches[BRANCH] == head


def test_open_pr_refuses_a_branch_that_appears_while_it_runs(world):
    gh, store, admitter = world
    admitter.during = lambda: gh.branches.__setitem__(BRANCH, gh.branches["main"])
    done = _done(_run(store))
    assert not done["success"] and done["code"] == "taken_over"
    assert gh.branches[BRANCH] == gh.branches["main"] and not gh.pulls


def test_a_dry_run_report_the_row_cannot_take_is_in_the_job_log(world, monkeypatch):
    gh, store, _admitter = world
    store.admission_column = False
    monkeypatch.setattr(adm, "deployed_lock_sha256",
                        lambda: adm.lock_sha256(gh.tarball(gh.branches["main"])))
    job = spec_pr.DryRunAdmission(store.record, "rev-1", run_key="admission|user-1|rec")
    events = list(job.run())
    assert events[-1]["report_stored"] is False and "job's log" in events[-1]["message"]
    (logged,) = store.job_logs  # the `running` envelope is not logged, the final one is
    assert logged["run_key"] == "admission|user-1|rec" and logged["model_id"] == MODEL
    envelope = spec_pr.report_from_log(logged["log"])
    assert envelope["job"] == "done" and envelope["mode"] == "dry_run"
    assert envelope["report"]["status"] == "ready"
    assert envelope["report"]["base"]["sha"] == gh.branches["main"]
