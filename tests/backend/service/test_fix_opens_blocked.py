"""Agent Fix and review on a record whose saved spec does not load.

A repairable refusal (a draft that clashes with a library namesake, a
migration that needs a declaration) opens Fix on the record as saved, with the
refusals as the work order; review reports it as fixable. A retired syntax is
not repairable and keeps the old answers.
"""
from __future__ import annotations

import pytest

import backend.service.routes.spec as _spec
import backend.spec_worker as _sw
from backend.arch.component_signature import component_structure_signature
from backend.arch.spec_migrate import SCHEMA_ERROR
from backend.service import agent_credentials as _creds
from backend.service import spec_generation_reviews as _reviews
from backend.service import spec_generations as _gens
from backend.spec_worker.bundle import (
    BlockedBundle,
    FormalComponent,
    FormalSource,
    NormalizedBundle,
)


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _component(bias: str) -> str:
    return f"""role: attention
params:
  inner: component
graph:
  out:
    op: inner
    in: x
    dims:
      bias: "{bias}"
outputs: [out]
"""


LIBRARY = _component("Qb")
DRAFT = _component("1")

NAMESAKE = {
    "spec_yaml": "match: {model_id: [acme/Foo]}\ndrafts: [same]\n",
    "new_component_yamls": [
        {"path": "backend/arch/components/drafts/same.yaml", "body": DRAFT}],
}

# Every refusal names a declaration (`SpecMigrationBlocked`).
DECLARATION = {
    "spec_yaml": """# the author's citation comment
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: scale, op: parameter_vector, dim: "1"}
  - {name: norm, op: rmsnorm, dim: d_model}
  - {name: head, op: linear, boundary_role: output_head, inputs: [norm], dims: {in: d_model, out: n_vocab}}
""",
    "new_component_yamls": [],
}

# A retired KV field: nothing to declare, regenerate only.
RETIRED = {
    "spec_yaml": "role: attention\nkv_cached: [k, v]\ngraph:\n  x: {op: identity, in: [x]}\n",
    "new_component_yamls": [],
}

LOADABLE = {"spec_yaml": "match: {model_id: [acme/Foo]}\n", "new_component_yamls": []}


def _record(kind: dict) -> dict:
    return {"id": "rec-1", "model_id": "acme/Foo", "user_id": "user-uuid-1", **kind}


@pytest.fixture(autouse=True)
def _fresh_quota():
    """Every /fix and /review here spends the real hourly quota of
    `user-uuid-1`; the bucket is process-global, so empty it around each
    test rather than leave later files with it full."""
    _spec._quota_hits.clear()
    yield
    _spec._quota_hits.clear()


@pytest.fixture(autouse=True)
def _formal(monkeypatch):
    """One library component named `same`, nothing else."""
    source = FormalSource(models={}, components={"same": FormalComponent(
        tier="library", body=LIBRARY,
        signature=component_structure_signature(LIBRARY, path="library/same.yaml"),
        path="backend/arch/components/library/same.yaml")})
    monkeypatch.setattr(FormalSource, "from_deployment",
                        classmethod(lambda cls, *a, **k: source))


@pytest.fixture
def jobs(monkeypatch):
    """Capture the job instead of running it; the agent never starts."""
    started = []
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    monkeypatch.setattr(_sw, "release", lambda k: None)
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: started.append(fn))
    monkeypatch.setattr(_creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    return started


@pytest.fixture
def prep(monkeypatch):
    """A fake prepare_fix that records what Fix was seeded with."""
    seen = {}

    def fake_prepare_fix(model_id, *, record_id, fixer_id, bundle,
                         review_report="", work_order=()):
        seen.update(bundle=bundle, review_report=review_report,
                    work_order=list(work_order))
        return {}
    monkeypatch.setattr(_sw, "prepare_fix", fake_prepare_fix)
    monkeypatch.setattr(_sw, "stream_fix", lambda *a, **k: iter(()))
    monkeypatch.setattr(_sw, "stream_review",
                        lambda *a, **k: pytest.fail("review agent ran"))
    return seen


def _post_fix(client, user_token, monkeypatch, record, reviews):
    monkeypatch.setattr(_gens, "get", lambda rid: record)
    monkeypatch.setattr(_reviews, "list_for_record_owner", lambda rid, uid: reviews)
    return client.post("/api/account/generations/rec-1/fix", headers=_auth(user_token))


def _review_done(client, user_token, monkeypatch, record, jobs):
    monkeypatch.setattr(_gens, "get", lambda rid: record)
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "started"
    done = list(jobs[-1]())[-1]
    assert done["type"] == "done" and done["success"] is False
    return done


NO_PASS = [{"verdict": "no_pass", "report": "- **a defect**: fix it"}]


# -- Fix ------------------------------------------------------------------


def test_fix_opens_a_namesake_draft_record_without_a_review(
        client, user_token, monkeypatch, jobs, prep):
    """A draft that clashes with a library namesake, so no review can
    run. Fix starts anyway, on the saved text, with the clash as its work
    order."""
    r = _post_fix(client, user_token, monkeypatch, _record(NAMESAKE), [])
    assert r.status_code == 200 and r.json()["status"] == "started"
    events = list(jobs[-1]())
    bundle = prep["bundle"]
    assert isinstance(bundle, BlockedBundle)
    assert bundle.spec_yaml == NAMESAKE["spec_yaml"]
    assert [f.body for f in bundle.serving_files] == [DRAFT]
    assert len(bundle.errors) == 1
    assert "conflicts with the different formal component" in bundle.errors[0]
    assert prep["review_report"] == ""
    assert any("refusal(s) as the work order" in (e.get("text") or "")
               for e in events)


def test_fix_opens_a_declaration_needed_record_as_saved(
        client, user_token, monkeypatch, jobs, prep):
    """A vision tower whose migration needs a declaration. Fix gets the
    UNmigrated text (comments and all) and the refusals, not a normalized
    rewrite; the NO_PASS review the record already had still rides along."""
    r = _post_fix(client, user_token, monkeypatch, _record(DECLARATION), NO_PASS)
    assert r.json()["status"] == "started"
    list(jobs[-1]())
    bundle = prep["bundle"]
    assert isinstance(bundle, BlockedBundle)
    assert bundle.spec_yaml == DECLARATION["spec_yaml"]
    assert "# the author's citation comment" in bundle.spec_yaml
    assert any("omits `inputs:`" in e for e in bundle.errors)
    assert prep["review_report"] == NO_PASS[0]["report"]


def test_fix_on_declaration_needed_record_needs_no_review(
        client, user_token, monkeypatch, jobs, prep):
    r = _post_fix(client, user_token, monkeypatch, _record(DECLARATION), [])
    assert r.json()["status"] == "started"
    list(jobs[-1]())
    assert isinstance(prep["bundle"], BlockedBundle)


def test_fix_on_retired_syntax_keeps_the_review_precondition(
        client, user_token, monkeypatch, jobs, prep):
    """Unrepairable: no review → no Fix, as before."""
    r = _post_fix(client, user_token, monkeypatch, _record(RETIRED), [])
    assert r.json()["status"] == "no_review"
    assert jobs == [] and prep == {}


def test_fix_on_retired_syntax_with_no_pass_fails_prep_as_before(
        client, user_token, monkeypatch, jobs, prep):
    r = _post_fix(client, user_token, monkeypatch, _record(RETIRED), NO_PASS)
    assert r.json()["status"] == "started"
    done = list(jobs[-1]())[-1]
    assert done["success"] is False
    assert done["error_kind"] == "spec_invalid"
    assert done["fixable"] is False
    assert "Agent Fix" not in done["message"]
    assert prep == {}


def test_fix_on_a_loadable_record_keeps_the_review_precondition(
        client, user_token, monkeypatch, jobs, prep):
    r = _post_fix(client, user_token, monkeypatch, _record(LOADABLE), [])
    assert r.json()["status"] == "no_review"
    r = _post_fix(client, user_token, monkeypatch, _record(LOADABLE),
                  [{"verdict": "pass", "report": ""}])
    assert r.json()["status"] == "no_review"
    assert jobs == []


def test_fix_on_a_loadable_no_pass_record_seeds_the_normalized_bundle(
        client, user_token, monkeypatch, jobs, prep):
    r = _post_fix(client, user_token, monkeypatch, _record(LOADABLE), NO_PASS)
    assert r.json()["status"] == "started"
    list(jobs[-1]())
    assert isinstance(prep["bundle"], NormalizedBundle)


# -- Review ---------------------------------------------------------------


def test_review_of_a_namesake_record_reports_a_fixable_error(
        client, user_token, monkeypatch, jobs, prep):
    done = _review_done(client, user_token, monkeypatch, _record(NAMESAKE), jobs)
    assert done["error_kind"] == "spec_bundle_incomplete"
    assert done["fixable"] is True
    assert "conflicts with the different formal component" in done["errors"][0]
    assert "Agent Fix" in done["message"]
    # the old dead end is gone
    assert "run Fix and ask it" not in done["message"]


def test_review_of_a_declaration_record_reports_a_fixable_error(
        client, user_token, monkeypatch, jobs, prep):
    done = _review_done(client, user_token, monkeypatch, _record(DECLARATION), jobs)
    assert done["error_kind"] == "spec_needs_declaration"
    assert done["fixable"] is True
    assert any("omits `inputs:`" in e for e in done["errors"])
    # the card names the kind; the refusals ride in errors[] for Agent Fix
    assert done["message"] == SCHEMA_ERROR


def test_review_of_a_retired_syntax_record_is_not_fixable(
        client, user_token, monkeypatch, jobs, prep):
    done = _review_done(client, user_token, monkeypatch, _record(RETIRED), jobs)
    assert done["error_kind"] == "spec_invalid"
    assert done["fixable"] is False
    assert "Regenerate it from the Inspector" in done["message"]
    assert "Agent Fix" not in done["message"]


def test_fix_leaves_a_record_blocked_by_engine_alone(
        client, user_token, monkeypatch, jobs, prep):
    """An engine-blocked record's spec is complete: no spec edit resolves an
    engine refusal, so Agent Fix has nothing to do. Decided on the record's
    status — a NO_PASS review whose report says nothing of the engine does
    not open it either."""
    record = {**_record(LOADABLE), "status": "blocked_by_engine", "revision": 3}
    r = _post_fix(client, user_token, monkeypatch, record, NO_PASS)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "blocked_by_engine"
    assert jobs == []


def test_fix_reads_the_status_not_the_report_text(
        client, user_token, monkeypatch, jobs, prep):
    """A working candidate whose NO_PASS report carries the human
    blocked-by-engine line still opens Fix on that report: the line is
    display text."""
    from backend.spec_worker._prompts.review import BLOCKED_BY_ENGINE_LINE
    report = [{"verdict": "no_pass",
               "report": f"## Deterministic validator\n{BLOCKED_BY_ENGINE_LINE}\n"}]
    record = {**_record(LOADABLE), "status": "generated"}
    r = _post_fix(client, user_token, monkeypatch, record, report)
    assert r.json()["status"] == "started"
    list(jobs[-1]())
    assert prep["review_report"] == report[0]["report"]


def test_fix_opens_a_needs_repair_record_on_its_errors(
        client, user_token, monkeypatch, jobs, prep):
    """A re-check stored the errors the writer resolves: they are the work
    order, with no review needed."""
    record = {**_record(LOADABLE), "status": "needs_repair", "revision": 7,
              "repairable_errors": ["node `x`: fix the dims"]}
    r = _post_fix(client, user_token, monkeypatch, record, [])
    assert r.status_code == 200 and r.json()["status"] == "started"
    list(jobs[-1]())
    assert prep["work_order"] == ["node `x`: fix the dims"]
    assert prep["review_report"] == ""


@pytest.mark.parametrize("status", ["blocked_by_engine", "needs_repair"])
def test_review_refuses_a_pending_record(
        client, user_token, monkeypatch, jobs, prep, status):
    monkeypatch.setattr(_gens, "get", lambda rid: {**_record(LOADABLE), "status": status})
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 200
    assert r.json()["status"] == status
    assert jobs == []
