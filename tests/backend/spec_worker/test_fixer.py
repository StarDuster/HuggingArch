"""Agent-fix runner: prompt work-order, bundle harvest, and the NO_PASS gate."""
from __future__ import annotations

import io
import json
from pathlib import Path

from backend._agent_runner import runner as R
from backend.spec_worker import fixer as FX
from backend.spec_worker import worker as W
from backend.spec_worker._prompts.fix import build_fix_task_body
from backend.spec_worker.bundle import FormalSource, normalize_bundle
from backend.spec_worker.fixer import _harvest_fixed_bundle
from backend.spec_worker.review_evidence import ReviewEvidence, SnapshotFacts


def test_fix_prompt_embeds_the_review_report_and_scopes_edits():
    report = "- **acme_linear_attention omits attn_variant**: add attn_variant: KDA"
    facts = SnapshotFacts.empty()
    body = build_fix_task_body(
        "acme/Foo-Flash", "match: {model_type: [acme_foo]}", [],
        {"valid": False, "errors": ["e1"]}, ReviewEvidence.audit(report, facts),
        facts=facts)
    # the report's finding reaches the fix run (untagged, so as a lead: its
    # title, not its suggested fix)
    assert "acme_linear_attention omits attn_variant" in body
    # scoped-edit contract: only this record's own model + drafts, minimal change
    assert "components/drafts" in body
    assert "no other models" in body
    assert "models/acme__Foo-Flash.yaml" in body
    assert "MINIMAL" in body
    # the deterministic validator result is folded in
    assert "valid: False" in body
    # the agent must NOT self-certify — a separate review does
    assert "do not emit a verdict" in body.lower()


_KDA_DRAFT = ("role: attention\nattn_variant: KDA\n"
              "graph:\n  o: {op: identity, in: x}\noutputs: [o]\n")


def test_harvest_reads_model_yaml_and_declared_drafts(tmp_path: Path):
    root = tmp_path / "overlay"
    (root / "models").mkdir(parents=True)
    (root / "components" / "drafts").mkdir(parents=True)
    (root / "models" / "glm5.yaml").write_text(
        "match: {model_type: [glm5]}\ndrafts: [glm5_kda]\n", encoding="utf-8")
    (root / "components" / "drafts" / "glm5_kda.yaml").write_text(
        _KDA_DRAFT, encoding="utf-8")
    # an UNdeclared draft in the same dir must NOT be harvested
    (root / "components" / "drafts" / "stray.yaml").write_text("x: 1\n", encoding="utf-8")

    seed = normalize_bundle(
        "glm5",
        "match: {model_type: [glm5]}\ndrafts: [glm5_kda]\n",
        [],
        FormalSource.from_deployment(),
        run_drafts={"glm5_kda": _KDA_DRAFT},
    )
    out = _harvest_fixed_bundle(str(root), seed)
    assert out is not None
    assert "model_type: [glm5]" in out.spec_yaml
    paths = {c["path"] for c in out.record_files()}
    assert paths == {"backend/arch/components/drafts/glm5_kda.yaml"}
    assert any("attn_variant: KDA" in c["body"] for c in out.record_files())


def test_harvest_none_when_no_model_yaml(tmp_path: Path):
    root = tmp_path / "empty"
    (root / "models").mkdir(parents=True)
    seed = normalize_bundle(
        "glm5", "match: {model_type: [glm5]}\n", [], FormalSource.from_deployment())
    assert _harvest_fixed_bundle(str(root), seed) is None


# -- a record whose saved spec does not load --------------------------------

_LOAD_ERROR = ("draft component 'same' for 'acme/Foo' conflicts with the "
               "different formal component of the same name; rename the draft "
               "and update the spec, or reuse the formal component")


def test_fix_prompt_on_load_refusals_alone_makes_them_the_work_order():
    facts = SnapshotFacts.empty()
    body = build_fix_task_body(
        "acme/Foo", "drafts: [same]\n", [], None, None, facts=facts,
        load_errors=[_LOAD_ERROR])
    assert "cannot be loaded under the current schema" in body
    assert "## Load refusals" in body
    assert f"- {_LOAD_ERROR}" in body
    assert "seeded exactly as it was saved" in body
    # no review was run, so none is quoted or claimed
    assert "The review's findings" not in body
    assert "NO_PASS" not in body.split("## Load refusals")[0]


def test_fix_prompt_carries_load_refusals_before_the_review():
    facts = SnapshotFacts.empty()
    report = "- **a defect**: fix it"
    body = build_fix_task_body(
        "acme/Foo", "drafts: [same]\n", [], None,
        ReviewEvidence.audit(report, facts), facts=facts,
        load_errors=[_LOAD_ERROR])
    assert "and a prior adversarial review returned **NO_PASS**" in body
    assert body.index("## Load refusals") < body.index("## The review's findings")
    assert "a defect" in body


def _component(bias: str, comment: str = "") -> str:
    return (f"{comment}role: attention\nparams:\n  inner: component\n"
            f"graph:\n  out:\n    op: inner\n    in: x\n    dims:\n"
            f"      bias: \"{bias}\"\noutputs: [out]\n")


def _library_source():
    from backend.arch.component_signature import component_structure_signature
    from backend.spec_worker.bundle import FormalComponent
    body = _component("Qb")
    return body, FormalSource(models={}, components={"same": FormalComponent(
        tier="library", body=body,
        signature=component_structure_signature(body, path="library/same.yaml"))})


def test_prepare_fix_stages_a_blocked_record_as_saved_and_harvest_rechecks_it(
        monkeypatch, tmp_path: Path):
    """A record shaped like this: an agent base, a draft saved under the retired root
    that clashes with a library namesake. Fix stages every file as saved, the
    prompt's work order is the refusal, and the harvest refuses the same
    record until the agent has renamed the draft."""
    import pytest
    from backend.spec_worker import fixer
    from backend.spec_worker.bundle import BlockedBundle, SpecBundleError, open_for_fix

    _library, source = _library_source()
    monkeypatch.setattr(FormalSource, "from_deployment",
                        classmethod(lambda cls, *a, **k: source))
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(fixer, "_build_prompt_context",
                        lambda model_id: ("sys", "gt", {"_snapshot_data": None}))
    monkeypatch.setattr(fixer, "_stage_source", lambda *a: [])
    monkeypatch.setattr(fixer, "_stage_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(fixer, "_invoke_validate", lambda *a, **k: {"valid": False})
    monkeypatch.setattr(fixer, "_code_revision", lambda: "sha")

    draft = _component("1", comment="# kept comment\n")
    spec = "# citation\nbase: parent\nmatch: {model_id: [acme/Foo]}\n"
    files = [
        {"path": "backend/arch/models/parent.yaml", "body": "drafts: [same]\n"},
        {"path": "backend/arch_spec/components/drafts/same.yaml", "body": draft},
    ]
    bundle = open_for_fix("acme/Foo", spec, files, source)
    assert isinstance(bundle, BlockedBundle)
    assert bundle.errors == (_LOAD_ERROR,)

    prep = fixer.prepare_fix("acme/Foo", record_id="rec-1", fixer_id="u",
                             bundle=bundle)
    root = Path(prep["target_root"])
    identity = (root / "models" / "acme__Foo.yaml").read_text()
    assert identity.startswith("# citation\nbase: parent\n")  # not flattened
    assert (root / "models" / "parent.yaml").read_text() == "drafts: [same]\n"
    assert (root / "components" / "drafts" / "same.yaml").read_text() == draft
    assert f"- {_LOAD_ERROR}" in prep["prompt"]
    assert "The review's findings" not in prep["prompt"]
    assert prep["seed_bundle"] is bundle

    # An agent that changed nothing: the harvest refuses the record again.
    with pytest.raises(SpecBundleError, match="conflicts with the different"):
        fixer._harvest_fixed_bundle(str(root), bundle)

    # The agent renames the draft in the base's list: the record loads.
    (root / "components" / "drafts" / "foo_same.yaml").write_text(draft)
    (root / "models" / "parent.yaml").write_text("drafts: [foo_same]\n")
    fixed = fixer._harvest_fixed_bundle(str(root), bundle)
    assert fixed is not None
    assert [f.path for f in fixed.serving_files] == [
        "backend/arch/components/drafts/foo_same.yaml"]


def test_renamed_namesake_draft_passes_the_fix_gate_with_the_old_file_left(
        monkeypatch, tmp_path: Path):
    """End to end through the verdict gate's validate
    subprocess: a draft saved under the retired root with a library name and
    a different structure. The agent cannot delete files, so after its rename
    the old file stays in the overlay; the spec's `drafts:` list no longer
    names it, so it is part of nothing and the gate passes."""
    import io
    import shutil
    import time

    from backend.spec_worker.bundle import BlockedBundle, open_for_fix
    from backend.spec_worker.overlay import seed_fix_overlay
    from backend.spec_worker.worker import _verify_spec_output

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    arch = Path(__file__).resolve().parents[3] / "backend" / "arch"
    library = (arch / "components" / "library" / "swiglu_ffn.yaml").read_text()
    draft = library.replace("in: [silu, up_proj]", "in: [up_proj, silu]")
    assert draft != library
    spec = ("base: qwen3\nmatch: {model_id: [acme/Foo]}\n"
            "drafts: [swiglu_ffn]\n"
            "components:\n  attention: gqa_qknorm_attention\n"
            "  ffn_dense: swiglu_ffn\n  norm: rmsnorm\n  block: pre_norm\n")
    bundle = open_for_fix(
        "acme/Foo", spec,
        [{"path": "backend/arch_spec/components/drafts/swiglu_ffn.yaml",
          "body": draft}],
        FormalSource.from_deployment())
    assert isinstance(bundle, BlockedBundle)

    root = Path(seed_fix_overlay("u", "rec-1", "acme/Foo", bundle))
    (root / "snapshots").mkdir(exist_ok=True)
    shutil.copy(arch / "snapshots" / "Qwen__Qwen3-8B.json",
                root / "snapshots" / "acme__Foo.json")
    drafts = root / "components" / "drafts"
    assert (drafts / "swiglu_ffn.yaml").read_text() == draft

    def gate():
        downgrade, _ = _verify_spec_output(
            "acme/Foo", io.StringIO(), time.time() - 5, root=str(root),
            seeded=True)
        return downgrade

    refused = gate()
    assert refused is not None and "swiglu_ffn" in refused.reason

    # The agent's rename: a new draft file, the new name in `drafts:` and in
    # the reference. Its tools write and edit; the old file stays.
    (drafts / "acme_swiglu_ffn.yaml").write_text(draft)
    (root / "models" / "acme__Foo.yaml").write_text(
        spec.replace("swiglu_ffn", "acme_swiglu_ffn"))
    assert (drafts / "swiglu_ffn.yaml").is_file()

    assert gate() is None
    fixed = _harvest_fixed_bundle(str(root), bundle)
    assert [f.path for f in fixed.serving_files] == [
        "backend/arch/components/drafts/acme_swiglu_ffn.yaml"]


# -- a NO_PASS review whose report orders nothing ---------------------------

_STUB_REPORT = "VERDICT: NO_PASS\n\n## Agent review **"


def test_a_review_with_no_checked_finding_hands_the_work_order_to_the_checks():
    """A review that is a validator-failure stub: the latest NO_PASS
    named nothing that checked out. The prompt says so, and the
    validator errors and load refusals of this run are the work order."""
    facts = SnapshotFacts.empty()
    stub = ReviewEvidence.audit(_STUB_REPORT, facts)
    assert not stub.gives_work_order()
    invalid = {"valid": False, "errors": ["D3 model.visual.blocks has no inputs:"]}

    body = build_fix_task_body("acme/Foo", "a: 1", [], invalid, stub,
                               facts=facts, load_errors=[_LOAD_ERROR])
    assert "No finding of this review checked out" in body
    assert ("The work order is the load refusals above and the errors in the "
            "deterministic validator result below") in body
    assert "EXACTLY the defects\nthe load refusals and the validator named" in body

    body = build_fix_task_body("acme/Foo", "a: 1", [], invalid, stub,
                               facts=facts)
    assert ("The work order is the errors in the deterministic validator "
            "result below") in body

    # Nothing failed and nothing checked out: leads only, else no edit.
    body = build_fix_task_body("acme/Foo", "a: 1", [], {"valid": True}, stub,
                               facts=facts)
    assert "make no edits and stop" in body
    assert "EXACTLY the defects\nyou confirm among the review's leads" in body



def test_fix_prompt_on_a_recheck_makes_its_errors_the_work_order():
    facts = SnapshotFacts.empty()
    body = build_fix_task_body(
        "acme/Foo", "match: {}\n", [], {"valid": False, "errors": ["e1"]}, None,
        facts=facts, validation_errors=["node `x`: fix the dims"])
    assert "A re-check found validation errors" in body
    assert "- node `x`: fix the dims" in body
    assert "NO_PASS" not in body.split("## Validation errors")[0]


# -- the fix run's verdict and write-back ------------------------------------

class _Proc:
    def __init__(self):
        self._lines = ['{"raw": 1}\n']
        self.stderr = io.StringIO("")
        self.pid, self.returncode = 1, 0
        self.stdout = self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


class _Backend:
    supports_resume = False

    def spawn(self, prompt, env=None, workspace=None):
        return _Proc()

    def extract_session_id(self, evt):
        return None

    def parse_event(self, evt):
        return [{"type": "done", "success": True}]


class _Bundle:
    model_id = "acme/Foo"
    spec_yaml = "fixed"

    def record_files(self, include_proposals=True):
        return [{"path": "p", "body": "b"}]


def _fix_done(monkeypatch, tmp_path, verdict, *, status, writes=None):
    monkeypatch.setattr(R, "get_backend", lambda cli: _Backend())
    for name in ("bind_job_log", "release", "_stage_snapshot", "_upload_run_log"):
        monkeypatch.setattr(FX, name, lambda *a, **k: None)
    monkeypatch.setattr(FX, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(FX, "_verify_spec_output", lambda *a, **k: (verdict, {}))
    monkeypatch.setattr(FX, "_harvest_fixed_bundle", lambda *a, **k: _Bundle())
    monkeypatch.setattr(FX, "_max_repair_rounds", lambda: 0)
    from backend import supabase_env
    from backend.service import spec_generations as G
    monkeypatch.setattr(supabase_env, "configured", lambda: True)
    calls = writes if writes is not None else {}

    def promote(rid, **kw):
        calls["promote"] = (rid, kw)
        return calls.get("answer", {"outcome": "recorded", "id": rid})
    monkeypatch.setattr(G, "promote_pending", promote)
    monkeypatch.setattr(G, "update_spec", lambda rid, **kw: calls.setdefault("update", (rid, kw)) and True)
    prep = {"prompt": "p", "target_root": str(tmp_path), "seed_bundle": _Bundle(),
            "record_id": "rec-1", "record_status": status, "record_revision": 3,
            "agent_cli": "claude"}
    frames = list(FX.stream_fix("acme/Foo", prep, run_key="fix|u|rec-1"))
    events = [json.loads(f[6:]) for f in frames if f.startswith("data: ")]
    (done,) = [e for e in events if e.get("type") == "done"]
    return done, calls


def test_a_fix_the_engine_blocks_carries_its_gaps_and_writes_nothing(monkeypatch, tmp_path):
    gap = {"capability": "gather extent", "node": "n", "site": "s", "exception": "E",
           "deployments": ["tp8"], "stages": ["decode"], "detail": "d"}
    verdict = W.Downgrade("blocked_by_engine", "blocked", ("d",),
                          error_kind=W.RunErrorKind.BLOCKED_BY_ENGINE, engine_gaps=(gap,))
    done, calls = _fix_done(monkeypatch, tmp_path, verdict, status="needs_repair")
    assert done["success"] is False
    assert done["error_kind"] == "blocked_by_engine" and done["engine_gaps"] == [gap]
    assert "promote" not in calls and "update" not in calls


def test_a_fix_of_a_needs_repair_record_lands_through_the_pending_promotion(monkeypatch, tmp_path):
    done, calls = _fix_done(monkeypatch, tmp_path, None, status="needs_repair")
    assert done["success"] is True
    rid, kw = calls["promote"]
    assert rid == "rec-1" and kw["expected_revision"] == 3
    assert kw["spec_yaml"] == "fixed" and kw["run_started_at"]
    assert done["record"] == {"outcome": "recorded", "id": "rec-1"}
    assert "update" not in calls


def test_a_fix_whose_pending_record_changed_meanwhile_fails(monkeypatch, tmp_path):
    done, _ = _fix_done(monkeypatch, tmp_path, None, status="needs_repair",
                            writes={"answer": {"outcome": "conflict"}})
    assert done["success"] is False
    assert "changed during the fix" in done["message"]
    assert done["record"] == {"outcome": "conflict"}


def test_a_fix_of_a_working_candidate_rewrites_it_in_place(monkeypatch, tmp_path):
    done, calls = _fix_done(monkeypatch, tmp_path, None, status="generated")
    assert done["success"] is True
    assert calls["update"][0] == "rec-1" and "promote" not in calls


def test_a_fix_stores_the_snapshot_it_ran_against(monkeypatch, tmp_path):
    """The record keeps the checkpoint facts the fix judged the spec on, so a
    record saved before snapshots carried the verbatim header gets one."""
    snap = {"config": {"model_type": "foo"}, "metadata": {"header": {"groups": []}}}
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "acme__Foo.json").write_text(json.dumps(snap))
    done, calls = _fix_done(monkeypatch, tmp_path, None, status="generated")
    assert done["success"] is True
    assert calls["update"][1]["snapshot_json"] == snap
