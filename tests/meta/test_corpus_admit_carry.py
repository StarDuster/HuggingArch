"""What the service passes the admitter (ADMIT §4.1, §4.4, §5.5): the base's
lock hash and the deployment's environment for a tree without `.git`, the
base kind, the interpreter the write entry accepts, and the carry of a
maintainer's decisions from a PR branch onto a new base."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools import corpus_admit
from tools.corpus_pins import env as env_mod
from tools.corpus_pins.carry import ARTIFACTS, carry
from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.states import Status

_STEM = "meta-llama__Llama-3.2-1B"
_TP = "tests/golden/pins/tp_split_decisions.json"
_REC = "tests/golden/pins/checkpoint_reconciliation.json"
_COPY = "tests/golden/pins/copy_contracts.json"


def _lock() -> str:
    return hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()


def _admission(**kw) -> corpus_admit.Admission:
    return corpus_admit.Admission(_STEM.replace("__", "/"), corpus_admit.PR,
                                  base_sha="0" * 40, **kw)


def test_a_tree_without_git_names_its_base_lock(monkeypatch):
    # `git show <sha>:uv.lock` cannot answer for an unknown commit; the
    # service passes the hash of the tree's own lock instead.
    monkeypatch.setattr(env_mod, "python_version", lambda: env_mod.ADMIT_PYTHON)
    run = _admission(base_lock_sha256=_lock())
    run.fix_inputs()
    assert not any("uv.lock" in g for g in run.gates.g0)
    run = _admission()
    run.fix_inputs()
    assert "write refused: the base's uv.lock cannot be read" in run.gates.g0


def test_a_base_lock_other_than_the_environments_refuses_the_write(monkeypatch):
    monkeypatch.setattr(env_mod, "python_version", lambda: env_mod.ADMIT_PYTHON)
    run = _admission(base_lock_sha256="f" * 64)
    run.fix_inputs()
    assert "write refused: uv.lock differs from the base's uv.lock" in run.gates.g0


def test_the_write_entry_refuses_another_interpreter(monkeypatch):
    monkeypatch.setattr(env_mod, "python_version", lambda: "3.10")
    run = _admission(base_lock_sha256=_lock())
    run.fix_inputs()
    assert f"write refused: interpreter 3.10 is not ADMIT_PYTHON {env_mod.ADMIT_PYTHON}" \
        in run.gates.g0
    dry = corpus_admit.Admission(_STEM.replace("__", "/"), corpus_admit.DRY_RUN,
                                 base_sha="0" * 40, base_lock_sha256=_lock())
    dry.fix_inputs()
    assert dry.gates.g0 == [] and dry.preview is True


def test_the_environment_is_the_deployments_and_the_kind_is_reported(monkeypatch, tmp_path):
    seen = []
    real = env_mod.Environment.here
    monkeypatch.setattr(corpus_admit.Environment, "here",
                        classmethod(lambda cls, root: seen.append(root) or real(PROJECT_ROOT)))
    run = _admission(base_lock_sha256=_lock(), env_root=tmp_path, base_kind="deployment",
                     orchestrator="a" * 40)
    candidate = run.fix_inputs()
    assert seen == [tmp_path]
    report = run.report(candidate, Status.UNAVAILABLE, None, [], [], {}, [], [])
    assert report["base"] == {"sha": "0" * 40, "kind": "deployment"}
    assert report["orchestrator"] == "a" * 40


def test_the_cli_takes_the_service_flags(monkeypatch, tmp_path):
    captured = {}

    class Fake:
        def __init__(self, model_id, mode, **kw):
            captured.update(kw, model_id=model_id, mode=mode)

        def run(self):
            return {"status": "ready"}
    monkeypatch.setenv(corpus_admit.CHILD_ENV, "1")
    monkeypatch.setattr(corpus_admit, "Admission", Fake)
    code = corpus_admit.main(["acme/M", "--dry-run", "--base-sha", "b" * 40,
                              "--base-kind", "deployment", "--base-lock-sha256", "c" * 64,
                              "--env-root", str(tmp_path), "--orchestrator", "d" * 40,
                              "--report", str(tmp_path / "r.json")])
    assert code == 0
    assert captured["base_kind"] == "deployment" and captured["base_lock_sha256"] == "c" * 64
    assert captured["env_root"] == tmp_path and captured["orchestrator"] == "d" * 40
    assert captured["mode"] == corpus_admit.DRY_RUN


# ── carry ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sides(tmp_path):
    """(old, new, tree): three copies of the decision files."""
    roots = []
    for name in ("old", "new", "tree"):
        root = tmp_path / name
        for rel in ARTIFACTS:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / rel, root / rel)
        roots.append(root)
    return roots


def _edit(root: Path, rel: str, fn) -> None:
    path = root / rel
    doc = json.loads(path.read_text())
    doc = fn(doc) or doc
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def test_unchanged_branch_carries_nothing(sides):
    old, new, tree = sides
    assert carry(CorpusModel(_STEM), old, new, tree) == ({}, [])


def test_the_models_own_decisions_move_onto_the_new_base(sides):
    old, new, tree = sides
    entry = {"(backbone)": {"verdict": "framework_split", "decided_by": "vllm", "source": "maintainer",
                              "nodes": None}}
    _edit(new, _TP, lambda d: d.__setitem__(_STEM, entry))
    _edit(new, _REC, lambda d: d["KNOWN"].__setitem__(_STEM, {"note": "why"}))
    # the new base moved on: another model's entry was added on main
    _edit(tree, _TP, lambda d: d.__setitem__("org__Other", {}))
    carried, problems = carry(CorpusModel(_STEM), old, new, tree)
    assert problems == []
    assert carried == {_TP: [_STEM], _REC: [f"KNOWN/{_STEM}"]}
    doc = json.loads((tree / _TP).read_text())
    assert doc[_STEM] == entry and doc["org__Other"] == {}
    assert json.loads((tree / _REC).read_text())["KNOWN"][_STEM] == {"note": "why"}
    assert (tree / _TP).read_text() == json.dumps(doc, indent=2, sort_keys=True) + "\n"


def test_a_change_to_another_models_entry_is_refused_and_writes_nothing(sides):
    old, new, tree = sides
    before = (tree / _TP).read_text()
    _edit(new, _TP, lambda d: d.__setitem__(_STEM, {}))
    _edit(new, _COPY, lambda d: d.append(["components/library/moe_ffn.yaml", "x", "instances",
                                          "unrolled", None, None]))
    carried, problems = carry(CorpusModel(_STEM), old, new, tree)
    assert carried == {} and len(problems) == 1
    assert _COPY in problems[0] and "moe_ffn.yaml" in problems[0]
    assert (tree / _TP).read_text() == before


def test_a_copy_contract_of_the_models_own_file_is_carried(sides):
    old, new, tree = sides
    own = f"models/{CorpusModel(_STEM).spec_file().name}"
    row = [own, "x", "instances", "unrolled", None, None]
    _edit(new, _COPY, lambda d: d.append(row))
    carried, problems = carry(CorpusModel(_STEM), old, new, tree)
    assert problems == [] and carried == {_COPY: [json.dumps(row)]}
    rows = json.loads((tree / _COPY).read_text())
    assert row in rows and rows == sorted(rows, key=json.dumps)


def test_a_changed_path_that_is_no_decision_file_is_refused(sides):
    old, new, tree = sides
    (new / "backend/arch/models").mkdir(parents=True)
    (new / "backend/arch/models/x.yaml").write_text("edited\n")
    carried, problems = carry(CorpusModel(_STEM), old, new, tree)
    assert carried == {} and problems == ["backend/arch/models/x.yaml: not a decision file"]
