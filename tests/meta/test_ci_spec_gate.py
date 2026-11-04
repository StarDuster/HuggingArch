"""The spec-only CI classifier stays a closed allowlist.

A contributed spec is YAML + snapshot (+ optional drafts). If this
allowlist grows to library components, dims.yaml, or Python, a spec PR
starts skipping the suite that actually covers those files.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _gate():
    path = REPO / "scripts" / "ci_spec_gate.py"
    spec = importlib.util.spec_from_file_location("ci_spec_gate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_a_contributed_spec_bundle_is_spec_only():
    g = _gate()
    assert g.is_spec_only([
        "backend/arch/models/zai-org__GLM-4.5.yaml",
        "backend/arch/snapshots/zai-org__GLM-4.5.json",
        "backend/arch/components/drafts/gqa_qknorm_partial_rope_bias_attention.yaml",
    ])


def test_supported_models_txt_alone_is_spec_only():
    g = _gate()
    assert g.is_spec_only(["backend/arch/supported_models.txt"])


def test_empty_diff_is_not_spec_only():
    g = _gate()
    assert not g.is_spec_only([])


def test_library_component_is_not_spec_only():
    g = _gate()
    assert not g.is_spec_only([
        "backend/arch/models/x.yaml",
        "backend/arch/components/library/mla_attention.yaml",
    ])


def test_python_or_tests_force_the_full_suite():
    g = _gate()
    spec = [
        "backend/arch/models/x.yaml",
        "backend/arch/snapshots/x.json",
    ]
    assert not g.is_spec_only([*spec, "backend/arch/ops.py"])
    assert not g.is_spec_only([*spec, "tests/backend/arch/test_loader.py"])
    assert not g.is_spec_only([*spec, "backend/arch/dims.yaml"])
    assert not g.is_spec_only([*spec, "web/src/App.tsx"])


def test_model_id_comes_from_the_snapshot_stem():
    g = _gate()
    ids = g.model_ids_to_validate(
        ["backend/arch/snapshots/Qwen__Qwen3-8B.json"],
        root=REPO,
    )
    assert ids == ["Qwen/Qwen3-8B"]


def test_changed_draft_expands_to_every_referer(tmp_path):
    g = _gate()
    arch = tmp_path / "backend" / "arch"
    (arch / "models").mkdir(parents=True)
    (arch / "components" / "drafts").mkdir(parents=True)
    (arch / "snapshots").mkdir(parents=True)
    for stem, model_id, drafts in (
        ("org__first", "org/first", ["shared_attention"]),
        ("org__second", "org/second", ["other", "shared_attention"]),
        ("org__unrelated", "org/unrelated", ["other"]),
    ):
        (arch / "models" / f"{stem}.yaml").write_text(
            f"match: {{model_id: [{model_id}]}}\ndrafts: {drafts}\n",
            encoding="utf-8",
        )
        (arch / "snapshots" / f"{stem}.json").write_text("{}", encoding="utf-8")

    ids = g.models_to_validate_for_paths([
        "backend/arch/components/drafts/shared_attention.yaml",
    ], root=tmp_path)

    assert set(ids) == {"org/first", "org/second"}


def test_changed_base_expands_through_identity_base_chains(tmp_path):
    g = _gate()
    arch = tmp_path / "backend" / "arch"
    (arch / "models").mkdir(parents=True)
    (arch / "snapshots").mkdir(parents=True)
    (arch / "models" / "family.yaml").write_text(
        "match: {model_type: [family]}\n", encoding="utf-8")
    (arch / "models" / "org__variant.yaml").write_text(
        "base: family\nmatch: {model_id: [org/variant]}\n", encoding="utf-8")
    (arch / "models" / "unrelated.yaml").write_text(
        "match: {model_id: [org/unrelated]}\n", encoding="utf-8")
    for stem in ("org__variant", "org__unrelated"):
        (arch / "snapshots" / f"{stem}.json").write_text("{}", encoding="utf-8")

    ids = g.models_to_validate_for_paths([
        "backend/arch/models/family.yaml",
    ], root=tmp_path)

    assert ids == ["org/variant"]


def test_workflow_wires_the_spec_gate():
    text = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "scripts/ci_spec_gate.py" in text
    assert "spec_only" in text
    assert "Run spec gate" in text


def test_schema_and_retired_parsing_require_the_stored_spec_check():
    g = _gate()
    assert g.requires_stored_spec_check(["backend/arch/schema/model.py"])
    assert g.requires_stored_spec_check(["backend/arch/spec_parsing.py"])
    assert not g.requires_stored_spec_check(["backend/arch/models/x.yaml"])


def test_workflow_wires_the_stored_spec_check():
    text = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "stored-spec-check" in text
    assert "tools/check_stored_specs.py" in text
    template = (REPO / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    assert "check_stored_specs.py" in template


def test_a_diff_that_adds_a_snapshot_runs_the_full_suite():
    """A new model owes a key to every corpus pin; only the full suite
    compares them all."""
    g = _gate()
    bundle = [
        "backend/arch/models/org__new.yaml",
        "backend/arch/snapshots/org__new.json",
    ]
    assert not g.is_spec_only(bundle, added=["backend/arch/snapshots/org__new.json"])
    # the same paths as a modification stay spec-only
    assert g.is_spec_only(bundle, added=["backend/arch/models/org__new.yaml"])


def test_workflow_runs_the_corpus_pin_verify_on_every_lane():
    text = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "--added-file /tmp/added.txt" in text
    assert "--diff-filter=AR" in text
    verify = text.index("python -m tools.corpus_admit verify\n")
    step = text.rfind("- name:", 0, verify)
    assert "if:" not in text[step:verify], "verify runs on every lane"
    assert "python -m tools.corpus_admit verify --changed --paths-file /tmp/changed.txt" in text
