"""The draft namespace is shared and cannot be claimed by silent override."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.arch.loader import SpecLoader, SpecLoadError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRE_NORM = (
    PROJECT_ROOT / "backend" / "arch" / "components" / "library" / "pre_norm.yaml"
).read_text(encoding="utf-8")


def _model(stem: str, model_id: str, block: str) -> str:
    return (
        "base: qwen2\n"
        f"drafts:\n  - {block}\n"
        f"match: {{model_id: [{model_id}]}}\n"
        f"components:\n  block: {block}\n"
    )


def test_same_structure_draft_is_reused_not_registered_twice(tmp_path):
    shared = tmp_path / "shared"
    (shared / "components" / "drafts").mkdir(parents=True)
    shared_draft = shared / "components" / "drafts" / "shared_pre_norm.yaml"
    shared_draft.write_text(PRE_NORM, encoding="utf-8")
    loader = SpecLoader(overlay_draft_dirs=[str(shared_draft.parent)])

    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    (run / "components" / "drafts").mkdir(parents=True)
    (run / "models" / "new-fork.yaml").write_text(
        _model("new-fork", "my-org/new-fork", "shared_pre_norm"),
        encoding="utf-8",
    )
    (run / "components" / "drafts" / "shared_pre_norm.yaml").write_text(
        PRE_NORM, encoding="utf-8"
    )

    assert loader.attach_overlay(str(run))
    assert loader._draft_sources["shared_pre_norm"] == str(shared_draft)
    config = json.loads(
        (PROJECT_ROOT / "backend/arch/snapshots/Qwen__Qwen3-8B.json").read_text()
    )["config"]
    assert loader.match(config, model_id="my-org/new-fork") is not None


def test_different_structure_conflict_rolls_the_whole_overlay_back(tmp_path):
    shared = tmp_path / "shared"
    (shared / "components" / "drafts").mkdir(parents=True)
    shared_draft = shared / "components" / "drafts" / "shared_pre_norm.yaml"
    shared_draft.write_text(PRE_NORM, encoding="utf-8")
    loader = SpecLoader(overlay_draft_dirs=[str(shared_draft.parent)])
    before = {
        "drafts": dict(loader._draft_lib),
        "sources": dict(loader._draft_sources),
        "signatures": dict(loader._draft_signatures),
        "specs": dict(loader._specs),
        "model_dirs": list(loader._overlay_model_dirs),
        "draft_dirs": list(loader._overlay_draft_dirs),
    }

    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    (run / "components" / "drafts").mkdir(parents=True)
    (run / "models" / "conflict.yaml").write_text(
        _model("conflict", "my-org/conflict", "shared_pre_norm"),
        encoding="utf-8",
    )
    conflicting = run / "components" / "drafts" / "shared_pre_norm.yaml"
    conflicting.write_text(
        PRE_NORM.replace("role: block", "role: aux_block"), encoding="utf-8"
    )

    with pytest.raises(SpecLoadError, match=r"different structure.*rename"):
        loader.attach_overlay(str(run))

    assert loader._draft_lib == before["drafts"]
    assert loader._draft_sources == before["sources"]
    assert loader._draft_signatures == before["signatures"]
    assert loader._specs == before["specs"]
    assert loader._overlay_model_dirs == before["model_dirs"]
    assert loader._overlay_draft_dirs == before["draft_dirs"]
    assert "conflict" not in loader._specs


def test_formal_name_still_cannot_be_shadowed_by_a_different_draft(tmp_path):
    loader = SpecLoader()
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    (run / "components" / "drafts").mkdir(parents=True)
    (run / "models" / "conflict.yaml").write_text(
        "base: qwen2\ndrafts: [pre_norm]\n"
        "match: {model_id: [my-org/conflict]}\n",
        encoding="utf-8",
    )
    (run / "components" / "drafts" / "pre_norm.yaml").write_text(
        PRE_NORM.replace("role: block", "role: aux_block"), encoding="utf-8"
    )

    with pytest.raises(SpecLoadError, match=r"library.*different structure"):
        loader.attach_overlay(str(run))


def test_an_unlisted_namesake_draft_is_part_of_no_spec(tmp_path):
    # A spec's `drafts:` list is what makes a draft file part of it. A
    # namesake no spec lists (the file a renamed draft left behind) neither
    # shadows the library nor stops the overlay; a spec that lists it is
    # still refused.
    loader = SpecLoader()
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    (run / "components" / "drafts").mkdir(parents=True)
    (run / "components" / "drafts" / "pre_norm.yaml").write_text(
        PRE_NORM.replace("role: block", "role: aux_block"), encoding="utf-8"
    )
    (run / "components" / "drafts" / "my_pre_norm.yaml").write_text(
        PRE_NORM, encoding="utf-8"
    )
    (run / "models" / "renamed.yaml").write_text(
        _model("renamed", "my-org/renamed", "my_pre_norm"), encoding="utf-8"
    )

    assert loader.attach_overlay(str(run))
    assert "renamed" in loader._specs
    assert "pre_norm" not in loader._draft_lib

    (run / "models" / "still_listed.yaml").write_text(
        _model("still_listed", "my-org/still-listed", "pre_norm"),
        encoding="utf-8",
    )
    with pytest.raises(SpecLoadError, match=r"library.*different structure"):
        SpecLoader().attach_overlay(str(run))
