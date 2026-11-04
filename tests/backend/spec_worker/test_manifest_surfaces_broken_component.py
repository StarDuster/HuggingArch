"""A malformed component must be surfaced to the spec-writer agent, not
silently dropped from the manifest.

If a component YAML fails to parse, `read_all_components()` used to `continue`
past it — the agent would never learn the component exists, and might reach
for it blind or be unable to explain why a spec referencing it is rejected.
The manifest now lists it under a loud "FAILED TO LOAD" bucket instead.
"""
from backend.spec_worker._prompts import components as C


def test_broken_component_appears_in_manifest_with_a_warning(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "good.yaml").write_text(
        "role: attention\nname: good_attn\ndescription: a fine component\n"
    )
    # Unbalanced flow sequence — a genuine YAML parse error.
    (lib / "broken.yaml").write_text("role: attention\n  bad: [unclosed\n")

    monkeypatch.setattr(C, "_COMPONENTS_DIR", str(tmp_path))
    manifest = C.read_all_components()

    # The good one renders normally...
    assert "good_attn" in manifest
    # ...and the broken one is surfaced loudly, named, not swallowed.
    assert "FAILED TO LOAD" in manifest
    assert "broken.yaml" in manifest
    assert "could not parse" in manifest


def test_manifest_surfaces_merged_drafts_as_read_only_reuse(tmp_path, monkeypatch):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "merged.yaml").write_text(
        "role: attention\ndescription: a merged component\n", encoding="utf-8"
    )
    monkeypatch.setattr(C, "_COMPONENTS_DIR", str(tmp_path))

    manifest = C.read_all_components()

    assert "### drafts/" in manifest
    assert "merged" in manifest
    assert "Merged but not promoted" in manifest
    assert "Read-only" in manifest
    assert "`drafts:` opt-in" in manifest


def test_formal_base_inventory_names_reviewed_model_stems(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "qwen2.yaml").write_text(
        "match: {model_type: [qwen2]}\n", encoding="utf-8"
    )
    (models / "org__model.yaml").write_text(
        "match: {model_id: [org/model]}\n", encoding="utf-8"
    )
    monkeypatch.setattr(C, "_MODELS_DIR", str(models))

    inventory = C.read_formal_bases()

    assert "`qwen2` — representative identity: qwen2" in inventory
    assert "`org__model` — representative identity: org/model" in inventory
