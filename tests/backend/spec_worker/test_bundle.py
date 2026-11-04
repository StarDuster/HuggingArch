"""Bundle provenance: one identity, self-contained serving files."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.arch.component_signature import component_structure_signature
from backend.arch.loader import SpecLoader, arch_fingerprint
from backend.spec_worker.bundle import (
    BundleFile,
    FormalComponent,
    FormalSource,
    SpecBundleError,
    normalize_bundle,
)


def _component(*, bias: str = "Qb", description: str = "one") -> str:
    return f"""# a formatting-only comment
role: attention
description: {description}
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


def _source(models=None, components=None) -> FormalSource:
    return FormalSource(
        models=dict(models or {}),
        components={
            name: FormalComponent(
                tier=tier,
                body=body,
                signature=component_structure_signature(body, path=f"{tier}/{name}.yaml"),
            )
            for name, tier, body in (components or [])
        },
    )


def _document(text: str) -> dict:
    return yaml.safe_load(text)


def test_structure_signature_ignores_prose_and_sees_binding_changes():
    base = _component()
    assert component_structure_signature(base) == component_structure_signature(
        base.replace("# a formatting-only comment", "# another comment")
        .replace("description: one", "description: two")
    )
    assert component_structure_signature(base) != component_structure_signature(
        _component(bias="1"))
    assert component_structure_signature(base) != component_structure_signature(
        base.replace("params:\n  inner: component", "params:\n  inner: dim"))
    assert component_structure_signature(base) != component_structure_signature(
        base.replace("op: inner", "op: other"))
    with pytest.raises(ValueError, match="component graph must be a mapping"):
        component_structure_signature("graph: [not-a-map]\n")

    with pytest.raises(ValueError, match="component document"):
        component_structure_signature("- not\n- a mapping\n")


def test_agent_chain_flattens_to_the_first_formal_base():
    source = _source(models={"qwen2": "model_type: qwen2\n"})
    bundle = normalize_bundle(
        "acme/Fork",
        "base: agent_two\nmatch: {model_id: acme/Fork}\nchild: yes\n",
        [],
        source,
        run_models={
            "agent_two": "base: agent_one\nchild: no\nparent: yes\n",
            "agent_one": "base: qwen2\nparent: yes\n",
        },
    )
    doc = _document(bundle.spec_yaml)
    assert doc["base"] == "qwen2"
    assert doc["child"] is True
    assert doc["parent"] is True
    assert bundle.record_files() == []
    assert any(note.startswith("flattened agent base chain") for note in bundle.notes)


def test_private_base_with_a_formal_name_loses_to_the_formal_layer():
    source = _source(models={"qwen2": "model_type: qwen2\n"})
    bundle = normalize_bundle(
        "acme/Fork", "base: qwen2\nmatch: {model_id: acme/Fork}\n", [], source,
        run_models={"qwen2": "match: private\n"})
    assert _document(bundle.spec_yaml)["base"] == "qwen2"
    assert all(file.path != "backend/arch/models/qwen2.yaml"
               for file in bundle.serving_files)


def test_no_formal_base_flattens_losslessly(tmp_path):
    formal_qwen = (Path(__file__).resolve().parents[3] / "backend" / "arch" / "models"
                   / "qwen2.yaml").read_text()
    source = FormalSource(models={}, components={})
    bundle = normalize_bundle(
        "acme/Fork",
        "base: fork_base\nmatch:\n  model_id: [acme/Fork]\n",
        [],
        source,
        run_models={"fork_base": formal_qwen},
    )
    identity = tmp_path / "acme__Fork.yaml"
    identity.write_text(bundle.spec_yaml, encoding="utf-8")
    normalized = SpecLoader().load_model_spec(identity)
    expected = SpecLoader().get("qwen2")
    assert arch_fingerprint(normalized) == arch_fingerprint(expected)
    assert "base" not in _document(bundle.spec_yaml)


_WRITTEN = """\
# MTP layer: modeling_mimo.py MiMoMTPLayers.forward
match: {model_id: [acme/Fork]}
blocks:
  mtp_block:
    graph:
      token_layernorm: {op: rmsnorm, in: x, dim: d_model}
      self_attn: {op: gqa_attention, in: token_layernorm}
      attn_residual: {op: add, in: [x, self_attn], dim: d_model}
    outputs: [attn_residual]
"""


def test_an_unchanged_spec_keeps_its_authored_text():
    """Comments are the author's citations; a spec normalization has nothing
    to rewrite in is stored exactly as written."""
    bundle = normalize_bundle("acme/Fork", _WRITTEN, [], _source())
    assert bundle.spec_yaml == _WRITTEN


def test_a_rewritten_spec_keeps_its_written_order():
    """A spec normalization does rewrite (here: a flattened agent base) is
    re-dumped in written order, so a graph still reads as its forward pass —
    producer before consumer — rather than alphabetically."""
    bundle = normalize_bundle(
        "acme/Fork", "base: agent\n", [], _source(),
        run_models={"agent": _WRITTEN})
    graph = list(_document(bundle.spec_yaml)["blocks"]["mtp_block"]["graph"])
    assert graph == ["token_layernorm", "self_attn", "attn_residual"]
    again = normalize_bundle(
        "acme/Fork", "base: agent\n", [], _source(),
        run_models={"agent": _WRITTEN})
    assert again.spec_yaml == bundle.spec_yaml


def test_a_renamed_spec_is_written_like_a_normalized_one():
    """A component rename is a rewrite too: it keeps the written text around
    the renamed entry, so normalizing it afterwards changes nothing and the
    text keeps its lists inline."""
    from backend.spec_worker.bundle import rename_component_references
    renamed = rename_component_references(
        "match: {model_id: [acme/Fork]}\ncomponents: {attention: toy}\n",
        {"toy": "main"})
    assert "model_id: [acme/Fork]" in renamed
    assert "attention: main" in renamed
    assert list(_document(renamed)) == ["match", "components"]
    assert normalize_bundle("acme/Fork", renamed, [], _source()).spec_yaml == renamed


_MIGRATED = """\
# Qwen3: modeling_qwen3.py Qwen3ForCausalLM
match: {model_id: [acme/Fork]}
model:
  # nn.Embedding(config.vocab_size, config.hidden_size)
  - {name: embed_tokens, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: toy}  # Qwen3DecoderLayer x N
  - name: lm_head
    op: linear  # lm_head = nn.Linear(hidden, vocab, bias=False)
    boundary_role: output_head
    dims: {in: d_model, out: n_vocab}
    tie_weights: config.tie_word_embeddings  # _tied_weights_keys
"""


def _comment_lines(text: str) -> list[str]:
    return [line[line.index("#"):] for line in text.splitlines() if "#" in line]


def test_a_migrated_spec_keeps_its_comments():
    """Migration rewrites parsed data; the rewritten spec goes back into the
    written text, so every citation survives, and normalizing the result
    again is the identity."""
    bundle = normalize_bundle("acme/Fork", _MIGRATED, [], _source())
    assert bundle.spec_yaml != _MIGRATED
    assert "tied_weight" in _document(bundle.spec_yaml)["model"][2]
    assert _comment_lines(bundle.spec_yaml) == _comment_lines(_MIGRATED)
    assert "  - {name: layers, repeat: n_layers, template: toy, inputs: [embed_tokens]" \
        in bundle.spec_yaml
    again = normalize_bundle("acme/Fork", bundle.spec_yaml, [], _source())
    assert again.spec_yaml == bundle.spec_yaml


def test_a_renamed_spec_keeps_its_comments():
    from backend.spec_worker.bundle import rename_component_references
    written = _MIGRATED.replace("template: toy", "template: toy_v1")
    renamed = rename_component_references(written, {"toy_v1": "toy"})
    assert renamed == _MIGRATED


def test_a_flattened_and_migrated_spec_keeps_each_comment_on_its_key():
    """A flattened base adds inherited keys and a migration replaces a
    retired one: the comment on the retired key goes to its replacement,
    the one on `base:` stays where it was written, and the inherited keys
    take none."""
    parent = """\
match: {model_type: [x]}
parent: yes
model:
  - {name: embed, op: embedding, boundary_role: input_embedding}
  - {name: out, op: linear, boundary_role: output_head, inputs: [embed]}
"""
    child = """\
# about inherited base
base: agent
# about ignored config
ignored_config_fields:
  - sliding_window  # no window in this model
match: {model_id: acme/Fork}
child: yes
"""
    bundle = normalize_bundle("acme/Fork", child, [], _source(),
                              run_models={"agent": parent})
    lines = bundle.spec_yaml.splitlines()
    assert lines[:3] == ["# about inherited base",
                         "match: {model_id: acme/Fork}", "parent: true"]
    at = lines.index("# about ignored config")
    assert lines[at - 1] == "child: yes"
    assert lines[at + 1] == "# no window in this model"
    assert lines[at + 2] == "validation_exemptions:"
    again = normalize_bundle("acme/Fork", bundle.spec_yaml, [], _source())
    assert again.spec_yaml == bundle.spec_yaml


def test_a_retired_key_written_twice_hands_on_only_its_live_comments():
    """`ignored_config_fields:` written twice reads as its last copy: that
    copy's comment goes to `validation_exemptions:`, the dead copy's stays
    where it was written."""
    parent = """\
match: {model_type: [x]}
parent: yes
model:
  - {name: embed, op: embedding, boundary_role: input_embedding}
  - {name: out, op: linear, boundary_role: output_head, inputs: [embed]}
"""
    child = """\
# base comment
base: agent
# dead retired list?
ignored_config_fields: [wrong]
# live retired list
ignored_config_fields:
  - sliding_window
match: {model_id: acme/Fork}
child: yes
"""
    bundle = normalize_bundle("acme/Fork", child, [], _source(),
                              run_models={"agent": parent})
    lines = bundle.spec_yaml.splitlines()
    assert lines[:4] == ["# base comment", "# dead retired list?",
                         "match: {model_id: acme/Fork}", "parent: true"]
    at = lines.index("# live retired list")
    assert lines[at - 1] == "child: yes"
    assert lines[at + 1] == "validation_exemptions:"
    assert lines.count("# dead retired list?") == 1


def test_missing_and_cyclic_bases_are_actionable():
    source = _source()
    with pytest.raises(SpecBundleError) as missing:
        normalize_bundle("acme/Fork", "base: gone\n", [], source)
    assert "base component 'gone'" in str(missing.value)
    assert "regenerate the spec" in str(missing.value)

    with pytest.raises(SpecBundleError) as cyclic:
        normalize_bundle(
            "acme/Fork", "base: a\n", [], source,
            run_models={"a": "base: b\n", "b": "base: a\n"})
    assert "cyclic base chain for acme/Fork: a -> b -> a" in str(cyclic.value)


def test_flattened_draft_list_harvests_run_drafts_and_checks_formal_names():
    same = _component()
    source = _source(components=[("same", "library", same)])
    bundle = normalize_bundle(
        "acme/Fork",
        "base: parent\n",
        [],
        source,
        run_models={"parent": "drafts: [same, new]\n"},
        run_drafts={"new": _component(), "same": same},
    )
    doc = _document(bundle.spec_yaml)
    assert doc["drafts"] == ["new"]
    assert [file.path for file in bundle.serving_files] == [
        "backend/arch/components/drafts/new.yaml"]
    assert "reused formal component same" in bundle.notes

    with pytest.raises(SpecBundleError, match="conflicts with the different formal"):
        normalize_bundle(
            "acme/Fork", "drafts: [same]\n", [], source,
            run_drafts={"same": _component(bias="1")})


def test_missing_draft_duplicate_path_and_formal_proposals_are_data_errors():
    source = _source()
    with pytest.raises(SpecBundleError, match="draft component 'missing'"):
        normalize_bundle("acme/Fork", "drafts: [missing]\n", [], source)

    with pytest.raises(SpecBundleError, match="duplicate path"):
        normalize_bundle(
            "acme/Fork", "match: {}", [
                BundleFile("backend/arch/components/drafts/x.yaml", "a: 1\n"),
                BundleFile("backend/arch/components/drafts/x.yaml", "a: 2\n"),
            ], source)

    proposal = BundleFile(
        "backend/arch/components/library/attention.yaml", _component(),
        formal_body_sha256="0" * 64)
    bundle = normalize_bundle("acme/Fork", "match: {}", [proposal], source)
    assert bundle.proposal_files == (proposal,)
    assert bundle.serving_files == ()
    assert bundle.record_files(include_proposals=False) == []
    assert bundle.record_files() == [proposal.as_json()]


@pytest.mark.parametrize("body", [
    "graph: [not-a-map]\n",
    "role: attention\nunknown_field: not-a-component-field\n",
    "role: attention\ncustom_ops: not-a-mapping\n",
])
def test_malformed_draft_signatures_are_bundle_errors(body):
    """A saved draft's parse/schema failure is repairable bundle data."""
    source = _source()
    with pytest.raises(SpecBundleError, match="draft component 'bad'"):
        normalize_bundle(
            "acme/Fork", "drafts: [bad]\n", [], source,
            run_drafts={"bad": body})


def test_a_draft_saved_under_the_retired_root_is_not_replaced_by_a_namesake():
    """14aac2b0 (08-28) renamed `backend/arch_spec/` to `backend/arch/`. A
    draft saved before it must still be the draft the record wrote: dropping
    the file left the LIBRARY component of the same name being validated in
    its place, with nothing said."""
    library = _component()
    source = _source(components=[("same", "library", library)])
    stored = BundleFile(
        "backend/arch_spec/components/drafts/same.yaml", _component(bias="1"))
    with pytest.raises(SpecBundleError, match="conflicts with the different formal"):
        normalize_bundle("acme/Fork", "drafts: [same]\n", [stored], source)

    # Unchanged since it was saved: the same component, so reusing the formal
    # one says so instead of shipping a copy.
    bundle = normalize_bundle(
        "acme/Fork", "drafts: [same]\n",
        [BundleFile("backend/arch_spec/components/drafts/same.yaml", library)],
        source)
    assert "reused formal component same" in bundle.notes
    assert bundle.serving_files == ()


def test_a_draft_and_a_base_under_the_retired_root_travel():
    """No library namesake: the draft used to hard-fail as missing, and an
    agent base under the old root was never flattened."""
    source = _source()
    bundle = normalize_bundle(
        "acme/Fork", "base: parent\n",
        [
            BundleFile("backend/arch_spec/models/parent.yaml",
                       "drafts: [own]\nmatch: {model_type: [x]}\n"),
            BundleFile("backend/arch_spec/components/drafts/own.yaml", _component()),
        ],
        source)
    doc = _document(bundle.spec_yaml)
    assert doc["drafts"] == ["own"]
    assert [file.path for file in bundle.serving_files] == [
        "backend/arch/components/drafts/own.yaml"]
    assert "flattened agent base chain: parent" in bundle.notes


def test_which_refusals_agent_fix_can_open():
    """A refusal about the saved text opens Fix on it; one about the stored
    file list itself (there is no file to stage) does not, and its message
    does not send the user there."""
    from backend.spec_worker.bundle import (
        BlockedBundle, RefusedRecord, is_repairable, normalize_error_payload,
        open_for_fix)
    source = _source()
    blocked = open_for_fix("acme/Fork", "drafts: [missing]\n", [], source)
    assert isinstance(blocked, BlockedBundle)
    assert blocked.spec_yaml == "drafts: [missing]\n"
    assert "recreate the missing draft" in blocked.errors[0]

    duplicate = [
        BundleFile("backend/arch/components/drafts/x.yaml", "a: 1\n"),
        BundleFile("backend/arch/components/drafts/x.yaml", "a: 2\n"),
    ]
    refused = open_for_fix("acme/Fork", "match: {}", duplicate, source)
    assert isinstance(refused, RefusedRecord)
    assert isinstance(refused.error, SpecBundleError)
    assert not is_repairable(refused.error)
    payload = normalize_error_payload(refused.error, model_id="acme/Fork")
    assert payload["fixable"] is False
    assert "Agent Fix" not in payload["message"]
    assert "Regenerate it from the Inspector" in payload["message"]
