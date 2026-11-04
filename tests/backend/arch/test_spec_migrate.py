"""Stored-spec schema rewrites are mechanical, idempotent, and refuse to guess."""
from __future__ import annotations

import pytest
import yaml

from backend.arch.spec_migrate import (
    SpecMigrationBlocked,
    migrate_yaml_text,
    refuse,
)


def _load(text: str) -> dict:
    return yaml.safe_load(text)


def test_tie_weights_becomes_tied_weight_to_the_embedding():
    src = """
match: {model_type: [qwen3]}
model:
  - {name: embed_tokens, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - {name: lm_head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}, tie_weights: config.tie_word_embeddings}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    doc = _load(out.text)
    head = next(layer for layer in doc["model"] if layer["name"] == "lm_head")
    assert "tie_weights" not in head
    assert head["tied_weight"] == {
        "to": "embed_tokens",
        "when": "config.tie_word_embeddings",
    }
    embed = next(layer for layer in doc["model"] if layer["name"] == "embed_tokens")
    assert embed["boundary_role"] == "input_embedding"
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed
    assert again.text == out.text


def test_vision_group_drops_backbone_inputs():
    src = """
match: {model_type: [hunyuan]}
model_inputs:
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head}
  - name: vit
    group: true
    aux_kind: vision
    in: embed_tokens
    layers:
      - {name: vit, template: tower, inputs: [pixel_values]}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    vit = next(layer for layer in _load(out.text)["model"] if layer["name"] == "vit")
    assert "in" not in vit and "inputs" not in vit
    assert "dropped in on vision group 'vit'" in " ".join(out.notes)


def test_retired_kv_fields_are_blockers_not_rewrites():
    src = """
role: attention
kv_cached: [k, v]
graph:
  x: {op: identity, in: [x]}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert "kv_cached" in out.blockers[0]
    assert "regenerate" in out.blockers[0]
    assert out.text == src


def test_ambiguous_embedding_is_a_blocker():
    src = """
match: {model_type: [k3]}
model:
  - {name: language_model.embed_tokens, op: embedding}
  - {name: speech_embeddings, op: embedding}
  - {name: lm_head, op: linear, boundary_role: output_head}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert "input_embedding" in out.blockers[0]
    # Naming the declaration to add is not the same as asking for a
    # regeneration, and neither is a promise that Fix can repair it.
    assert out.needs_declaration == out.blockers
    error = refuse("moonshotai/Kimi-K3", out)
    assert isinstance(error, SpecMigrationBlocked)
    assert str(error) == "Schema error."
    assert "regenerate" not in str(error).lower()


def test_tie_weights_false_constant_is_dropped():
    src = """
match: {model_type: [longcat]}
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head, tie_weights: "0"}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    head = next(layer for layer in _load(out.text)["model"] if layer["name"] == "lm_head")
    assert "tie_weights" not in head
    assert "tied_weight" not in head


@pytest.mark.parametrize("spelling", ["yes", "on", "no", "off"])
def test_tie_weights_legacy_non_config_spellings_are_dropped(spelling):
    """Old `_resolve_config_bool` returned false for any string that did
    not start with `config.`. Guessing `"yes"` / `"on"` into an
    unconditional tie would bind a head that used to be free."""
    src = f"""
match: {{model_type: [x]}}
model:
  - {{name: embed_tokens, op: embedding, boundary_role: input_embedding}}
  - {{name: lm_head, op: linear, boundary_role: output_head, tie_weights: "{spelling}"}}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    head = next(layer for layer in _load(out.text)["model"] if layer["name"] == "lm_head")
    assert "tie_weights" not in head
    assert "tied_weight" not in head


def test_tie_weights_true_constant_is_unconditional():
    src = """
match: {model_type: [x]}
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head, tie_weights: "1"}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    head = next(layer for layer in _load(out.text)["model"] if layer["name"] == "lm_head")
    assert "tie_weights" not in head
    assert head["tied_weight"] == {"to": "embed_tokens"}


def test_two_declared_input_embeddings_are_a_blocker():
    src = """
match: {model_type: [k3]}
model:
  - {name: language_model.embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: speech_embeddings, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head, tie_weights: config.tie_word_embeddings}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("exactly one" in b for b in out.blockers)
    assert out.text == src


# -- the carried stream `hidden` -------------------------------------------
#
# A stack used to thread the hidden state through its block's first formal
# by default; that is now a stream the stack declares. A stored spec written
# before gets the declaration on every chain stack, and nothing is guessed.

HIDDEN = {"carries": True}

CHAINS = """
match: {model_type: [x]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams:
      dsa_topk: {src: "threads prev_topk across layers"}
    per_layer:
      assign: [{type: a, rest: true}]
      map: {a: {template: blk}}
  - {name: norm, op: rmsnorm, inputs: [layers]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [norm]}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    repeat: n_mtp_layers
    layers:
      - {name: mtp, template: head_blk, inputs: [norm]}
  - name: nested
    group: true
    aux_kind: mtp
    inputs: [norm]
    repeat: n_mtp_layers
    layers:
      - {name: nested.layers, template: blk, repeat: "2", inputs: [norm]}
"""


def _layer(doc: dict, *path: str) -> dict:
    layers = doc["model"]
    for name in path:
        layer = next(l for l in layers if l["name"] == name)
        layers = layer.get("layers") or []
    return layer


def test_every_chain_stack_gets_the_carried_stream_where_it_repeats():
    out = migrate_yaml_text(CHAINS)
    assert out.ok and out.changed
    doc = _load(out.text)
    # A per-layer stack keeps its inner stream beside the carried one.
    assert _layer(doc, "layers")["streams"] == {
        "hidden": HIDDEN, "dsa_topk": {"src": "threads prev_topk across layers"}}
    # A repeated group around a block that does not repeat: on the group.
    assert _layer(doc, "mtp")["streams"] == {"hidden": HIDDEN}
    assert "streams" not in _layer(doc, "mtp", "mtp")
    # A repeated group around a repeated stack: on the innermost.
    assert "streams" not in _layer(doc, "nested")
    assert _layer(doc, "nested", "nested.layers")["streams"] == {
        "hidden": HIDDEN}


def test_the_carried_stream_rewrite_is_idempotent():
    once = migrate_yaml_text(CHAINS)
    twice = migrate_yaml_text(once.text)
    assert twice.ok and not twice.changed
    assert twice.text == once.text


def test_a_stream_already_named_hidden_is_a_blocker():
    src = CHAINS.replace("dsa_topk: {src:", "hidden: {src:")
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("named `hidden`" in b for b in out.blockers)
    assert out.text == src


# -- `keep:` ports become appended streams ----------------------------------
#
# A port published the stack's hidden state after chosen iterations as
# `<stack>.<port>`, and its reader was a stack whose iteration j read element
# j. That is an appended stream now, and the reader names it in `inputs:`.

KEPT = """
match: {model_type: [x]}
model_inputs:
  input_ids: {kind: tokens}
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids]}
  - name: visual
    group: true
    aux_kind: vision
    layers:
      - {name: visual.patch, op: linear, inputs: [pixel_values]}
      - name: visual.blocks
        template: vblk
        repeat: n_vis_layers
        inputs: [visual.patch]
        keep: {deepstack: {layers: config.deepstack_visual_indexes, index: block, src: "tap"}}
      - {name: visual.mergers, template: merger, repeat: n_deepstack, inputs: [visual.blocks.deepstack]}
  - {name: layers, template: blk, repeat: n_layers, inputs: [embed]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers]}
"""

APPENDED = """
match: {model_type: [x]}
model_inputs:
  input_ids: {kind: tokens}
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids]}
  - name: visual
    group: true
    aux_kind: vision
    layers:
      - {name: visual.patch, op: linear, inputs: [pixel_values]}
      - name: visual.blocks
        template: vblk
        repeat: n_vis_layers
        inputs: [visual.patch]
        streams:
          hidden: {carries: true}
          deepstack: {from: hidden, after: config.deepstack_visual_indexes, index: block, src: "tap"}
      - {name: visual.mergers, template: merger, repeat: n_deepstack, inputs: [deepstack]}
  - name: layers
    template: blk
    repeat: n_layers
    inputs: [embed]
    streams:
      hidden: {carries: true}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers]}
"""


def test_a_keep_port_becomes_an_appended_stream():
    """The port is the stream, its reader names the stream and carries
    nothing — iteration j reads element j — and the writer carries the
    hidden state it appends."""
    out = migrate_yaml_text(KEPT)
    assert out.ok and out.changed
    assert _load(out.text) == _load(APPENDED)


def test_the_appended_stream_rewrite_is_idempotent():
    once = migrate_yaml_text(KEPT)
    twice = migrate_yaml_text(once.text)
    assert twice.ok and not twice.changed
    assert not migrate_yaml_text(APPENDED).changed


@pytest.mark.parametrize("edit,match", [
    # The port's name becomes model-wide: a layer already has it.
    (lambda s: s.replace("{name: visual.patch,", "{name: deepstack,")
     .replace("inputs: [visual.patch]", "inputs: [deepstack]"),
     "already names a layer"),
    # ... or a stream does.
    (lambda s: s.replace(
        "{name: layers, template: blk, repeat: n_layers, inputs: [embed]}",
        "{name: layers, template: blk, repeat: n_layers, inputs: [embed], "
        "streams: {hidden: {carries: true}, deepstack: {src: a}}}"),
     "already names a layer"),
    # A reader that carries a stream does not read element by element.
    (lambda s: s.replace(
        "inputs: [visual.blocks.deepstack]}",
        "inputs: [visual.blocks.deepstack], "
        "streams: {hidden: {carries: true}}}"),
     "carries a stream"),
    # Under a repeating group, element j would have a reader per copy.
    (lambda s: s.replace("    aux_kind: vision\n",
                         "    aux_kind: vision\n    repeat: n_vis\n"),
     "repeats"),
])
def test_what_the_keep_rewrite_cannot_decide_is_a_blocker(edit, match):
    src = edit(KEPT)
    assert src != KEPT
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any(match in b for b in out.blockers), out.blockers
    assert out.text == src


def test_a_component_stream_input_is_a_blocker():
    src = """
role: attention
inputs: [w, x]
stream_input: x
graph:
  s: {op: add, in: [x, w]}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert "stream_input" in out.blockers[0]
    assert out.text == src


def test_a_retired_weight_copies_declaration_is_a_blocker_naming_expert_weight():
    src = """
role: ffn
inputs: [x]
graph:
  experts:
    op: swiglu_ffn
    in: [x]
    repeat: n_expert_copies
    weight_copies: {world: moe_copies, parts: n_expert_copies,
                    carrier: instances, checkpoint: unrolled}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("weight_copies" in b and "expert_weight" in b
               for b in out.blockers), out.blockers
    assert out.text == src


from tools.chain_stacks import MODELS


def _strip_carried(layers: list) -> None:
    for layer in layers:
        streams = layer.get("streams")
        if isinstance(streams, dict):
            for name in [n for n, d in streams.items() if d.get("carries")]:
                del streams[name]
            if not streams:
                del layer["streams"]
        _strip_carried(layer.get("layers") or [])


def test_the_current_corpus_is_untouched_the_way_serving_calls_it():
    """Zero effect on what is committed — asked the way `normalize_bundle`
    asks it, with the components in hand.

    The rules that read them (a dim a component renamed, a switch a component
    asks for) see a different document than a bare `migrate_yaml_text(text)`
    does, so this is the check that has to carry them: a closure built from
    any identifier a body spells, rather than from the channels a component
    is called through, reaches components the spec never calls — which is how
    a first cut of the router-switch rule was zero-effect without the
    components and rewrote seven committed models with them.
    """
    from backend.spec_worker.bundle import FormalSource
    source = FormalSource.from_deployment()
    components = {name: component.body
                  for name, component in source.components.items()}
    tiers = ("models", "library", "adapters", "drafts")
    scanned = [path for path in sorted((MODELS.parent).rglob("*.yaml"))
               if path.parent.name in tiers]
    assert len(scanned) > 200, len(scanned)
    for path in scanned:
        out = migrate_yaml_text(path.read_text(), components, source.models)
        assert out.ok, (path.name, out.blockers)
        assert not out.changed, (path.name, out.notes)


def test_the_shared_corpus_round_trips():
    """Every model file that owns `model:`: remove its carried streams and
    migrate, and the document comes back as committed — so the rewrite a
    stored spec gets is the one the corpus was given — and migrating the
    committed file changes nothing."""
    for path in sorted(MODELS.glob("*.yaml")):
        committed = _load(path.read_text())
        if "model" not in committed:
            continue
        assert not migrate_yaml_text(path.read_text()).changed, path.name
        stripped = _load(path.read_text())
        _strip_carried(stripped["model"])
        out = migrate_yaml_text(yaml.safe_dump(stripped, sort_keys=False))
        assert out.ok, (path.name, out.blockers)
        assert _load(out.text)["model"] == committed["model"], path.name


def test_omitted_inputs_become_the_retired_sequential_default():
    src = """
match: {model_type: [deepseek_v3]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - {name: rotary_inv_freq, op: rotary_inv_freq_buffer, dim: rope_inv_freq_len}
  - {name: norm, op: rmsnorm, dim: d_model}
  - {name: head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    layers:
      - {name: mtp.enorm, op: rmsnorm, dim: d_model}
      - {name: mtp.layers, template: block}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    reads = {layer["name"]: layer.get("inputs")
             for layer in _load(out.text)["model"]}
    assert reads["embed"] == ["input_ids"]   # the token ids it looked up
    assert reads["rotary_inv_freq"] is None
    assert reads["layers"] == ["embed"]
    assert reads["norm"] == ["layers"]      # the buffer does not advance
    assert reads["head"] == ["norm"]
    mtp = _load(out.text)["model"][-1]["layers"]
    assert mtp[0]["inputs"] == ["norm"]     # the group attaches to its first stage
    assert mtp[1]["inputs"] == ["mtp.enorm"]


def test_omitted_inputs_after_a_parameter_source_is_a_blocker():
    src = """
match: {model_type: [deepseek_v4]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block, inputs: [embed]}
  - {name: scale, op: parameter_vector, dim: "1"}
  - {name: norm, op: rmsnorm, dim: d_model}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("'norm'" in b and "'scale'" in b for b in out.blockers)
    assert out.text == src


# -- the dim symbol vocabulary ---------------------------------------------
#
# 59db3827 renamed every single-letter symbol, and 0f0516bd made an
# unregistered one a hard error. A spec stored before that speaks the old
# vocabulary in its `dims:` block, in every expression, and in the callee
# parameter keys of its nodes.

OLD_VOCABULARY = """
match: {model_type: [x]}
dims:
  H: config.hidden_size
  N: config.num_attention_heads
  D: config.qk_nope_head_dim
  P: config.qk_rope_head_dim
  R: config.kv_lora_rank
  V: config.vocab_size
  L: config.num_hidden_layers
  Ld: config.first_k_dense_replace
  I: config.intermediate_size
  Nlv: config.linear_attn_config.num_heads
  Dlv: config.linear_attn_config.head_dim
blocks:
  blk:
    inputs: [x]
    graph:
      norm: {op: rmsnorm, in: x, dim: H}
      q: {op: linear, in: norm, dims: {in: H, out: "N * D"}}
      q_split: {op: view_split, in: q, dims: {N: N, D: D}}
      mlp: {op: swiglu_ffn, in: norm, dims: {H: H, I: I}}
    outputs: [mlp]
    cache:
      family: MLA
      entries: [{state: conv_state, kind: conv_state, shape: [Nlv], carrier: q}]
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {V: V, H: H}}
  - name: layers
    repeat: L
    inputs: [embed]
    per_layer:
      assign: [{type: dense, range: ["0", "Ld - 1"]}, {type: moe, rest: true}]
      map: {dense: {template: blk}, moe: {template: blk}}
  - {name: head, op: linear, boundary_role: output_head, inputs: [layers], dims: {in: H, out: V}, weight_axes: [{world: attention, parts: V, fit: pad}, {replicated: true}]}
"""


def test_the_retired_dim_vocabulary_is_renamed():
    out = migrate_yaml_text(OLD_VOCABULARY)
    assert out.ok and out.changed
    doc = _load(out.text)
    assert list(doc["dims"]) == [
        "d_model", "n_heads", "d_qk_nope", "d_qk_rope", "d_kv_lora",
        "n_vocab", "n_layers", "n_dense_layers", "d_ffn",
        "n_lin_v_heads", "d_lin_v_head"]
    graph = doc["blocks"]["blk"]["graph"]
    assert graph["norm"]["dim"] == "d_model"
    assert graph["q"]["dims"] == {"in": "d_model", "out": "n_heads * d_qk_nope"}
    # A node's `dims:` keys are the callee's formal parameters: the per-head
    # width is the family-neutral `d_head`, its value the family's symbol.
    assert graph["q_split"]["dims"] == {"n_heads": "n_heads", "d_head": "d_qk_nope"}
    assert graph["mlp"]["dims"] == {"d_model": "d_model", "d_ffn": "d_ffn"}
    assert doc["blocks"]["blk"]["cache"]["entries"][0]["shape"] == ["n_lin_v_heads"]
    layers = _layer(doc, "layers")
    assert layers["repeat"] == "n_layers"
    assert layers["per_layer"]["assign"][0]["range"] == ["0", "n_dense_layers - 1"]
    head = _layer(doc, "head")
    assert head["dims"] == {"in": "d_model", "out": "n_vocab"}
    assert head["weight_axes"][0]["parts"] == "n_vocab"
    assert "renamed" in " ".join(out.notes)


def test_the_vocabulary_rename_is_idempotent():
    once = migrate_yaml_text(OLD_VOCABULARY)
    twice = migrate_yaml_text(once.text)
    assert twice.ok and not twice.changed
    assert twice.text == once.text


def test_a_binding_of_head_dim_reads_the_ambiguous_symbol_as_the_head_dim():
    """`D` named three quantities; the config key its binding reads says
    which. `head_dim` is the plain head width."""
    src = OLD_VOCABULARY.replace("  P: config.qk_rope_head_dim\n", "") \
        .replace("  R: config.kv_lora_rank\n", "") \
        .replace("  D: config.qk_nope_head_dim", "  D: config.head_dim")
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    doc = _load(out.text)
    assert "d_head" in doc["dims"] and "d_qk_nope" not in doc["dims"]
    assert doc["blocks"]["blk"]["graph"]["q"]["dims"]["out"] == "n_heads * d_head"


def test_an_mla_model_whose_d_reads_head_dim_keeps_the_head_width():
    """Ling 2.6.1T is MLA and binds `D = head_dim`: its binding, not its MLA
    markers, says which quantity `D` is."""
    src = OLD_VOCABULARY.replace(
        "  D: config.qk_nope_head_dim",
        "  D: config.head_dim\n  mla_qk_nope_dim: config.qk_nope_head_dim")
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    dims = _load(out.text)["dims"]
    assert dims["d_head"] == "config.head_dim"
    assert dims["d_qk_nope"] == "config.qk_nope_head_dim"
    assert _load(out.text)["blocks"]["blk"]["graph"]["q"]["dims"]["out"] == (
        "n_heads * d_head")


def test_the_binding_names_the_quantity_without_the_mla_markers():
    src = OLD_VOCABULARY.replace("  P: config.qk_rope_head_dim\n", "") \
        .replace("  R: config.kv_lora_rank\n", "") \
        .replace("  D: config.qk_nope_head_dim", "  D: config.text_config.qk_nope_head_dim")
    out = migrate_yaml_text(src)
    assert out.ok
    assert _load(out.text)["dims"]["d_qk_nope"] == "config.text_config.qk_nope_head_dim"


def test_a_binding_that_names_no_known_quantity_is_a_blocker():
    for binding in ('"128"', "config.attention_head_dim",
                    "config.head_dim ?? config.qk_nope_head_dim"):
        src = OLD_VOCABULARY.replace("  D: config.qk_nope_head_dim", f"  D: {binding}")
        out = migrate_yaml_text(src)
        assert not out.ok, binding
        assert out.blockers == out.needs_declaration
        assert len([b for b in out.blockers if "'D'" in b]) == 1, out.blockers
        assert out.text == src


def test_the_ambiguous_symbol_in_a_component_body_is_a_blocker():
    """A component binds no dims of its own, so it names no family: the two
    readings are different quantities and neither may be guessed."""
    src = """
role: attention
inputs: [x]
graph:
  q: {op: linear, in: x, dims: {in: H, out: "N * D"}}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("D" in b and "d_qk_nope" in b and "d_head" in b
               for b in out.blockers), out.blockers
    assert out.text == src


# -- the exit boundary and the reviewer's notes -----------------------------
#
# a6020cca made an authored model declare its output head; f0c58501 removed
# `ignored_config_fields:`. Both landed on 08-23, and a spec stored before
# them fails to load on the missing declaration and the removed key.

NO_EXIT = """
match: {model_type: [x]}
ignored_config_fields:
  - max_position_embeddings   # positional — no params
dims:
  d_model: config.hidden_size
  n_vocab: config.vocab_size
model:
  - {name: embed, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: gate_proj, op: linear, inputs: [layers], dims: {in: d_model, out: "1"}}
  - {name: norm, op: rmsnorm, inputs: [gate_proj], dim: d_model}
  - {name: lm_head, op: linear, inputs: [norm], dims: {in: d_model, out: n_vocab}}
"""


def test_the_vocabulary_projection_becomes_the_output_head():
    out = migrate_yaml_text(NO_EXIT)
    assert out.ok and out.changed
    doc = _load(out.text)
    assert _layer(doc, "lm_head")["boundary_role"] == "output_head"
    assert "boundary_role" not in _layer(doc, "gate_proj")
    assert "ignored_config_fields" not in doc
    assert "stamped boundary_role: output_head on 'lm_head'" in out.notes
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_the_one_real_reader_of_the_retired_list_is_carried_over():
    """`ignored_config_fields:` was reviewer prose with one consumer: a
    `sliding_window` entry turned the config-window check off, which f0c58501
    replaced with a reason-bearing `validation_exemptions`. Dropping the key
    would turn the declared exemption back into a warning."""
    src = NO_EXIT.replace(
        "  - max_position_embeddings   # positional — no params",
        "  - text_config.sliding_window   # local attention not modelled\n"
        "  - max_position_embeddings")
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    doc = _load(out.text)
    assert "ignored_config_fields" not in doc
    reason = doc["validation_exemptions"]["sliding_window"]
    assert "ignored_config_fields" in reason and "restate it" in reason
    assert any("exemption the retired list held" in note for note in out.notes)
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_an_exemption_the_spec_already_declares_is_not_overwritten():
    src = NO_EXIT.replace(
        "ignored_config_fields:",
        "validation_exemptions: {sliding_window: the author's own reason}\n"
        "ignored_config_fields:\n  - sliding_window").replace(
        "  - max_position_embeddings   # positional — no params", "")
    out = migrate_yaml_text(src)
    assert out.ok
    assert _load(out.text)["validation_exemptions"]["sliding_window"] == (
        "the author's own reason")


def test_the_head_is_the_stage_the_retired_engine_took():
    """Before the role, the head was the top-level `lm_head` linear; another
    linear onto the vocabulary is not it."""
    src = NO_EXIT.replace(
        '{name: gate_proj, op: linear, inputs: [layers], dims: {in: d_model, out: "1"}}',
        "{name: other_head, op: linear, inputs: [layers], dims: {in: d_model, out: n_vocab}}")
    out = migrate_yaml_text(src)
    assert out.ok
    doc = _load(out.text)
    assert _layer(doc, "lm_head")["boundary_role"] == "output_head"
    assert "boundary_role" not in _layer(doc, "other_head")


def test_two_lm_head_linears_are_a_blocker():
    """Stamping the wrong one moves the logits and the tied weight."""
    src = NO_EXIT.replace(
        '{name: gate_proj, op: linear, inputs: [layers], dims: {in: d_model, out: "1"}}',
        "{name: mtp_lm_head, op: linear, inputs: [layers], dims: {in: d_model, out: n_vocab}}")
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("output_head" in b and "mtp_lm_head" in b for b in out.blockers)
    assert out.text == src


def test_a_declared_output_head_is_left_alone():
    src = NO_EXIT.replace(
        "{name: lm_head, op: linear, inputs: [norm], dims: {in: d_model, out: n_vocab}}",
        "{name: lm_head, op: linear, boundary_role: output_head, inputs: [norm], "
        "dims: {in: d_model, out: n_vocab}}")
    out = migrate_yaml_text(src)
    assert out.ok
    doc = _load(out.text)
    assert _layer(doc, "lm_head")["boundary_role"] == "output_head"
    assert not any("output_head" in note for note in out.notes)


# -- the encoder's rows enter the embedding stream --------------------------
#
# f40c3ad5 (09-29) made a stack's `inputs:` bind its block's formals one to
# one. A decoder that named `[embed_tokens, vision]` used to have the tower
# silently dropped; what it meant is a `scatter_rows_replace` it reads.

TWO_INPUTS = """
match: {model_type: [x]}
blocks:
  blk:
    inputs: [x]
    graph:
      norm: {op: rmsnorm, in: x, dim: d_model}
model_inputs:
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: language_model.model.embed_tokens, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - name: vision
    group: true
    aux_kind: vision
    layers:
      - {name: vision, template: tower, inputs: [pixel_values]}
  - name: language_model.model.layers
    repeat: n_layers
    inputs: [language_model.model.embed_tokens, vision]
    template: blk
  - {name: language_model.model.norm, op: rmsnorm, inputs: [language_model.model.layers], dim: d_model}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [language_model.model.norm], dims: {in: d_model, out: n_vocab}}
"""


def test_a_stacks_second_input_becomes_the_row_merge():
    out = migrate_yaml_text(TWO_INPUTS)
    assert out.ok and out.changed
    doc = _load(out.text)
    names = [layer["name"] for layer in doc["model"]]
    assert names.index("language_model.model.embed_merge") == names.index(
        "language_model.model.layers") - 1
    merge = _layer(doc, "language_model.model.embed_merge")
    assert merge["op"] == "scatter_rows_replace"
    assert merge["inputs"] == ["language_model.model.embed_tokens", "vision"]
    stack = _layer(doc, "language_model.model.layers")
    assert stack["inputs"] == ["language_model.model.embed_merge"]
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_a_stack_whose_block_takes_two_formals_is_left_alone():
    """f40c3ad5 KEPT two inputs binding two formals
    (`ArchSpec._check_stack_call_arity`); what it refused is a count that
    does not match. Only the stack whose block declares ONE formal lost its
    second input, so only that one is rewritten."""
    src = TWO_INPUTS.replace("    inputs: [x]\n", "    inputs: [x, v]\n")
    out = migrate_yaml_text(src)
    assert out.ok
    doc = _load(out.text or src)
    assert not any("embed_merge" in str(layer.get("name"))
                   for layer in doc["model"])
    assert _layer(doc, "language_model.model.layers")["inputs"] == [
        "language_model.model.embed_tokens", "vision"]


def test_a_stack_whose_formals_nothing_declares_is_left_alone():
    """The block is neither inline nor in the library handed to the migrator,
    so how many formals it takes is not in evidence."""
    src = TWO_INPUTS.replace("template: blk", "template: absent_block")
    out = migrate_yaml_text(src)
    assert out.ok
    doc = _load(out.text or src)
    assert not any("embed_merge" in str(layer.get("name"))
                   for layer in doc["model"])


def test_the_row_merge_rewrite_keeps_a_taken_name():
    src = TWO_INPUTS.replace(
        "{name: language_model.model.norm, op: rmsnorm",
        "{name: language_model.model.embed_merge, op: rmsnorm")
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("already taken" in b for b in out.blockers), out.blockers
    assert out.text == src


def test_a_view_split_gets_the_axis_it_used_to_default_to():
    """be09cb7b (09-28) made `view_split` declare its `axis:`; the default it
    replaced was the last axis, wherever the node sits."""
    src = """
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
blocks:
  blk:
    inputs: [x]
    graph:
      split: {op: view_split, in: x, dims: {n_heads: n_heads, d_head: d_head}}
      merge: {op: view_merge, in: split, dims: {n_heads: n_heads, d_head: d_head, out: d_model}}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: hc_split, op: view_split, inputs: [layers], dims: {n_heads: n_hidden_copies, d_head: d_model}}
  - {name: window_fold, op: view_split, axis: 1, inputs: [layers], dims: {n_heads: n_slots, d_head: s_window}}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [hc_split], dims: {in: d_model, out: n_vocab}}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    doc = _load(out.text)
    assert doc["blocks"]["blk"]["graph"]["split"]["axis"] == -1
    assert "axis" not in doc["blocks"]["blk"]["graph"]["merge"]
    assert _layer(doc, "hc_split")["axis"] == -1
    # A node that declares one keeps it.
    assert _layer(doc, "window_fold")["axis"] == 1
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


# -- what the library renamed ----------------------------------------------

VISION = """
match: {model_type: [qwen3_5]}
dims:
  d_model: config.text_config.hidden_size
  n_vocab: config.text_config.vocab_size
  d_vis_model: config.vision_config.hidden_size
  n_vis_heads: config.vision_config.num_heads
  d_v_head: "d_vis_model // n_vis_heads"
drafts: [dense_block]
model_inputs:
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: dense_block, inputs: [embed]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers], dims: {in: d_model, out: n_vocab}}
  - name: vision
    group: true
    aux_kind: vision
    checkpoint: {prefix: model.visual.blocks, index: local}
    repeat: "1"
    layers:
      - {name: model.visual.blocks, template: qwen35_vision_block, repeat: n_vis_layers, inputs: [pixel_values]}
"""

# The two components the spec reaches, as the library body of each would read.
COMPONENTS = {
    "qwen35_vision_block": "graph:\n  q: {op: linear, dims: {d_head: d_vis_head}}\n",
    "dense_block": "graph:\n  layer: {op: pre_norm, params: "
                   "{attn: gqa_qknorm_partial_rope_silu_gated_attention}}\n",
    "mla_attention": "graph:\n  v: {op: linear, dims: {out: d_v_head}}\n",
}


def test_a_renamed_component_is_called_by_its_current_name():
    """1299577b (09-27) renamed the silu-gated attention to sigmoid, with
    every caller. A stored draft still names the component that is gone."""
    body = COMPONENTS["dense_block"]
    out = migrate_yaml_text(body, COMPONENTS)
    assert out.ok and out.changed
    params = _load(out.text)["graph"]["layer"]["params"]
    assert params["attn"] == "gqa_qknorm_partial_rope_sigmoid_gated_attention"
    assert any("renamed the component" in note for note in out.notes)
    assert not migrate_yaml_text(out.text, COMPONENTS).changed


def test_the_vision_blocks_per_head_dim_is_bound_under_its_new_name():
    """7f7f0267 (09-25) renamed `qwen35_vision_block`'s `d_v_head` to
    `d_vis_head`, and the binding lives in the calling model."""
    out = migrate_yaml_text(VISION, COMPONENTS)
    assert out.ok and out.changed
    dims = _load(out.text)["dims"]
    assert dims["d_vis_head"] == "d_vis_model // n_vis_heads"
    assert "d_v_head" not in dims
    assert not migrate_yaml_text(out.text, COMPONENTS).changed


def test_a_model_whose_mla_also_reads_the_old_dim_is_a_blocker():
    """`d_v_head` is still the MLA value-head width. Renaming the one
    binding would unbind the attention that reads it."""
    src = VISION.replace("drafts: [dense_block]", "components: {attention: mla_attention}").replace(
        "template: dense_block", "template: block")
    out = migrate_yaml_text(src, COMPONENTS)
    assert not out.ok
    assert any("mla_attention" in b and "d_vis_head" in b for b in out.blockers), out.blockers
    assert out.text == src


def test_a_reached_component_that_does_not_parse_is_a_blocker():
    """An unparseable draft is not a component that reads nothing: taking it
    as empty would rename `d_v_head` from under an attention nobody read."""
    src = VISION.replace("drafts: [dense_block]", "components: {attention: mla_attention}").replace(
        "template: dense_block", "template: block")
    out = migrate_yaml_text(src, {**COMPONENTS, "mla_attention": "graph: [unclosed\n"})
    assert not out.ok
    assert any("mla_attention" in b and "does not parse" in b for b in out.blockers), out.blockers
    assert out.text == src


def test_without_the_component_bodies_the_dim_rename_stays_put():
    """`migrate_yaml_text` is also called with no library in hand (the
    read-only stored-spec report). A rewrite that needs to read the
    components then has nothing to decide on, and decides nothing."""
    out = migrate_yaml_text(VISION)
    assert out.ok
    assert "d_v_head" in _load(out.text or VISION)["dims"]


# -- a step that became a config switch ------------------------------------

ROUTED = """
match: {model_type: [their_flash]}
dims:
  d_model: config.hidden_size
  n_vocab: config.vocab_size
  n_experts: config.n_routed_experts
  n_experts_active: config.moe_topk
components: {ffn_moe: routed_ffn}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: moe_block, inputs: [embed]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers], dims: {in: d_model, out: n_vocab}}
"""

ROUTERS = {
    "moe_block": "graph:\n  layer: {op: pre_norm, params: {ffn: routed_ffn}}\n",
    "routed_ffn": "graph:\n  gate: {op: topk_router, in: x}\n",
    "topk_router": "graph:\n  topk_norm: {op: normalize_sum_one, in: topk_gather, "
                   "when: n_topk_renorm}\n",
    "dense_ffn": "graph:\n  up: {op: linear, in: x}\n",
    "dense_block": "graph:\n  layer: {op: pre_norm, params: {ffn: dense_ffn}}\n",
}

# What the corpus declares for two different config classes: the family is how
# a stored spec reads its own switch.
FAMILIES = {
    "their_flash": "match: {model_type: [their_flash]}\n"
                   'dims: {n_topk_renorm: \'config.norm_topk_prob ?? "0"\'}\n',
    "their_next": "match: {model_type: [their_next]}\n"
                  'dims: {n_topk_renorm: \'config.text_config.norm_topk_prob ?? "1"\'}\n',
}


def test_the_renormalization_switch_is_bound_as_its_family_reads_it():
    """603f85f0 / d3ae42d6 (09-27) made the renormalization a `when:` switch.
    What to bind is the config class's own default, which the corpus model of
    the spec's family declares — not one global fallback: these two families
    differ in both the default and the config path."""
    out = migrate_yaml_text(ROUTED, ROUTERS, FAMILIES)
    assert out.ok and out.changed
    assert _load(out.text)["dims"]["n_topk_renorm"] == (
        'config.norm_topk_prob ?? "0"')
    assert not migrate_yaml_text(out.text, ROUTERS, FAMILIES).changed

    other = ROUTED.replace("[their_flash]", "[their_next]")
    out = migrate_yaml_text(other, ROUTERS, FAMILIES)
    assert out.ok and out.changed
    assert _load(out.text)["dims"]["n_topk_renorm"] == (
        'config.text_config.norm_topk_prob ?? "1"')


def test_a_switch_no_family_declares_is_a_refusal_not_a_guess():
    """A fallback of 0 here would silently turn a renormalization off for a
    config class that defaults it on."""
    orphan = ROUTED.replace("[their_flash]", "[nobodys_family]")
    out = migrate_yaml_text(orphan, ROUTERS, FAMILIES)
    assert not out.ok
    assert any("n_topk_renorm" in b and "config class" in b
               for b in out.blockers), out.blockers
    assert out.text == orphan


def test_a_bound_switch_and_a_spec_with_no_router_are_left_alone():
    bound = ROUTED.replace(
        "  n_experts: config.n_routed_experts",
        '  n_topk_renorm: config.norm_topk_prob ?? "1"\n  n_experts: config.n_routed_experts')
    assert _load(migrate_yaml_text(bound, ROUTERS, FAMILIES).text or bound)[
        "dims"]["n_topk_renorm"] == 'config.norm_topk_prob ?? "1"'
    dense = ROUTED.replace("routed_ffn", "dense_ffn").replace(
        "template: moe_block", "template: dense_block")
    out = migrate_yaml_text(dense, ROUTERS, FAMILIES)
    assert "n_topk_renorm" not in _load(out.text or dense)["dims"]


def test_a_component_that_only_documents_the_switch_does_not_read_it():
    """`# Required dims: … n_topk_renorm` in a component's prose is not a
    read. Counting it fired this rule on seven committed corpus models whose
    routers never ask for the switch."""
    mentions = dict(ROUTERS)
    mentions["routed_ffn"] = ("# Required dims: d_model, n_topk_renorm\n"
                              "graph:\n  gate: {op: linear, in: x}\n")
    out = migrate_yaml_text(ROUTED, mentions, FAMILIES)
    assert out.ok
    assert "n_topk_renorm" not in _load(out.text or ROUTED)["dims"]


# -- a dim that lost its registry default ---------------------------------

SCALED_ROUTER = dict(ROUTERS, topk_router=(
    "graph:\n  topk_scale: {op: scale_const, in: topk_norm, "
    "when: n_routed_scale}\n"))


def test_a_dim_that_lost_its_default_comes_back_to_the_author():
    """`n_routed_scale` used to fill in as 0 for a spec that bound nothing,
    which dropped DeepSeek-V2's routed scale. Whether the step exists is the
    model's forward source, so the record is refused with a declaration
    naming the dim and how to bind it — never bound for the author."""
    bound = ROUTED.replace(
        "  n_experts: config.n_routed_experts",
        '  n_topk_renorm: config.norm_topk_prob ?? "0"\n'
        "  n_experts: config.n_routed_experts")
    out = migrate_yaml_text(bound, SCALED_ROUTER, FAMILIES)
    assert out.only_needs_declarations, out.blockers
    assert any("n_routed_scale" in b and "routed_scaling_factor" in b
               for b in out.blockers), out.blockers
    assert out.text == bound

    stated = bound.replace("  n_experts: config.n_routed_experts",
                           '  n_routed_scale: "0"\n'
                           "  n_experts: config.n_routed_experts")
    assert migrate_yaml_text(stated, SCALED_ROUTER, FAMILIES).ok


RENORM_BOUND = ROUTED.replace(
    "  n_experts: config.n_routed_experts",
    '  n_topk_renorm: config.norm_topk_prob ?? "0"\n'
    "  n_experts: config.n_routed_experts")


def test_a_base_binds_only_what_the_served_document_keeps():
    """`base:` is the loader's shallow merge: a child that writes `dims:`
    replaces its base's `dims:` whole, so the base's binding is not the
    child's and the record is refused. A child that writes no `dims:` serves
    its base's, binding included."""
    family = dict(FAMILIES, their_flash=FAMILIES["their_flash"].replace(
        "dims: {", 'dims: {n_routed_scale: "0", '))
    replaces = RENORM_BOUND.replace(
        "match: {model_type: [their_flash]}",
        "match: {model_type: [their_flash]}\nbase: their_flash")
    out = migrate_yaml_text(replaces, SCALED_ROUTER, family)
    assert out.only_needs_declarations, out.blockers
    assert any("n_routed_scale" in b for b in out.blockers), out.blockers

    doc = _load(replaces)
    del doc["dims"]
    inherits = yaml.safe_dump(doc, sort_keys=False)
    assert migrate_yaml_text(inherits, SCALED_ROUTER, family).ok


def test_a_slot_no_layer_reaches_reads_nothing():
    """The loader loads every `components:` slot, but only what a layer
    reaches runs. A model whose layers run a dense block while it still
    declares a routed `ffn_moe` slot (LongCat's shape) binds no router dim."""
    dense = RENORM_BOUND.replace("template: moe_block", "template: dense_block")
    assert "ffn_moe: routed_ffn" in dense
    assert migrate_yaml_text(dense, SCALED_ROUTER, FAMILIES).ok


def test_a_call_site_binding_binds():
    """A layer's `dims:` binds the dim for the block it runs, so a record
    that says it there is not sent back."""
    at_site = RENORM_BOUND.replace(
        "template: moe_block, inputs: [embed]}",
        'template: moe_block, inputs: [embed], dims: {n_routed_scale: "0"}}')
    assert migrate_yaml_text(at_site, SCALED_ROUTER, FAMILIES).ok


def test_a_stored_record_missing_a_router_switch_opens_agent_fix():
    """The serving boundary for a stored private record: the library router
    reads the switch, the record does not bind it, and the refusal is a
    declaration Agent Fix can write (`spec_needs_declaration`, fixable)
    with the dim and its binding in `errors[]`."""
    from backend.spec_worker.bundle import (
        FormalSource, normalize_bundle, normalize_error_payload)
    record = """
match: {model_id: [acme/Routed]}
components: {ffn_moe: moe_ffn}
dims:
  d_model: config.hidden_size
  n_vocab: config.vocab_size
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_topk_renorm: config.norm_topk_prob ?? "0"
  n_expert_groups: "1"
  n_groups_active: "1"
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: moe_ffn, inputs: [embed]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers], dims: {in: d_model, out: n_vocab}}
"""
    with pytest.raises(SpecMigrationBlocked) as refused:
        normalize_bundle("acme/Routed", record, [],
                         FormalSource.from_deployment())
    payload = normalize_error_payload(refused.value, model_id="acme/Routed")
    assert payload["status"] == "spec_needs_declaration"
    assert payload["fixable"] is True
    assert len(payload["errors"]) == 1
    assert "n_routed_scale" in payload["errors"][0]


# -- where an indexed aux stack lives --------------------------------------

AUX = """
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
model_inputs:
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: norm, op: rmsnorm, inputs: [layers], dim: d_model}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [norm], dims: {in: d_model, out: n_vocab}}
  - name: vision
    group: true
    aux_kind: vision
    repeat: "1"
    layers:
      - {name: model.visual.patch_embed.proj, op: linear, inputs: [pixel_values], dims: {in: d_model, out: d_vis_model}}
      - {name: model.visual.blocks, template: vblk, repeat: n_vis_layers, inputs: [model.visual.patch_embed.proj]}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    layers:
      - {name: mtp, template: fc_blk, inputs: [norm]}
      - {name: mtp.layers, template: blk, repeat: n_mtp_layers, inputs: [mtp]}
      - {name: mtp.norm, op: rmsnorm, inputs: [mtp.layers], dim: d_model}
  - name: speculative
    group: true
    aux_kind: speculative
    inputs: [norm]
    repeat: n_spec_layers
    layers:
      - {name: layers, template: blk, inputs: [norm]}
"""


def test_an_indexed_aux_group_gets_the_address_it_used_to_derive():
    out = migrate_yaml_text(AUX)
    assert out.ok and out.changed
    doc = _load(out.text)
    # A dotted child name IS the structural path.
    assert _layer(doc, "vision")["checkpoint"] == {
        "prefix": "model.visual.blocks", "index": "local"}
    # The one indexed child of the group, likewise.
    assert _layer(doc, "mtp")["checkpoint"] == {
        "prefix": "mtp.layers", "index": "local"}
    # A bare child label under a group that carries the count: the group.
    assert _layer(doc, "speculative")["checkpoint"] == {
        "prefix": "speculative", "index": "local"}
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_two_indexed_stacks_in_one_aux_group_are_left_to_the_schema():
    """The retired rule gave each stack its own prefix; one declaration on
    the group cannot say that, and the schema's refusal names what to write."""
    src = AUX.replace(
        "{name: mtp, template: fc_blk, inputs: [norm]}",
        '{name: mtp.heads, template: fc_blk, repeat: "2", inputs: [norm]}')
    out = migrate_yaml_text(src)
    assert out.ok
    assert "checkpoint" not in _layer(_load(out.text), "mtp")


# -- a custom op's layout contract -----------------------------------------

CUSTOM = """
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
custom_ops:
  packed_expert_linear:
    flops: "2 * B * s_ctx * In * Out"
    params: "In * Out"
    signature:
      x: [B, s_ctx, In]
    formula_symbols: [x]
    shape_kind: axis_replace
    out_key: Out
    weight_tensor_suffix: ""
    weight_shape: [Out, In]
    formula: "y = W[e] . x"
    description: "A packed expert projection, one naked 3-D Parameter."
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: experts, op: packed_expert_linear, inputs: [embed], dims: {In: d_model, Out: d_ffn_expert}}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [experts], dims: {in: d_model, out: n_vocab}}
"""


def test_a_custom_linear_maps_coupling_comes_from_its_weight():
    out = migrate_yaml_text(CUSTOM)
    assert out.ok and out.changed
    op = _load(out.text)["custom_ops"]["packed_expert_linear"]
    assert "formula_symbols" not in op
    assert op["couples_on"] == ["In"]
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_a_geometry_custom_ops_coupling_is_a_blocker():
    """A zero-parameter fold joins axes only its author can name; an empty
    `couples_on` would tell every layout that a shard may cut them."""
    src = CUSTOM.replace('    params: "In * Out"', '    params: "0"') \
        .replace("    weight_shape: [Out, In]\n", "")
    out = migrate_yaml_text(src)
    assert not out.ok
    assert any("packed_expert_linear" in b and "couples_on" in b
               and "axis_use" in b for b in out.blockers), out.blockers
    assert out.text == src


def test_a_tagged_geometry_op_names_its_coupled_axes():
    src = CUSTOM.replace('    params: "In * Out"', '    params: "0"') \
        .replace("    weight_shape: [Out, In]\n",
                 "    axis_use:\n"
                 "      s_ctx: {category: window, param: \"m*m patch fold\"}\n"
                 "      In: {category: contract, param: sum}\n")
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    op = _load(out.text)["custom_ops"]["packed_expert_linear"]
    assert op["couples_on"] == ["In", "s_ctx"]


def test_a_documents_own_op_labels_are_not_the_registrys_symbols():
    """`In` is the DSA indexer head count in the registry and the input width
    of this document's own op. A custom op's labels are local names bound at
    each call site, so neither they nor the keys that bind them are renamed
    with the vocabulary — a label that became a registered symbol is what
    breaks the axis declarations of the op that owns it."""
    out = migrate_yaml_text(CUSTOM)
    assert out.ok
    doc = _load(out.text)
    assert doc["custom_ops"]["packed_expert_linear"]["signature"]["x"] == [
        "B", "s_ctx", "In"]
    assert list(_layer(doc, "experts")["dims"]) == ["In", "Out"]


def test_the_refusal_after_a_source_names_both_readings():
    """D3 of the private-spec audit: threading the predecessor here would
    validate and be wrong (a vision tower's position table is ADDED to the
    patch embedding, not replaced by it), so the refusal has to tell the
    author what to declare."""
    src = """
match: {model_type: [qwen3_5]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers], dims: {in: d_model, out: n_vocab}}
  - name: vision
    group: true
    aux_kind: vision
    layers:
      - {name: model.visual.patch_embed.proj, op: linear, dims: {in: d_model, out: d_vis_model}}
      - {name: model.visual.pos_embed, op: embedding, dims: {n_vocab: n_vis_pos_vocab, d_model: d_vis_model}}
      - {name: model.visual.blocks, template: vblk, repeat: n_vis_layers}
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    [blocker] = [b for b in out.blockers
                 if "'model.visual.blocks' omits `inputs:`" in b]
    # The tower's first stage names no media either: that is the other refusal.
    assert [b for b in out.blockers if b != blocker] == [
        b for b in out.blockers if "'model.visual.patch_embed.proj'" in b
        and "Declare each media input" in b]
    # the table, the stage before it, and the op that reads both
    assert "inputs: [model.visual.patch_embed.proj]" in blocker
    assert ("inputs: [model.visual.patch_embed.proj, model.visual.pos_embed]"
            in blocker)
    assert "position_table_add" in blocker
    assert out.text == src


# -- which channel a refusal comes back through ----------------------------


def test_a_refusal_that_names_a_declaration_is_not_a_spec_to_regenerate():
    """The record is served again as soon as its author writes the one line
    the refusal names, so the serving boundary asks for that line instead of
    telling them to regenerate — and `errors[]` carries every refusal back to
    the spec writer either way."""
    src = """
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: scale, op: parameter_vector, dim: "1"}
  - {name: norm, op: rmsnorm, dim: d_model}
  - {name: head, op: linear, boundary_role: output_head, inputs: [norm], dims: {in: d_model, out: n_vocab}}
"""
    out = migrate_yaml_text(src)
    assert out.blockers and out.needs_declaration == out.blockers
    assert out.only_needs_declarations
    error = refuse("acme/Foo", out)
    assert isinstance(error, SpecMigrationBlocked)
    assert error.errors == list(out.blockers)
    from backend.spec_worker.bundle import normalize_error_payload
    payload = normalize_error_payload(error, model_id="acme/Foo")
    assert payload["error_kind"] == "spec_needs_declaration"
    assert payload["message"] == "Schema error."
    assert payload["errors"] == list(out.blockers)


def test_a_retired_syntax_is_still_a_spec_to_regenerate():
    """Nothing can be added to a document that declares a channel the
    language no longer has: that one is regenerate-only, as before."""
    src = """
role: attention
kv_cached: [k, v]
graph:
  x: {op: identity, in: [x]}
"""
    out = migrate_yaml_text(src)
    assert out.blockers and out.needs_declaration == ()
    assert not out.only_needs_declarations
    error = refuse("acme/Foo", out)
    assert not isinstance(error, SpecMigrationBlocked)
    assert "Regenerate it from the Inspector" in str(error)
    from backend.spec_worker.bundle import normalize_error_payload
    assert normalize_error_payload(
        error, model_id="acme/Foo")["error_kind"] == "spec_invalid"


# -- a rewrite keeps the author's comments -----------------------------------
#
# Comments carry the source citations a review reads. Every rule rewrites
# parsed data; the rewritten document goes back into the written text, so
# the comments stay on the entries they annotate and untouched lines stay
# byte for byte.

TIE_WEIGHTS = """
match: {model_type: [qwen3]}
model:
  - {name: embed_tokens, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - {name: lm_head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}, tie_weights: config.tie_word_embeddings}
"""

VISION_IN = """
match: {model_type: [hunyuan]}
model_inputs:
  pixel_values: {kind: image, axes: [n_img, img_h, img_w]}
model:
  - {name: embed_tokens, op: embedding, boundary_role: input_embedding}
  - {name: lm_head, op: linear, boundary_role: output_head}
  - name: vit
    group: true
    aux_kind: vision
    in: embed_tokens
    layers:
      - {name: vit, template: tower, inputs: [pixel_values]}
"""

SEQUENTIAL = """
match: {model_type: [deepseek_v3]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - {name: norm, op: rmsnorm, dim: d_model}
  - {name: head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    layers:
      - {name: mtp.enorm, op: rmsnorm, dim: d_model}
      - {name: mtp.layers, template: block}
"""

VIEW_SPLIT = """
match: {model_type: [x]}
dims: {d_model: config.hidden_size, n_vocab: config.vocab_size}
blocks:
  blk:
    inputs: [x]
    graph:
      split: {op: view_split, in: x, dims: {n_heads: n_heads, d_head: d_head}}
      merge: {op: view_merge, in: split, dims: {n_heads: n_heads, d_head: d_head, out: d_model}}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed]}
  - {name: hc_split, op: view_split, inputs: [layers], dims: {n_heads: n_hidden_copies, d_head: d_model}}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [hc_split], dims: {in: d_model, out: n_vocab}}
"""

#: One stored spec per rule class, with what the rule reads.
RULE_CLASSES = {
    "dim vocabulary": (lambda: OLD_VOCABULARY, None, None),
    "tie_weights": (lambda: TIE_WEIGHTS, None, None),
    "vision group inputs": (lambda: VISION_IN, None, None),
    "output head, retired document key": (lambda: NO_EXIT, None, None),
    "omitted inputs": (lambda: SEQUENTIAL, None, None),
    "view_split axis": (lambda: VIEW_SPLIT, None, None),
    "carried stream": (lambda: CHAINS, None, None),
    "keep port": (lambda: KEPT, None, None),
    "row merge": (lambda: TWO_INPUTS, None, None),
    "renamed component and dim": (lambda: VISION, lambda: COMPONENTS, None),
    "config switch": (lambda: ROUTED, lambda: ROUTERS, lambda: FAMILIES),
    "aux checkpoint": (lambda: AUX, None, None),
    "custom op keys": (lambda: CUSTOM, None, None),
}


def _commented(text: str) -> str:
    """`text` with a comment above and on every line that can hold one,
    each naming the line: `# above <n>`, `# on <n>`."""
    lines = text.splitlines()
    doc = _load(text)
    for n in range(len(lines) - 1, -1, -1):
        line = lines[n]
        if not line.strip():
            continue
        indent = line[:len(line) - len(line.lstrip())]
        for edit in (lambda: [*lines[:n], line + f"  # on {n}", *lines[n + 1:]],
                     lambda: [*lines[:n], indent + f"# above {n}", *lines[n:]]):
            edited = edit()
            try:
                if _load("\n".join(edited)) == doc:
                    lines = edited
            except yaml.YAMLError:
                pass
    return "\n".join(lines) + "\n"


def _comments(text: str) -> list[str]:
    import re
    return re.findall(r"# (?:above|on) \d+", text)


@pytest.mark.parametrize("rule", sorted(RULE_CLASSES))
def test_a_rewrite_keeps_every_comment_on_its_entry(rule):
    import re
    source, components, models = RULE_CLASSES[rule]
    components = components and components()
    models = models and models()
    plain = migrate_yaml_text(source(), components, models)
    assert plain.ok and plain.changed
    written = _commented(source())
    out = migrate_yaml_text(written, components, models)
    assert out.ok and out.changed
    assert _load(out.text) == _load(plain.text)
    # Not one comment lost or repeated.
    assert sorted(_comments(out.text)) == sorted(_comments(written))
    lines = out.text.splitlines()
    written_lines = written.splitlines()
    for i, line in enumerate(lines):
        on = re.search(r"# on (\d+)$", line)
        if not on:
            continue
        n = int(on.group(1))
        # A line's comments stay together: the one above it first.
        above = i - 1
        while above >= 0 and lines[above].lstrip().startswith("#") \
                and f"# above {n}" not in lines[above]:
            above -= 1
        original = next(l for l in written_lines if l.endswith(f"# on {n}"))
        if f"# above {n}" in "\n".join(written_lines):
            assert above >= 0 and f"# above {n}" in lines[above], (rule, n)
        # A comment still on a line is on the line of the same entry.
        content = line[:on.start()].strip()
        if content and not content.startswith("#"):
            assert content.split(":")[0] == original.split("#")[0].strip().split(":")[0], (
                rule, line, original)
    again = migrate_yaml_text(out.text, components, models)
    assert again.ok and not again.changed


def test_a_removed_keys_comments_move_to_its_replacement():
    """`tie_weights:` becomes `tied_weight:`: the citation on the removed key
    is the replacement's; the entries around it keep theirs where they are."""
    src = """\
match: {model_type: [qwen3]}
model:
  - {name: embed_tokens, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: block}
  - name: lm_head
    op: linear  # modeling_qwen3.py:512
    boundary_role: output_head
    dims: {in: d_model, out: n_vocab}
    # tied unless the config says otherwise
    tie_weights: config.tie_word_embeddings  # config.json
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    assert out.text.endswith("""\
  - name: lm_head
    op: linear  # modeling_qwen3.py:512
    boundary_role: output_head
    dims: {in: d_model, out: n_vocab}
    # tied unless the config says otherwise
    # config.json
    tied_weight:
      to: embed_tokens
      when: config.tie_word_embeddings
    inputs: [layers]
model_inputs:
  input_ids:
    kind: tokens
""")
    assert migrate_yaml_text(out.text).text == out.text


def test_a_rewrite_changes_only_the_rewritten_entries():
    src = """\
# Qwen3 — https://huggingface.co/Qwen/Qwen3-8B
match: {model_type: [qwen3]}
model:
  # embedding table, config.vocab_size rows
  - name: embed_tokens
    op: embedding  # nn.Embedding
    dims: {n_vocab: n_vocab, d_model: d_model}
  - {name: layers, repeat: n_layers, template: block}  # 36 layers
  - {name: lm_head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}}
# end of spec
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    assert out.text == """\
# Qwen3 — https://huggingface.co/Qwen/Qwen3-8B
match: {model_type: [qwen3]}
model:
  # embedding table, config.vocab_size rows
  - name: embed_tokens
    op: embedding  # nn.Embedding
    dims: {n_vocab: n_vocab, d_model: d_model}
    boundary_role: input_embedding
    inputs: [input_ids]
  - {name: layers, repeat: n_layers, template: block, inputs: [embed_tokens], streams: {hidden: {carries: true}}}  # 36 layers
  - {name: lm_head, op: linear, boundary_role: output_head, dims: {in: d_model, out: n_vocab}, inputs: [layers]}
model_inputs:
  input_ids:
    kind: tokens
# end of spec
"""


def test_comments_move_only_to_a_declared_replacement():
    """Several entries removed, one unrelated entry added: a removed entry's
    comments stay where it was written (right after the entry before it),
    and only a key declared as the replacement takes a removed key's."""
    from backend.arch.authored_yaml import AuthoredYaml
    written = """\
# about a
a: 1
# about b
b: 2  # b inline
c: 3
# about d
d: 4
# about e
e: 5
"""
    doc = {"c": 3, "added": 0, "e2": 5}
    assert AuthoredYaml(written).rewrite(doc, replaced={"e": "e2"}) == """\
# about a
# about b
# b inline
c: 3
# about d
added: 0
# about e
e2: 5
"""
    # With no declared replacement nothing is handed to an added key.
    assert AuthoredYaml(written).rewrite(doc) == """\
# about a
# about b
# b inline
c: 3
# about d
# about e
added: 0
e2: 5
"""


def test_a_rewrite_that_splices_logs_nothing_and_a_fallback_warns(caplog):
    import logging
    from backend.arch.authored_yaml import AuthoredYaml
    with caplog.at_level(logging.WARNING, logger="backend.arch.authored_yaml"):
        out = migrate_yaml_text(_commented(KEPT))
    assert out.ok and out.changed
    assert not caplog.records
    aliased = "# a comment\nx: &a {k: 1}\ny: *a\n"
    with caplog.at_level(logging.WARNING, logger="backend.arch.authored_yaml"):
        text = AuthoredYaml(aliased).rewrite({"x": {"k": 2}, "y": {"k": 1}})
    assert _load(text) == {"x": {"k": 2}, "y": {"k": 1}}
    assert "# a comment" not in text
    assert [r.getMessage() for r in caplog.records] == [
        "rewritten YAML dumped without its comments: anchors and aliases"]


def test_a_dead_duplicate_or_a_contested_replacement_keeps_its_place():
    """A replacement takes the comments of one source: the key's live
    occurrence. The dead first copy of a key written twice stays where it
    was written, and two keys declared onto one replacement both stay."""
    from backend.arch.authored_yaml import AuthoredYaml
    written = """\
# dead duplicate comment
e: dead
# live e comment
e: live
c: 3
"""
    assert AuthoredYaml(written).rewrite(
        {"c": 3, "e2": "live"}, replaced={"e": "e2"}) == """\
# dead duplicate comment
c: 3
# live e comment
e2: live
"""
    written = "# about a\na: 1\n# about b\nb: 2\nc: 3\n"
    assert AuthoredYaml(written).rewrite(
        {"c": 3, "x": 0}, replaced={"a": "x", "b": "x"}) == """\
# about a
# about b
c: 3
x: 0
"""


# -- what the model receives is declared ------------------------------------
#
# A lookup read the token ids without naming them; that is `input_ids:
# {kind: tokens}` now. An encoder's first stage read its media the same way,
# but no width in the record states the media's own axes or the processor's
# cut of them, so that one is a declaration for the author to write.

UNNAMED_INPUTS = """
match: {model_type: [x]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed], streams: {hidden: {carries: true}}}
  - {name: norm, op: rmsnorm, dim: d_model, inputs: [layers]}
  - {name: head, op: linear, boundary_role: output_head, inputs: [norm], dims: {in: d_model, out: n_vocab}}
  - name: vis
    group: true
    aux_kind: vision
    layers:
      - {name: vis.patch, op: linear, dims: {in: "n_vis_channels * s_patch * s_patch", out: d_model}}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    layers:
      - {name: mtp.embed, op: embedding, dims: {n_vocab: n_vocab, d_model: d_model}}
      - {name: mtp.layers, template: blk, inputs: [norm, mtp.embed]}
"""


def test_a_lookup_and_an_encoder_entry_read_declared_model_inputs():
    """Every lookup outside the encoder reads `input_ids`; the encoder's
    media is a declaration the author writes, not one read off a width."""
    from backend.arch.spec_migrate import migrate_mapping
    out = migrate_yaml_text(UNNAMED_INPUTS)
    assert not out.ok and out.text == UNNAMED_INPUTS
    assert out.needs_declaration == out.blockers
    [blocker] = out.blockers
    assert "encoder group 'vis' reads its media at 'vis.patch'" in blocker
    assert "Declare each media input" in blocker
    doc, _ = migrate_mapping(_load(UNNAMED_INPUTS))
    assert doc["model_inputs"] == {"input_ids": {"kind": "tokens"}}
    assert _layer(doc, "embed")["inputs"] == ["input_ids"]
    assert _layer(doc, "mtp", "mtp.embed")["inputs"] == ["input_ids"]
    assert "inputs" not in _layer(doc, "vis", "vis.patch")


def test_an_encoder_template_stage_is_a_media_input_to_declare():
    src = UNNAMED_INPUTS.replace(
        '{name: vis.patch, op: linear, dims: {in: "n_vis_channels * s_patch * s_patch", out: d_model}}',
        "{name: vis.tower, template: tower}")
    out = migrate_yaml_text(src)
    assert not out.ok
    assert out.needs_declaration == out.blockers
    assert any("'vis'" in b and "'vis.tower'" in b
               and "Declare each media input" in b
               for b in out.blockers), out.blockers


def test_a_vision_input_is_an_image_whose_axes_are_to_declare():
    """`kind: vision` is `kind: image`, and its width goes: an image's rows
    come from its own axes through a `patchify`, which the author declares."""
    from backend.arch.spec_migrate import migrate_mapping
    src = VISION_IN.replace("{kind: image, axes: [n_img, img_h, img_w]}",
                            "{kind: vision, width: d_model}")
    out = migrate_yaml_text(src)
    assert not out.ok and out.text == src
    assert out.needs_declaration == out.blockers
    [blocker] = out.blockers
    assert "model input 'pixel_values' (image) declares no `axes:`" in blocker
    doc, _ = migrate_mapping(_load(src))
    assert doc["model_inputs"]["pixel_values"] == {"kind": "image"}


def test_a_library_block_that_takes_the_token_ids_is_passed_them():
    src = """
match: {model_type: [x]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids]}
  - {name: norm, op: rmsnorm, dim: d_model, inputs: [embed]}
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [norm]
    layers:
      - {name: mtp, template: head_block, inputs: [norm]}
model_inputs:
  input_ids: {kind: tokens}
"""
    library = {"head_block": "inputs: [x, input_ids]\ngraph:\n"
                             "  e: {op: embedding, in: input_ids}\n"
                             "outputs: [e]\n"}
    out = migrate_yaml_text(src, library)
    assert out.ok and out.changed
    assert _layer(_load(out.text), "mtp", "mtp")["inputs"] == [
        "norm", "input_ids"]


def test_a_view_merge_states_no_width():
    src = """
role: attention
graph:
  merge: {op: view_merge, in: x, dims: {n_heads: n_heads, d_head: d_head, out: "n_heads * d_head"}}
  fold: {op: view_merge, in: merge, dims: {out: d_model}}
outputs: [fold]
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    graph = _load(out.text)["graph"]
    assert graph["merge"]["dims"] == {"n_heads": "n_heads", "d_head": "d_head"}
    assert "dims" not in graph["fold"]
    assert not migrate_yaml_text(out.text).changed


def test_a_graph_declares_the_outputs_the_retired_parser_chose():
    """Every node no other node reads, in graph order; the last node when
    every node is read."""
    src = """
role: attention
graph:
  n: {op: rmsnorm, in: x, dim: d_model}
  o: {op: add, in: [x, n], dim: d_model}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    assert _load(out.text)["outputs"] == ["o"]
    two = migrate_yaml_text(src + "  p: {op: rmsnorm, in: x, dim: d_model}\n")
    assert two.ok and _load(two.text)["outputs"] == ["o", "p"]


def test_a_lookup_wired_to_an_activation_is_a_blocker():
    """A lookup reads ids; one an old record wired to a computed tensor is a
    wrong edge, returned as a declaration to fix rather than kept."""
    model = UNNAMED_INPUTS.replace(
        "{name: mtp.embed, op: embedding, dims:",
        "{name: mtp.embed, op: embedding, inputs: [norm], dims:")
    out = migrate_yaml_text(model)
    assert not out.ok
    assert out.needs_declaration == out.blockers
    assert any("mtp.embed" in b and "'norm'" in b for b in out.blockers)
    block = """
role: aux_block
inputs: [x]
graph:
  n: {op: rmsnorm, in: x, dim: d_model}
  e: {op: embedding, in: n, dims: {n_vocab: n_vocab, d_model: d_model}}
outputs: [e]
"""
    out = migrate_yaml_text(block)
    assert not out.ok
    assert any("<component>.e" in b and "'n'" in b for b in out.blockers)


# --- inner streams: a reader names the layer it reads ----------------------

INNER = """
match: {model_type: [x]}
model_inputs:
  input_ids: {kind: tokens}
blocks:
  w: {stream_out: {s: n}, graph: {n: {op: rmsnorm, in: x, dim: d_model}}, outputs: [n]}
  r: {reads: [s], graph: {g: {op: mul, in: [x, s], dim: d_model}}, outputs: [g]}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids]}
  - name: layers
    repeat: n_layers
    inputs: [embed]
    streams:
      hidden: {carries: true}
      s: {src: "toy", window: "4", init: zeros}
    per_layer:
      assign:
        - {type: w, layers: ["0"]}
        - {type: r, rest: true}
      map:
        w: {template: w}
        r: {template: r}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers]}
"""


def test_inner_stream_annotations_go_and_a_reader_must_name_its_layer():
    """`window:` / `init:` described the retired latest-writer pairing and
    are dropped. The layer a reader reads is the reader's to declare: a
    reading assignment without `reads_from` is a declaration to write, never
    filled in from where the latest writer happened to be."""
    out = migrate_yaml_text(INNER)
    assert any("dropped window and init on inner stream 's'" in note
               for note in out.notes)
    (blocker,) = out.blockers
    assert "per_layer.assign[1] (type r)" in blocker
    assert "declare reads_from: { s: <layer> }" in blocker
    assert out.only_needs_declarations
    assert "reads_from" not in out.text


def test_a_declared_reader_migrates_clean():
    src = INNER.replace("{type: r, rest: true}",
                        '{type: r, rest: true, reads_from: {s: "0"}}')
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    stream = _load(out.text)["model"][1]["streams"]["s"]
    assert stream == {"src": "toy"}
    assert not migrate_yaml_text(out.text).changed


COMPRESSOR = """
role: attention
cache:
  entries:
    - { node: pooled_out, kind: compressed_kv }
    - { state: kv_state,    kind: compressor_state, carrier: weighted, elements: "compress_ratio * Cdim_io", dtype: fp32 }
    - { state: score_state, kind: compressor_state, carrier: smax,     elements: "compress_ratio * Cdim_io", dtype: fp32 }
graph:
  weighted: {op: identity, in: [x]}
  smax: {op: identity, in: [x]}
  pooled_out: {op: identity, in: [weighted]}
outputs: [pooled_out]
"""


def test_unique_cache_elements_become_shape():
    """DeepSeek-V4 compressor buffers were `elements: compress_ratio * Cdim_io`;
    every library copy stores that as `[compress_ratio, Cdim_io]`."""
    out = migrate_yaml_text(COMPRESSOR)
    assert out.ok and out.changed
    entries = _load(out.text)["cache"]["entries"]
    assert entries[0] == {"node": "pooled_out", "kind": "compressed_kv"}
    assert entries[1]["shape"] == ["compress_ratio", "Cdim_io"]
    assert "elements" not in entries[1]
    assert entries[2]["shape"] == ["compress_ratio", "Cdim_io"]
    assert not migrate_yaml_text(out.text).changed


def test_ambiguous_cache_elements_are_a_declaration():
    """GDN and Kimi-K3 wrote the same product in different stored orders."""
    src = """
role: attention
cache:
  entries:
    - { state: recurrent_state, kind: recurrent_state, carrier: delta_rule, elements: "n_lin_v_heads * d_lin_k_head * d_lin_v_head" }
graph:
  delta_rule: {op: identity, in: [x]}
outputs: [delta_rule]
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert out.only_needs_declarations
    assert "elements: n_lin_v_heads * d_lin_k_head * d_lin_v_head" in out.blockers[0]
    assert "shape:" in out.blockers[0]
    assert out.text == src


# -- a carried stream's `close:` (every carried stream settles) ------------

def test_a_carried_streams_close_key_is_dropped():
    """Every carried stream settles its layout at each iteration exit, so a
    stored `close: true` says nothing `carries: true` does not."""
    src = """
match: {model_type: [x]}
model_inputs: {input_ids: {kind: tokens}}
model:
  - {name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids]}
  - {name: layers, repeat: n_layers, template: blk, inputs: [embed], streams: {hidden: {carries: true, close: true}}}
  - {name: lm_head, op: linear, boundary_role: output_head, inputs: [layers]}
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    assert _layer(_load(out.text), "layers")["streams"] == {
        "hidden": {"carries": True}}
    again = migrate_yaml_text(out.text)
    assert again.ok and not again.changed


def test_a_recurrence_carried_state_declares_its_kind():
    """A state carried by a kernel that declares `carries_recurrent_state` is
    that recurrence's matrix; the rewrite says so."""
    src = """
role: attention
cache:
  family: Mamba2
  entries:
    - { state: recurrent_state, carrier: scan, shape: [n_ssm_heads, d_ssm_head, d_ssm_state] }
graph:
  scan: {op: mamba2_scan, in: [x]}
outputs: [scan]
"""
    out = migrate_yaml_text(src)
    assert out.ok and out.changed
    (entry,) = _load(out.text)["cache"]["entries"]
    assert entry["kind"] == "recurrent_state"
    assert not migrate_yaml_text(out.text).changed


def test_any_other_entry_without_a_kind_is_a_declaration():
    """Which node is a key, a latent or an index key is the author's
    statement; the record is opened for it, naming the kinds to choose."""
    src = """
role: attention
cache:
  family: GQA
  entries:
    - k
    - { state: window, carrier: conv, shape: ["kernel_size - 1", C] }
graph:
  k: {op: identity, in: [x]}
  conv: {op: identity, in: [x]}
outputs: [k]
"""
    out = migrate_yaml_text(src)
    assert not out.ok
    assert out.only_needs_declarations
    assert len(out.blockers) == 2
    assert "'k'" in out.blockers[0] and "{node: k, kind: <kind>" in out.blockers[0]
    assert "'kv'" in out.blockers[0]
    assert "'window'" in out.blockers[1] and "'conv_state'" in out.blockers[1]
