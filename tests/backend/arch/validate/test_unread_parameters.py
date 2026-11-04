"""A stored parameter that no node reads is a repairable validate error,
unless the spec declares it `stored_without_graph_reader` with its kind.

The corpus had this bug four times: a position table the model samples by
interpolation written as a lookup the blocks never read (Qwen3.8-27B,
Qwen3.5-35B-A3B; vLLM qwen3_vl.py `fast_pos_embed_interpolate`), MiMo-V2.5's
speech-code banks left beside an audio encoder that started from nothing
(vLLM mimo_audio.py `apply_speech_embeddings` sums them into its input), and
LongCat-Flash's MTP `embed_tokens` while `enorm` read the hidden state
(vLLM longcat_flash_mtp.py embeds the ids with its own table). The byte
anchor sees none of them: the table's bytes are stored either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.validate._unread_parameters import run_unread_parameter_check
from tests.snapshots import shared_loader

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolved(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    return shared_loader().match(config, model_id=stem.replace("__", "/", 1))


def _findings(resolved):
    """The check on the checkpoint walk validate runs: every stack, each in
    its own phase."""
    return run_unread_parameter_check(Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=8).checkpoint())


def test_the_tables_the_forward_source_reads_are_read():
    """Each model that carried the bug reads its table now."""
    for stem in ("Qwen__Qwen3.5-35B-A3B", "XiaomiMiMo__MiMo-V2.5",
                 "meituan-longcat__LongCat-Flash-Thinking-2601"):
        assert _findings(_resolved(stem)) == [], stem


def test_a_lookup_nothing_reads_is_named_with_its_repair():
    """A table looked up at the model root whose rows no node reads — the
    shape of MiMo-V2.5's speech banks left beside an encoder that read
    something else: the table is the finding, and the message says how to
    repair it."""
    resolved = _resolved("XiaomiMiMo__MiMo-V2.5")
    spec = resolved.spec
    embed = next(e for e in spec.model if e.name == "embed_tokens")
    orphan = embed.model_copy(update={"name": "orphan_table",
                                      "boundary_role": None})
    resolved = resolved.with_spec(spec.model_copy(
        update={"model": [*spec.model, orphan]}))
    (finding,) = _findings(resolved)
    assert finding.kind == "unread_lookup_table"
    assert finding.node == "orphan_table"
    assert "no node reads" in finding.detail
    assert "`inputs:`" in finding.detail
    assert "parameter_tensor" in finding.detail


def _block_node(resolved, block: str, node: str, **update):
    """The spec with one inline block node's fields replaced."""
    spec = resolved.spec
    graph = dict(spec.blocks[block].graph)
    graph[node] = graph[node].model_copy(update=update)
    blocks = {**spec.blocks,
              block: spec.blocks[block].model_copy(update={"graph": graph})}
    return resolved.with_spec(spec.model_copy(update={"blocks": blocks}))


def test_the_declaring_models_are_clean():
    """Kimi-K3's layer-0 AttnRes scoring weights and HunyuanOCR's
    `image_sep` are never read by the forward; Qwen3.8-Flash-Next's PLE
    index tables are read by the n-gram id hash, a step the
    graph does not model. All are stored, have no graph reader, and are
    declared with their kind: no finding."""
    kinds = {}
    for stem in ("moonshotai__Kimi-K3", "tencent__HunyuanOCR",
                 "Qwen__Qwen3.8-Flash-Next"):
        program = Program.of(_resolved(stem))
        for n in program.nodes():
            if n.stored_without_graph_reader is not None:
                kinds[n.name] = n.stored_without_graph_reader.kind
        assert _findings(_resolved(stem)) == [], stem
    assert kinds == {
        "self_attention_res_norm.weight": "forward_unread",
        "self_attention_res_proj.weight": "forward_unread",
        "vit.perceive.image_sep": "forward_unread",
        "ple_embedding.layer_multipliers": "unmodelled_reader",
        "ple_embedding.ngram_heads_offsets": "unmodelled_reader",
        "ple_embedding.ngram_heads_vocab_sizes": "unmodelled_reader",
    }


def test_an_undeclared_unread_parameter_is_named_with_both_repairs():
    """Kimi-K3 without its layer-0 declaration: the two weights are the
    findings, and the message offers the reader and the declaration."""
    resolved = _resolved("moonshotai__Kimi-K3")
    for name in ("self_attention_res_norm.weight",
                 "self_attention_res_proj.weight"):
        resolved = _block_node(resolved, "block_kda_dense_first", name,
                               stored_without_graph_reader=None)
    findings = _findings(resolved)
    assert {f.kind for f in findings} == {"unread_stored_parameter"}
    assert sorted(f.node.rsplit(".", 2)[-2] for f in findings) == [
        "self_attention_res_norm", "self_attention_res_proj"]
    assert all("stored_without_graph_reader" in f.detail and "source_anchor" in f.detail
               and "forward_unread" in f.detail and "unmodelled_reader" in f.detail
               and "`in:`" in f.detail for f in findings)


def test_a_declared_parameter_that_is_read_is_refused():
    """Kimi-K3's layer-1+ scoring weight declared unread while the combine
    reads it: the declaration is the error."""
    resolved = _resolved("moonshotai__Kimi-K3")
    declared = Program.of(resolved).by_uid(next(
        n.uid for n in Program.of(resolved).nodes()
        if n.stored_without_graph_reader is not None)).stored_without_graph_reader
    resolved = _block_node(resolved, "block_kda_moe", "mlp_res_norm.weight",
                           stored_without_graph_reader=declared)
    # one finding per place the template is instantiated (a visibility
    # group of the AttnRes history), each the same refusal
    findings = _findings(resolved)
    assert findings and {f.kind for f in findings} == {"stored_without_graph_reader_is_read"}
    assert all("Drop the declaration" in f.detail for f in findings)


def test_stored_without_graph_reader_on_a_compute_node_is_refused():
    """Only a parameter source declares it: a norm that owns a weight but
    computes is not one."""
    resolved = _resolved("moonshotai__Kimi-K3")
    declared = next(n for n in Program.of(resolved).nodes()
                    if n.stored_without_graph_reader is not None).stored_without_graph_reader
    resolved = _block_node(resolved, "block_kda_moe", "input_layernorm",
                           stored_without_graph_reader=declared)
    findings = _findings(resolved)
    assert findings and {f.kind for f in findings} == {
        "stored_without_graph_reader_not_a_parameter"}
    assert all("parameter_tensor" in f.detail for f in findings)


def test_the_declaration_needs_a_kind_a_reason_and_an_anchor():
    import pytest
    from backend.arch.schema.weights import StoredWithoutGraphReaderSpec
    for kind in ("forward_unread", "unmodelled_reader"):
        StoredWithoutGraphReaderSpec(kind=kind, reason="x", source_anchor="f.py:1")
    ok = {"kind": "forward_unread", "reason": "x", "source_anchor": "f.py:1"}
    for bad in ({**ok, "reason": " "}, {**ok, "source_anchor": ""},
                {"kind": "forward_unread", "reason": "x"},
                {"reason": "x", "source_anchor": "f.py:1"},
                {**ok, "kind": "unread"}, {**ok, "why": "y"}):
        with pytest.raises(ValueError):
            StoredWithoutGraphReaderSpec(**bad)


def test_declaring_it_moves_no_stored_byte():
    """The declaration is about readers only: the stored rows, with and
    without it, are the same rows."""
    from backend.arch.eval.transforms import Deployment
    resolved = _resolved("moonshotai__Kimi-K3")
    bare = resolved
    for name in ("self_attention_res_norm.weight",
                 "self_attention_res_proj.weight"):
        bare = _block_node(bare, "block_kda_dense_first", name,
                           stored_without_graph_reader=None)

    def rows(r):
        analysis = Deployment(Program.of(r)).at(batch=1, input_tokens=8).checkpoint()
        return sorted((row.name, tuple(row.shape), str(row.storage_dtype),
                       str(row.count))
                      for row in analysis.iter_checkpoint())

    assert rows(resolved) == rows(bare)
