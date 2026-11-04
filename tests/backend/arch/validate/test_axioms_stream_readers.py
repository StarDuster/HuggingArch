"""Claims on a component that reads a cross-layer stream.

Claims are judged at each call of the component in the model's analysis, so
the stream it reads arrives from its real producer — with or without a
`shape:` on the stack's declaration.

The model: a Llama backbone whose layer 0 publishes its keys as `shared_kv`
and whose other layers run sliding-window attention that also reads them.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.arch.loader import SpecLoader
from backend.arch.validate._axioms import run_axiom_checks

ARCH = Path(__file__).resolve().parents[4] / "backend" / "arch"
LIB = ARCH / "components" / "library"

_STREAM = "      shared_kv: { shape: [B, n_kv_heads, s_kv, d_head], src: test }\n"


def _model(stream_line: str) -> str:
    text = (ARCH / "models" / "llama.yaml").read_text()
    text = text.replace("match:\n  model_type: [llama]",
                        "match:\n  model_id: [test/shared-kv]\n\n"
                        "drafts: [kv_source_attention, kv_reader_attention]")
    text = text.replace("  s_max: config.max_position_embeddings\n",
                        "  s_max: config.max_position_embeddings\n"
                        '  s_window: "128"\n')
    stack = ("  - { name: layers, repeat: n_layers, template: block, "
             "inputs: [embed_tokens], streams: { hidden: { carries: true } }}\n")
    assert stack in text
    return text.replace(stack, (
        "  - name: layers\n"
        "    inputs: [embed_tokens]\n"
        "    repeat: n_layers\n"
        "    streams:\n"
        "      hidden: { carries: true }\n"
        f"{stream_line}"
        "    per_layer:\n"
        "      assign:\n"
        '        - { type: source, layers: ["0"] }\n'
        "        - { type: reader, rest: true,"
        " reads_from: { shared_kv: \"0\" } }\n"
        "      map:\n"
        "        source: { attention: kv_source_attention }\n"
        "        reader: { attention: kv_reader_attention }\n"))


def _source() -> str:
    return ((LIB / "gqa_attention.yaml").read_text()
            + "stream_out: { shared_kv: k_rope }\n")


def _reader() -> str:
    text = (LIB / "swa_attention.yaml").read_text()
    text = text.replace("graph:\n", "reads: [shared_kv]\ngraph:\n", 1)
    return text.replace(
        "outputs: [o_proj]",
        "  k_window:        { op: sliding_window_kv, in: k_rope, "
        "dims: { s_window: s_window } }\n"
        "  k_union:         { op: kv_stream_concat, in: [k_window, shared_kv] }\n"
        "outputs: [o_proj]")


def _claims(storage: str) -> str:
    return ("schema: huggingarch.claims/v1\n"
            "subject: kv_reader_attention\n"
            "axioms:\n"
            "  checks:\n"
            f"    - {{ kind: kv_growth, node: k_rope, {storage}, src: test }}\n")


def _resolved(tmp_path, *, storage: str, stream_line: str = _STREAM):
    models = tmp_path / "models"
    drafts = tmp_path / "components" / "drafts"
    claims = tmp_path / "claims" / "components"
    for d in (models, drafts, claims):
        d.mkdir(parents=True)
    (models / "test__shared-kv.yaml").write_text(_model(stream_line))
    (drafts / "kv_source_attention.yaml").write_text(_source())
    (drafts / "kv_reader_attention.yaml").write_text(_reader())
    (claims / "kv_reader_attention.yaml").write_text(_claims(storage))
    config = json.loads(
        (ARCH / "snapshots" / "meta-llama__Llama-3.2-1B.json").read_text())["config"]
    loader = SpecLoader(overlay_model_dirs=[str(models)],
                        overlay_draft_dirs=[str(drafts)])
    return loader.resolve(loader.get("test__shared-kv"), config)


def _reader_findings(resolved):
    return [f for f in run_axiom_checks("test/shared-kv", resolved)
            if f.template == "kv_reader_attention"]


def test_a_stream_readers_kv_growth_axiom_is_checked(tmp_path):
    resolved = _resolved(tmp_path, storage="storage: windowed, bound: s_window")
    assert _reader_findings(resolved) == []


def test_a_wrong_axiom_on_a_stream_reader_fires(tmp_path):
    # The reader's own cache is its window: declaring it per_token is wrong,
    # and the check has to be able to say so.
    resolved = _resolved(tmp_path, storage="storage: per_token")
    kinds = {f.kind for f in _reader_findings(resolved)}
    assert kinds == {"axiom_kv_growth_violation"}


def test_a_stream_with_no_declared_shape_is_still_judged(tmp_path):
    resolved = _resolved(
        tmp_path, storage="storage: per_token",
        stream_line="      shared_kv: { src: test }\n")
    kinds = {f.kind for f in _reader_findings(resolved)}
    assert kinds == {"axiom_kv_growth_violation"}
