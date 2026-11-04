"""Inner streams read where the reader's declaration says.

A layer that reads an inner stream names the layer whose write it reads —
`reads_from` on the declaration that selects it: a layer number, `"i - k"`,
`{stack, layer}`, or `{nearest: <set>}`, the nearest earlier layer of a set
the config lists — and that layer writes the stream from a node.
`SpecIRTree.stream_reads` / `stream_writes` hold the declarations;
`Program.stream_origin` reads them. Every refusal names `reads_from` and the
layers that write the stream.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from backend.arch.eval.program import Program
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval.transforms import DECODE, PREFILL, Deployment
from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import StreamReadRefusal
from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config


# --- GLM-5.2: the shared layers reuse the top-k of the full layer before ----

@pytest.mark.parametrize("phase", [PREFILL, DECODE])
def test_glm52_shared_layer_reads_the_topk_of_the_full_layer_before_it(phase):
    """HF's decoder loop hands every layer the top-k the layer before it
    returned (`prev_topk_indices=topk_indices`), and a shared layer returns
    the top-k it was handed. So each shared layer i in
    `config.indexer_types` reads the `dsa_topk_mask` of the nearest full
    (or dense) layer before i — declared `{nearest: {from:
    config.indexer_types, is: full}}` — in the prefill program and the
    decode one."""
    resolved = resolve_snapshot("zai-org/GLM-5.2", shared_loader())
    kinds = snapshot_config("zai-org/GLM-5.2")["indexer_types"]
    analysis = Deployment(Program.of(resolved)).during(phase).analyze()
    program = analysis.program
    masks = {}
    for template in program.blocks:
        for node in template.nodes():
            if node.op == "dsa_topk_mask" and template.meta.aux_kind is None:
                for layer in template.layer_indices:
                    masks[layer] = node.uid
    (stack,) = {t.meta.iteration_owner
                for t in program.blocks if t.meta.aux_kind is None}
    shared = [i for i, kind in enumerate(kinds) if kind == "shared"]
    assert len(shared) == 57
    for i in shared:
        full = max(j for j in range(i) if kinds[j] == "full")
        origin = program.stream_origin(stack, i, "dsa_topk")
        assert (origin.uid, origin.iteration) == (masks[full], full), i
    assert program.stream_distances() == {"dsa_topk": 3}


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5", "zai-org/GLM-5.1"])
def test_a_stream_written_and_never_read_is_no_finding(model_id):
    """Writing a stream is what a component can do: DeepSeek-V3.2 declares
    none, GLM-5 / 5.1 declare `dsa_topk` and run no reader. The model
    analyses in both phases and nothing reads a stream."""
    resolved = resolve_snapshot(model_id, shared_loader())
    program = Program.of(resolved)
    assert program.ir.stream_reads == {}
    for phase in (PREFILL, DECODE):
        Deployment(program).during(phase).analyze()


# --- a small stack ----------------------------------------------------------

_BLOCKS = """\
  w:
    stream_out: { s: n }
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
  r:
    reads: [s]
    graph:
      g: { op: mul, in: [x, s], dim: d_model }
    outputs: [g]
  plain:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
"""

_HEAD = """\
  - { name: head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _spec(name: str, assign: str, *, blocks: str = _BLOCKS,
          streams: str = "s: {}", extra: str = "", before: str = "") -> str:
    """A six-layer stack; each `assign` row's type runs the block of that
    name."""
    types = dict.fromkeys(re.findall(r"type: (\w+)", assign))
    mapping = "".join(f"        {t}: {{ template: {t} }}\n" for t in types)
    return f"""\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims: {{ d_model: config.hidden_size, n_layers: "6" }}
blocks:
{blocks}model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
{before}  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }}, {streams} }}
    per_layer:
      assign:
{assign}      map:
{mapping}{_HEAD}{extra}"""


def _load(name: str, text: str, **config):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(text)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8, **config}, model_id=name)


def _program(name: str, assign: str, config=None, **kw) -> Program:
    return Program.of(_load(name, _spec(name, assign, **kw), **(config or {})))


def _refused(name: str, assign: str, **kw) -> StreamReadRefusal:
    with pytest.raises(StreamReadRefusal) as refused:
        _program(name, assign, **kw)
    assert refused.value.declaration == "reads_from"
    return refused.value


def _rows(*rows: str) -> str:
    return "".join(f"        - {row}\n" for row in rows)


_KINDS = {"kinds": ["full", "shared", "shared", "full", "shared", "shared"]}


@pytest.mark.parametrize("nearest, config", [
    ("{ from: config.kinds, is: full }", _KINDS),
    ("{ from: config.sources }", {"sources": [0, 3]}),
])
def test_a_reader_reads_the_nearest_layer_of_the_declared_set(nearest, config):
    """Layers 0 and 3 write; 1, 2, 4, 5 read the nearest of the set the
    config lists — a per-layer list selected `is: full`, the selector
    `per_layer.assign` uses, or a list of layer numbers."""
    program = _program("srd_nearest", _rows(
        '{ type: w, layers: ["0", "3"] }',
        f'{{ type: r, rest: true, reads_from: {{ s: {{ nearest: {nearest} }} }} }}'),
        config=config)
    (reader,) = [t for t in program.blocks if t.name == "r"]
    stack = reader.meta.iteration_owner
    assert {layer: program.stream_origin(stack, layer, "s").iteration
            for layer in reader.layer_indices} == {1: 0, 2: 0, 4: 3, 5: 3}
    (form, rows), = program.declared_stream_reads(
        program.block_index(reader)).values()
    assert form.startswith("nearest config.")
    assert rows == ((1, 0), (2, 0), (4, 3), (5, 3))
    for phase in (PREFILL, DECODE):
        Deployment(program).during(phase).analyze()


@pytest.mark.parametrize("mapping, config", [
    ('{ map: { "3": "2", "4": "0", "5": "2" } }', {}),
    ("{ map: config.sources_by_reader }", {"sources_by_reader": [2, 0, 2]}),
])
def test_a_map_reads_any_earlier_layer_per_reader(mapping, config):
    """The general form: each reading layer's source written out — here an
    irregular one, readers 3, 4, 5 reading 2, 0, 2 — or a config list
    aligned with the selected layers. One reading template, no split."""
    program = _program("srd_map", _rows(
        '{ type: w, layers: ["0", "2"] }',
        f'{{ type: r, layers: ["3", "4", "5"], reads_from: {{ s: {mapping} }} }}',
        '{ type: plain, rest: true }'), config=config)
    (reader,) = [t for t in program.blocks if t.name == "r"]
    stack = reader.meta.iteration_owner
    assert {layer: program.stream_origin(stack, layer, "s").iteration
            for layer in reader.layer_indices} == {3: 2, 4: 0, 5: 2}
    for phase in (PREFILL, DECODE):
        Deployment(program).during(phase).analyze()


@pytest.mark.parametrize("mapping, why", [
    ('{ map: { "3": "2", "4": "0" } }', "no source for 5"),
    ('{ map: { "3": "2", "4": "0", "5": "2", "1": "0" } }',
     "1 are not selected"),
    ('{ map: { "3": "2", "4": "1", "5": "2" } }',
     "from layer 1 of stack 'layers', which does not write it"),
])
def test_a_map_covers_exactly_the_selected_layers_from_writers(mapping, why):
    refusal = _refused("srd_map_bad", _rows(
        '{ type: w, layers: ["0", "2"] }',
        f'{{ type: r, layers: ["3", "4", "5"], reads_from: {{ s: {mapping} }} }}',
        '{ type: plain, rest: true }'))
    assert why in str(refusal)
    assert "layers 0, 2 write it" in refusal.fix


def test_a_set_member_that_does_not_write_is_refused():
    """Every member of the set writes the stream: layer 3 runs `plain`."""
    refusal = _refused("srd_set_silent", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: plain, layers: ["3"] }',
        '{ type: r, rest: true, reads_from: { s: { nearest: { from: config.kinds, is: full } } } }'),
        config=_KINDS)
    assert "layers 3 of stack 'layers' are in the set and do not write" in str(
        refusal)
    assert "layers 0 write it" in refusal.fix


def test_a_reader_with_no_set_member_before_it_is_refused():
    refusal = _refused("srd_set_first", _rows(
        '{ type: w, layers: ["3"] }',
        '{ type: r, rest: true, reads_from: { s: { nearest: { from: config.sources } } } }'),
        config={"sources": [3]})
    assert "the set has none before it" in str(refusal)
    assert "config.sources" in refusal.fix


def test_a_missing_reads_from_names_the_writers_and_the_config_list():
    """The fix lists the layers that write the stream and the config field
    that lists exactly them."""
    refusal = _refused("srd_missing", _rows(
        '{ type: w, layers: ["0", "3"] }',
        '{ type: r, rest: true }'), config=_KINDS)
    assert "names no layer it reads it from" in str(refusal)
    assert "layers 0, 3 write it" in refusal.fix
    assert "{nearest: {from: config.kinds, is: full}}" in refusal.fix
    assert refusal.meta.template_name == "r"


def test_a_source_that_does_not_write_is_refused():
    refusal = _refused("srd_nowrite", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: plain, layers: ["1"] }',
        '{ type: r, layers: ["2"], reads_from: { s: "i - 1" } }',
        '{ type: plain, rest: true }'))
    assert "from layer 1 of stack 'layers', which does not write it" in str(
        refusal)
    assert "layers 0 write it" in refusal.fix


@pytest.mark.parametrize("value, why", [
    ('"2"', "from layer 2, which is not a layer before it"),
    ('"i - 3"', "from layer -1, which is not a layer before it"),
    ('"i - 0"', "is not a layer before the reading one"),
])
def test_a_source_not_before_the_reader_is_refused(value, why):
    refusal = _refused("srd_order", _rows(
        '{ type: w, layers: ["0"] }',
        f'{{ type: r, layers: ["2"], reads_from: {{ s: {value} }} }}',
        '{ type: plain, rest: true }'))
    assert why in str(refusal)


def test_a_source_past_the_other_stack_is_refused():
    extra = """\
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [layers]
    repeat: "1"
    checkpoint: { prefix: mtp, index: local }
    streams: { hidden: { carries: true } }
    layers:
      - { name: blk, template: r, inputs: [layers],
          reads_from: { s: { stack: layers, layer: "n_layers" } } }
"""
    refusal = _refused("srd_past", _rows(
        '{ type: w, rest: true }'), extra=extra)
    assert "from layer 6 of stack 'layers', which has layers 0..5" in str(
        refusal)


def test_a_cross_stack_read_crosses_a_pipeline_cut():
    """An MTP-like head reads the stream layer 0 of the backbone wrote,
    `{stack: layers, layer: "0"}`. At pp = 2 layer 0 is on stage 0 and the
    head on stage 1: the value is sent across cut 0."""
    from backend.arch.layout.parallel_config import ParallelConfig
    from backend.arch.layout.stages import StagePlacement
    from dataclasses import replace
    extra = """\
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [layers]
    repeat: "1"
    checkpoint: { prefix: mtp, index: local }
    streams: { hidden: { carries: true } }
    layers:
      - { name: blk, template: r, inputs: [layers],
          reads_from: { s: { stack: layers, layer: "0" } } }
"""
    program = _program("srd_cross", _rows('{ type: w, rest: true }'),
                       extra=extra)
    (reader,) = [t for t in program.blocks if t.name == "r"]
    stack = reader.meta.iteration_owner
    origin = program.stream_origin(stack, 0, "s")
    assert origin.iteration == 0
    deployment = Deployment(program).at(batch=1, input_tokens=4)
    analysis = replace(deployment, config=ParallelConfig(pp=2)).analyze()
    placement = StagePlacement.of(analysis, 2)
    (hop,) = [c for c in placement.crossings
              if c.producer[0] == origin.uid and c.source_iteration == 0]
    assert hop.cut == 0
    assert {placement.stage(s) for s, _ in hop.sinks} == {1}


def test_stacks_that_read_each_other_are_refused():
    """The backbone reads the head's stream and the head reads the
    backbone's output: no order walks both."""
    blocks = _BLOCKS + """\
  hw:
    stream_out: { t: n }
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
  tr:
    reads: [t]
    graph:
      g: { op: mul, in: [x, t], dim: d_model }
    outputs: [g]
"""
    extra = """\
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [layers]
    repeat: "1"
    checkpoint: { prefix: mtp, index: local }
    streams: { hidden: { carries: true }, t: {} }
    layers:
      - { name: blk, template: hw, inputs: [layers] }
"""
    text = _spec("srd_cycle", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: tr, layers: ["1"], reads_from: { t: { stack: mtp, layer: "0" } } }',
        '{ type: plain, rest: true }'), blocks=blocks, extra=extra)
    with pytest.raises(SpecRefusal) as refused:
        _ = Program.of(_load("srd_cycle", text)).walk_stacks
    assert refused.value.declaration == "reads_from"
    message = str(refused.value)
    assert "backbone -> mtp -> backbone" in message
    assert "reads stream 't' from layer 0 of 'mtp'" in message
    assert "'mtp' reads 'layers'" in message
    assert "reads_from" in refused.value.fix


def test_a_head_stage_named_like_a_backbone_stage_is_its_own():
    """An MTP head whose stack is called `layers`, like the backbone's
    (MiMo): the head's `mtp_logits` reads its own `layers`, the backbone's
    `head` reads the backbone's, and the head reads the backbone's stream.
    The head evaluates after the backbone, and both phases analyse."""
    extra = """\
  - name: mtp
    group: true
    aux_kind: mtp
    inputs: [head]
    checkpoint: { prefix: mtp, index: local }
    layers:
      - { name: layers, repeat: "1", template: r, inputs: [head],
          streams: { hidden: { carries: true } },
          reads_from: { s: { stack: layers, layer: "5" } } }
      - { name: mtp_logits, op: linear, boundary_role: output_head,
          inputs: [layers], dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
"""
    program = _program("srd_named_head", _rows('{ type: w, rest: true }'),
                       extra=extra)
    assert [stack.name for stack in program.walk_stacks] == ["", "mtp"]
    for phase in (PREFILL, DECODE):
        Deployment(program).during(phase).analyze()


def test_an_encoder_and_the_backbone_that_read_each_other_are_refused():
    """The tower's stage reads the backbone's embedding and the backbone
    reads the tower's stage. Only a modality encoder's stages are read by
    name from outside it, and an encoder reads its media and no backbone
    tensor (`ArchSpec.encoder_groups`); this group reads no media, so the
    backbone's read names no layer it may read, and the two stacks never
    wait for each other: refused at load."""
    text = f"""\
match: {{ model_type: [srd_named_cycle], model_id: [srd_named_cycle] }}
dims: {{ d_model: config.hidden_size, n_layers: "2" }}
blocks:
{_BLOCKS}model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: vision
    group: true
    aux_kind: vision
    layers:
      - {{ name: tower, op: rmsnorm, dim: d_model, inputs: [embed],
          weight_axes: [{{ replicated: true }}] }}
  - name: layers
    inputs: [tower]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    template: plain
{_HEAD}"""
    with pytest.raises(Exception, match=r"references unknown layer 'tower'"):
        _load("srd_named_cycle", text)


def test_a_reads_from_beside_a_named_read_between_two_stacks_is_named():
    """The tower reads the backbone's embedding by name AND the backbone's
    stream through `reads_from`, and the backbone reads the tower. Only a
    modality encoder's stages are read by name from outside it, and an
    encoder reads no backbone tensor or stream: refused at load, before any
    order is sought."""
    extra = """\
  - name: vision
    group: true
    aux_kind: vision
    checkpoint: { prefix: vision, index: local }
    layers:
      - { name: tower, repeat: "1", template: r, inputs: [embed],
          streams: { hidden: { carries: true } },
          reads_from: { s: { stack: layers, layer: "0" } } }
"""
    text = _spec("srd_parallel", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: plain, rest: true }'), extra=extra)
    # The backbone reads the tower, declared after it.
    text = text.replace("  - name: layers\n    inputs: [embed]",
                        extra + "  - name: layers\n    inputs: [tower]")
    text = text[:text.rindex(extra)]
    with pytest.raises(Exception, match=r"references unknown layer 'tower'"):
        _load("srd_parallel", text)


def test_a_reads_from_no_component_reads_is_refused():
    refusal = _refused("srd_unread", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: plain, rest: true, reads_from: { s: "i - 1" } }'))
    assert "reads no stream 's'" in str(refusal)


def test_a_reads_from_of_an_appended_stream_is_refused():
    refusal = _refused("srd_appended", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: plain, rest: true, reads_from: { hist: "i - 1" } }'),
        streams=("s: {}, hist: { from: hidden, after: every, index: block, "
                 "src: toy }"))
    assert "'hist' is not an inner stream" in str(refusal)
    assert "read: each" in refusal.fix


@pytest.mark.parametrize("value", ['"i + 1"', '"2 * i"', '["0", "1"]'])
def test_a_reads_from_that_names_no_single_layer_is_refused(value):
    """A layer reads one layer's write; reading several (V4.1's three-layer
    `main_hidden`) is an appended stream."""
    with pytest.raises(Exception, match="appended stream"):
        _program("srd_grammar", _rows(
            '{ type: w, layers: ["0"] }',
            f'{{ type: r, rest: true, reads_from: {{ s: {value} }} }}'))


def test_reads_from_on_a_per_layer_stack_is_refused():
    text = _spec("srd_stack_level", _rows(
        '{ type: w, layers: ["0"] }',
        '{ type: r, rest: true, reads_from: { s: "0" } }'))
    text = text.replace("    repeat: n_layers\n",
                        '    repeat: n_layers\n    reads_from: { s: "0" }\n')
    with pytest.raises(Exception, match=r"write it on the `per_layer\.assign`"):
        _load("srd_stack_level", text)


def test_a_layer_writes_a_stream_once():
    blocks = _BLOCKS + """\
  both:
    graph:
      a: { op: w, in: x }
      b: { op: w, in: a }
    outputs: [b]
"""
    text = _spec("srd_twice", _rows('{ type: both, rest: true }'),
                 blocks=blocks)
    with pytest.raises(StreamReadRefusal, match="writes stream 's' 2 times"):
        Program.of(_load("srd_twice", text))


def test_one_template_reading_differing_sources_is_refused():
    """A template is evaluated once for all its layers: the layers it
    stands for read one shape. Layer 2 reads a write of `d_model` and layer
    4 one of `2 * d_model`."""
    blocks = _BLOCKS + """\
  wide:
    stream_out: { s: n }
    graph:
      n: { op: linear, in: x, dims: { in: d_model, out: "2 * d_model" },
           weight_axes: [{ replicated: true }, { replicated: true }] }
      o: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [o]
  rr:
    reads: [s]
    graph:
      g: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      r: { op: identity, in: s }
    outputs: [g]
"""
    text = _spec("srd_differ", _rows(
        '{ type: w, layers: ["1"] }',
        '{ type: wide, layers: ["3"] }',
        '{ type: rr, layers: ["2", "4"], reads_from: { s: "i - 1" } }',
        '{ type: plain, rest: true }'), blocks=blocks)
    program = Program.of(_load("srd_differ", text))
    with pytest.raises(SpecRefusal, match="from layers whose values differ"
                       ) as refused:
        Deployment(program).analyze()
    assert refused.value.declaration == "reads_from"
    assert "adapter template" in refused.value.fix


# --- DeepSeek-V4.1-Flash: one shared slot, five inner streams ---------------

_V41 = "deepseek-ai/DeepSeek-V4.1-Flash"


def _v41():
    """The library model and its text config, from its snapshot."""
    return (resolve_snapshot(_V41, shared_loader()),
            snapshot_config(_V41)["text_config"])


def test_v41_layers_read_the_source_layers_its_config_names():
    """The reference implementation keeps one `SharedAttentionRuntime` slot
    (source/model.py:1166-1180): the kv sources write the compressed KV and
    the index keys, the index sources the top-k, the candidate source the
    candidate mask, and every other layer reads the slot as the nearest
    source left it. Declared as `reads_from: {nearest: {from: <the config's
    source list>}}`, the producers each read resolves to are the config's
    source layers — and the model analyses in
    both phases, its caches traced through them."""
    resolved, text = _v41()
    program = Program.of(resolved)
    (stack,) = {t.meta.iteration_owner
                for t in program.blocks
                if t.meta.aux_kind is None and t.meta.stack_name == "layers"}
    produced = {}
    for (at, layer, name) in program.ir.stream_writes:
        if at == stack:
            produced.setdefault(name, set()).add(layer)
    kv = set(text["kv_source_layer_ids"])
    index = set(text["index_source_layer_ids"])
    candidate = {text["candidate_source_layer_id"]}
    assert produced["compress_kv"] == kv
    assert produced["index_k"] == kv
    assert produced["csa_topk"] == index
    assert produced["candidates"] == candidate
    read = {}
    for (at, layer, name) in program.ir.stream_reads:
        if at == stack and name != "hc_pre_mix":
            read.setdefault(name, set()).add(
                program.stream_origin(at, layer, name).iteration)
    assert read == {"compress_kv": kv, "index_k": {20}, "candidates": candidate,
                    "csa_topk": index}
    for phase in (PREFILL, DECODE):
        Deployment(program).during(phase).analyze()


# --- a seed before the first write, the final value after the last -------

_RW = _BLOCKS + """\
  rw:
    reads: [s]
    stream_out: { s: g }
    graph:
      g: { op: mul, in: [x, s], dim: d_model }
    outputs: [g]
"""

_SEED = """\
  - { name: seed, op: rmsnorm, inputs: [embed], dim: d_model,
      weight_axes: [{ replicated: true }] }
"""

_FINAL = """\
  - { name: fin, op: mul, inputs: [head, s], dim: d_model }
"""


def test_a_seed_is_what_the_first_layer_reads_and_the_last_write_is_read_after():
    """DeepSeek-V4.1's single-pass hyper-connections: every layer reads the
    mix the layer before it wrote (`"i - 1"`), layer 0 reads the stream's
    `seed` — a model-level op, run once — and the exit reduce after the
    loop reads the last layer's write by the stream's name."""
    program = _program("srd_seed", _rows(
        '{ type: rw, rest: true, reads_from: { s: "i - 1" } }'),
        blocks=_RW, streams="s: { seed: seed }", before=_SEED, extra=_FINAL)
    (reader,) = [t for t in program.blocks if t.name == "rw"]
    stack = reader.meta.iteration_owner
    seed = next(n for n in program.ir.top_level if n.name == "seed")
    (writer,) = [n for n in reader.nodes() if n.name == "g"]
    origins = {layer: program.stream_origin(stack, layer, "s")
               for layer in reader.layer_indices}
    assert (origins[0].uid, origins[0].iteration) == (seed.uid, None)
    assert {layer: (o.uid, o.iteration) for layer, o in origins.items()
            if layer} == {k: (writer.uid, k - 1) for k in range(1, 6)}
    (_form, rows), = program.declared_stream_reads(
        program.block_index(reader)).values()
    assert rows[0] == (0, "seed")
    assert program.stream_distances() == {"s": 1}
    fin = next(n for n in program.ir.top_level if n.name == "fin")
    for phase in (PREFILL, DECODE):
        analysis = Deployment(program).during(phase).analyze()
        (source,) = analysis.input_references.at(fin.uid, 1).table[0][1]
        assert (source.occurrence.uid, source.occurrence.iteration,
                source.via) == (writer.uid, 5, "named")


def test_the_first_layer_reads_nothing_before_it_without_a_seed():
    """With no `seed`, layer 0 has no layer before it to read; the refusal
    names `seed` as the way to say what it reads."""
    refused = _refused("srd_noseed", _rows(
        '{ type: rw, rest: true, reads_from: { s: "i - 1" } }'),
        blocks=_RW)
    assert "`seed`" in refused.fix


@pytest.mark.parametrize("streams, before, why", [
    ("s: { seed: nowhere }", "", "names no one earlier `op:` layer"),
    ("s: { seed: embed, carries: true }", "", "`seed` is the value an inner"),
])
def test_a_seed_names_one_earlier_op_layer(streams, before, why):
    text = _spec("srd_badseed", _rows(
        '{ type: rw, rest: true, reads_from: { s: "i - 1" } }'),
        blocks=_RW, streams=streams, before=before)
    with pytest.raises(Exception, match=re.escape(why)):
        _load("srd_badseed", text)


def test_a_stack_does_not_read_an_inner_stream_by_name():
    """Only a model-level op reads an inner stream's final value by name;
    a stack binds `inputs:` to its block's formals."""
    extra = ("  - { name: more, inputs: [s], repeat: \"1\", template: plain,"
             " streams: { hidden: { carries: true } } }\n")
    text = _spec("srd_stackread", _rows(
        '{ type: w, layers: ["0"] }', '{ type: plain, rest: true }'),
        extra=extra)
    with pytest.raises(Exception, match="only a model-level `op:` reads"):
        _load("srd_stackread", text)
