"""Whether a block reads a formal (`BlockTemplate.reads_formal`, via
`program._consumes`), bound as the walk binds it.

The tree grafts a component's own formals onto what its caller binds them
to, so a read inside a call is written in the caller's name whatever the
component calls its formal (`SpecTreeWalker._resolve_input_ref`, step 3);
inside the call that name is the formal only if the call hands it in — else
it is a stream of the same spelling (`Scope.calling` / `Scope.read`). A
component call that only lists the formal among its inputs hands it on.

The unit cases build the IR the graft leaves; the end-to-end cases load a
spec, walk it, and read the architecture view's edges."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.arch.eval.program import _consumes
from backend.arch.spec_tree.nodes import InputRef, SpecIRNode


def _refs(inputs, grafted):
    return [InputRef(None, "", name, grafted) for name in inputs]


def _call(children, inputs, formals, grafted=False):
    """A call; `grafted` as its refs are: False at the block's top, True
    for a call inside another call that reads the caller's formals."""
    return SpecIRNode(name="w", op="inner", is_component=True,
                      formals=tuple(formals), children=list(children),
                      inputs_obj=_refs(inputs, grafted))


def _op(*inputs, grafted=True):
    """An op inside a call: its reads of the call's formals are grafted."""
    return SpecIRNode(name="u", op="embedding", is_component=False,
                      inputs_obj=_refs(inputs, grafted))


def test_a_read_inside_a_call_counts_whatever_the_callee_names_its_formal():
    """`inner` declares `inputs: [x, renamed]`; grafted, its embedding reads
    the caller's `input_ids`."""
    nodes = [_call([_op("input_ids")], ["x", "input_ids"], ["x", "renamed"])]
    assert _consumes(nodes, "input_ids")


def test_a_call_that_only_hands_a_formal_on_does_not_read_it():
    """DeepSeek-V4's score-routed MoE takes `input_ids` and reads nothing
    from it."""
    nodes = [_call([_op("x")], ["x", "input_ids"], ["x", "input_ids"])]
    assert not _consumes(nodes, "input_ids")
    assert _consumes(nodes, "x")


def test_a_read_two_calls_deep_counts():
    inner = _call([_op("input_ids")], ["input_ids"], ["ids"], grafted=True)
    assert _consumes([_call([inner], ["x", "input_ids"], ["x", "tokens"])], "input_ids")


def test_a_stream_of_the_formals_name_read_by_a_call_not_handed_it_is_no_read():
    """`reader` is handed `x` and reads stream `s`; the block's formal `s`
    is not what it reads."""
    nodes = [_call([_op("s", grafted=False)], ["x"], ["x"])]
    assert not _consumes(nodes, "s")


def test_a_stream_spelled_like_a_handed_formal_is_still_a_stream():
    """`reader` is handed the block's `s` as its formal `renamed` and also
    reads stream `s`: its op reads the stream (not grafted), not the
    formal, so the block reads nothing through it."""
    nodes = [_call([_op("s", grafted=False)], ["x", "s"], ["x", "renamed"])]
    assert not _consumes(nodes, "s")


# --- end to end: load, walk, draw -----------------------------------------

_BLOCKS = {
    # Codex, round 4: `outer`'s formal `s` (bound to `extra`) and the stream
    # `s` its nested `reader` reads share a spelling.
    "stream_clash": """
  writer:
    inputs: [x]
    graph:
      y: { op: identity, in: x, dim: d_model }
    outputs: [y]
    stream_out: { s: y }
  reader:
    inputs: [x]
    reads: [s]
    graph:
      r: { op: identity, in: s, dim: d_model }
    outputs: [r]
  outer:
    inputs: [x, s]
    graph:
      w: { op: writer, in: [x] }
      r: { op: reader, in: [x] }
    outputs: [r]
""",
    # Codex, round 5: `reader` is handed `outer`'s `s` as its formal
    # `renamed`, reads nothing from it, and reads a stream `s`.
    "renamed_beside_stream": """
  writer:
    inputs: [x]
    graph:
      y: { op: identity, in: x, dim: d_model }
    outputs: [y]
    stream_out: { s: y }
  reader:
    inputs: [x, renamed]
    reads: [s]
    graph:
      r: { op: identity, in: s, dim: d_model }
    outputs: [r]
  outer:
    inputs: [x, s]
    graph:
      w: { op: writer, in: [x] }
      r: { op: reader, in: [x, s] }
    outputs: [r]
""",
    # `inner` calls the tensor it is handed `renamed`; it reads it.
    "renamed": """
  inner:
    inputs: [x, renamed]
    graph:
      y: { op: identity, in: renamed, dim: d_model }
      out: { op: add, in: [x, y], dim: d_model }
    outputs: [out]
  outer:
    inputs: [x, s]
    graph:
      w: { op: inner, in: [x, s] }
    outputs: [w]
""",
}

_SPEC = """match: {{ model_type: [_reads_formal] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers, n_vocab: config.vocab_size }}
model_inputs: {{ input_ids: {{ kind: tokens }} }}
blocks:{blocks}
model:
  - {{ name: embed, op: embedding, boundary_role: input_embedding, inputs: [input_ids], dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - {{ name: extra, op: linear, inputs: [embed], dims: {{ in: d_model, out: d_model }} }}
  - name: layers
    inputs: [embed, extra]
    repeat: n_layers
    template: outer
{streams}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [layers], dims: {{ in: d_model, out: n_vocab }} }}
"""

_CLASH = """    reads_from: { s: "i - 1" }
    streams:
      hidden: { carries: true }
      s: { seed: embed }"""

_STREAMS = {
    "stream_clash": _CLASH,
    "renamed_beside_stream": _CLASH,
    "renamed": """    streams:
      hidden: { carries: true }""",
}


def _walked(case: str):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.loader import SpecLoader

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_SPEC.format(blocks=_BLOCKS[case], streams=_STREAMS[case]))
        path = f.name
    loader = SpecLoader()
    try:
        spec = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(spec, {
        "model_type": "_reads_formal", "torch_dtype": "bfloat16",
        "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 16})
    program = Program.of(resolved)
    # The walk binds it: the spec is one the engine runs.
    analysis = Deployment(program).at(batch=1, input_tokens=16).analyze()
    return resolved, program, analysis


def _edges(resolved):
    from backend.rendering import build_architecture_view

    summary = {"model_type": "x", "hidden_size": 8, "vocab_size": 16,
               "num_layers": 2, "attention_info": {}, "ffn_info": {},
               "architectures": ["x"]}
    view = build_architecture_view(summary, [], resolved, str, {})
    names = {c["id"]: c["name"] for c in view["layers"]}
    names.update({m["id"]: m["name"] for m in view["model_inputs"]})
    return {(names[e["from"]].split(" ")[0], names[e["to"]].split(" ")[0], e["kind"])
            for e in view["edges"]}


def test_end_to_end_a_same_named_stream_draws_a_stream_edge_not_a_data_edge():
    resolved, program, _ = _walked("stream_clash")
    (outer,) = [t for t in program.blocks if t.meta.template_name == "outer"]
    assert not outer.reads_formal("s")
    edges = _edges(resolved)
    assert not [e for e in edges if e[0] == "extra" and e[2] == "data"
                and e[1] != "head"]
    assert any(e[0] == "embed" and e[2] == "stream" for e in edges)


def test_end_to_end_a_renamed_formal_is_read():
    resolved, program, _ = _walked("renamed")
    (outer,) = [t for t in program.blocks if t.meta.template_name == "outer"]
    assert outer.reads_formal("s")
    assert any(e[0] == "extra" and e[2] == "data" for e in _edges(resolved))


def test_end_to_end_a_stream_beside_a_handed_formal_of_its_spelling_is_the_stream():
    """The walk binds `reader`'s read of `s` to the stream, not to the
    `outer` formal it was handed as `renamed` (`Scope.read`): the read's
    reference is not the formal's, the block reads nothing through `s`, and
    the view draws a stream edge and no data edge from `extra`."""
    resolved, program, analysis = _walked("renamed_beside_stream")
    (outer,) = [t for t in program.blocks if t.meta.template_name == "outer"]
    formal = analysis.stage_references[program.block_index(outer)][1]
    (leaf,) = [n for n in outer.nodes() if n.name == "r" and not n.is_component]
    assert analysis.input_reference(leaf.uid, 0) != formal
    assert not outer.reads_formal("s")
    edges = _edges(resolved)
    assert not [e for e in edges if e[0] == "extra" and e[1] == "outer"]
    assert any(e[0] == "embed" and e[1] == "outer" and e[2] == "stream" for e in edges)
