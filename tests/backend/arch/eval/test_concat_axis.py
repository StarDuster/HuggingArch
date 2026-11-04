"""A `concat` joins along the axis its node names.

HunyuanOCR wraps each image's patch sequence in learned begin/end tokens:
`concat axis: 1` over `[B, 1, D]`, `[B, N, D]`, `[B, 1, D]`. The operands
disagree on the sequence axis by construction — that is the axis being
joined — and agree on the feature axis. The label binding behind every FLOP
evaluation used to read the registry template, which joins along the last
axis, and refused the node ("input label 's_ctx' has conflicting extents");
it now reads the templates of the op resolved with the node's axis
(`OpRef.input_templates`).

Three operands is also where a join's FAN-OUT is first observable. A concat
hands on the least repetition its branches share (`traffic._fanout_factor`),
and a fold over branches reads every branch only if its accumulator is the
same thing on the last round as on the first. With two operands a fold runs
once and a seed of the wrong type is never read back, so the law and a fold
that answers only the first two agree; with three they do not, and the
accumulator's type is the difference between an answer and an
`AttributeError`. Arity, not any one model, is what makes this visible —
the shipped specs that join three ways happen to be two vision towers
wrapping their image rows in a begin and an end row (HunyuanOCR's
`begin_end_concat`, DeepSeek-V4.1-Flash's `span_rows`), and would be none
tomorrow — so it is written here against a join of its own.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from backend.arch.eval.flops import bind_labels

SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
blocks:
  blk:
    graph:
      first: {{ op: slice, in: x, axis: 1, dims: {{ in: s_ctx, out: "1" }} }}
      cat: {{ op: concat, in: [first, x, first], axis: {axis} }}
      # Three learned rows, each expanded over the batch from a stored
      # vector, so each branch of `reps` arrives with a repetition of its
      # own: 4B elements per stored one, then 2B, then B.  The least is the
      # last, which only a fold that reaches the last branch can find.
      p4: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p2: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      p1: {{ op: parameter_vector, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      rows4: {{ op: broadcast, in: p4, axis: 0, dims: {{ width: "4" }} }}
      rows2: {{ op: broadcast, in: p2, axis: 0, dims: {{ width: "2" }} }}
      rows1: {{ op: broadcast, in: p1, axis: 0, dims: {{ width: "1" }} }}
      wide: {{ op: broadcast, in: rows4, axis: 0, dims: {{ width: "B" }} }}
      mid: {{ op: broadcast, in: rows2, axis: 0, dims: {{ width: "B" }} }}
      narrow: {{ op: broadcast, in: rows1, axis: 0, dims: {{ width: "B" }} }}
      reps: {{ op: concat, in: [wide, mid, narrow], axis: 1 }}
      joined: {{ op: concat, in: [cat, reps], axis: 1 }}
      n: {{ op: rmsnorm, in: joined, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: layers, template: blk, inputs: [embed] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


def _analyze(name: str, axis: int, batch: Any = 2):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(
            SPEC.format(name=name, axis=axis))
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8}, model_id=name)
    program = Program.of(resolved)
    return program, Deployment(program).at(batch=batch, input_tokens=4).analyze()


@pytest.mark.parametrize("axis", [1, -2])
def test_a_sequence_concat_evaluates(axis):
    program, analysis = _analyze(f"concat_seq_{axis + 2}", axis)
    cat = next(n for n in program.nodes() if n.name == "cat")
    assert [int(e) for e in analysis.shapes.values[cat.uid].out] == [2, 6, 8]


#: The two bindings one walk is read under (`Binding`): the batch as a number
#: and the batch as the symbol every number is a point of.
BATCHES = [(2, "numeric"), (sp.Symbol("B", positive=True, integer=True),
                            "symbolic")]


@pytest.mark.parametrize("batch,binding", BATCHES, ids=[b for _, b in BATCHES])
def test_a_three_way_join_hands_on_the_least_fan_out_of_every_branch(
        batch, binding):
    """A join's fan-out is the repetition ALL of its branches have.

    Three branches that repeat 4B, 2B and B times leave B: a reader of the
    join reads each stored row B times over, and no more, because one branch
    has only that much to give. The least of the first two branches is 2B,
    so an answer of 2B would mean the third branch was never looked at — and
    the fold that answered 2B was the one whose accumulator stopped being a
    `_Fanout` after the first round, which could not get even that far: it
    asked an extent for the `.total` only a `_Fanout` has.
    """
    from backend.arch.eval.cache import Edges
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.traffic import _fanout

    _program, analysis = _analyze(f"concat_fanout_{binding}", 1, batch=batch)
    edges, shapes = Edges(analysis), analysis.shapes.values
    nodes = {node.name: node for node in analysis.program.walked_nodes()}

    def fanout(name: str) -> Any:
        return value_of(_fanout(nodes[name], shapes, edges=edges))

    assert [fanout("wide"), fanout("mid"), fanout("narrow")] == [
        4 * batch, 2 * batch, batch]
    assert fanout("reps") == batch
    # And a branch that repeats nothing leaves nothing to clamp, whether it
    # is one of two (`joined`) or one of three (`cat`, around a real stream
    # as HunyuanOCR's begin/end rows are).
    assert fanout("joined") == 1
    assert fanout("cat") == 1


def _concat(axis):
    from backend.arch.eval.op_ref import OpRef
    from backend.arch.ops import get_op
    return OpRef("concat", get_op("concat"), False, axis)


def test_the_joined_axis_binds_per_operand_and_the_rest_must_agree():
    labels, batch = bind_labels(_concat(1), [(2, 1, 8), (2, 4, 8)])
    assert (labels["B"], batch) == (2, 1)
    assert sorted(value for name, value in labels.items()
                  if name.startswith("s_ctx")) == [1, 4]
    # Operands that disagree on an axis not being joined still conflict.
    with pytest.raises(ValueError, match="'B' has conflicting extents"):
        bind_labels(_concat(1), [(1, 1, 8), (2, 4, 8)])
