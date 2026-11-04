"""A `derived:` name follows its source into every frame.

`dims.yaml` derives `n_expert_copies` from `n_experts` and `n_route_targets`
from `n_experts + n_zero_experts`. A component call or a block assignment
that rebinds `n_experts` opens a frame, and in it the two derived names are
derived again (`Binding.frame`): the call's bank has its own expert axis and
its dispatch its own denominator. Before, the derivation ran once, at model
resolution, and a call that rebound `n_experts` kept the enclosing block's
axis and denominator — DeepSeek-V4.1-Flash's draft stages divided their
tokens by 384 destinations instead of 128 until the call wrote all four
names by hand.

The model here is synthetic so that every factor the rule turns on is
separable: the zero-compute destinations are not zero (LongCat's case, which
makes "against the whole frame" observable), `d_ffn_shared` is a DEFAULT left
unbound (so "a default does not follow" is observable), and a primitive
rebinds a root (so "a primitive's `dims:` is not a frame" is observable).
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import sympy as sp

from backend.arch.eval.dim import expr_of, value_of
from backend.arch.eval.program import Program
from backend.arch.eval.storage import is_expert_axis
from backend.arch.eval.transforms import Deployment

COMPONENTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "components"
CONFIG = {
    "model_type": "frames_toy",
    "torch_dtype": "bfloat16", "hidden_size": 8,
    "moe_intermediate_size": 4,
    "draft_intermediate_size": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "draft_num_experts": 4,
    "vocab_size": 32,
}

# `moe_ffn` with its `permute` rebinding `n_experts` in the primitive's own
# `dims:`. A primitive's map is not a frame: the permute's expert axis stays
# the enclosing frame's 8, which is what the bank's weights lead with.
PRIMITIVE_REBINDS = """\
role: ffn_moe
description: "Synthetic: moe_ffn whose permute rebinds n_experts in its own dims."
graph:
  gate:          { op: router_softmax_topk, in: x }
  permute:       { op: permute, in: [x, gate], dims: { n_experts: n_draft_experts } }
  experts:       { op: gated_experts, role: ffn_moe, in: permute, dims: { d_model: d_model, d_ffn: d_ffn_expert }, expert_weight: unrolled }
  combine:       { op: combine, in: [experts, gate] }
  moe_out:       { op: add, in: [combine, x], dim: d_model }
outputs: [moe_out]
"""

MODEL = """\
match: { model_type: [frames_toy], model_id: [frames_toy] }
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  n_zero_experts: "2"
  n_vocab: config.vocab_size
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_expert_groups: "1"
  n_groups_active: "1"
  n_routed_scale: "0"
  n_topk_renorm: "0"
  n_draft_experts: config.draft_num_experts
  d_draft_ffn: config.draft_intermediate_size
dim_symbols:
  n_draft_experts: { kind: whole, group: mtp_speculative, meaning: "a second pool's size" }
  d_draft_ffn: { kind: split_width, shardable_by: [moe_expert], group: mtp_speculative, meaning: "a second pool's expert width" }
blocks:
  call_blk:
    inputs: [x]
    graph:
      ffn: { op: moe_ffn, in: x, dims: { n_experts: n_draft_experts, d_ffn_expert: d_draft_ffn } }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  identity_blk:
    inputs: [x]
    graph:
      ffn: { op: moe_ffn, in: x, dims: { n_experts: n_experts } }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  plain_blk:
    inputs: [x]
    graph:
      ffn: { op: moe_ffn, in: x }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
  prim_blk:
    inputs: [x]
    graph:
      ffn: { op: frames_toy_primitive_rebinds, in: x }
      o: { op: add, in: [x, ffn], dim: d_model }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: call_layers, repeat: "1", template: call_blk, inputs: [embed_tokens],
      streams: { hidden: { carries: true } } }
  - { name: prim_layers, repeat: "1", template: prim_blk, inputs: [call_layers],
      streams: { hidden: { carries: true } } }
  - { name: identity_layers, repeat: "1", template: identity_blk, inputs: [prim_layers],
      streams: { hidden: { carries: true } } }
  - name: assigned_layers
    inputs: [identity_layers]
    repeat: "2"
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: a, rest: true }
      map:
        a:
          template: plain_blk
          dims: { n_experts: n_draft_experts }
  - { name: norm, op: rmsnorm, dim: d_model, inputs: [assigned_layers],
      weight_axes: [{ replicated: true }] }
  - { name: lm_head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: { in: d_model, out: n_vocab },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _resolve():
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        (base / "components" / "library"
         / "frames_toy_primitive_rebinds.yaml").write_text(PRIMITIVE_REBINDS)
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "frames_toy.yaml").write_text(MODEL)
        return SpecLoader(base_dir=str(base)).match(CONFIG, model_id="frames_toy")


@pytest.fixture(scope="module")
def resolved():
    return _resolve()


@pytest.fixture(scope="module")
def walk(resolved):
    return Deployment(Program.of(resolved)).at(batch=1, input_tokens=4).analyze()


def _in(analysis, node, template):
    """Whether `node` belongs to a block of template `template`."""
    program = analysis.program
    block = program.block_of(node.uid)
    return block is not None and program.blocks[block].name == template


def _ffn_frame(analysis, template):
    """The frame the `ffn` call of the `template` block runs in."""
    (uid,) = {n.uid for n in analysis.program.nodes()
              if n.name == "ffn" and _in(analysis, n, template)}
    return analysis.node_dims.values[uid]


def _bank(analysis, template):
    """The `gate_proj` bank leaf of the `template` block."""
    nodes = {n.uid: n for n in analysis.program.nodes()
             if n.name == "gate_proj" and n.parent is not None
             and n.parent.name == "experts" and _in(analysis, n, template)}
    (node,) = nodes.values()
    return node


def test_the_model_frame_is_what_it_was(resolved):
    dims = resolved.dim_values
    assert (dims["n_expert_copies"], dims["n_route_targets"],
            dims["d_ffn_shared"]) == (8, 10, 4)


def test_a_component_call_derives_the_expert_axis_and_denominator_again(walk):
    frame = _ffn_frame(walk, "call_blk")
    copies, targets = frame["n_expert_copies"], frame["n_route_targets"]
    # Its own identity, not the source's (`n_draft_experts`), and the call's
    # count, not the model's 8.
    assert expr_of(copies) == sp.Symbol("n_expert_copies")
    assert value_of(copies) == 4
    # Against the WHOLE frame: the inherited 2 zero-compute destinations.
    assert expr_of(targets) == sp.Symbol("n_route_targets")
    assert value_of(targets) == 6
    bank = _bank(walk, "call_blk")
    (weight,) = walk.storage.values[bank.uid][:1]
    assert is_expert_axis(weight.shape[0])
    assert int(value_of(weight.shape[0])) == 4


def test_a_default_does_not_follow_its_source(walk):
    """`d_ffn_shared` is a `default:` (= `d_ffn_expert`), left unbound by the
    model, so the model's value is 4. The call rebinds `d_ffn_expert` to 2;
    its `d_ffn_shared` is still the model's 4 — a frame cannot tell a
    defaulted value from one the model bound."""
    frame = _ffn_frame(walk, "call_blk")
    assert value_of(frame["d_ffn_expert"]) == 2
    assert value_of(frame["d_ffn_shared"]) == 4


def test_a_block_assignment_derives_them_again(walk):
    frame = _ffn_frame(walk, "plain_blk")
    assert expr_of(frame["n_expert_copies"]) == sp.Symbol("n_expert_copies")
    assert (value_of(frame["n_expert_copies"]),
            value_of(frame["n_route_targets"])) == (4, 6)


def test_a_primitives_own_dims_are_not_a_frame(walk):
    """The permute rebinds `n_experts` in its own map; its expert axis is
    still the enclosing frame's 8, the axis the bank's weights lead with."""
    (permute,) = [n for n in walk.program.nodes()
                  if n.name == "permute" and _in(walk, n, "prim_blk")]
    lead = walk.shapes.values[permute.uid].out[0]
    assert is_expert_axis(lead)
    assert int(value_of(lead)) == 8


def test_expert_parallelism_divides_the_frames_axis_not_its_denominator(resolved):
    from backend.arch.layout.physical_cost import physical_cost_plan

    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=4).on(ep=2).analyze()
    bank = _bank(analysis, "plain_blk")
    local = physical_cost_plan(analysis).cost.values[bank.uid]
    # Global [4, d_ffn 4, d_model 8]; a rank holds 2 of the assignment's 4.
    assert [int(value_of(e)) for e in analysis.storage.values[bank.uid][0].shape] == [4, 4, 8]
    assert int(local.params) == 2 * 4 * 8
    frame = _ffn_frame(analysis, "plain_blk")
    assert value_of(frame["n_route_targets"]) == 6


def _counts_walk(deployment):
    """The walk over the workload's counts, unread, and the point it is
    read at."""
    workload, point = deployment.workload.symbolic()
    return replace(deployment, workload=workload).analyze(), point


def test_the_symbolic_frame_is_the_axis_valued_in_the_frames_count(resolved):
    """In the walk over the workload's counts the call's expert axis is
    still `n_expert_copies` by identity, and its value is the count the
    frame bound `n_experts` to; the walk's total, read at the workload's
    point, is the numeric total."""
    deployment = Deployment(Program.of(resolved)).at(batch=1, input_tokens=4)
    numeric = deployment.analyze()
    walk, point = _counts_walk(deployment)
    frame = _ffn_frame(walk, "call_blk")
    assert expr_of(frame["n_expert_copies"]) == sp.Symbol("n_expert_copies")
    assert value_of(frame["n_expert_copies"]) == \
        value_of(numeric.binding["n_draft_experts"])
    for field in ("flops", "params", "active_params"):
        law = sp.sympify(value_of(getattr(walk.total(), field)))
        got = law.subs({s: point[s.name] for s in law.free_symbols})
        want = value_of(getattr(numeric.total(), field))
        assert sp.simplify(got - want) == 0, field


def test_an_identity_rebinding_moves_nothing(resolved):
    """`n_experts: n_experts` (LongCat's block) is written as its own name:
    it rebinds nothing, so the derived names keep the enclosing frame's
    values and spelling — re-deriving would respell the axis as
    `n_experts` and the denominator as `n_experts + n_zero_experts`."""
    walk, _ = _counts_walk(Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=4))
    frame = _ffn_frame(walk, "identity_blk")
    assert expr_of(frame["n_expert_copies"]) == sp.Symbol("n_expert_copies")
    assert expr_of(frame["n_route_targets"]) == sp.Symbol("n_route_targets")
