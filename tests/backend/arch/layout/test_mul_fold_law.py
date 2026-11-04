"""A pending sum through an elementwise multiply, on a synthetic TP block.

`mul` is linear in each operand separately: `sum_r(x_r) * g = sum_r(x_r * g)`
when `g` is the same on every rank. So a row-parallel output scaled by a
replicated gate (Qwen3-Next's sigmoid on the shared expert) stays unreduced
and the one all_reduce lands at the residual add, as vLLM and sglang run it.
Two pending operands (`(Σa)(Σb) != Σ ab`), or a factor cut across the
ranks, must reduce before the multiply.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.transition import RedistributeSite
from backend.inference.parallel_config import ParallelConfig

COL = "{ world: dense_ffn, parts: d_ffn, fit: exact }"
REP = "{ replicated: true }"

#: The row-parallel exit `down` holds a pending sum `[B, S, d_ffn]`.
PAIR = f"""\
      up:   {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }}, weight_axes: [{COL}, {REP}] }}
      down: {{ op: linear, in: up, dims: {{ in: d_ffn, out: d_ffn }}, weight_axes: [{REP}, {COL}] }}
"""

#: The multiply's other operand, by kind.
FACTORS = {
    # A [B, S, 1] gate from replicated weights: invariant, broadcast.
    "invariant_gate": f"""\
      gl:   {{ op: linear, in: x, dims: {{ in: d_model, out: "1" }}, weight_axes: [{REP}, {REP}] }}
      f:    {{ op: sigmoid, in: gl, dim: "1" }}
""",
    # A full-width factor from replicated weights: invariant.
    "invariant_full": f"""\
      f:    {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }}, weight_axes: [{REP}, {REP}] }}
""",
    # A second row-parallel exit: itself a pending sum.
    "partial": f"""\
      up2:  {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }}, weight_axes: [{COL}, {REP}] }}
      f:    {{ op: linear, in: up2, dims: {{ in: d_ffn, out: d_ffn }}, weight_axes: [{REP}, {COL}] }}
""",
    # A full-width invariant factor beside a pending [B, S, 1] row-parallel
    # exit `gp` (`down` is then a dead end the multiply does not read).
    "pending_gate": f"""\
      gp:   {{ op: linear, in: up, dims: {{ in: d_ffn, out: "1" }}, weight_axes: [{REP}, {COL}] }}
      f:    {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }}, weight_axes: [{REP}, {REP}] }}
""",
    # A column-parallel output: each rank holds a different slice.
    "sharded": f"""\
      f:    {{ op: linear, in: x, dims: {{ in: d_model, out: d_ffn }}, weight_axes: [{COL}, {REP}] }}
""",
}

SPEC = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims:
  d_model: config.hidden_size
  d_ffn: config.intermediate_size
  n_layers: "1"
blocks:
  blk:
    graph:
{pair}{factor}      s:    {{ op: mul, in: [{operands}], dim: d_ffn }}
      back: {{ op: linear, in: s, dims: {{ in: d_ffn, out: d_model }}, weight_axes: [{rep}, {rep}] }}
      o:    {{ op: add, in: [x, back], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: d_model, d_model: d_model }},
      weight_axes: [{rep}, {rep}] }}
  - {{ name: layers, repeat: n_layers, streams: {{ hidden: {{ carries: true }} }}, template: blk, inputs: [embed] }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: d_model }}, inputs: [layers],
      weight_axes: [{rep}, {rep}] }}
"""


def _analysis(factor: str, partial_first: bool, pending: str = "down"):
    from backend.arch.loader import SpecLoader
    name = f"mulfold_{factor}_{int(partial_first)}_{pending}"
    operands = f"{pending}, f" if partial_first else f"f, {pending}"
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(SPEC.format(
            name=name, pair=PAIR, factor=FACTORS[factor],
            operands=operands, rep=REP))
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16",
             "hidden_size": 64, "intermediate_size": 128}, model_id=name)
    program = Program.of(resolved)
    config = ParallelConfig(tp_attention=4, tp_dense_ffn=4)
    return Deployment(program, config=config).at(
        batch=1, input_tokens=16).analyze()


def _reductions(analysis):
    """`(consumer name, input index, collective)` of every reduction on a
    serving edge inside the block."""
    program = analysis.program
    out = []
    for edge in analysis.redistributes:
        if edge.site != RedistributeSite.EDGE:
            continue
        name = program.by_uid(edge.node).name
        if name in ("embed", "head"):
            continue
        out.extend((name, edge.input_index, op)
                   for _world, op in edge.collectives
                   if op in ("all_reduce", "reduce_scatter"))
    return sorted(out)


@pytest.mark.parametrize("partial_first", [False, True])
@pytest.mark.parametrize("factor", ["invariant_gate", "invariant_full"])
def test_an_invariant_factor_defers_the_reduction_to_the_residual(
        factor, partial_first):
    """One all_reduce, at the residual add — not at the multiply — with
    the pending operand in either position."""
    analysis = _analysis(factor, partial_first)
    assert _reductions(analysis) == [("o", 1, "all_reduce")]
    s = next(n for n in analysis.program.nodes() if n.name == "s")
    assert analysis.states[s.uid].carries_partial()


def test_a_pending_size_one_operand_defers_under_a_full_width_factor():
    """The pending sum is the [B, S, 1] side and broadcasts: still one
    all_reduce, at the residual add."""
    analysis = _analysis("pending_gate", True, pending="gp")
    assert _reductions(analysis) == [("o", 1, "all_reduce")]


def test_two_pending_operands_reduce_before_the_multiply():
    """(Σa)(Σb) is not Σ(ab): both operands reduce at the multiply."""
    analysis = _analysis("partial", True)
    assert _reductions(analysis) == [("s", 0, "all_reduce"),
                                      ("s", 1, "all_reduce")]


@pytest.mark.parametrize("partial_first", [True, False])
def test_a_factor_cut_across_the_ranks_reduces_before_the_multiply(
        partial_first):
    """Each rank would scale its part by a different slice: the pending
    sum resolves at the multiply — scattered onto the factor's cut, which
    is all the product needs — and the product is not a pending sum."""
    analysis = _analysis("sharded", partial_first)
    pending = 0 if partial_first else 1
    assert _reductions(analysis) == [("s", pending, "reduce_scatter")]
    s = next(n for n in analysis.program.nodes() if n.name == "s")
    assert not analysis.states[s.uid].carries_partial()


def test_qwen35_moe_decode_tp8_reduces_twice_per_layer():
    """Qwen3.5-35B-A3B decode at TP8, against vLLM v0.32.0: per decoder
    layer the attention exit (`o_proj` / GatedDeltaNet `out_proj`,
    RowParallelLinear with reduce_results=True) reduces once, and the MoE
    block reduces once — the shared expert is built with
    reduce_results=False and multiplies its sigmoid gate into the
    unreduced output (models/qwen2_moe.py Qwen2MoeMLP.forward), the
    routed experts add it in and reduce the sum (models/qwen3_next.py
    Qwen3NextSparseMoeBlock). Two all_reduces per layer, plus the
    vocab-parallel embedding's one."""
    import json

    from backend.inference.pipeline import analyze
    from tests.snapshots import resolve_snapshot, shared_loader
    model_id = "Qwen/Qwen3.5-35B-A3B"
    resolved = resolve_snapshot(model_id, shared_loader())
    snapshot = PROJECT_ROOT / "backend" / "arch" / "snapshots" / (
        model_id.replace("/", "__") + ".json")
    config = json.loads(snapshot.read_text())["config"]
    n_layers = (config.get("text_config") or config)["num_hidden_layers"]
    result = analyze(
        resolved_spec=resolved,
        parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                tp_moe_expert=8),
        batch_size=1, avg_input_tokens=2048, avg_output_tokens=0,
        phase="decode", breakdown=False)
    all_reduces = sum(
        float(row.get("count_per_step") or 0)
        for row in result.per_op_timing
        if int(row.get("comm_world", 1) or 1) > 1
        and row.get("op") == "all_reduce")
    assert all_reduces == 2 * n_layers + 1
