"""`stored_axes`: a weight the checkpoint stores in another axis order.

Qwen3-VL-MoE stores each packed expert as `[in, out]` under a `(out, in)`
`linear_param` whose declared weight is `[E, out, in]`. The toy below has
the same shape of problem with no square matrix, so a wrong axis order cannot
pass by coincidence: `d_model = 6`, `d_ffn_expert = 4`, four experts on the
leading axis.
"""
from __future__ import annotations

import shutil
import tempfile
from math import prod
from pathlib import Path

import pytest

from backend.arch.layout import copy_checkpoint as ckpt
from backend.arch.layout import weight_endpoints
from backend.arch.layout.persistent import STORED
from backend.arch.layout.layout import Replicated, Segmented
from backend.arch.layout.persistent import physical_artifacts
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

COMPONENTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "components"
CONFIG = {
    "model_type": "qt_toy",
    "torch_dtype": "bfloat16", "hidden_size": 6,
    "moe_intermediate_size": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 16,
}
TP2 = ParallelConfig(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2)

# The checkpoint-order declaration: the expert axis stays leading and the
# last two swap.
GATE_UP = """\
    stored_axes: [0, 2, 1]
    weight_axes:
      - { world: moe_copies, parts: n_expert_copies, fit: exact }
      - { replicated: true }
      - segments:
          - { segment: gate, extent: d_ffn_expert, world: moe_expert, parts: d_ffn_expert, fit: exact }
          - { segment: up,   extent: d_ffn_expert, world: moe_expert, parts: d_ffn_expert, fit: exact }"""
DOWN = """\
    stored_axes: [0, 2, 1]
    weight_axes:
      - { world: moe_copies, parts: n_expert_copies, fit: exact }
      - { world: moe_expert, parts: d_ffn_expert, fit: exact }
      - { replicated: true }"""
# The op-order declaration every corpus spec writes.
GATE_UP_OP_ORDER = """\
    weight_axes:
      - { world: moe_copies, parts: n_expert_copies, fit: exact }
      - segments:
          - { segment: gate, extent: d_ffn_expert, world: moe_expert, parts: d_ffn_expert, fit: exact }
          - { segment: up,   extent: d_ffn_expert, world: moe_expert, parts: d_ffn_expert, fit: exact }
      - { replicated: true }"""
DOWN_OP_ORDER = """\
    weight_axes:
      - { world: moe_copies, parts: n_expert_copies, fit: exact }
      - { replicated: true }
      - { world: moe_expert, parts: d_ffn_expert, fit: exact }"""

DRAFT = """\
role: ffn_moe
graph:
  gate:    {{ op: router_softmax_topk, in: x }}
  permute: {{ op: permute, in: [x, gate] }}
  experts.gate_up_proj:
    op: linear_param
    role: ffn_moe
    in: permute
    dims: {{ in: d_model, out: "2 * d_ffn_expert" }}
    shape: [n_expert_copies, "2 * d_ffn_expert", d_model]
{gate_up}
    expert_weight: leading_axis
  experts.gate_slice: {{ op: slice, in: experts.gate_up_proj, dims: {{ in: "2 * d_ffn_expert", out: d_ffn_expert }} }}
  experts.up_slice:   {{ op: slice, in: experts.gate_up_proj, dims: {{ in: "2 * d_ffn_expert", out: d_ffn_expert }} }}
  experts.silu:       {{ op: silu, in: experts.gate_slice, dim: d_ffn_expert{silu} }}
  experts.gate_mul:   {{ op: mul, in: [experts.silu, experts.up_slice], dim: d_ffn_expert }}
  experts.down_proj:
    op: {down_op}
    role: ffn_moe
    in: experts.gate_mul
    dims: {{ in: d_ffn_expert, out: d_model{down_bias} }}
    shape: [n_expert_copies, d_model, d_ffn_expert]
{down}
    expert_weight: leading_axis
  combine: {{ op: combine, in: [experts.down_proj, gate] }}
outputs: [combine]
"""

MODEL = """\
match: {{ model_type: [qt_toy], model_id: [{name}] }}
drafts: [toy_packed_moe]
dims:
  d_model: config.hidden_size
  d_ffn_expert: config.moe_intermediate_size
  n_vocab: config.vocab_size
  n_layers: "1"
  n_experts: config.num_experts
  n_experts_active: config.num_experts_per_tok
  n_topk_renorm: "1"
  n_expert_groups: "1"
  n_groups_active: "1"
  n_routed_scale: "0"
{quant}blocks:
  blk:
    graph:
      ffn: {{ op: toy_packed_moe, in: x }}
      o: {{ op: add, in: [x, ffn], dim: d_model }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: layers, repeat: n_layers, template: blk, inputs: [embed_tokens],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: lm_head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""
FP8 = """\
quant_context:
  per_role:
    ffn_moe: { method: fp8, block_h: 2, block_w: 2 }
    default: { method: none }
"""


def _load(name, *, gate_up=GATE_UP, down=DOWN, down_op="linear_param",
          down_bias="", silu="", quant=""):
    from backend.arch.loader import SpecLoader
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        # The router is a library component; the toy reuses the real library.
        shutil.copytree(COMPONENTS / "library", base / "components" / "library")
        for sub in ("components/adapters", "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "components" / "drafts" / "toy_packed_moe.yaml").write_text(
            DRAFT.format(gate_up=gate_up, down=down, down_op=down_op,
                         down_bias=down_bias, silu=silu))
        (base / "models" / f"{name}.yaml").write_text(
            MODEL.format(name=name, quant=quant))
        return SpecLoader(base_dir=str(base)).match(CONFIG, model_id=name)


def _step(resolved, pc=TP2):
    return evaluate_step(resolved, pc, phase="prefill", batch=1, input_tokens=4, output_tokens=0,
)


def _analyze(resolved):
    """The walk itself: a step evaluates lazily, on `.analysis`."""
    return _step(resolved).analysis


def _by_name(rows, suffix):
    found = [row for row in rows if row.tensor_name.endswith(suffix)]
    assert len(found) == 1, [row.tensor_name for row in rows]
    return found[0]


@pytest.fixture(scope="module")
def reordered():
    return _step(_load("qt_toy_reordered"))


@pytest.fixture(scope="module")
def op_order():
    return _step(_load("qt_toy_op_order", gate_up=GATE_UP_OP_ORDER,
                       down=DOWN_OP_ORDER))


def _header(step, *, transposed):
    """A safetensors header for the toy's two expert banks: one layer, so
    one leading-axis tensor per bank, at layer index 0."""
    from backend.hf.header import SafetensorsHeader
    tensors = {}
    for row in weight_endpoints.instantiate(step):
        if ".experts." not in row.tensor_name:
            continue
        experts, out, inner = row.global_shape
        payload = [inner, out] if transposed else [out, inner]
        name = ".".join("0" if s == "X" else s
                        for s in row.tensor_name.split("."))
        tensors[name] = ("BF16", [experts, *payload])
    return SafetensorsHeader.from_tensors(tensors)


class TestReorderedGeometry:
    def test_checkpoint_shape_is_the_payload_in_stored_order(self, reordered):
        rows = weight_endpoints.instantiate(reordered)
        gate_up = _by_name(rows, "experts.gate_up_proj")
        down = _by_name(rows, "experts.down_proj")
        # Runtime (op) order: the experts, then linear_param's (out, in).
        assert gate_up.global_shape == (4, 8, 6)
        assert down.global_shape == (4, 6, 4)
        # Checkpoint order: the experts, then (in, out).
        assert gate_up.checkpoint_shape == (4, 6, 8)
        assert down.checkpoint_shape == (4, 4, 6)

    def test_runtime_is_per_rank_in_op_order(self, reordered):
        rows = weight_endpoints.instantiate(reordered)
        gate_up = _by_name(rows, "experts.gate_up_proj")
        down = _by_name(rows, "experts.down_proj")
        # TP2 on moe_expert halves d_ffn_expert: each of gate/up keeps 2
        # rows; no EP, so every rank holds all four experts.
        assert gate_up.runtime_shape == (4, 4, 6)
        assert down.runtime_shape == (4, 6, 2)

    def test_layout_axes_follow_their_own_order(self, reordered):
        rows = weight_endpoints.instantiate(reordered)
        gate_up = _by_name(rows, "experts.gate_up_proj")
        down = _by_name(rows, "experts.down_proj")
        runtime = gate_up.runtime_layout.axes
        checkpoint = gate_up.checkpoint_layout.axes
        # Runtime: the experts (whole, no EP), then the fused gate/up axis,
        # sharded, then d_model whole.
        assert isinstance(runtime[0], Replicated)
        assert isinstance(runtime[1], Segmented)
        assert isinstance(runtime[2], Replicated)
        # Checkpoint: the experts, then d_model, then the fused axis.
        assert isinstance(checkpoint[0], Replicated)
        assert isinstance(checkpoint[1], Replicated)
        assert isinstance(checkpoint[2], Segmented)
        assert gate_up.axis_segment_names == (("",), ("gate", "up"), ("",))
        assert not isinstance(down.runtime_layout.axes[2], Replicated)
        assert isinstance(down.runtime_layout.axes[1], Replicated)

    def test_artifacts_are_runtime_order_and_bytes(self, reordered):
        artifacts = [
            row for row in physical_artifacts(reordered.analysis)
            if row.surface == STORED
            and ".experts." in row.artifact_name
        ]
        shapes = {row.artifact_name.rsplit(".", 1)[-1]:
                  (row.global_shape, row.runtime_shape, row.runtime_instances,
                   row.resident_bytes_per_rank) for row in artifacts}
        # All four experts in one tensor per rank (no EP), BF16.
        assert shapes == {
            "gate_up_proj": ((4, 8, 6), (4, 4, 6), 1, prod((4, 4, 6)) * 2),
            "down_proj": ((4, 6, 4), (4, 6, 2), 1, prod((4, 6, 2)) * 2),
        }

    def test_the_checkpoint_ledger_carries_the_stored_order(self, reordered):
        rows = [
            row for row in ckpt.expert_artifacts(reordered.analysis)
            if row.surface == STORED
        ]
        shapes = {row.artifact_name.rsplit(".", 1)[-1]: row.checkpoint_shape
                  for row in rows}
        assert shapes == {"gate_up_proj": (4, 6, 8), "down_proj": (4, 4, 6)}

    def test_header_oracle_expects_the_leading_copy_axis_then_payload(
            self, reordered):
        rows = ckpt.compare_checkpoint(
            reordered, _header(reordered, transposed=True))
        experts = {row.artifact_key.rsplit(".", 1)[-1]: row for row in rows
                   if ".experts." in row.artifact_key}
        assert {key: row.expected_variants[0].shape
                for key, row in experts.items()} == {
            "gate_up_proj": (4, 6, 8), "down_proj": (4, 4, 6)}
        assert {(row.verdict, row.evidence) for row in experts.values()} == {
            (ckpt.MATCH, ckpt.EVIDENCE_HEADER_OK)}

    def test_the_op_order_declaration_is_what_the_transposed_header_refutes(
            self, op_order):
        # The same header against the declaration that omits `stored_axes`.
        rows = ckpt.compare_checkpoint(
            op_order, _header(op_order, transposed=True))
        experts = [row for row in rows if ".experts." in row.artifact_key]
        assert {(row.verdict, row.evidence) for row in experts} == {
            (ckpt.UNEXPLAINED, ckpt.EVIDENCE_HEADER_OK)}

    def test_order_changes_no_runtime_fact(self, reordered, op_order):
        def runtime(step):
            return sorted(
                (row.artifact_name, row.global_shape, row.runtime_shape,
                 row.runtime_layout, row.resident_bytes_per_rank)
                for row in physical_artifacts(step.analysis))

        assert runtime(reordered) == runtime(op_order)
        assert (reordered.analysis.cost.values.keys()
                == op_order.analysis.cost.values.keys())
        assert (
            [reordered.analysis.cost.values[uid].flops
             for uid in sorted(reordered.analysis.cost.values)]
            == [op_order.analysis.cost.values[uid].flops
                for uid in sorted(op_order.analysis.cost.values)])

    def test_identity_order_is_not_serialized(self):
        from backend.arch.schema.graph import GraphNode
        node = GraphNode(op="linear", inputs=["x"],
                         weight_axes=[{"replicated": True}] * 2)
        assert "stored_axes" not in node.model_dump()
        reordered = GraphNode(op="linear", inputs=["x"], stored_axes=[1, 0],
                              weight_axes=[{"replicated": True}] * 2)
        assert reordered.model_dump()["stored_axes"] == [1, 0]


class TestRefusals:
    @pytest.mark.parametrize("order", ["[0, 0]", "[1, 2]", "[]"])
    def test_not_a_permutation(self, order):
        with pytest.raises(Exception, match="permutation"):
            _load("qt_toy_not_perm",
                  gate_up=GATE_UP.replace("[0, 2, 1]", order))

    def test_rank_differs_from_the_weight_axes(self):
        with pytest.raises(Exception, match="names 4 axes but `weight_axes`"):
            _load("qt_toy_rank",
                  gate_up=GATE_UP.replace("[0, 2, 1]", "[3, 2, 1, 0]"))

    def test_rank_differs_from_the_op_weight_shape(self):
        down = DOWN.replace("[0, 2, 1]", "[3, 2, 1, 0]") + \
            "\n      - { replicated: true }"
        with pytest.raises(Exception, match="weight_axes declares 4 axes"):
            _analyze(_load("qt_toy_op_rank", down=down))

    def test_op_without_a_weight(self):
        # `silu` stores nothing: there is no weight shape to reorder.
        spec = _load(
            "qt_toy_no_weight",
            silu=", stored_axes: [0], weight_axes: [{ replicated: true }]")
        with pytest.raises(Exception, match="weight shape"):
            _analyze(spec)

    def test_missing_weight_axes(self):
        with pytest.raises(Exception, match="requires `weight_axes`"):
            _load("qt_toy_no_axes", down="    stored_axes: [1, 0]")

    def test_bias_sibling(self):
        with pytest.raises(Exception, match="sibling tensors"):
            _analyze(_load("qt_toy_bias", down_op="linear",
                           down_bias=', bias: "1"'))

    def test_quantized_saved_format(self):
        with pytest.raises(Exception, match="quant method 'fp8'"):
            physical_artifacts(_analyze(_load("qt_toy_fp8", quant=FP8)))
