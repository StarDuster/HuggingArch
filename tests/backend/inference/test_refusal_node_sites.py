"""A refusal raised while the serving analysis handles one node names that
node: the recurrent-state projection of the physical cache, the per-op row
keying, and the two per-op folds `_capacity_finish` builds its phases from."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.arch.eval.access import LeafAccess
from backend.arch.eval.dim import value_of
from backend.arch.eval.layer import Layer
from backend.arch.eval.refusal import refusal_node
from backend.arch.layout.physical_cache import cache_geometry
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import _flops_rows, compute_per_op_timing
from backend.inference.pipeline import analyze
from backend.inference.step_eval import evaluate_step
from tests.serving_fixture import numbers_as_laws
from tests.snapshots import shared_loader

SNAPSHOTS = Path(__file__).resolve().parents[3] / "backend" / "arch" / "snapshots"
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


def _step(snapshot: str):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return evaluate_step(resolved, TP8, phase="prefill", batch=1, input_tokens=4096, output_tokens=0, s_prefix_cached=3968,
)


@pytest.fixture(scope="module")
def llama():
    return _step("meta-llama__Llama-3.2-3B.json")


def _leaves(program):
    return [node for node in program.nodes() if not node.children]


def test_a_recurrent_state_projection_refusal_names_its_carrier():
    analysis = _step("Qwen__Qwen3-Coder-Next.json").analysis
    seen = []

    def refuse(analysis, kernel_shapes, component, carrier, *rest):
        seen.append(carrier.display_path)
        raise ValueError("state extent does not divide")

    with (patch("backend.arch.layout.physical_cache._project_state", refuse),
          pytest.raises(ValueError) as caught):
        cache_geometry(analysis)
    assert seen and refusal_node(caught.value) == seen[0]


def test_a_row_keying_refusal_names_the_rows_node():
    # Rows are keyed off the serving walk (`pipeline.analyze`), the result
    # `RowIdentity` reads its fused program and cache access from.
    payload = json.loads((SNAPSHOTS / "meta-llama__Llama-3.2-3B.json").read_text())
    walk = analyze(shared_loader().match(payload["config"],
                                         model_id="meta-llama/Llama-3.2-3B"),
                   parallel=TP8, avg_input_tokens=128, avg_output_tokens=0,
                   phase="prefill", breakdown=False)
    program = walk.program
    node = next(n for n in _leaves(program) if n.aggregation_id)
    first = next(n for n in program.nodes()
                 if str(n.aggregation_id) == str(node.aggregation_id))
    row = {"aggregation_id": node.aggregation_id, "op": node.op,
           "shape_in": None}
    refuse = ValueError("no kernel for this operand rank")
    with (patch("backend.inference.benchmark_identity.resolve_op",
                side_effect=refuse),
          pytest.raises(ValueError) as caught):
        RowIdentity(walk).of(row)
    assert refusal_node(caught.value) == first.display_path


def test_a_flops_row_refusal_names_its_node(llama):
    empty = SimpleNamespace(values={})
    step = SimpleNamespace(analysis=llama.analysis, has_deployment=True,
                           physical_flops=lambda: empty)
    with pytest.raises(ValueError) as caught:
        _flops_rows(step)
    node = refusal_node(caught.value)
    assert node is not None
    assert str(caught.value) == f"physical FLOPs have no value for {node}"


def test_a_per_op_row_refusal_names_its_leaf(llama):
    analysis = llama.analysis
    program = analysis.program
    leaf = next(
        n for n in _leaves(program)
        if analysis.cost.values.get(n.uid) is not None
        and value_of(analysis.cost.values[n.uid].flops or 0) > 0
    )
    # The row type itself, at its defaults: no activation dtype declared. A
    # hand-listed stub of its fields went stale the day the row grew one.
    undeclared = LeafAccess(full=0, per_rank=0)
    with pytest.raises(ValueError) as caught:
        compute_per_op_timing(
            program, analysis.cost, analysis.shapes,
            Layer(program, {leaf.uid: undeclared}),
            instances=analysis.instances,
            weight_shards=analysis.weight_shards,
            laws=numbers_as_laws(
                cost=analysis.cost, shapes=analysis.shapes,
                access=Layer(program, {leaf.uid: undeclared}),
                access_shapes=analysis.shapes, step_rows=1),
            pre=False,
            at={"B": 1, "s_input": 1, "s_output": 0, "s_prefix": 0,
                "s_verify": 1, "rho": 1},
            gpus_per_node=None,
            breakdown=False,
        )
    assert "activation dtype" in str(caught.value)
    assert refusal_node(caught.value) == leaf.display_path
