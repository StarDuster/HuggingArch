"""A call binds what its callee takes (`ArchSpec.check_call_bindings`): a
component's open slots — `params:` of kind component with no default
(`BlockGraph.open_slots`) — are bound at every call of it, whether a graph
node, a stack's `template:` or a model-level layer makes the call. Left
unbound, the slot's node runs no op; the spec is refused at load instead."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.schema import (ArchSpec, BlockAssignment, BlockGraph, GraphNode,
                                 ModelLayer, StreamSpec)
from backend.arch.schema.graph import ParamSpec

_NORM = BlockGraph(graph={"n": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
                   outputs=["n"])
#: A wrapper whose `inner` slot only its caller fills.
_SLOTTED = BlockGraph(params={"inner": ParamSpec(kind="component")},
                      graph={"y": GraphNode(op="<inner>", inputs=["x"])},
                      outputs=["y"])


def _spec(model, extra=None):
    """The token ids, embedded, then `model` (a stack reading the embedding)."""
    return ArchSpec(
        name="t", dims={"d_model": "8", "n_vocab": "16", "n_layers": "2"},
        blocks={"norm_block": _NORM, "slotted": _SLOTTED, **(extra or {})},
        model_inputs={"input_ids": {"kind": "tokens"}},
        model=[ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                          boundary_role="input_embedding",
                          dims={"n_vocab": "n_vocab", "d_model": "d_model"}),
               *model])


def _stack(**kw):
    return ModelLayer(name="layers", repeat="n_layers", inputs=["embed"],
                      streams={"hidden": StreamSpec(carries=True)}, **kw)


def test_open_slots_are_the_component_params_without_default():
    with_default = BlockGraph(
        params={"inner": ParamSpec(kind="component", default="norm_block"),
                "k": ParamSpec(kind="dim")},
        graph={"y": GraphNode(op="<inner>", inputs=["x"])}, outputs=["y"])
    assert _SLOTTED.open_slots == ["inner"]
    assert with_default.open_slots == []


def test_a_stack_template_with_an_unbound_slot_is_refused():
    with pytest.raises(ValidationError, match=r"leaves its slots \['inner'\]"):
        _spec([_stack(template="slotted")])


def test_a_graph_node_calling_it_without_params_is_refused():
    caller = BlockGraph(graph={"c": GraphNode(op="slotted", inputs=["x"])},
                        outputs=["c"])
    with pytest.raises(ValidationError, match=r"Node 'c' in 'caller'"):
        _spec([_stack(template="caller")], extra={"caller": caller})


def test_an_assignment_with_an_unbound_slot_is_refused():
    with pytest.raises(ValidationError, match=r"leaves its slots \['inner'\]"):
        _spec([_stack(assignments=[
            BlockAssignment(template="norm_block", range=["0", "0"]),
            BlockAssignment(template="slotted", rest=True)])])


def test_a_model_level_op_with_an_unbound_slot_is_refused():
    with pytest.raises(ValidationError, match=r"Model layer 'tail'"):
        _spec([ModelLayer(name="tail", op="slotted", inputs=["embed"])])


def test_a_call_that_binds_the_slot_loads():
    caller = BlockGraph(
        graph={"c": GraphNode(op="slotted", inputs=["x"],
                              params={"inner": "norm_block"})},
        outputs=["c"])
    spec = _spec([_stack(template="caller")], extra={"caller": caller})
    spec.check_call_bindings()
