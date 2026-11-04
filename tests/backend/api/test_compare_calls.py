"""The compare page lines models up by the calls their layers make
(`analyze._component_call_diff`), not by component names: one component
bound with different slots is two calls, and a cell names the sub-calls the
calls of one component differ in."""
from __future__ import annotations

from backend.api.routes.analyze import _component_call_diff


def _call(role, component, layers, **calls):
    return {"role": role, "layers": layers, "params": 0, "active_params": 0,
            "signature": {"component": component,
                          "calls": {name: {"component": sub, "calls": {}}
                                    for name, sub in calls.items()}}}


def _result(*rows):
    return {"composition": {"rows": [
        {"stack": stack, "aux_kind": aux, "calls": list(calls)}
        for stack, aux, calls in rows]}}


def test_one_component_bound_two_ways_is_two_calls():
    a = _result(("layers", None, [
        _call("attention", "csa2", 4, kv="pool", q_split="head_split"),
        _call("attention", "csa2", 31, kv="shared_read", q_split="head_split"),
    ]))
    b = _result(("layers", None, [
        _call("attention", "csa2", 35, kv="pool", q_split="head_split"),
    ]))
    (row,) = _component_call_diff([a, b])
    assert row["slot"] == "attention" and row["same"] is False
    # The sub-call every csa2 call makes alike is not shown; the slot is.
    assert row["values"][0] == ["csa2 [kv=pool] ×4", "csa2 [kv=shared_read] ×31"]
    assert row["values"][1] == ["csa2 [kv=pool] ×35"]


def test_an_aux_stack_call_is_named_by_its_stack_and_counts_apart():
    model = _result(("layers", None, [_call("ffn", "moe", 40)]),
                    ("mtp", "mtp", [_call("ffn", "moe", 1)]))
    (row,) = _component_call_diff([model, model])
    assert row["same"] is True
    assert row["values"][0] == ["moe ×40", "mtp: moe ×1"]


def test_the_same_calls_at_other_layer_counts_differ():
    a = _result(("layers", None, [_call("ffn", "moe", 40)]))
    b = _result(("layers", None, [_call("ffn", "moe", 58)]))
    (row,) = _component_call_diff([a, b])
    assert row["same"] is False
