"""OpSpec.input_roles — what each operand is to the op.

Structure views group a component by the operand each node ultimately feeds
(the query path, the key path, the routing …). That grouping is only as
trustworthy as these declarations, so they are checked against the other
declarations and against every place the ops are wired.
"""
from __future__ import annotations

import glob
import pathlib

import pytest
import yaml

import backend.arch.ops_registry  # noqa: F401  (registers the ops)
from backend.arch.ops_base import OP_REGISTRY, OpSpec, INPUT_ROLE_VOCABULARY

ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown input role"):
        OpSpec("x_bad_role", flops_fn=lambda d: 0, params_fn=lambda d: 0,
               input_roles=("query",))


def test_duplicate_role_is_refused():
    with pytest.raises(ValueError, match="duplicate input role"):
        OpSpec("x_dup_role", flops_fn=lambda d: 0, params_fn=lambda d: 0,
               input_roles=("q", "q"))


def test_a_mask_edge_is_the_selection():
    with pytest.raises(ValueError, match="not 'select'"):
        OpSpec("x_mask_role", flops_fn=lambda d: 0, params_fn=lambda d: 0,
               input_kinds=("activation", "activation", "mask"),
               input_roles=("q", "k", "v"))


def test_attention_kernels_declare_their_operands():
    """Score reads Q at 0 and K at 1; apply reads V at 1 — the positions the
    score/apply shape templates already consume."""
    for name, spec in OP_REGISTRY.items():
        kernel = getattr(spec, "attention_kernel", None)
        if kernel == "score":
            assert spec.input_roles[:2] == ("q", "k"), name
        elif kernel == "apply":
            assert spec.input_role(1) == "v", name


def _wired_arities():
    """op name -> set of input counts it is wired with, across every
    library / adapter component graph."""
    seen: dict = {}

    def walk(obj):
        if isinstance(obj, dict):
            op = obj.get("op")
            if isinstance(op, str) and op in OP_REGISTRY:
                ins = obj.get("in")
                n = len(ins) if isinstance(ins, list) else (1 if ins else 0)
                seen.setdefault(op, set()).add(n)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for f in glob.glob(str(ROOT / "backend/arch/components/**/*.yaml"), recursive=True):
        walk(yaml.safe_load(pathlib.Path(f).read_text()) or {})
    return seen


def test_every_wired_input_of_a_role_declaring_op_has_a_role():
    wired = _wired_arities()
    short = {name: (spec.input_roles, sorted(wired[name]))
             for name, spec in OP_REGISTRY.items()
             if spec.input_roles and name in wired
             and max(wired[name]) > len(spec.input_roles)}
    assert not short, short


def test_vocabulary_is_used():
    used = {r for s in OP_REGISTRY.values() for r in s.input_roles if r is not None}
    assert used <= INPUT_ROLE_VOCABULARY
    assert {"q", "k", "v", "tokens", "routing"} <= used


def test_a_position_may_play_no_part():
    """None marks a position with no role (a parameter between operands);
    it is neither validated as a role nor counted as a duplicate."""
    spec = OP_REGISTRY["mamba2_scan"]
    assert spec.input_role(2) is None and spec.input_role(3) == "k"
    assert spec.input_roles.count(None) == 3


def test_a_negative_position_is_not_a_declared_input():
    spec = OP_REGISTRY["attention_score"]
    assert spec.input_role(-1) is None
    assert spec.input_kind(-1) == "activation"
