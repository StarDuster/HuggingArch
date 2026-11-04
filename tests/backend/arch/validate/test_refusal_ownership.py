"""Who resolves an analysis refusal is stated where it is raised.

A spec writer is told to repair what the spec can repair, and a run stops
only on what the engine cannot yet do. A `SpecRefusal` names the declaration
and the fix; an `EngineGap` names the capability the engine lacks; a refusal
raised as neither claims neither side — the writer is sent to compare the node
with the source, not told that no spec edit can help. Two V4.1-Flash
refusals that were spec errors (a mislabelled width, a row scatter where the
source selects a bias per token) used to read as the engine's.

A literal a node binds under a model dim's name, at another value, is caught
where the spec makes it rather than at the per-rank projection.
"""
from __future__ import annotations

import pytest

from backend.arch.eval.refusal import EngineGap, SpecRefusal
from backend.arch.loader import ResolvedSpec
from backend.arch.schema import ArchSpec, BlockGraph, GraphNode, ModelLayer
from backend.arch.validate._analysable import (
    ENGINE_KIND, OPEN_KIND, SPEC_KIND, Refusal, refusal_finding)


def _refusal(exc: BaseException) -> Refusal:
    try:
        raise exc
    except BaseException as raised:
        return Refusal.of(raised, deployment="tp8", stage="phases")


def test_a_spec_refusal_names_the_declaration_and_fix():
    r = _refusal(SpecRefusal("bad", declaration="n.in", fix="write x"))
    assert r.kind == SPEC_KIND
    assert "Fix `n.in`: write x" in refusal_finding([r]).detail


def test_an_engine_gap_names_the_capability_it_lacks():
    r = _refusal(EngineGap("stuck", capability="two writers of one stream"))
    assert r.kind == ENGINE_KIND
    detail = refusal_finding([r]).detail
    assert "does not yet support two writers of one stream" in detail


def test_not_implemented_is_the_engines_own_statement_of_a_gap():
    r = _refusal(NotImplementedError("DCP sparse top-k"))
    assert r.kind == ENGINE_KIND
    assert "does not yet support DCP sparse top-k" in refusal_finding([r]).detail


def test_a_refusal_raised_as_neither_claims_no_owner():
    r = _refusal(ValueError("rows arrive cut"))
    assert r.kind == OPEN_KIND
    detail = refusal_finding([r]).detail
    assert "forward source" in detail
    assert "engine" not in detail.split(":", 1)[0]


def test_operands_that_do_not_broadcast_are_the_spec_writers():
    """One operand names a dim, the other a literal at another value: they
    disagree, and the operands are the spec's choice — so analyze offers
    Agent Fix rather than an ownerless failure."""
    import sympy as sp

    from backend.arch.eval.dim import Dim
    from backend.arch.ops_base import elementwise_broadcast_axes

    try:
        elementwise_broadcast_axes(
            ((Dim(sp.Symbol("n_hidden_copies"), 3),), (4,)))
    except ValueError as exc:
        r = Refusal.of(exc, deployment="the requested analysis",
                       stage="analyze")
    else:
        pytest.fail("incompatible extents broadcast")
    assert r.kind == SPEC_KIND
    assert r.declaration == "in"


def _model(embed_dims: dict) -> ResolvedSpec:
    block = BlockGraph(
        inputs=["x"],
        graph={"e": GraphNode(op="embedding", inputs=["x"], dims=embed_dims)},
        outputs=["e"])
    spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
        name="demo", dims={"d_model": "config.hidden_size"},
        model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"]),
        ],
        blocks={"block": block})
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 1}, {}, {})


def test_a_literal_under_a_model_dims_name_is_refused_where_it_is_bound():
    from backend.arch.eval.program import Program
    with pytest.raises(ValueError, match="is the model's dim"):
        Program.of(_model({"n_vocab": "32", "d_model": "4"}))


def test_the_models_own_value_written_as_a_number_is_refused():
    """At the model's value the literal is right only while the config is."""
    from backend.arch.eval.program import Program
    for literal in ("16", "2 * 8"):
        with pytest.raises(ValueError, match="written as a number"):
            Program.of(_model({"n_vocab": "32", "d_model": literal}))


def test_a_named_dim_binds():
    from backend.arch.eval.program import Program
    Program.of(_model({"n_vocab": "32", "d_model": "d_model"}))


def _assigned(layer_dims: dict) -> ResolvedSpec:
    """`_model`'s graph, its layers chosen by one assignment that binds
    `layer_dims`."""
    from backend.arch.schema import BlockAssignment
    resolved = _model({"n_vocab": "32", "d_model": "d_model"})
    spec = resolved.spec
    layers = spec.model[1].model_copy(update={
        "template": None, "repeat": "n_layers",
        "assignments": [BlockAssignment(template="block", range=["0", "0"],
                                        dims=layer_dims)]})
    return ResolvedSpec(spec.model_copy(update={"model": [spec.model[0], layers]}),
                        resolved.dim_values, {}, {})


def test_an_assignment_restating_the_models_value_as_a_number_is_refused():
    """An assignment's `dims:` rebinds the model's dim for its layers; at
    the model's own value it rebinds nothing and is the dim written as a
    number. Raised as a `SpecRefusal`, so a saved spec carrying one is
    offered Agent Fix."""
    from backend.arch.eval.program import Program
    with pytest.raises(SpecRefusal, match="written as a number") as caught:
        Program.of(_assigned({"d_model": "16"}))
    r = Refusal.of(caught.value, deployment="tp8", stage="phases")
    assert r.kind == SPEC_KIND
    assert r.declaration.endswith("dims.d_model")


def test_an_assignment_may_give_the_models_dim_another_value():
    """Another value for the layers an assignment selects is what an
    assignment's `dims:` is for (a playground group at 128 experts)."""
    from backend.arch.eval.program import Program
    Program.of(_assigned({"d_model": "8"}))


def test_an_assignment_may_bind_a_dim_the_model_leaves_unbound():
    """A dim stated only per layer (DeepSeek-V4's `compress_ratio`, selected
    by the same `compress_ratios` entry its assignment matches) has no
    model-wide value to contradict."""
    from backend.arch.eval.program import Program
    Program.of(_assigned({"compress_ratio": "4"}))
