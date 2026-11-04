"""What a spec-scoped custom op may declare, and what it may not.

`CustomOpSpec` holds the inputs `build_custom_op_spec()` needs to CONSTRUCT an
`OpSpec` — the signature (arity + operand shapes), the shape kind, the cost
expressions, and the weight geometry. Everything else about an op is the
registration flow's to give, so this locks both halves: the declarations that
must compile into a real OpSpec, and the rejections that send an author to
`ops.py` instead of letting a behavior go silently undeclared.
"""
import re

import pytest

from backend.arch.schema import CustomOpSpec
from backend.arch.axis_propagation import compute_shape_out
from backend.arch.ops import OpSpec, build_custom_op_spec


def _cop(**kw):
    """A minimally-complete custom op."""
    base = dict(signature={"x": ["B", "s_ctx", "d_model"]},
                couples_on=[], shape_kind="preserve",
                description="test-only novel primitive")
    base.update(kw)
    return CustomOpSpec(**base)


def test_signature_becomes_the_ops_positional_shape_in():
    op = build_custom_op_spec("custom_mixer", _cop())
    assert isinstance(op, OpSpec)
    # One operand: the flat shape, as registered primitives declare it.
    assert op.shape_in == ("B", "s_ctx", "d_model")
    assert op.shape_kind == "preserve"


def test_multi_operand_signature_nests_in_call_order():
    cop = _cop(
        signature={"q": ["B", "N", "Sq", "D"], "kt": ["B", "N", "D", "Lq"]},
        shape_kind="contract",
    )
    op = build_custom_op_spec("custom_score", cop)
    assert op.shape_in == (("B", "N", "Sq", "D"), ("B", "N", "D", "Lq"))
    assert cop.arity() == 2


def test_coupling_label_not_present_in_signature_is_rejected():
    with pytest.raises(ValueError, match="not present in signature"):
        _cop(signature={"q": ["B", "S", "D"], "k": ["B", "S", "D"]},
             couples_on=["missing"])


def test_flops_params_compile_from_strings():
    # A weight-bearing axis rewrite is a linear map over the operand's last
    # axis (the default `(out, in)` geometry), so the fixture says it couples.
    cop = _cop(flops="2*B*s_ctx*In*Out + B*s_ctx*Out", params="In*Out",
               shape_kind="axis_replace", out_key="Out", couples_on=["d_model"])
    op = build_custom_op_spec("custom_lin", cop)
    assert int(op.flops_fn({"B": 1, "s_ctx": 8, "In": 2, "Out": 3})) \
        == 2 * 8 * 2 * 3 + 8 * 3
    assert int(op.params_fn({"In": 2, "Out": 3})) == 6


def test_weight_shape_declares_nonlinear_weight_leaf():
    # A norm-scale-like custom op: weight is the [D] vector, not (out,in) linear.
    op = build_custom_op_spec("custom_norm",
                              _cop(params="d_head", weight_shape=["d_head"]))
    from backend.arch.ops import resolve_shape_template
    assert op.weight_shape == ("d_head",)
    assert resolve_shape_template(op.weight_shape, {"d_head": 4096}) == (4096,)
    # unbound dim → None (not a spurious leaf)
    assert resolve_shape_template(op.weight_shape, {"d_head": 0}) is None


def test_embedding_shape_two_dims():
    # `V` rather than `n_vocab`: the table's row count is an axis of THIS op's
    # weight, and a name vocab parallelism divides would size it from the
    # policy instead of from the call. `params` may still say `n_vocab` — a
    # cost expression is allowed to read the model's own quantity.
    op = build_custom_op_spec(
        "custom_embed",
        _cop(params="n_vocab * d_model", weight_shape=["V", "d_model"]))
    from backend.arch.ops import resolve_shape_template
    assert resolve_shape_template(
        op.weight_shape, {"V": 152064, "d_model": 4096}) == (152064, 4096)


def test_axis_rewrite_without_out_key_is_rejected():
    for kind in ("axis_replace", "axis_select"):
        with pytest.raises(ValueError, match="out_key"):
            _cop(shape_kind=kind)


def test_axis_select_clamps_to_input_extent():
    op = build_custom_op_spec(
        "custom_topk",
        _cop(shape_kind="axis_select", out_key="k"),
    )
    assert compute_shape_out(op, [(2, 3, 4)], {"k": 8}) == (2, 3, 4)
    assert compute_shape_out(op, [(2, 3, 4)], {"k": 2}) == (2, 3, 2)


def test_broadcast_shape_kind_is_available_to_custom_elementwise_ops():
    op = build_custom_op_spec(
        "custom_gate",
        _cop(
            signature={
                "gate": ["dim"],
                "activation": ["B", "s_ctx", "dim"],
            },
            shape_kind="broadcast",
        ),
    )
    assert compute_shape_out(op, [(32,), (2, 16, 32)], {}) == (2, 16, 32)


def test_shape_kinds_needing_a_dim_map_are_rejected():
    # `explicit` / `source` ARE OpSpec kinds — they just aren't reachable from
    # YAML, because their DimMap cannot be written there.
    for kind in ("explicit", "source"):
        with pytest.raises(ValueError, match=re.escape("ops.py")):
            _cop(shape_kind=kind)


def test_input_roles_compile_into_the_op():
    # A custom recurrence is an attention core: it reads a query, keys and
    # values, and says so the way a registered score does.
    cop = _cop(signature={"q": ["B", "S", "D"], "k": ["B", "S", "D"],
                          "v": ["B", "S", "D"]},
               input_roles=["q", "k", "v"])
    assert build_custom_op_spec("custom_recurrence", cop).input_roles \
        == ("q", "k", "v")
    assert build_custom_op_spec("custom_mixer", _cop()).input_roles == ()


@pytest.mark.parametrize("roles, match", [
    (["query"], "unknown role"),
    (["q", "q"], "duplicate role"),
    (["q", "k"], "signature operand"),
])
def test_input_roles_are_the_closed_vocabulary_one_per_operand(roles, match):
    with pytest.raises(ValueError, match=match):
        _cop(input_roles=roles)


def test_missing_signature_is_rejected():
    with pytest.raises(ValueError, match="signature"):
        CustomOpSpec(couples_on=[], shape_kind="preserve")


def test_undeclarable_behaviors_are_rejected_by_name():
    # Each of these was once a field mirroring OpSpec. An op needing one is an
    # op for ops.py — and asking for it here fails loudly rather than being
    # dropped on the floor.
    for field, value in (("sharding", [{"name": "col", "out": "sharded:Out"}]),
                         ("kv_positions_per_query", "Min(Lq, W)"),
                         ("is_view_op", True),
                         ("is_comm_op", True),
                         ("output_dtype", "I32"),
                         ("required_dims", ["out"]),
                         ("shape_out", ["B", "s_ctx", "d_model"]),
                         ("shape_in", ["B", "s_ctx", "d_model"]),
                         ("permute_to", [0, 2, 1]),
                         ("kv_block_anchor", False)):
        with pytest.raises(ValueError, match=r"[Ee]xtra"):
            _cop(**{field: value})


def test_shape_parameter_named_after_a_divisible_symbol_is_rejected():
    """The name an axis is bound under is what the shard policy divides.

    Two shipped custom ops called their contraction axis after a head count and
    bound it to `d_model`: a head count is a name TP divides, so at TP=8 the
    axis shrank by eight while the width it was bound to did not, and the op
    reported an eighth of its FLOPs and an eighth of its weight read. The
    sibling call binding `d_ffn_expert` came out right — that one really does
    split by the same factor — which is what made the pair look consistent.
    """
    with pytest.raises(ValueError, match="a parallelism divides"):
        build_custom_op_spec("custom_lin", _cop(
            signature={"x": ["B", "s_ctx", "n_kv_heads"]},
            shape_kind="axis_replace", out_key="Out"))
    with pytest.raises(ValueError, match="a parallelism divides"):
        # A `[Out, d_ffn]` weight under an axis rewrite is a linear map, and a
        # linear map couples on its in axis — say so, so the refusal under
        # test (the divisible symbol) is the one that fires.
        build_custom_op_spec("custom_lin", _cop(
            signature={"x": ["B", "s_ctx", "d_ffn"]},
            weight_shape=["Out", "d_ffn"], couples_on=["d_ffn"],
            shape_kind="axis_replace", out_key="Out"))
    with pytest.raises(ValueError, match="a parallelism divides"):
        build_custom_op_spec("custom_lin", _cop(
            shape_kind="axis_replace", out_key="n_heads"))


def _custom_ir(**kw):
    from types import SimpleNamespace
    base = dict(
        op="packed_lin",
        from_custom_op=True,
        dims={"In": 8, "Out": 16},
        dim=None,
        dim_raw=None,
        dims_raw={"In": "d_model", "Out": "d_ffn"},
        multiplicity=1,
        role="ffn_moe",
        template="t",
        component_path=(),
        has_bias=False,
        stack_name="layers",
        parent=None,
        name="experts.w13",
        weight_name="experts.w13",
        display_path="layers.X.experts.w13",
        repeat=None,
        repeat_raw=None,
        checkpoint_indexed=False,
        expert_weight=None,
        shape=None,
        tensor_suffix={},
        provenance="spec",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_weight_bearing_custom_op_without_shape_raises():
    """Call sites bind In/Out. Guessing n_indexer_heads used to drop the leaf."""
    from backend.arch.spec_tree import ir_to_leaf
    cop = _cop(flops="2*B*s_ctx*In*Out", params="In*Out",
               signature={"x": ["B", "s_ctx", "In"]},
               shape_kind="axis_replace", out_key="Out",
               couples_on=["In"])
    with pytest.raises(ValueError, match="weight_shape"):
        ir_to_leaf(_custom_ir(), "", {"packed_lin": cop})


def test_zero_param_custom_op_is_not_a_weight_leaf():
    from backend.arch.spec_tree import ir_to_leaf
    cop = _cop(params="0", signature={"x": ["B", "s_ctx", "d_model"]},
               shape_kind="preserve", couples_on=[])
    assert ir_to_leaf(_custom_ir(op="merge", dims={"d_model": 4}),
                       "", {"merge": cop}) is None


def test_declared_weight_shape_emits_the_custom_op_leaf():
    from backend.arch.spec_tree import ir_to_leaf
    cop = _cop(params="In*Out", weight_shape=["Out", "In"],
               signature={"x": ["B", "s_ctx", "In"]},
               shape_kind="axis_replace", out_key="Out",
               couples_on=["In"])
    leaf = ir_to_leaf(_custom_ir(), "", {"packed_lin": cop})
    assert leaf is not None
    assert leaf.shape == (16, 8)
    assert leaf.in_dim == 8
    assert leaf.out_dim == 16


def test_in_out_linear_default_still_emits():
    from backend.arch.spec_tree import ir_to_leaf
    cop = _cop(params="in*out",
               signature={"x": ["B", "s_ctx", "in"]},
               shape_kind="axis_replace", out_key="out",
               couples_on=["in"])
    leaf = ir_to_leaf(
        _custom_ir(dims={"in": 4, "out": 8},
                   dims_raw={"in": "d_model", "out": "d_ffn"}),
        "", {"packed_lin": cop})
    assert leaf is not None
    assert leaf.shape == (8, 4)


def test_engine_supplied_symbols_stay_legal_in_a_signature():
    """`s_ctx` is divisible too, and naming it is the only honest thing to do.

    CP really does divide the sequence axis. What separates it from `n_heads`
    above is that no call site binds it — it arrives from the enclosing frame —
    so there is no second identity for the policy to disagree with.
    """
    op = build_custom_op_spec("custom_mixer", _cop(
        signature={"q": ["B", "s_ctx", "In"], "k": ["B", "s_kv", "In"]},
        shape_kind="preserve"))
    assert op.shape_in == (("B", "s_ctx", "In"), ("B", "s_kv", "In"))
