"""The formula-display substitution helpers.

The substituted line of a formula popover binds the symbolic expression with
the substitution record the evaluation itself used (`flops_from_shape`'s
`record` / `physical_cost_plan`' params record — identity → value),
keeps the products uncollapsed, and is shown only when the bound form
reproduces the engine's value exactly.
"""
from __future__ import annotations

import sympy as sp

from backend.arch.eval.dim import (
    Dim, bind_keep_products, bound_formula_text, bound_or_folded,
)
from backend.arch.eval.flops import merge_bindings, record_binding

B, s_ctx, d_model, d_ffn = sp.symbols("B s_ctx d_model d_ffn")
n_heads, d_head = sp.symbols("n_heads d_head")
n_route_targets, n_experts_active, dp_attention = sp.symbols(
    "n_route_targets n_experts_active dp_attention")


def test_bind_keep_products_keeps_products():
    expr = 2 * B * d_ffn * d_model * s_ctx
    bound = bind_keep_products(
        expr, {"B": 4, "d_ffn": 8192, "d_model": 3072, "s_ctx": 1024})
    assert str(bound) == "2*4*8192*3072*1024"


def test_unbound_symbol_stays_symbolic():
    expr = 2 * B * d_ffn
    bound = bind_keep_products(expr, {"B": 4})
    assert bound.free_symbols == {d_ffn}


def test_record_binding_keys_on_identity():
    record = {}
    record_binding(record, Dim(n_heads, 12))
    record_binding(record, Dim(n_heads * d_head, 1536))
    record_binding(record, Dim(sp.Integer(1), 1))   # no identity: not recorded
    assert record == {n_heads: 12, n_heads * d_head: 1536}


def test_record_binding_drops_conflicting_identities():
    record = {}
    record_binding(record, Dim(d_ffn, 8192))
    record_binding(record, Dim(d_ffn, 4096))   # same identity, other scope
    record_binding(record, Dim(d_model, 3072))
    assert record == {d_model: 3072}


def test_record_binding_overwrite_tier():
    # The tensor-measured tier overrides the environment tier on an identity
    # collision (CP: env's s_ctx is the shard, the query axis is full).
    # Within one evaluation the env tier is always recorded first.
    record = {}
    record_binding(record, Dim(s_ctx, 512))                      # env tier
    record_binding(record, Dim(s_ctx, 1024), overwrite=True)     # axis tier
    assert record == {s_ctx: 1024}


def test_merge_bindings_unions_and_drops_conflicts():
    merged = merge_bindings([
        {n_heads: 8, d_head: 128},
        {n_heads: 8, d_ffn: 2048},
        {d_head: 64},                          # disagrees with the first
    ])
    assert merged == {n_heads: 8, d_ffn: 2048}


def test_composite_binds_as_unit_inside_flattened_product():
    # q_b_proj under TP: the sharded head count exists only as the
    # product `n_heads*(d_qk_nope + d_qk_rope)` — never as a bare axis.
    d_q_lora, d_qk_nope, d_qk_rope = sp.symbols(
        "d_q_lora d_qk_nope d_qk_rope")
    expr = d_q_lora * n_heads * (d_qk_nope + d_qk_rope)
    text = bound_formula_text(
        expr, {"d_q_lora": 1536}, 2359296,
        {n_heads * (d_qk_nope + d_qk_rope): 1536})
    assert text == "1536*1536"


def test_composite_with_divisor_binds_as_unit():
    # MoE dispatch zone: the token axis is
    # `B*dp_attention*n_experts_active*s_ctx/n_route_targets` — its symbols
    # are rebound scope-locally in the node dims, so only the composite
    # value is trustworthy.
    expr = (2 * B * d_ffn * d_model * dp_attention * n_experts_active * s_ctx
            / n_route_targets)
    composite = B * dp_attention * n_experts_active * s_ctx / n_route_targets
    text = bound_formula_text(
        expr, {"d_ffn": 2048, "d_model": 7168}, 2 * 128 * 2048 * 7168,
        {composite: 128})
    assert text == "2*2048*7168*128"


def test_wrong_binding_is_refused():
    # A record that cannot reproduce the engine's value must not display:
    # the folded number stands instead.
    expr = 2 * B * d_head * d_model * n_heads * s_ctx
    text = bound_formula_text(
        expr,
        {"B": 4, "d_head": 128, "d_model": 3072, "n_heads": 24,
         "s_ctx": 1024},
        2 * 4 * 128 * 3072 * 12 * 1024,   # per-rank value: n_heads is 12
    )
    assert text is None


def test_bound_or_folded_keeps_products():
    d_q_lora, d_qk_nope, d_qk_rope = sp.symbols(
        "d_q_lora d_qk_nope d_qk_rope")
    expr = d_q_lora * n_heads * (d_qk_nope + d_qk_rope)
    expect = 1536 * 128 * 192
    extent = Dim(expr, expect)
    record = {
        d_q_lora: 1536, n_heads: 128, d_qk_nope: 128, d_qk_rope: 64,
    }
    text = bound_or_folded(extent, record, expect)
    assert "*" in text
    assert text != str(expect)
    assert sp.sympify(text) == expect
    assert bound_or_folded(extent, None, expect) == str(expect)


def test_record_names_every_factor_when_the_symbols_reproduce_it():
    """A record holding both the axis product and each symbol's value binds
    the symbols — every factor named — when they reproduce the figure, in
    the order the symbolic line prints its terms."""
    d_q_lora, d_qk_nope, d_qk_rope = sp.symbols(
        "d_q_lora d_qk_nope d_qk_rope")
    expr = d_q_lora * n_heads * (d_qk_nope + d_qk_rope)
    record = {d_q_lora: 1536, n_heads * (d_qk_nope + d_qk_rope): 24576,
              n_heads: 128, d_qk_nope: 128, d_qk_rope: 64}
    text = bound_or_folded(Dim(expr, 1536 * 24576), record)
    assert text == "1536*128*(128 + 64)"
    assert str(expr) == "d_q_lora*n_heads*(d_qk_nope + d_qk_rope)"


def test_a_shard_keeps_the_axis_product():
    """Per rank the head count is the shard's: the symbols' values do not
    reproduce the figure, so the axis product binds as before."""
    d_q_lora, d_qk_nope, d_qk_rope = sp.symbols(
        "d_q_lora d_qk_nope d_qk_rope")
    expr = d_q_lora * n_heads * (d_qk_nope + d_qk_rope)
    record = {d_q_lora: 1536, n_heads * (d_qk_nope + d_qk_rope): 3072,
              n_heads: 128, d_qk_nope: 128, d_qk_rope: 64}
    assert bound_or_folded(Dim(expr, 1536 * 3072), record) == "1536*3072"
