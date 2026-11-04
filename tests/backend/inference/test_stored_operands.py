"""A measurement key reads a head-broadcast K/V as stored
(`benchmark_identity.stored_operands`): the kernel takes one copy per KV
head, not the view the engine broadcast to every query head."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

from backend.inference.benchmark_identity import (
    identity_shape_from_shape_in, stored_operands)
from backend.inference.op_signatures import get_signature


def test_absorbed_mla_keys_its_one_latent():
    # DeepSeek-V3 decode: Q per head, the latent broadcast to 128 heads as K
    # and as V — a store the walk marks headless (`_Fanout.headless_store`),
    # which is what drops the axis: a fan-out ratio alone leaves a store's
    # heads divided, never removed (one head per KV head is still a head).
    walked = [[1, 128, 1, 576], [1, 128, 1152, 576], [1, 128, 1152, 512]]
    forms = stored_operands(get_signature("mla_attention"), walked,
                            {0: Fraction(1), 1: Fraction(128), 2: Fraction(128)},
                            frozenset({1, 2}))
    assert forms[0] == [[1, 128, 1, 576], [1, 1152, 576], [1, 1152, 512]]
    key = identity_shape_from_shape_in("mla_attention", forms[0],
                                       shape_out=[1, 128, 1, 512],
                                       spec_dims={"kv_block_size": 16})
    assert '"n_kv_heads"' not in key and '"n_heads":128' in key


def test_the_sink_spans_value_is_its_one_latent_too():
    # Hy4 decode (`flash_dsa_sink`): the stored latent is the score's K
    # (input 1) and, broadcast to all 64 heads, the apply's value (input 4).
    # Both are dropped to the store; the value used to keep its 64 heads,
    # and the launch gate then held every build to a [2,64,1088,512]
    # operand no kernel reads.
    walked = [[2, 64, 1, 576], [2, 64, 1088, 576], [2, 1, 1088], [64],
              [2, 64, 1088, 512]]
    sig = get_signature("mla_sparse_sink_attention")
    assert sig.head_axis(4, 4) == 1
    forms = stored_operands(sig, walked, {1: Fraction(64), 4: Fraction(64)},
                            frozenset({1, 4}))
    assert forms[0][1] == [2, 1088, 576]
    assert forms[0][4] == [2, 1088, 512]
    key = identity_shape_from_shape_in(
        "mla_sparse_sink_attention", forms[0], shape_out=[2, 64, 1, 512],
        spec_dims={"kv_block_size": 16, "n_topk_keys": 2048})
    assert key is not None and '"d_v_head":512' in key, key


def test_a_gqa_kv_broadcast_to_every_head_is_one_kv_head():
    # DeepSeek-V4: one KV head broadcast to 128 query heads.
    walked = [[1, 128, 1, 512], [1, 128, 140, 512], [128]]
    forms = stored_operands(get_signature("gqa_attention"), walked,
                            {1: Fraction(128)})
    keys = [identity_shape_from_shape_in("gqa_attention", f,
                                         shape_out=[1, 128, 1, 512],
                                         spec_dims={"kv_block_size": 16})
            for f in forms]
    assert next(k for k in keys if k).count('"n_kv_heads":1') == 1


def test_a_real_per_head_kv_is_left_as_it_is():
    walked = [[1, 32, 64, 128], [1, 8, 64, 128], [1, 8, 64, 128]]
    assert stored_operands(get_signature("gqa_attention"), walked,
                           {1: Fraction(1)}) == [walked]


def test_a_fanout_on_an_operand_without_a_head_axis_is_not_a_head_share():
    walked = [[1, 1024, 4, 4096], [1, 1024, 4, 4096]]
    assert stored_operands(get_signature("gemm"), walked,
                           {1: Fraction(4)}) == [walked]


def test_a_fanout_that_does_not_divide_the_heads_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        stored_operands(get_signature("gqa_attention"),
                        [[1, 6, 1, 64], [1, 6, 9, 64]], {1: Fraction(4)})


def test_absorbed_mla_at_one_head_per_rank_keys_the_same_latent():
    # DeepSeek-V3 decode at 128-way attention TP: one query head per rank,
    # so the latent's view repeats its store exactly once and the fan-out
    # alone cannot say the store is headless. The walk says it
    # (`traffic._Fanout.headless_store`); without it this cell came out as
    # the non-absorbed prefill geometry, `n_kv_heads`, and no decode kernel
    # serves that — 1,600 MLA cells were refused instead of measured.
    walked = [[1, 1, 1, 576], [1, 1, 1152, 576], [1, 1, 1152, 512]]
    forms = stored_operands(get_signature("mla_attention"), walked,
                            {0: Fraction(1), 1: Fraction(1), 2: Fraction(1)},
                            frozenset({1, 2}))
    assert forms[0] == [[1, 1, 1, 576], [1, 1152, 576], [1, 1152, 512]]
    key = identity_shape_from_shape_in("mla_attention", forms[0],
                                       shape_out=[1, 1, 1, 512],
                                       spec_dims={"kv_block_size": 32})
    assert '"n_kv_heads"' not in key and '"n_heads":1' in key


def test_a_headless_store_is_squeezed_and_offers_no_other_reading():
    # The second form exists for a head extent that MIGHT be one stored
    # copy. A store the walk calls headless is not a maybe, and the
    # variant geometry must not be reachable from it.
    walked = [[1, 1, 1, 576], [1, 1, 1152, 576], [1, 1, 1152, 512]]
    forms = stored_operands(get_signature("mla_attention"), walked,
                            {1: Fraction(1), 2: Fraction(1)},
                            frozenset({1, 2}))
    assert all(form == forms[0] for form in forms)


def test_one_stored_head_the_walk_does_not_call_headless_keeps_its_axis():
    # The reading the ratio cannot distinguish: an MQA store with its own
    # head axis, read at one head. Nothing is squeezed without the walk's
    # fact, so a real per-head cell is still keyed as one.
    walked = [[1, 1, 1, 64], [1, 1, 9, 64], [1, 1, 9, 64]]
    assert stored_operands(get_signature("gqa_attention"), walked,
                           {1: Fraction(1)}) == [walked]


def test_the_head_axis_is_the_layout_of_that_rank():
    # indexer_logits lays out a rank-3 weights [B, s, heads] and a rank-4 q
    # [B, heads, s, d] at one position; each operand reads its own.
    sig = get_signature("indexer_logits")
    assert sig.head_axis(2, 3) == 2
    assert sig.head_axis(0, 4) == 1


def test_a_read_term_past_the_operands_is_refused():
    with pytest.raises(ValueError, match="different launches"):
        stored_operands(get_signature("gqa_attention"),
                        [[1, 8, 1, 64], [1, 8, 9, 64]], {5: Fraction(8)})
