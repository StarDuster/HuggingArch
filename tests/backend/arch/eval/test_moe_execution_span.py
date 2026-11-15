from types import SimpleNamespace
import pytest
from backend.arch.eval.fusion import _routed_span, FusedProgram
from backend.arch.eval.launch import FusedNode


def node(uid, op, *parents, routed=False):
    return SimpleNamespace(uid=uid, op=op, routed=routed, children=(),
        inputs_obj=tuple(SimpleNamespace(src=p, port='', external_sid='') for p in parents))


PROGRAM = SimpleNamespace(in_zone=lambda n: n.routed)


def test_whole_expert_span_uses_original_tokens_not_per_expert_padding():
    x = node(1, 'linear')
    dispatch = node(2, 'permute', x)
    first = node(3, 'linear', dispatch, routed=True)
    quant = node(4, 'quantise_activation', first)
    second = node(5, 'linear', quant, routed=True)
    combine = node(6, 'combine', second)
    reduction = node(7, 'all_reduce', combine)
    norm = node(8, 'rmsnorm', reduction)
    assert not _routed_span(PROGRAM, [dispatch, first, quant, second, combine, reduction, norm])


def test_partial_crossing_still_refuses_a_missing_combine():
    dispatch = node(2, 'permute')
    expert = node(3, 'linear', dispatch, routed=True)
    norm = node(4, 'rmsnorm', expert)
    with pytest.raises(ValueError, match='straddles'):
        _routed_span(PROGRAM, [dispatch, expert, norm])


def test_partial_crossing_still_refuses_a_missing_dispatch():
    external = node(1, 'linear')
    expert = node(3, 'linear', external, routed=True)
    combine = node(4, 'combine', expert)
    with pytest.raises(ValueError, match='straddles'):
        _routed_span(PROGRAM, [expert, combine])


def test_expert_only_launch_retains_routed_geometry():
    first = node(1, 'linear', routed=True)
    second = node(2, 'situ', first, routed=True)
    assert _routed_span(PROGRAM, [first, second])


def test_whole_span_does_not_inherit_internal_dispatch_membership():
    base=SimpleNamespace(routed_uids=frozenset({3}), dispatch_members=frozenset({2,3,4}))
    span=FusedNode(uid=10,op='test_complete_moe',origins=(2,3,4),owns_dispatch=True)
    graph=FusedProgram(base=base,fused=(span,))
    assert 10 not in graph.routed_uids
    assert 10 not in graph.dispatch_members


def test_whole_span_weight_demand_reads_original_tokens():
    from backend.inference.row_roofline import weight_demand
    span=FusedNode(uid=10,op='test_complete_moe',origins=(2,3,4),owns_dispatch=True)
    demand=weight_demand(None,span,None,active_ratio=.25,step_rows=2,total=1200)
    assert demand.asked == 600
    assert demand.saturates
