import pytest

torch = pytest.importorskip('torch')
from kernel_bench.refs import _kimi_attn_res_norm_reference, _attn_res_reference


def test_zero_score_weights_give_uniform_combine_then_output_norm():
    prefix = torch.tensor([[[1., 2., 3., 4.]]])
    stack = torch.tensor([[[[4., 3., 2., 1.], [2., 4., 1., 3.]]]])
    inputs = dict(prefix=prefix, stack=stack, score_gain=torch.ones(4),
                  score_proj=torch.zeros(1, 4), out_norm_w=torch.arange(1., 5.))
    mean = torch.cat([stack, prefix.unsqueeze(2)], 2).mean(2)
    expected = mean / torch.sqrt(mean.square().mean(-1, keepdim=True) + 1e-6)
    expected *= inputs['out_norm_w']
    torch.testing.assert_close(_kimi_attn_res_norm_reference(inputs), expected)


def test_learned_score_projection_changes_the_result():
    inputs = dict(prefix=torch.tensor([[[2., 0., 0., 0.]]]),
                  stack=torch.tensor([[[[0., 2., 0., 0.]]]]),
                  score_gain=torch.ones(4), score_proj=torch.tensor([[2., 0., 0., 0.]]),
                  out_norm_w=torch.ones(4))
    first = _kimi_attn_res_norm_reference(inputs)
    inputs['score_proj'] = torch.tensor([[0., 2., 0., 0.]])
    second = _kimi_attn_res_norm_reference(inputs)
    assert first[0, 0, 0] > first[0, 0, 1]
    assert second[0, 0, 1] > second[0, 0, 0]


def test_unfused_reference_reads_both_learned_weights():
    inputs = dict(prefix=torch.tensor([[[2., 0.]]]),
                  stack=torch.tensor([[[[0., 2.]]]]),
                  score_gain=torch.tensor([1., 0.]),
                  score_proj=torch.tensor([[1., 1.]]))
    first = _attn_res_reference(inputs)
    inputs['score_gain'] = torch.tensor([0., 1.])
    second = _attn_res_reference(inputs)
    assert first[0, 0, 0] > first[0, 0, 1]
    assert second[0, 0, 1] > second[0, 0, 0]
    inputs['score_proj'].zero_()
    torch.testing.assert_close(_attn_res_reference(inputs), torch.ones(1, 1, 2))


def test_tma_provider_declines_unaudited_geometry():
    from types import SimpleNamespace
    from kernel_bench.providers.norm_act import _sglang_kimi_attn_res_norm_applies
    inv = SimpleNamespace(dtype='bf16', dims=dict(d_model=7168, Sar=8, s_query=1))
    env = SimpleNamespace(sm='sm_103')
    assert _sglang_kimi_attn_res_norm_applies(inv, env)
    inv.dims['s_query'] = 2
    assert not _sglang_kimi_attn_res_norm_applies(inv, env)
    inv.dims['s_query'] = 1
    assert not _sglang_kimi_attn_res_norm_applies(inv, SimpleNamespace(sm='sm_120'))
