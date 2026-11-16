from types import SimpleNamespace


def test_current_source_router_registers_and_declines_unaudited_geometries():
    # Import itself catches conflicting Provider scope declarations.
    from kernel_bench.providers.moe_gate import _sglang_current_moe_gate_applies
    def invocation(**changes):
        dims = dict(n_experts=896, n_experts_active=16, n_expert_groups=1)
        dims.update(changes)
        return SimpleNamespace(signature='moe_gate', dtype='bf16',
                               dims=dims, kernel_params={})
    assert _sglang_current_moe_gate_applies(invocation(), None)
    assert not _sglang_current_moe_gate_applies(invocation(n_expert_groups=8), None)
    assert not _sglang_current_moe_gate_applies(invocation(n_experts=512), None)
