import pytest
import sympy as sp


def test_conv1d_params_project_from_declared_tensors(monkeypatch):
    from backend.arch.eval.op_ref import OpRef
    from backend.arch.ops import get_op

    op = get_op("conv1d")
    depthwise = {
        "C_in": 128, "C_out": 128, "groups": 128,
        "kernel_size": 4, "bias": 0,
    }
    grouped = {
        "C_in": 16, "C_out": 32, "groups": 4,
        "kernel_size": 3, "bias": 1,
    }

    monkeypatch.setattr(
        op, "params_fn",
        lambda dims: (_ for _ in ()).throw(AssertionError("params_fn read")))
    ref = OpRef("conv1d", op, False)
    unused_resolver = lambda expr, dims: None

    assert ref.params(depthwise, unused_resolver) == 128 * 4
    assert ref.params(grouped, unused_resolver) == 32 * 4 * 3 + 32
    assert op.params_expr_override == ""


def test_depthwise_symbolic_params_collapse_through_the_declared_group_axis():
    from backend.arch.eval.storage import declared_elements
    from backend.arch.ops import get_op

    channels, width = sp.symbols("channels width", positive=True, integer=True)
    got = declared_elements(get_op("conv1d"), {
        "C_in": channels, "C_out": channels, "groups": channels,
        "kernel_size": width, "bias": 0,
    })

    assert sp.simplify(sp.sympify(got) - channels * width) == 0


def test_conv1d_registry_display_is_derived_from_the_group_declaration():
    from backend.arch.ops import get_op

    op = get_op("conv1d")
    assert op.flops_expr_override == ""
    assert op.params_expr_override == ""
    assert "groups" in op.flops_expr
    assert "groups" in op.params_expr


def test_conv1d_declares_group_division_exact():
    from backend.arch.eval.storage import Undeclarable, declared_elements
    from backend.arch.ops import get_op

    with pytest.raises(Undeclarable, match="declared exact"):
        declared_elements(get_op("conv1d"), {
            "C_in": 10, "C_out": 16, "groups": 4,
            "kernel_size": 3, "bias": 0,
        })
