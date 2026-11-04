"""Mechanical auto-merge gate: backend/op_workers/review/_rules.py.

The LLM reviewer's verdict is necessary but not sufficient; these rules
are the deterministic envelope that decides whether `auto_merge` can
actually promote a proposal. Each rejection branch gets a real assertion
on the reason text so a silent rule drift is caught.
"""
from __future__ import annotations

import pytest

from backend.op_workers.review._rules import check_auto_merge_eligibility

_VALID_REGISTRATION = '''\
register_op(OpSpec("rope_partial",
    flops_fn=lambda d: d["d_model"] * d["d_seq"],
    params_fn=_zero,
    shape_in=("d_seq", "d_model"),
    shape_out=("d_seq", "d_model"),
    shape_kind="preserve",
    layout_family="linear",
    formula="y = rotate_half(x)",
    description="Partial rotary embedding.",
    required_dims=("d_seq",),
))'''


def _write_op_dir(tmp_path, registrations=_VALID_REGISTRATION, new_symbols=()):
    """Write a minimal proposal dir with a loadable patch.py."""
    patch = tmp_path / "patch.py"
    patch.write_text(
        f"NEW_DIM_SYMBOLS = {list(new_symbols)!r}\n"
        f"OP_REGISTRATIONS = {registrations!r}\n",
        encoding="utf-8",
    )
    return tmp_path


def _registration(**overrides) -> str:
    """Build a register_op(OpSpec(...)) source string with kwargs swapped."""
    kwargs = {
        "flops_fn": 'lambda d: d["d_model"] * d["d_seq"]',
        "params_fn": "_zero",
        "shape_in": '("d_seq", "d_model")',
        "shape_out": '("d_seq", "d_model")',
        "shape_kind": '"preserve"',
        "layout_family": '"linear"',
        "formula": '"y = f(x)"',
        "description": '"test op"',
        "required_dims": '("d_seq",)',
    }
    kwargs.update(overrides)
    body = ",\n    ".join(f"{k}={v}" for k, v in kwargs.items())
    return f'register_op(OpSpec("rope_partial",\n    {body}\n))'


def test_minimal_valid_proposal_passes(tmp_path):
    op_dir = _write_op_dir(tmp_path)

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is True
    assert reasons == []


@pytest.mark.parametrize("shape_kind", ['"preserve"', '"axis_replace"'])
def test_string_literal_shape_kinds_pass(tmp_path, shape_kind):
    op_dir = _write_op_dir(tmp_path, _registration(shape_kind=shape_kind))

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is True
    assert reasons == []


@pytest.mark.parametrize("shape_kind", ["ShapeKind.PRESERVE", "ShapeKind.AXIS_REPLACE"])
def test_shapekind_enum_form_passes(tmp_path, shape_kind):
    op_dir = _write_op_dir(tmp_path, _registration(shape_kind=shape_kind))

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is True
    assert reasons == []


def test_accepts_str_op_dir_path(tmp_path):
    _write_op_dir(tmp_path)

    eligible, reasons = check_auto_merge_eligibility(str(tmp_path))

    assert eligible is True
    assert reasons == []


def test_missing_patch_py_rejected(tmp_path):
    eligible, reasons = check_auto_merge_eligibility(tmp_path / "nope")

    assert eligible is False
    assert len(reasons) == 1
    assert reasons[0].startswith("missing ")
    assert reasons[0].endswith("patch.py")


def test_patch_py_that_fails_to_load_rejected(tmp_path):
    (tmp_path / "patch.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )

    eligible, reasons = check_auto_merge_eligibility(tmp_path)

    assert eligible is False
    assert len(reasons) == 1
    assert "failed to load patch.py" in reasons[0]
    assert "boom" in reasons[0]


_VALID_FRAGMENT = (
    "d_expert:\n"
    "  kind: split_width\n"
    "  shardable_by: [moe_expert]\n"
    "  group: backbone\n"
    "  meaning: expert FFN width\n"
)


def test_new_dim_symbols_allowed_when_fragment_declares_them(tmp_path):
    op_dir = _write_op_dir(tmp_path, new_symbols=["d_expert"])
    (op_dir / "dims.yaml").write_text(_VALID_FRAGMENT, encoding="utf-8")

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is True
    assert reasons == []


def test_new_dim_symbols_rejected_when_fragment_cannot_splice(tmp_path):
    """group must be a dims.yaml group slug, not a world slug."""
    op_dir = _write_op_dir(tmp_path, new_symbols=["d_expert"])
    (op_dir / "dims.yaml").write_text(
        "d_expert:\n  kind: split_width\n  group: moe_expert\n"
        "  meaning: expert FFN width\n",
        encoding="utf-8",
    )

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert any("does not splice" in r for r in reasons)


def test_new_dim_symbols_rejected_when_list_empty_but_fragment_present(tmp_path):
    op_dir = _write_op_dir(tmp_path, new_symbols=[])
    (op_dir / "dims.yaml").write_text(_VALID_FRAGMENT, encoding="utf-8")

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert any("NEW_DIM_SYMBOLS" in r and "fragment keys" in r for r in reasons)


def test_new_dim_symbols_rejected_on_unknown_fragment_field(tmp_path):
    op_dir = _write_op_dir(tmp_path, new_symbols=["d_expert"])
    (op_dir / "dims.yaml").write_text(
        _VALID_FRAGMENT + "  bogus_field: 3\n", encoding="utf-8"
    )

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert any("does not splice" in r for r in reasons)


def test_new_dim_symbols_rejected_without_fragment(tmp_path):
    op_dir = _write_op_dir(tmp_path, new_symbols=["d_expert", "d_topk"])

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert len(reasons) == 1
    assert "NEW_DIM_SYMBOLS" in reasons[0]
    assert "dims.yaml fragment" in reasons[0]


def test_empty_op_registrations_rejected(tmp_path):
    _write_op_dir(tmp_path, registrations="")

    eligible, reasons = check_auto_merge_eligibility(tmp_path)

    assert eligible is False
    assert reasons == ["OP_REGISTRATIONS is empty"]


def test_missing_op_registrations_attribute_rejected(tmp_path):
    (tmp_path / "patch.py").write_text("NEW_DIM_SYMBOLS = []\n", encoding="utf-8")

    eligible, reasons = check_auto_merge_eligibility(tmp_path)

    assert eligible is False
    assert reasons == ["OP_REGISTRATIONS is empty"]


def test_unparseable_op_registrations_rejected(tmp_path):
    op_dir = _write_op_dir(tmp_path, registrations="register_op(OpSpec(  # unbalanced")

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert len(reasons) == 1
    assert reasons[0].startswith("OP_REGISTRATIONS does not parse:")


def test_multiple_register_op_calls_rejected(tmp_path):
    two_ops = _registration() + "\n" + _registration().replace(
        '"rope_partial"', '"rope_full"'
    )
    op_dir = _write_op_dir(tmp_path, two_ops)

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert (
        "OP_REGISTRATIONS contains 2 register_op call(s); "
        "auto-merge requires exactly 1"
    ) in reasons


def test_zero_register_op_calls_rejected(tmp_path):
    op_dir = _write_op_dir(tmp_path, "# nothing registered here\n")

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert (
        "OP_REGISTRATIONS contains 0 register_op call(s); "
        "auto-merge requires exactly 1"
    ) in reasons
    assert "no OpSpec(...) call found in OP_REGISTRATIONS" in reasons


def test_register_op_without_opspec_rejected(tmp_path):
    op_dir = _write_op_dir(tmp_path, 'register_op({"name": "dict_style"})')

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert reasons == ["no OpSpec(...) call found in OP_REGISTRATIONS"]


def test_secondary_outputs_rejected(tmp_path):
    op_dir = _write_op_dir(
        tmp_path,
        _registration(secondary_outputs='(("d_seq", "d_model"),)'),
    )

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert "OpSpec has secondary_outputs (multi-output op)" in reasons


def test_omitted_shape_kind_rejected(tmp_path):
    kwargs = {
        "flops_fn": 'lambda d: d["d_model"]',
        "params_fn": "_zero",
        "shape_in": '("d_model",)',
        "shape_out": '("d_model",)',
        "formula": '"y = f(x)"',
        "description": '"test op"',
        "required_dims": '("d_model",)',
        "layout_family": '"linear"',
    }
    body = ",\n    ".join(f"{k}={v}" for k, v in kwargs.items())
    op_dir = _write_op_dir(tmp_path, f'register_op(OpSpec("rope_partial",\n    {body}\n))')

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert len(reasons) == 1
    assert reasons[0].startswith("OpSpec omits shape_kind")
    assert "preserve or axis_replace" in reasons[0]


@pytest.mark.parametrize(
    "shape_kind",
    ['"explicit"', '"contract"', '"source"', "ShapeKind.EXPLICIT", "ShapeKind.CONTRACT"],
)
def test_disallowed_shape_kinds_rejected(tmp_path, shape_kind):
    op_dir = _write_op_dir(tmp_path, _registration(shape_kind=shape_kind))

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert len(reasons) == 1
    assert "auto-merge requires one of ['axis_replace', 'preserve']" in reasons[0]


def test_unrecognizable_shape_kind_form_rejected(tmp_path):
    op_dir = _write_op_dir(tmp_path, _registration(shape_kind="SHAPE_KIND_PRESERVE"))

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert reasons == [
        "OpSpec.shape_kind is not a recognizable string or "
        "ShapeKind.<NAME> form"
    ]


def test_all_disqualifiers_reported_in_one_pass(tmp_path):
    """Reasons accumulate: bad shape_kind + secondary_outputs."""
    op_dir = _write_op_dir(
        tmp_path,
        _registration(
            shape_kind='"explicit"',
            secondary_outputs='(("d_model",),)',
        ),
    )

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert len(reasons) == 2
    assert any("secondary_outputs" in r for r in reasons)
    assert any("shape_kind='explicit'" in r for r in reasons)


def test_a_proposal_without_a_layout_family_is_refused_with_the_fix(tmp_path):
    """No hand-written layout handler exists for a proposed op: its declared
    family is what every multi-GPU analysis places it by."""
    registration = _registration()
    registration = registration.replace(',\n    layout_family="linear"', "")
    op_dir = _write_op_dir(tmp_path, registration)

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert any("declares no layout family" in r and 'layout_family="linear"' in r
               for r in reasons), reasons


def test_a_recurrent_proposal_declares_its_axis_use(tmp_path):
    op_dir = _write_op_dir(tmp_path, _registration(layout_family='"recurrent"'))

    eligible, reasons = check_auto_merge_eligibility(op_dir)

    assert eligible is False
    assert any("declares axis_use" in r for r in reasons), reasons
