"""HA_PENDING_OP_PATCH registers a proposal op in THIS process only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.ops_registry import OP_REGISTRY, apply_pending_op_patch


_PATCH = '''\
NEW_DIM_SYMBOLS = []
OP_REGISTRATIONS = """
register_op(OpSpec("zzz_pending_probe",
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y = x",
    description="test probe",
    required_dims=(),
))
"""
'''


def test_apply_pending_op_patch_registers_and_is_reversible(tmp_path):
    patch = tmp_path / "patch.py"
    patch.write_text(_PATCH, encoding="utf-8")
    assert "zzz_pending_probe" not in OP_REGISTRY
    added = apply_pending_op_patch(str(patch))
    assert added == ("zzz_pending_probe",)
    assert "zzz_pending_probe" in OP_REGISTRY
    OP_REGISTRY.pop("zzz_pending_probe")
    assert "zzz_pending_probe" not in OP_REGISTRY


def test_apply_pending_op_patch_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        apply_pending_op_patch(str(tmp_path / "nope.py"))


_FRAGMENT = """\
d_zzz_probe:
  kind: split_width
  shardable_by: [moe_expert]
  group: backbone
  meaning: pending-op probe width
"""


def test_apply_pending_op_patch_rejects_non_mapping_fragment(tmp_path):
    from backend.arch import dims

    patch = tmp_path / "patch.py"
    patch.write_text(_PATCH, encoding="utf-8")
    (tmp_path / "dims.yaml").write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        apply_pending_op_patch(str(patch))
    assert "zzz_pending_probe" not in OP_REGISTRY
    assert not dims.is_registered("d_zzz_probe")


def test_apply_pending_op_patch_overlays_dims_fragment(tmp_path):
    from backend.arch import dims, ops_base

    patch = tmp_path / "patch.py"
    patch.write_text(
        _PATCH.replace("NEW_DIM_SYMBOLS = []", 'NEW_DIM_SYMBOLS = ["d_zzz_probe"]'),
        encoding="utf-8",
    )
    (tmp_path / "dims.yaml").write_text(_FRAGMENT, encoding="utf-8")
    assert not dims.is_registered("d_zzz_probe")
    try:
        apply_pending_op_patch(str(patch))
        assert dims.is_registered("d_zzz_probe")
        assert dims.get("d_zzz_probe").kind == "split_width"
        assert "d_zzz_probe" in ops_base._CANONICAL_DIMS
    finally:
        OP_REGISTRY.pop("zzz_pending_probe", None)
        ops_base._CANONICAL_DIMS.pop("d_zzz_probe", None)
        dims.overlay_pending(None)
        assert not dims.is_registered("d_zzz_probe")
