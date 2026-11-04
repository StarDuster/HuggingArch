"""Open PR splices a proposal package into canonical files at PR time."""
from __future__ import annotations

from backend.arch.promote_op_proposal import OPS_REGISTRY_PATH, REPO_ROOT
from backend.service.github import _OPS_REGISTRY_REL, _bundle_files


_PATCH = '''\
NEW_DIM_SYMBOLS = ["d_zzz_probe"]
OP_REGISTRATIONS = """
register_op(OpSpec("zzz_contrib_probe",
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y = x",
    description="contribute probe",
    required_dims=(),
))
"""
'''

_FRAGMENT = """\
d_zzz_probe:
  kind: split_width
  shardable_by: [moe_expert]
  group: backbone
  meaning: contribute-splice probe width
"""


def test_bundle_splices_ops_registry_and_keeps_the_package():
    files = _bundle_files(
        "org/M",
        "name: m\n",
        [
            {"path": "backend/arch/ops_proposals/zzz_contrib_probe/patch.py",
             "body": _PATCH},
            {"path": "backend/arch/ops_proposals/zzz_contrib_probe/dims.yaml",
             "body": _FRAGMENT},
            {"path": "backend/arch/components/drafts/foo.yaml",
             "body": "op: zzz_contrib_probe\n"},
        ],
        None,
    )
    assert str(OPS_REGISTRY_PATH.relative_to(REPO_ROOT)) == _OPS_REGISTRY_REL
    assert "zzz_contrib_probe" in files[_OPS_REGISTRY_REL]
    assert "d_zzz_probe" in files["backend/arch/ops_base.py"]
    assert "d_zzz_probe" in files["backend/arch/dims.yaml"]
    assert files["backend/arch/ops_proposals/zzz_contrib_probe/patch.py"] == _PATCH
    assert files["backend/arch/ops_proposals/zzz_contrib_probe/dims.yaml"] == _FRAGMENT
    assert files["backend/arch/components/drafts/foo.yaml"] == "op: zzz_contrib_probe\n"
    assert "backend/arch/models/org__M.yaml" in files
