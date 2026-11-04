"""spliced_files returns new texts without writing the serving tree."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.promote_op_proposal import (
    OPS_REGISTRY_PATH, spliced_files,
)

_REG = '''\
register_op(OpSpec("zzz_splice_probe",
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y = x",
    description="splice probe",
    required_dims=(),
))
'''


def test_spliced_files_does_not_touch_disk():
    before = OPS_REGISTRY_PATH.read_text(encoding="utf-8")
    files = spliced_files({
        "section_comment": "",
        "symbols": [],
        "registration": _REG,
        "registry_entries": "",
        "tests_src": "",
    })
    assert OPS_REGISTRY_PATH.read_text(encoding="utf-8") == before
    rel = str(OPS_REGISTRY_PATH.relative_to(PROJECT_ROOT))
    assert rel in files
    assert "zzz_splice_probe" in files[rel]
    assert "zzz_splice_probe" not in before
