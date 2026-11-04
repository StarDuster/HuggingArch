"""Every git-managed snapshot JSON must parse.

`load_snapshot` now raises on a snapshot file that exists but is corrupt
(rather than returning None, which callers read as "no snapshot" and
silently fall back). Gate the shipped store at the source so that raise can
only ever fire for a corrupt overlay/local file, never a checked-in one.
"""
import glob
import json
import pathlib

import pytest

_SNAP_DIR = pathlib.Path(__file__).resolve().parents[2] / "backend/arch/snapshots"
_SNAPSHOTS = sorted(glob.glob(str(_SNAP_DIR / "*.json")))


def test_there_are_snapshots_to_check():
    assert _SNAPSHOTS, "no snapshot JSONs found — has the snapshots dir moved?"


@pytest.mark.parametrize(
    "path", _SNAPSHOTS, ids=[pathlib.Path(p).name for p in _SNAPSHOTS]
)
def test_snapshot_json_parses(path):
    json.loads(pathlib.Path(path).read_text())
