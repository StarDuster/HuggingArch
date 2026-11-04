"""The encode phase, pinned for every snapshot with an encoder, at TP=1 and
TP=8 and two reference media (`tools/corpus_pins/pins/encode_surface.py`).

An encoder's work is its own phase: its FLOPs, its activation traffic and the
rows it hands the language model grow from its media inputs alone, and none
of it is in a prefill or decode number. This golden is the one place those
numbers are written down, so a change that moves them goes red here.

Regenerating
------------
    ENCODE_SURFACE_GOLDEN_REGEN=1 .venv/bin/python -m pytest \\
        tests/backend/arch/eval/test_encode_surface_golden.py

Regeneration writes the file and fails; read the diff, rerun without the
flag, and say in the commit why every changed row moved. It is refused on CI.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import SNAPSHOT_DIR, any_snapshot
from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pins.encode_surface import ENCODE_SURFACE, measure_snapshot

pytestmark = pytest.mark.regression  # resolves every snapshot to find the encoders

GOLDEN_PATH = ENCODE_SURFACE.path
REGEN_ENV = "ENCODE_SURFACE_GOLDEN_REGEN"


def _measure_all() -> Dict[str, Any]:
    any_snapshot()
    loader = shared_loader()
    table: Dict[str, Any] = {}
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        row = measure_snapshot(CorpusModel(path.stem), loader)
        if row:
            table[path.stem] = row
    return table


def _diff(got: Any, want: Any, prefix: str) -> List[str]:
    if isinstance(got, dict) and isinstance(want, dict):
        out: List[str] = []
        for key in sorted(set(got) | set(want)):
            if key not in want:
                out.append(f"{prefix}.{key}: added {got[key]}")
            elif key not in got:
                out.append(f"{prefix}.{key}: removed (was {want[key]})")
            else:
                out += _diff(got[key], want[key], f"{prefix}.{key}")
        return out
    return [] if got == want else [f"{prefix}: {want} -> {got}"]


class TestEncodeSurfaceGolden(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.measured = _measure_all()

    def test_every_encoder_holds_its_pinned_surface(self) -> None:
        if os.environ.get(REGEN_ENV) == "1":
            if os.environ.get("CI"):
                self.fail(f"{REGEN_ENV} must not be used on CI")
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_bytes(ENCODE_SURFACE.render(self.measured))
            self.fail(
                f"regenerated {GOLDEN_PATH.name} ({len(self.measured)} "
                f"snapshots); rerun without {REGEN_ENV} and justify every "
                "changed row")
        self.assertTrue(
            GOLDEN_PATH.exists(),
            f"{GOLDEN_PATH} missing; regenerate locally with {REGEN_ENV}=1 "
            "(never on CI)")
        moved = _diff(self.measured, ENCODE_SURFACE.load(), "encode")
        self.assertEqual(moved, [], "\n".join(moved[:40]))


if __name__ == "__main__":
    unittest.main()
