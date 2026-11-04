"""Per-rank encoder weight bytes, pinned for every snapshot with a vision or
audio encoder, at TP=1 and TP=8.

No other golden holds these numbers. `weight_bytes_per_rank` in the SOL and
capacity goldens leaves the auxiliary stacks out, and the storage surface is
the TP=1 logical checkpoint. So when the vision towers were aligned with
vLLM's TP split (Qwen3.5 merger, MiMo-V2.5 merger, Kimi-K2.5 attention,
Gemma-4 patch_dense), the per-rank figures quoted in those commits were
checked by nothing, and a later change could move them back without any test
going red.

What is pinned, per (snapshot x TP x encoder stack): the stack's resident
bytes on one rank (`resident_weights(...).by_stack`, i.e. parameters, quant
metadata and buffers), and each weight node's resident bytes on one rank
(`physical_weight_leaves`, summed over the node's layer instances). Both come
from the persistent-artifact join of the encoder's own encode walk, and are
the same whatever media that walk runs on.

Two structural facts are checked on the measured numbers, not the golden:
splitting over TP never makes a rank hold more of a stack or a node than TP=1
does, the towers the framework runs tensor-parallel hold strictly less at TP=8
(the ones it replicates the same), and every stack that shrinks at TP=8 is one
the framework splits.

Regenerating
------------
    ENCODER_WEIGHT_GOLDEN_REGEN=1 .venv/bin/python -m pytest \\
        tests/backend/arch/layout/test_encoder_weight_per_rank_golden.py

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
from tools.corpus_pins.checks.encoder_tp import grew
from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.decisions.tp_split import encoder_entries
from tools.corpus_pins.pins.encoder_weight import ENCODER_WEIGHT, measure_snapshot

pytestmark = pytest.mark.regression  # resolves every snapshot to find the encoders

GOLDEN_PATH = ENCODER_WEIGHT.path
REGEN_ENV = "ENCODER_WEIGHT_GOLDEN_REGEN"

#: Encoder stacks the framework runs tensor-parallel at TP=8 (vLLM's default
#: `mm_encoder_tp_mode="weights"`), so a TP=8 rank must hold strictly less of
#: them than TP=1, and encoder stacks it replicates, which hold the same
#: bytes at TP=8 as at TP=1. Keyed (snapshot stem, stack name); the value
#: names the framework's classes (class names, not line numbers, so the
#: citation survives an edit upstream). Both are read from the TP-split
#: decisions (`tests/golden/pins/tp_split_decisions.json`). Encoders absent
#: from them are held only to "no more than TP=1": MiMo-V2.5's audio encoder
#: and Gemma-4's audio embedder are plain or replicated linears in vLLM.
SPLIT_TOWERS = encoder_entries("framework_split")
REPLICATED_TOWERS = encoder_entries("framework_replicated")

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


class TestEncoderWeightPerRankGolden(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.measured = _measure_all()

    def test_every_encoder_holds_its_pinned_bytes_per_rank(self) -> None:
        if os.environ.get(REGEN_ENV) == "1":
            # Local-only: a suite-wide regen on CI would launder a regression
            # into a green baseline.
            if os.environ.get("CI"):
                self.fail(f"{REGEN_ENV} must not be used on CI")
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_bytes(ENCODER_WEIGHT.render(self.measured))
            self.fail(
                f"regenerated {GOLDEN_PATH.name} ({len(self.measured)} "
                f"snapshots); rerun without {REGEN_ENV} and justify every "
                "changed row")
        self.assertTrue(
            GOLDEN_PATH.exists(),
            f"{GOLDEN_PATH} missing; regenerate locally with {REGEN_ENV}=1 "
            "(never on CI)")
        golden = ENCODER_WEIGHT.load()
        moved = _diff(self.measured, golden, "encoder")
        self.assertEqual(
            moved, [],
            "per-rank encoder weight bytes moved (want -> got):\n  "
            + "\n  ".join(moved))

    def test_the_media_moves_no_weight_byte(self) -> None:
        """The bytes come from each encoder's encode walk; what that walk
        holds is the same on the encode surface's reference media as on
        none (`encode_surface.REFERENCES`)."""
        from backend.arch.eval.program import Program
        from tools.corpus_pins.pins.encode_surface import (
            REFERENCES, reference_media)
        from tools.corpus_pins.pins.encoder_weight import (
            TP_DEGREES, encoder_weight_leaves)
        loader = shared_loader()
        for stem, row in sorted(self.measured.items()):
            resolved = CorpusModel(stem).resolved(loader)
            program = Program.of(resolved)
            for reference in REFERENCES:
                media = reference_media(program, reference)
                for degree in TP_DEGREES:
                    for stack, pinned in row[str(degree)].items():
                        by_stack, _ = encoder_weight_leaves(
                            resolved, degree, stack, media)
                        with self.subTest(snapshot=stem, stack=stack,
                                          tp=degree, media=reference):
                            self.assertEqual(int(by_stack.get(stack, 0)),
                                             pinned["bytes_per_rank"])

    def test_tp8_never_holds_more_than_tp1(self) -> None:
        found: List[str] = []
        for stem, row in self.measured.items():
            found += grew(stem, row)
        self.assertEqual(found, [], "TP=8 holds more per rank than TP=1")

    def test_towers_the_framework_splits_hold_strictly_less_at_tp8(self) -> None:
        for (stem, stack), source in sorted(SPLIT_TOWERS.items()):
            with self.subTest(snapshot=stem, stack=stack):
                row = self.measured.get(stem)
                self.assertIsNotNone(
                    row, f"{stem} no longer has an encoder stack ({source})")
                self.assertIn(stack, row["1"], f"{stem} lost stack {stack!r}")
                self.assertLess(
                    row["8"][stack]["bytes_per_rank"],
                    row["1"][stack]["bytes_per_rank"],
                    f"the framework splits this tower over TP ({source})")
        for (stem, stack), source in sorted(REPLICATED_TOWERS.items()):
            with self.subTest(snapshot=stem, stack=stack):
                row = self.measured.get(stem)
                self.assertIsNotNone(
                    row, f"{stem} no longer has an encoder stack ({source})")
                self.assertEqual(
                    row["8"][stack]["bytes_per_rank"],
                    row["1"][stack]["bytes_per_rank"],
                    f"the framework replicates this tower over TP ({source})")

    def test_every_stack_that_shrinks_at_tp8_is_accounted_for(self) -> None:
        shrinks = {
            (stem, stack)
            for stem, row in self.measured.items()
            for stack, one in row["1"].items()
            if row["8"][stack]["bytes_per_rank"] < one["bytes_per_rank"]
        }
        self.assertEqual(
            sorted(shrinks - set(SPLIT_TOWERS)), [],
            "encoder stacks split at TP=8 with no framework source in "
            "SPLIT_TOWERS")


if __name__ == "__main__":
    unittest.main()
