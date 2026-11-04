"""Freeze which architecture every snapshot currently resolves to.

This is the migration safety net for the spec-matching rewrite: before the matcher changes from
model_type scoring to model_id lookup, pin — for every checkpoint snapshot —
the STRUCTURE of the spec `SpecLoader().match(config)` returns today.

The key is a `resolved_fingerprint`: a canonical hash of the resolved
`ArchSpec` with its identity/location fields (`name`, `source_path`) EXCLUDED.
The rewrite gives each model_id its own file, so both the spec name and its
path change by design; the fingerprint is the structure that must stay
identical — the same checkpoint must keep resolving to the same architecture.
A drift here during migration is a real regression; a rename is not, and must
not show up as drift.

Behaviour-neutral: reads snapshots and today's matcher, writes nothing to the
product. On-disk custom specs under ~/.huggingarch are deliberately ignored so
the fingerprint is a pure function of the repo, not the machine.
"""
from __future__ import annotations

import os
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.pins.spec_match import SPEC_MATCH

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
GOLDEN = SPEC_MATCH.path

# The fingerprint and what it excludes live on the pin
# (`tools/corpus_pins/pins/spec_match.py`), shared with the admitter.


def _current() -> dict[str, str | None]:
    return SPEC_MATCH.measure_all()


class TestSpecMatchGolden(unittest.TestCase):
    @pytest.mark.regression  # every snapshot
    def test_every_snapshot_resolves_to_its_frozen_architecture(self):
        snapshots = sorted(SNAPSHOTS.glob("*.json"))
        # Snapshots are committed files, not an optional submodule — an empty
        # dir means the net silently disappeared, not a legitimate skip.
        self.assertTrue(snapshots, f"no snapshots under {SNAPSHOTS}")

        current = _current()
        regen = os.environ.get("SPEC_MATCH_GOLDEN_REGEN") == "1"

        if regen:
            # Explicit, local-only escape hatch for an intended matcher change.
            # Refuse on CI so a suite-wide REGEN can never launder a regression
            # into a green baseline.
            if os.environ.get("CI"):
                self.fail("SPEC_MATCH_GOLDEN_REGEN must not be used on CI")
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_bytes(SPEC_MATCH.render(current))
            self.fail(
                f"regenerated {GOLDEN} ({len(current)} snapshots); rerun without "
                "SPEC_MATCH_GOLDEN_REGEN and justify each changed row"
            )

        self.assertTrue(
            GOLDEN.exists(),
            f"{GOLDEN} missing — regenerate locally with "
            "SPEC_MATCH_GOLDEN_REGEN=1 (never on CI)",
        )
        golden = SPEC_MATCH.load()
        self.assertEqual(
            set(current), set(golden),
            "snapshot set changed — regenerate the baseline intentionally",
        )
        drifted = {
            stem: (golden[stem], current[stem])
            for stem in current
            if current[stem] != golden[stem]
        }
        self.assertEqual(
            drifted, {},
            "spec-match architecture drifted for: "
            + ", ".join(sorted(drifted))
            + " (want→got fingerprints). If a matcher change intends this, "
            "regen with SPEC_MATCH_GOLDEN_REGEN=1 and justify per checkpoint.",
        )


if __name__ == "__main__":
    unittest.main()
