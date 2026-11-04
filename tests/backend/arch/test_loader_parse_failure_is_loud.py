"""A spec file that does not parse cannot make `match()` answer None.

`_refuse_if_load_failed` names a failed file by the `match.model_id` it
still declares even when the spec did not build; asking for that id gets
the load error, not a miss, so a broken spec is never mistaken for a
missing one and quietly sent to spec-gen (contract D).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader, SpecLoadError


class TestUnparsableSpecIsNamed(unittest.TestCase):
    def test_match_raises_when_a_file_never_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for sub in ("components/library", "components/adapters",
                        "components/drafts", "models"):
                (base / sub).mkdir(parents=True)
            (base / "models" / "broken.yaml").write_text(
                "match: { model_type: [broken], model_id: [broken] }\nmodel:\n  - { name: x, op: rmsnorm, dim: d_model ?? 1 }\n")
            loader = SpecLoader(base_dir=str(base))
            with self.assertRaisesRegex(SpecLoadError, "broken.yaml"):
                loader.match({"model_type": "broken", "hidden_size": 8},
                             model_id="broken")


if __name__ == "__main__":
    unittest.main()
