"""The Finding protocol ratchet.

Validator findings are `Finding` objects, not convention-shaped dicts. This
test is what keeps it that way: a bare `{"kind": ...}` literal anywhere in
the validate/structural producers means someone bypassed the type, and the
kind/detail/extra discipline is back to being upheld by nothing.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

_SCAN_DIRS = (
    PROJECT_ROOT / "backend" / "arch" / "validate",
    PROJECT_ROOT / "backend" / "arch" / "structural",
)
# The type's own home is exempt — it legitimately names the key.
_EXEMPT = {"_finding.py"}

#: A `"kind":` key in a dict literal. `Finding(kind=...)` does not match.
_DICT_KIND = re.compile(r'"kind"\s*:')


class TestFindingProtocol(unittest.TestCase):
    def test_no_bare_kind_dicts_in_producers(self):
        offenders = []
        for d in _SCAN_DIRS:
            for path in sorted(d.glob("*.py")):
                if path.name in _EXEMPT:
                    continue
                for lineno, line in enumerate(
                        path.read_text().splitlines(), 1):
                    if _DICT_KIND.search(line):
                        offenders.append(f"{path.name}:{lineno}")
        self.assertEqual(
            offenders, [],
            "bare 'kind' dict literals in validator producers — construct "
            f"a Finding instead: {offenders}",
        )

    def test_as_dict_drops_unset_fields(self):
        from backend.arch.validate._finding import Finding
        f = Finding(kind="k", detail="d")
        self.assertEqual(f.as_dict(), {"kind": "k", "detail": "d"})

    def test_as_dict_merges_extra_and_context(self):
        from backend.arch.validate._finding import Finding
        f = Finding(kind="k", detail="d", template="t",
                    extra={"custom": 1})
        self.assertEqual(
            f.as_dict(),
            {"kind": "k", "detail": "d", "template": "t", "custom": 1},
        )


if __name__ == "__main__":
    unittest.main()
