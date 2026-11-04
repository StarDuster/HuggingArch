"""The README's headline counts are derived from the repository, not typed.

Each count below is computed from the same files the product loads, then
looked for verbatim in both READMEs. A new model family, component, GPU SKU
or fusion plan that lands without touching the READMEs fails here rather
than leaving the front page stale.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.pins import readme_counts

READMES = (PROJECT_ROOT / "README.md", PROJECT_ROOT / "README.zh-CN.md")

# The counts and the phrases that state them are the pin's (P8,
# `tools/corpus_pins/pins/readme_counts.py`), shared with the admitter.
_counts = readme_counts.counts
PHRASES = readme_counts.PHRASES


class TestReadmeCounts(unittest.TestCase):
    def test_counts_match_repository(self) -> None:
        counts = _counts()
        for readme in READMES:
            text = readme.read_text(encoding="utf-8")
            for name, template in PHRASES[readme.name].items():
                pattern = template.format(n=counts[name])
                with self.subTest(readme=readme.name, count=name):
                    self.assertRegex(
                        text, pattern,
                        f"{readme.name} does not state {name}={counts[name]}")

    def test_no_stale_hosts_or_modules(self) -> None:
        for readme in READMES:
            text = readme.read_text(encoding="utf-8")
            with self.subTest(readme=readme.name):
                self.assertNotIn("huggingarch.fly.dev", text)
                for module in re.findall(r"[\w/]+\.py", text):
                    if module.startswith(("backend/", "kernel_bench/", "scripts/")):
                        self.assertTrue(
                            (PROJECT_ROOT / module).exists(),
                            f"{readme.name} references missing {module}")


if __name__ == "__main__":
    unittest.main()
