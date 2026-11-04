"""The op-promotion tool checks proposals against the dim registry's own
vocabulary and points at files that exist.

Both legs were dead: the validator read a `world:` key that left dims.yaml
when the registry moved to `shardable_by` (so every non-whole proposal was
refused for a field nobody could write), and the test-insertion path named a
test file that had been moved. A tool whose gate cannot pass and whose
writer cannot find its target is not a tool.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dims
from backend.arch import promote_op_proposal as promote

REGISTRY_TEXT = (PROJECT_ROOT / "backend" / "arch" / "dims.yaml").read_text()


class TestProposalsSpeakTheRegistrysVocabulary(unittest.TestCase):
    def test_a_world_the_registry_declares_is_accepted(self):
        world = sorted(dims.worlds())[0]
        merged = promote._insert_registry(REGISTRY_TEXT, (
            "d_probe_axis:\n"
            "  kind: split_width\n"
            f"  shardable_by: [{world}]\n"
            "  group: backbone\n"
            "  meaning: a probe\n"))
        self.assertIn("d_probe_axis", merged)

    def test_an_undeclared_world_is_refused_by_name(self):
        with self.assertRaisesRegex(RuntimeError, "shardable by \\['nowhere'\\]"):
            promote._insert_registry(REGISTRY_TEXT, (
                "d_probe_axis:\n"
                "  kind: split_width\n"
                "  shardable_by: [nowhere]\n"
                "  group: backbone\n"
                "  meaning: a probe\n"))

    def test_whole_and_shardable_by_cannot_both_be_said(self):
        world = sorted(dims.worlds())[0]
        with self.assertRaisesRegex(RuntimeError, "whole"):
            promote._insert_registry(REGISTRY_TEXT, (
                "d_probe_axis:\n"
                "  kind: whole\n"
                f"  shardable_by: [{world}]\n"
                "  group: backbone\n"
                "  meaning: a probe\n"))


class TestTheToolPointsAtRealFiles(unittest.TestCase):
    def test_test_targets_exist(self):
        self.assertTrue(promote.TESTS_PATH.is_file(), promote.TESTS_PATH)
        for target in promote.DEFAULT_PYTEST_TARGETS:
            self.assertTrue((PROJECT_ROOT / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
