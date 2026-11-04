"""A stored tensor the spec names is a tensor the checkpoint names.

A parameter node named after its module (`pos_emb`) against the header's
`pos_emb.weight` stores the right bytes under the wrong name. The
checkpoint-name check compares the emitted name itself, and names the
header's spelling of the same module.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

MODEL = "moonshotai/Kimi-K2.5"
POS_EMB = "vision_tower.patch_embed.pos_emb"


class TestCheckpointNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.snapshots import shared_program
        from backend.arch.eval.transforms import Deployment
        from backend.arch.validate._matching import _get_safetensors_data
        # The vision tower's tensors are the checkpoint's, not the language
        # model step's: the whole-checkpoint walk names them.
        cls.analysis = Deployment(shared_program(MODEL)).at(
            batch=1, input_tokens=8).checkpoint()
        cls.header = _get_safetensors_data(MODEL)["header"]

    def _check(self, header):
        from backend.arch.validate._checkpoint_names import run_checkpoint_name_check
        return run_checkpoint_name_check(self.analysis, header)

    def test_the_spec_names_what_the_header_holds(self):
        self.assertTrue(self.header.select(POS_EMB + ".weight"))
        self.assertEqual(self._check(self.header), [])

    def test_a_name_off_by_its_tensor_suffix_is_refused_with_the_header_spelling(self):
        # The header holds the module's tensor under the bare name; the spec
        # (`pos_emb.weight`) no longer equals it.
        from backend.hf.header import SafetensorsHeader
        header = SafetensorsHeader.from_tensors({
            (POS_EMB if name == POS_EMB + ".weight" else name):
                (group.dtype, group.shape)
            for name, group in self.header.names()})
        findings = self._check(header)
        self.assertEqual([f.kind for f in findings], ["checkpoint_name_absent"])
        detail = findings[0].detail
        self.assertIn(f"`{POS_EMB}.weight`", detail)
        self.assertIn(f"['{POS_EMB}']", detail)


if __name__ == "__main__":
    unittest.main()
