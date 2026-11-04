"""The checkpoint's native dtype is what the config says, or nothing.

`native_dtype` used to answer None for a `torch_dtype` it did not
recognise, and the caller's `native or "BF16"` then priced the checkpoint
at two bytes — the mechanism behind gemma-2's factor-of-two error. A key
that is present and unknown is an error; a key that is absent is the
honest None the quantisation declaration chain handles.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.dtypes import UnknownDtype
from backend.arch.dtypes import native_dtype


class TestNativeDtypeIsDeclaredOrAbsent(unittest.TestCase):
    def test_a_known_name_is_translated(self):
        self.assertEqual(native_dtype({"torch_dtype": "bfloat16"}), "BF16")

    def test_an_absent_key_is_none(self):
        self.assertIsNone(native_dtype({}))
        self.assertIsNone(native_dtype(None))

    def test_a_present_unknown_name_is_refused(self):
        with self.assertRaisesRegex(UnknownDtype, "float8_e4m3fnuz"):
            native_dtype({"torch_dtype": "float8_e4m3fnuz"})


if __name__ == "__main__":
    unittest.main()
