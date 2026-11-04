"""One row per dtype; every other spelling table is a column of it.

The nine tables this replaced never contradicted each other on a width;
they contradicted each other on *coverage* — a spelling one knew and
another did not — and each answered its own miss its own way. These pin
the columns the consumers now read, with the values the old tables held.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dtypes as D


class TestTheTableIsWellFormed(unittest.TestCase):
    def test_every_token_is_unique_and_priced(self):
        tokens = [r.token for r in D.DTYPES.values() if r.token]
        self.assertEqual(len(tokens), len(set(tokens)))
        for token in tokens:
            self.assertGreater(D.dtype_bytes(token), 0)

    def test_every_spelling_belongs_to_one_row(self):
        seen = {}
        for name, row in D.DTYPES.items():
            for sp in row.spellings:
                self.assertNotIn(sp, seen, f"{sp} in both {seen.get(sp)} and {name}")
                seen[sp] = name
            self.assertIn(name, row.spellings)


class TestTheOldTablesAreColumns(unittest.TestCase):
    def test_torch_spellings(self):
        for torch, canon in {"float32": "F32", "float": "F32", "float16": "F16", "half": "F16",
                             "bfloat16": "BF16", "float64": "F64", "double": "F64",
                             "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2",
                             "int8": "I8", "uint8": "U8"}.items():
            self.assertEqual(D.from_torch_name(torch), canon, torch)
        self.assertIsNone(D.from_torch_name("float8_e4m3fnuz"))

    def test_ui_and_header_aliases_reach_the_corpus_token(self):
        for spelling, token in {"fp8": "fp8_e4m3", "f8_e4m3": "fp8_e4m3", "f8_e5m2": "fp8_e5m2",
                                "fp4": "fp4_e2m1", "nvfp4": "fp4_e2m1", "f16": "fp16", "f32": "fp32",
                                "f8_e8m0": "ue8m0", "f8_e8m0fnu": "ue8m0", "i8": "int8",
                                "i4": "int4", "bf16": "bf16", "mxfp4": "mxfp4"}.items():
            self.assertEqual(D.corpus_token(spelling), token, spelling)

    def test_the_corpus_vocabulary(self):
        self.assertEqual(D.CORPUS_TOKENS, {"bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2",
                                           "int8", "int4", "fp4_e2m1", "mxfp4", "mxfp8", "ue8m0",
                                           "int32"})
        self.assertEqual(D.UNQUANTIZED_TOKENS, {"bf16", "fp16", "fp32"})

    def test_display_and_canonical(self):
        self.assertEqual(D.display_name("F8_E4M3"), "FP8 (E4M3)")
        self.assertEqual(D.display_name("INT4"), "INT4")
        self.assertEqual(D.canonical("F8_E8M0FNU"), "F8_E8M0")
        self.assertEqual(D.canonical("float16"), "F16")

    def test_widths_and_peaks_did_not_move(self):
        for spelling, width in {"BF16": 2.0, "FP8": 1.0, "F8_E8M0": 1.0, "MXFP4": 0.5,
                                "INT4": 0.5, "I32": 4.0, "U8": 1.0, "BOOL": 1.0}.items():
            self.assertEqual(D.dtype_bytes(spelling), width, spelling)
        self.assertEqual(D.peak_kind("I32"), "fp32")
        with self.assertRaises(D.UnknownDtype):
            D.peak_kind("F8_E8M0")

    def test_a_dtype_with_no_corpus_token_is_refused_as_one(self):
        with self.assertRaises(D.UnknownDtype):
            D.corpus_token("U8")
        with self.assertRaises(D.UnknownDtype):
            D.corpus_token("nope")


if __name__ == "__main__":
    unittest.main()
