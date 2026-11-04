"""One implementation, one identity — across both recorded spellings.

A row records `Provider.impl_id`: the registered kernel name, then the
parameters that resolved for that invocation. The separator between the two
was introduced after rows had already been published without it, so the same
implementation appears in the stored corpus as `flashmla_decode_b64` and as
`flashmla_decode@_b64`. `catalog_kernel` is the one place that turns a
recorded name back into the registered one; anything that instead cuts the
name at a character reads every row of the older spelling as belonging to no
kernel at all — which is how three by-implementation tools came to report
kernels with tens of thousands of rows as never run, and how a purge keyed on
an implementation revoked nothing for them.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernel_bench.providers import all_providers, catalog_kernel


class TestRecordedSpellings(unittest.TestCase):
    def test_both_spellings_of_a_parameterised_kernel_name_one_kernel(self):
        # The pair that exists in the stored corpus today.
        for recorded in ("flashmla_decode_b64", "flashmla_decode@_b64"):
            self.assertEqual(catalog_kernel(recorded), "flashmla_decode",
                             f"{recorded} must classify as its kernel")

    def test_a_longer_registered_name_wins_over_a_shorter_prefix(self):
        # `torch_gelu` is registered and is a prefix of `torch_gelu_tanh`,
        # which is registered too: the longer name is the kernel, not a
        # parameterisation of the shorter one.
        names = {p.kernel_impl for p in all_providers()}
        self.assertIn("torch_gelu", names)
        self.assertIn("torch_gelu_tanh", names)
        self.assertEqual(catalog_kernel("torch_gelu_tanh"), "torch_gelu_tanh")

    def test_a_name_no_kernel_claims_stays_unclaimed(self):
        # An orphan must keep reading as an orphan: this is the signal that a
        # row was written by something no longer registered.
        self.assertIsNone(catalog_kernel("no_such_kernel_anywhere"))

    def test_every_registered_name_classifies_as_itself(self):
        for provider in all_providers():
            self.assertEqual(catalog_kernel(provider.kernel_impl),
                             provider.kernel_impl)


class TestNothingCutsTheNameAtACharacter(unittest.TestCase):
    """The ratchet: the three tools that regressed each held their own
    `split("@")[0]`, and nothing went red when a spelling stopped matching.
    A recorded name is classified by the registry or not at all."""

    def test_no_tool_splits_a_recorded_kernel_name_itself(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        pattern = re.compile(r"kernel_impl[^\n]*split\(\s*[\"']@[\"']")
        offenders = []
        for path in sorted((root / "tools").glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(root)}:{number}")
        self.assertEqual(offenders, [], "ask `catalog_kernel` instead: "
                         "cutting a recorded name at `@` loses every row "
                         "written before the separator existed")


if __name__ == "__main__":
    unittest.main()
