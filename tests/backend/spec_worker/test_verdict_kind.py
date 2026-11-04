"""Outcomes carry their kind; nobody reads it back out of a message.

Three places sniffed a string for a fact the producer knew: the repair
loop asked whether a reason `startswith("generated spec failed
validation")`, the service route asked whether a loader error mentioned
`user_specs`, and the spec route asked whether a matched spec's name
started with `custom_`. Each producer now says what it knows — a
`Downgrade.kind`, a `SpecLoadError.path`, the custom model's own spec
file — and the consumer asks that.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoadError
from backend.spec_worker.overlay import is_user_spec_path
from backend.spec_worker.worker import Downgrade, _repair_prompt_fn


class TestRepairDispatchesOnKind(unittest.TestCase):
    def test_only_a_validation_downgrade_is_repairable(self):
        make = _repair_prompt_fn("org/model")
        validation = Downgrade("validation", "generated spec failed validation: a; b", ("a", "b"))
        prompt = make(validation.reason, {"downgrade": validation})
        self.assertIsNotNone(prompt)
        self.assertIn("a; b", prompt)
        for kind, reason in (("crash", "generated spec validation crashed: X"),
                             ("no_output", "agent exited cleanly but wrote no model spec")):
            self.assertIsNone(make(reason, {"downgrade": Downgrade(kind, reason)}))

    def test_a_reason_that_merely_reads_like_a_validation_failure_is_not_one(self):
        """The string is the runner's; the verdict is the worker's."""
        make = _repair_prompt_fn("org/model")
        self.assertIsNone(make("generated spec failed validation: looks like it", {}))


class TestSpecLoadErrorNamesItsFile(unittest.TestCase):
    def test_path_travels_on_the_error(self):
        err = SpecLoadError("spec 'x' exists at /tmp/x.yaml but failed", path="/tmp/x.yaml")
        self.assertEqual(err.path, "/tmp/x.yaml")
        self.assertIsNone(SpecLoadError("no single file").path)

    def test_a_users_overlay_is_recognised_by_where_it_lives(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"HA_USER_SPECS_ROOT": tmp}):
            inside = Path(tmp) / "u1" / "org__model" / "models" / "x.yaml"
            self.assertTrue(is_user_spec_path(inside))
            self.assertFalse(is_user_spec_path("/somewhere/else/user_specs/x.yaml"))


if __name__ == "__main__":
    unittest.main()
