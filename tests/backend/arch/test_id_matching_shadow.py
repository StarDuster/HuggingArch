"""Shadow check for the id-keyed matching migration.

Every checkpoint snapshot has an identity file (`base: <family>` +
`match.model_id`). Until the flip (step 4) the model_type scorer is still the
authority; this test asserts the id-lookup that will replace it resolves each
snapshot to the SAME architecture the scorer picks today. A non-zero
divergence here means base inheritance, the id index, or a family mapping is
wrong — and would become a real regression at the flip.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import SpecLoader, SpecLoadError, arch_fingerprint

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


class TestIdMatchingShadow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = SpecLoader()

    def test_every_snapshot_id_is_registered(self):
        snaps = sorted(SNAPSHOTS.glob("*.json"))
        self.assertTrue(snaps, "no snapshots checked out")
        for snap in snaps:
            model_id = snap.stem.replace("__", "/", 1)
            self.assertIn(
                model_id, self.loader._id_index,
                f"{model_id} has no identity file (id index miss)")

    def test_id_lookup_matches_the_scorer_for_every_snapshot(self):
        snaps = sorted(SNAPSHOTS.glob("*.json"))
        self.assertTrue(snaps, "no snapshots checked out")
        divergent = []
        for snap in snaps:
            model_id = snap.stem.replace("__", "/", 1)
            config = json.loads(snap.read_text())["config"]
            scored = self.loader.match(config, model_id=model_id)
            self.assertIsNotNone(scored, f"scorer missed {model_id}")
            id_spec_name = self.loader._id_index.get(model_id)
            id_spec = self.loader.get(id_spec_name) if id_spec_name else None
            self.assertIsNotNone(id_spec, f"id index -> missing spec for {model_id}")
            if arch_fingerprint(id_spec) != arch_fingerprint(scored.spec):
                divergent.append(
                    f"{model_id}: id-lookup {id_spec.name} != scorer {scored.spec.name}")
        self.assertEqual(
            divergent, [],
            "shadow divergence — id-lookup and scorer disagree on architecture:\n"
            + "\n".join(divergent))


class TestBrokenIdentityFile(unittest.TestCase):
    """A broken identity file (bad `base:`) is recorded, not fatal (contract D):
    it does not crash boot, it is refused loudly on request by its own id, and
    it does not taint spec-gen for unrelated unregistered ids."""

    def _loader_with_broken_identity(self, tmp: str) -> SpecLoader:
        models = Path(tmp) / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "_broken.yaml").write_text(
            "base: no_such_family\nmatch: {model_id: [broken/id]}\n")
        return SpecLoader(overlay_model_dirs=[str(models)])

    def test_broken_base_does_not_crash_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader_with_broken_identity(tmp)
            self.assertTrue(loader.all_specs())  # booted with the real corpus

    def test_broken_id_is_refused_loudly_not_silently_family_matched(self):
        config = json.loads((SNAPSHOTS / "Qwen__Qwen3-8B.json").read_text())["config"]
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader_with_broken_identity(tmp)
            with self.assertRaises(SpecLoadError):
                loader.match(config, model_id="broken/id")

    def test_a_broken_identity_does_not_dos_unrelated_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = self._loader_with_broken_identity(tmp)
            # An unknown model with no spec must still return None -> spec-gen,
            # not inherit the broken file's failure.
            self.assertIsNone(
                loader.match({"model_type": "zzz_unknown_arch"},
                             model_id="unrelated/unknown"))


class TestRuntimeIndexRebuild(unittest.TestCase):
    """A runtime overlay/custom add rebuilds the id index. A colliding id keeps
    the shared claim (contract E) and never raises — one cached spec must not
    DoS every match()."""

    @staticmethod
    def _overlay(tmp: str, stem: str, body: str) -> str:
        root = Path(tmp)
        (root / "models").mkdir(parents=True, exist_ok=True)
        (root / "components" / "drafts").mkdir(parents=True, exist_ok=True)
        (root / "models" / f"{stem}.yaml").write_text(body)
        return str(root)

    def test_overlay_registers_a_new_id(self):
        loader = SpecLoader()
        with tempfile.TemporaryDirectory() as tmp:
            root = self._overlay(
                tmp, "myfork", "base: llama\nmatch: {model_id: [my-org/my-fork]}\n")
            self.assertTrue(loader.attach_overlay(root))
            self.assertEqual(loader._id_index.get("my-org/my-fork"), "myfork")

    def test_overlay_colliding_id_keeps_shared_and_does_not_crash_match(self):
        loader = SpecLoader()
        cfg = json.loads((SNAPSHOTS / "Qwen__Qwen3-8B.json").read_text())["config"]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._overlay(
                tmp, "collide",
                "base: llama\nmatch: {model_id: [meta-llama/Llama-3.2-1B]}\n")
            self.assertTrue(loader.attach_overlay(root))  # mounted, no raise
            # shared identity still owns the id
            self.assertEqual(
                loader._id_index["meta-llama/Llama-3.2-1B"],
                "meta-llama__Llama-3.2-1B")
            # matcher is not DoS'd — an unrelated match still resolves
            self.assertIsNotNone(loader.match(cfg, model_id="Qwen/Qwen3-8B"))

    def test_boot_with_a_user_overlay_colliding_on_a_shared_id_does_not_crash(self):
        # analyze.py builds SpecLoader(overlay_model_dirs=[user dir]) per
        # request. A user spec sharing an id with a shared identity file must
        # not 500 the constructor — shared wins (contract E).
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir(parents=True)
            (models / "user.yaml").write_text(
                "base: llama\nmatch: {model_id: [meta-llama/Llama-3.2-1B]}\n")
            loader = SpecLoader(overlay_model_dirs=[str(models)])
            self.assertEqual(
                loader._id_index["meta-llama/Llama-3.2-1B"],
                "meta-llama__Llama-3.2-1B")  # shared, not the user file


if __name__ == "__main__":
    unittest.main()
