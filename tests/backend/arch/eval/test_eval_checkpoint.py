"""The checkpoint invariant, against real safetensors headers.

The oracle next door compares this package to the engine, which shares its
assumptions. This one compares it to the FILE: every byte the spec says a
checkpoint holds, against the byte count `fetch_metadata` read out of the shard
headers. Nothing here can be right for the wrong reason, because nothing on the
other side of the comparison came from a spec.

What the residue means, and what it does not
--------------------------------------------
A model absent from `EXACT` means the two sides disagree. It does NOT mean the
spec is wrong, and this file used to say it did — "each is a statement the SPEC
makes that the checkpoint contradicts". Three of those statements have since
been read properly and none of them was the spec:

  * MiMo x3, 15.4 GB — this package had no slot-to-role map, so
    `per_role.attention` never reached leaves carrying `attention_swa`;
  * V4-Pro, 49 GB — `F8_E8M0FNU` had no entry in the byte table, and an
    unknown dtype cost zero;
  * V4-Pro again, 147 GB — the SNAPSHOT costs this model's E8M0 scales at F32.

So a delta localises a disagreement and says nothing about which side owns it.
That is still the whole value: before the invariant, a wrong dtype on one tensor
family was indistinguishable from a wrong dtype everywhere, and both were
indistinguishable from nothing being wrong. But a cause written beside a number
is a claim, and the ones here were wrong more often than they were right — hence
`undiagnosed`, which is what most of them honestly are.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.checkpoint import (
    Reconciliation,
)
from backend.arch.eval.transforms import Deployment
from tests.snapshots import shared_program, shared_walk
from tools.corpus_pins.checks import checkpoint_invariants as invariants
from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pins.checkpoint_reconciliation import (
    CHECKPOINT_RECONCILIATION as PIN,
    reconcile as _reconcile,
)

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"

# The four sections live in `tests/golden/pins/checkpoint_reconciliation.json`,
# each reasoned entry with its note:
#   EXACT      models whose saved formats reproduce the checkpoint's size
#              EXACTLY — to the byte, over a file of up to 1.5 TB;
#   KNOWN      `model -> (weight delta, metadata delta, note)`. Two columns,
#              because they fail for unrelated reasons: a weight delta means
#              the spec has the wrong tensor or the wrong dtype, a metadata
#              delta means a quant scheme's scale cardinality or width is
#              wrong and every weight is fine;
#   MEASURED   `model -> measured delta` against `metadata.index_total_size`,
#              the byte count the hub computed from the shards themselves —
#              the only figure on the other side that is not a
#              reconstruction;
#   NO_ANCHOR  snapshots with NO anchor, and why. An absent anchor is honest;
#              a wrong one is not.
_DOC = PIN.load()
EXACT = sorted(PIN.section(_DOC, "EXACT"))
KNOWN: Dict[str, Tuple[int, int, str]] = {
    stem: (entry["weight_delta"], entry["metadata_delta"], entry["note"])
    for stem, entry in PIN.section(_DOC, "KNOWN").items()}
MEASURED: Dict[str, int] = {
    stem: entry["measured_delta"]
    for stem, entry in PIN.section(_DOC, "MEASURED").items()}
NO_ANCHOR: Dict[str, str] = {
    stem: entry["reason"] for stem, entry in PIN.section(_DOC, "NO_ANCHOR").items()}


class TestCheckpointInvariant(unittest.TestCase):
    def test_saved_formats_reproduce_the_file_size(self):
        for name in EXACT:
            with self.subTest(model=name):
                result = _reconcile(name)
                self.assertGreater(result.header_total, 0, "no header bytes")
                self.assertEqual(
                    int(result.delta), 0,
                    f"{name}: {result.total:,.0f} vs {result.header_total:,.0f}")

    def test_known_disagreements_have_not_moved(self):
        for name, (weight, meta, why) in KNOWN.items():
            with self.subTest(model=name):
                result = _reconcile(name)
                self.assertEqual(
                    (int(result.weight_delta), int(result.metadata_delta)),
                    (weight, meta), f"{name}: {why}")

    def test_the_weight_column_is_clean_almost_everywhere(self):
        """What the spec says the checkpoint HOLDS, against what it holds.

        39 of 39, with nothing left in it. Llama-2's F32 `inv_freq` closed when
        the spec recorded the dtype, V4-Pro's 256 MB closed with the
        scale-layout work, and this assertion kept both pinned for a while
        after they were gone.

        Stated separately from the total because it is the column the incumbent
        also checks, and agreeing with it there is what makes the metadata
        column's findings credible rather than noise. Empty is the strong form
        of that: what the spec says a checkpoint HOLDS is what it holds, for
        every model here.
        """
        off = {}
        for name in sorted({p.stem for p in SNAPSHOTS.glob("*.json")}):
            delta = invariants.weight_delta(CorpusModel(name))
            if delta:
                off[name] = delta
        self.assertEqual(off, {})

    def test_every_quant_sibling_is_named_after_a_tensor_that_exists(self):
        """Every scale / zero-point / index tensor a format implies must land
        on a name the checkpoint header actually contains.

        The byte columns cannot see this. A sibling contributes its bytes to
        the metadata total whatever it is called, so a scale named after a
        module the checkpoint has never heard of reconciles perfectly. Neither
        can the coverage gate, which counts checkpoint tensors no spec node
        claimed — a spec stamp with NO counterpart is not a tensor and does not
        appear. Names were the one thing about a checkpoint nothing checked,
        and two families had drifted:

          * expert banks — every spec called the node `expert` / `shared_expert`
            where the checkpoints write `experts` / `shared_experts`, and the
            validator folded the two spellings together, which made
            the spec-side spelling unfalsifiable;
          * DeepSeek + GLM MTP heads — an `<inner>` invocation named
            `transformer` contributed a path segment to every leaf under it,
            and those checkpoints store an MTP layer flat in the main `layers`
            index space with nothing at that position.

        Both were invisible for the same reason and both are gone. This is the
        check that would have caught them on the day they were written.
        """
        # The sibling KNOWS its name; the check reads it off the format rather
        # than stripping `.weight|.bias` off the weight and appending a suffix.
        orphans: Dict[str, list] = {}
        for name in sorted({p.stem for p in SNAPSHOTS.glob("*.json")}):
            found = invariants.orphan_siblings(CorpusModel(name))
            if found:
                orphans[name] = found
        self.assertEqual(orphans, {})

    def test_every_quant_override_decides_a_stored_tensor(self):
        """A `quant_context.overrides` row is a statement about tensors the
        checkpoint holds. One that decides no stored tensor's format — its
        pattern names nothing the spec stores, or an earlier row always wins
        first — changes no byte, so neither byte column can see it, and it
        still reads as the dtype those tensors have."""
        dead: Dict[str, list] = {}
        for name in sorted({p.stem for p in SNAPSHOTS.glob("*.json")}):
            found = invariants.dead_overrides(CorpusModel(name))
            if found:
                dead[name] = found
        self.assertEqual(dead, {})

    def test_every_snapshot_is_accounted_for(self):
        """No snapshot may be silently neither exact nor known."""
        listed = set(EXACT) | set(KNOWN)
        on_disk = {p.stem for p in SNAPSHOTS.glob("*.json")}
        self.assertEqual(on_disk - listed, set(), "unaccounted snapshots")

    def test_every_dtype_the_headers_use_has_a_width(self):
        """An unknown dtype silently contributes zero bytes, which reads as
        agreement. `F8_E8M0` — MXFP8's block-scale exponent — was missing, and
        DeepSeek-V4-Pro was the only model that would have shown it."""
        for name in sorted({p.stem for p in SNAPSHOTS.glob("*.json")}):
            with self.subTest(model=name):
                self.assertEqual(invariants.unknown_dtypes(CorpusModel(name)), ())


class TestSavedVersusIntent(unittest.TestCase):
    """The split itself: `saved` is the file, `intent` is the deployment."""

    def test_intent_defaults_to_saved_and_changes_nothing(self):
        result = _reconcile("deepseek-ai__DeepSeek-V2-Lite")
        self.assertEqual(int(result.delta), 0)

    def test_serving_a_checkpoint_at_a_different_width(self):
        """Dequantising to BF16 doubles what is RESIDENT and leaves the file
        alone — which is the whole reason the two are separate fields."""
        from backend.arch.eval.precision import PrecisionProfile
        from backend.arch.eval import traffic as traffic_mod

        program = shared_program('deepseek-ai/DeepSeek-V2-Lite')
        base = Deployment(program).at(batch=1, input_tokens=128)

        as_saved = shared_walk(base)
        as_fp8 = shared_walk(base.with_precision(PrecisionProfile.canonicalize(
            {"weights": [{"select": {"roles": "all"},
                          "format": {"scheme": "fp8"}}]})))

        # The file is the file, whatever anyone serves it as.
        from backend.hf.metadata import fetch_metadata
        header = fetch_metadata("deepseek-ai/DeepSeek-V2-Lite")
        self.assertEqual(
            int(Reconciliation.of(as_saved, header).total),
            int(Reconciliation.of(as_fp8, header).total))

        # HBM is not.
        saved_reads = sum(
            t.weight_reads for t in
            traffic_mod.derive(as_saved).values.values())
        fp8_reads = sum(
            t.weight_reads for t in
            traffic_mod.derive(as_fp8).values.values())
        self.assertGreater(saved_reads, 0)
        self.assertLess(fp8_reads, saved_reads)

    def test_a_converted_representation_says_so(self):
        from backend.arch.eval.formats import (
            Representation, TensorFormat)

        native = TensorFormat(storage_dtype="BF16")
        self.assertFalse(Representation(saved=native).is_converted)
        self.assertIs(Representation(saved=native).resident, native)

        fp8 = TensorFormat(storage_dtype="F8_E4M3", num_bits=8)
        converted = Representation(saved=native, intent=fp8)
        self.assertTrue(converted.is_converted)
        self.assertIs(converted.resident, fp8)
        # Metadata elements are the SAVED format's: they are what the file
        # holds, and a deployment's choice does not rewrite the file.
        self.assertEqual(converted.metadata_elements, native.metadata_elements)
        # And the whole representation can be asked for as the file has it.
        self.assertIs(converted.as_saved().resident, native)
        self.assertIsNone(converted.as_saved().intent)
        self.assertIs(Representation(saved=native).as_saved().saved, native)

    def test_the_checkpoint_projection_reports_the_stored_width(self):
        """gemma-2 is saved F32 and loaded BF16 (`dtypes.served_dtype`), and
        the two projections answer two questions: `tensors()` the deployment's
        width, `checkpoint_tensors()` the file's. With the loaded width on the
        checkpoint rows the byte gate compared 18.5 GB against a 37.0 GB
        header and called a correct spec wrong by exactly half."""
        from tests.snapshots import shared_program, shared_walk
        from backend.arch.eval.transforms import Deployment
        analysis = shared_walk(
            Deployment(shared_program("google__gemma-2-9b")).at(
                batch=1, input_tokens=1))
        resident = {row.rep.resident.storage_dtype
                    for row in analysis.tensors() if row.rep is not None}
        stored = {row.rep.resident.storage_dtype
                  for row in analysis.checkpoint_tensors()
                  if row.rep is not None}
        self.assertEqual(resident, {"BF16"})
        self.assertEqual(stored, {"F32"})
        self.assertEqual(analysis.checkpoint_param_bytes(),
                         sum(int(row.tensor.elements) * 4 * int(row.count)
                             for row in analysis.checkpoint_tensors()
                             if not row.is_metadata and not row.is_buffer))

class TestAgainstTheMeasuredTotal(unittest.TestCase):
    """The anchor: what the hub says the files weigh.

    Every other comparison in this file is between two reconstructions — this
    package's dtype table against `backend/hf/metadata`'s — and two tables
    missing the same row agree while both are wrong. `index_total_size` is
    summed by the hub from the shards, so it can contradict both.
    """

    def test_every_anchored_snapshot_matches_or_is_pinned(self):
        exact, off = [], {}
        for path in sorted(SNAPSHOTS.glob("*.json")):
            result = _reconcile(path.stem)
            if not result.is_anchored:
                continue
            delta = int(result.measured_delta)
            (exact.append(path.stem) if delta == 0
             else off.__setitem__(path.stem, delta))
        self.assertEqual(off, MEASURED, "measured deltas moved")
        self.assertEqual(
            len(exact) + len(off), len(list(SNAPSHOTS.glob("*.json"))),
            "a snapshot has no measured anchor")

    def test_only_consistent_anchors_are_kept(self):
        """A snapshot without an anchor is one nobody can check; a snapshot with
        the WRONG anchor is one everybody checks against the wrong thing."""
        missing = sorted(p.stem for p in SNAPSHOTS.glob("*.json")
                         if not _reconcile(p.stem).is_anchored)
        self.assertEqual(missing, sorted(NO_ANCHOR))


if __name__ == "__main__":
    unittest.main()
