"""P4: every snapshot's bytes against its checkpoint header.

`Reconciliation.of` compares what the saved formats say the checkpoint holds
with what `fetch_metadata` reads out of the snapshot's shard headers. A model
whose two sides agree to the byte is `EXACT`; one that does not is `KNOWN`
with its (weight, metadata) delta and a note that says why. Separately, a
snapshot with a hub anchor (`index_total_size`) that disagrees with it is
`MEASURED`, and one without an anchor is `NO_ANCHOR` with the reason.

Keys name the section: `EXACT/<stem>`, `KNOWN/<stem>`, `MEASURED/<stem>`,
`NO_ANCHOR/<stem>`. `EXACT` needs no one's judgement; the other three carry
a reason only a person can give (decision D3), so they wait for it.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pin import Doc, Pin, Row, Scope, json_bytes

SECTIONS = ("EXACT", "KNOWN", "MEASURED", "NO_ANCHOR")
#: the sections whose entry is a person's statement of why (decision D3)
REASONED = ("KNOWN", "MEASURED", "NO_ANCHOR")


@functools.lru_cache(maxsize=None)
def reconcile(stem: str) -> Any:
    """The snapshot's `Reconciliation`, against the header as `fetch_metadata`
    reads it — bucketed by the current classifier, as every other consumer
    sees it — not the snapshot's raw buckets, which are as old as the
    snapshot."""
    from backend.arch.eval.checkpoint import Reconciliation
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.hf.metadata import fetch_metadata
    model = CorpusModel(stem)
    analysis = Deployment(Program.of(model.resolved(shared_loader()))).at(
        batch=1, input_tokens=128).checkpoint()
    return Reconciliation.of(analysis, fetch_metadata(model.model_id), stem)


def key(section: str, stem: str) -> str:
    return f"{section}/{stem}"


def split(k: str) -> Tuple[str, str]:
    section, stem = k.split("/", 1)
    return section, stem


@dataclass(frozen=True)
class CheckpointReconciliationPin(Pin):
    def parse(self, data: bytes) -> Doc:
        raw = json.loads(data.decode("utf-8"))
        if sorted(raw) != sorted(SECTIONS):
            raise ValueError(f"{self.artifact}: sections {sorted(raw)}")
        if raw["EXACT"] != sorted(set(raw["EXACT"])):
            raise ValueError(f"{self.artifact}: EXACT is not a sorted set")
        doc: Doc = {key("EXACT", stem): True for stem in raw["EXACT"]}
        for section in REASONED:
            doc.update({key(section, stem): value
                        for stem, value in raw[section].items()})
        return doc

    def render(self, doc: Doc) -> bytes:
        raw: Dict[str, Any] = {section: {} for section in REASONED}
        exact = []
        for k, value in doc.items():
            section, stem = split(k)
            if section == "EXACT":
                exact.append(stem)
            else:
                raw[section][stem] = value
        raw["EXACT"] = sorted(exact)
        return json_bytes(raw)

    def key_waits_on(self, k: str) -> Tuple[str, ...]:
        return ("D3",) if split(k)[0] in REASONED else ()

    def measure(self, model: CorpusModel) -> Row:
        """The model's keys with the numbers the engine measures. A reasoned
        entry's `note` / `reason` is D3's to write; the measured value holds
        only the numbers."""
        result = reconcile(model.stem)
        row: Row = {}
        if int(result.delta) == 0:
            row[key("EXACT", model.stem)] = True
        else:
            row[key("KNOWN", model.stem)] = {
                "weight_delta": int(result.weight_delta),
                "metadata_delta": int(result.metadata_delta),
            }
        if not result.is_anchored:
            row[key("NO_ANCHOR", model.stem)] = {}
        elif int(result.measured_delta) != 0:
            row[key("MEASURED", model.stem)] = {
                "measured_delta": int(result.measured_delta)}
        return row

    def missing_keys(self, doc: Doc, model: CorpusModel) -> list:
        """Every snapshot is EXACT or KNOWN; its anchor keys depend on a
        measurement."""
        keys = (key("EXACT", model.stem), key("KNOWN", model.stem))
        return [] if any(k in doc for k in keys) else list(keys)

    def key_exists(self, k: str, stems, components) -> bool:
        return split(k)[1] in stems

    def value_matches(self, recorded: Any, measured: Any) -> bool:
        """The numbers; a reasoned entry's note is the person's."""
        if isinstance(measured, dict):
            return {f: recorded.get(f) for f in measured} == measured
        return recorded == measured

    def section(self, doc: Doc, name: str) -> Dict[str, Any]:
        """One section as `{stem: value}` (the sorted stems for `EXACT`)."""
        return {split(k)[1]: v for k, v in doc.items() if split(k)[0] == name}


CHECKPOINT_RECONCILIATION = CheckpointReconciliationPin(
    id="P4",
    title="checkpoint reconciliation",
    scope=Scope.REQUIRED,
    artifact="tests/golden/pins/checkpoint_reconciliation.json",
    owner_tests=tuple(
        "tests/backend/arch/eval/test_eval_checkpoint.py::" + name for name in (
            "TestCheckpointInvariant::test_saved_formats_reproduce_the_file_size",
            "TestCheckpointInvariant::test_known_disagreements_have_not_moved",
            "TestCheckpointInvariant::test_every_snapshot_is_accounted_for",
            "TestAgainstTheMeasuredTotal::test_every_anchored_snapshot_matches_or_is_pinned",
            "TestAgainstTheMeasuredTotal::test_only_consistent_anchors_are_kept",
        )),
)
