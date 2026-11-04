"""P1: which architecture every snapshot resolves to, as a fingerprint.

The fingerprint is a canonical hash of the resolved `ArchSpec` with its
identity and location fields excluded:
  name, source_path   — which file, stamped at load
  unresolved_note     — the file's `# UNRESOLVED:` comment block, same
                        file-metadata class as source_path
  derived_from        — lineage, not merged into the resolved graph
A rename or a re-parent changes those and keeps the architecture, so none of
them may move the fingerprint.

Fields left at their schema default are excluded too. The schema growing a
field (`ModelLayer.role: Optional[str] = None`) adds that field, at its
default, to every spec's dump; hashing it moved all ~60 fingerprints for a
change no spec made. A spec that states a value hashes it; one that leaves
the default hashes nothing for that field, so only what the specs say can
move a row. The price: changing a schema default is not seen here — that is
a schema change, and the eval goldens are what see its effect.

On-disk custom specs under ~/.huggingarch are ignored so the fingerprint is
a function of the repository alone.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from unittest import mock

from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.pin import JsonMapPin, Row, Scope

EXCLUDE = {"name", "source_path", "unresolved_note", "derived_from"}


def resolved_fingerprint(config: dict, model_id: str) -> Optional[str]:
    """The architecture `match` picks for this checkpoint, as a stable hash,
    or None when nothing matches (pinned too, so a future accidental match
    is caught)."""
    from backend.arch.loader import SpecLoader
    with mock.patch.object(SpecLoader, "_load_custom_spec_if_any",
                           lambda self: None):
        resolved = SpecLoader().match(config, model_id=model_id)
    if resolved is None:
        return None
    payload = json.dumps(
        resolved.spec.model_dump(mode="json", exclude=EXCLUDE,
                              exclude_defaults=True),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpecMatchPin(JsonMapPin):
    def measure(self, model: CorpusModel) -> Row:
        return {model.stem: resolved_fingerprint(model.config, model.model_id)}


SPEC_MATCH = SpecMatchPin(
    id="P1",
    title="spec-match fingerprints",
    scope=Scope.REQUIRED,
    artifact="tests/golden/spec_match_fingerprints.json",
    owner_tests=(
        "tests/backend/arch/test_spec_match_golden.py::TestSpecMatchGolden::"
        "test_every_snapshot_resolves_to_its_frozen_architecture",
    ),
    waits_on=("G4",),
)
