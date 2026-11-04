"""A pin: a closed per-model table the corpus tests compare against.

Each pin owns one artifact under `tests/golden/` (or, for the README counts,
two text files). A pin can parse its artifact into a document and render the
document back to the same bytes, measure the row one model owns, merge that
row into a document, and check that a rewritten artifact differs from the old
one exactly by the keys the model owns. The test that guards the pin and the
admitter that extends it both go through these methods, so what the admitter
writes is what the test compares.

Documents are plain mappings from key to value. A pin's file may lay its
keys out in several sections (the checkpoint reconciliation's `EXACT` list
and `KNOWN` map); the key names the section (`EXACT/<stem>`), so the
mapping stays flat and the layout stays the pin's business.
"""
from __future__ import annotations

import abc
import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

from tools.corpus_pins.corpus import ROOT, Corpus, CorpusModel

Doc = Dict[str, Any]
Row = Dict[str, Any]


class Scope(str, enum.Enum):
    #: every snapshot has a key
    REQUIRED = "required"
    #: a snapshot has a key when the pin's predicate holds for it
    CONDITIONAL = "conditional"
    #: cells derived from the whole corpus, not keyed by model
    REQUIRED_AGGREGATE = "required_aggregate"


@dataclass(frozen=True)
class PinBase:
    """What every pin declares. `artifact` is relative to the repository
    root; `owner_tests` are the pytest node ids that compare it; `waits_on`
    names the decisions a new key needs before it can be written (`G4`: the
    TP-split and copy-contract decisions are in place)."""

    id: str
    title: str
    scope: Scope
    artifact: str
    owner_tests: Tuple[str, ...]
    waits_on: Tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return ROOT / self.artifact

    def artifacts(self) -> Tuple[str, ...]:
        """Every file this pin owns, relative to the repository root."""
        return (self.artifact,)


@dataclass(frozen=True)
class Pin(PinBase, abc.ABC):
    """A pin keyed by model (or by a model's component file)."""

    def key_waits_on(self, key: str) -> Tuple[str, ...]:
        """The decisions one new key waits on; the pin's own by default."""
        return self.waits_on

    def read(self) -> bytes:
        return self.path.read_bytes()

    def load(self) -> Doc:
        return self.parse(self.read())

    @abc.abstractmethod
    def parse(self, data: bytes) -> Doc: ...

    @abc.abstractmethod
    def render(self, doc: Doc) -> bytes: ...

    def round_trips(self, data: bytes) -> bool:
        """`render(parse(data)) == data`, the precondition for any write."""
        return self.render(self.parse(data)) == data

    # ── per-model keys ──────────────────────────────────────────────────
    def applies(self, model: CorpusModel) -> bool:
        """Whether `model` owns keys here; always true for a required pin."""
        return True

    @abc.abstractmethod
    def measure(self, model: CorpusModel) -> Row:
        """The keys `model` owns and their values, measured on this tree."""

    def measure_all(self) -> Doc:
        doc: Doc = {}
        for model in Corpus.models():
            if self.applies(model):
                doc.update(self.measure(model))
        return doc

    # ── what verify asks without measuring ──────────────────────────────
    def missing_keys(self, doc: Doc, model: CorpusModel) -> List[str]:
        """The keys a required pin must hold for `model`, known without
        measuring it, that `doc` lacks. A conditional pin's keys depend on a
        measurement, so it asks nothing here."""
        if self.scope is not Scope.REQUIRED:
            return []
        return [] if model.stem in doc else [model.stem]

    def key_exists(self, key: str, stems: Set[str], components: Set[str]) -> bool:
        """Whether a key still names something in the tree."""
        return key in stems

    def value_matches(self, recorded: Any, measured: Any) -> bool:
        return canonical(recorded) == canonical(measured)

    def merge(self, doc: Doc, row: Row) -> Doc:
        """`doc` with `row`'s keys inserted. A key already present is a
        conflict: merging never rewrites a value."""
        clash = sorted(set(doc) & set(row))
        if clash:
            raise ValueError(f"{self.id}: keys already present: {clash}")
        return {**doc, **row}

    def check_diff(self, old: bytes, new: bytes, owned: Mapping[str, Any]) -> List[str]:
        """Why `new` is not `old` plus exactly the keys `owned`: a removed
        key, a key not owned, a changed old value, bytes moved outside the
        inserted keys. Empty when the write is exactly the insertion."""
        problems: List[str] = []
        if not self.round_trips(old):
            return [f"{self.id}: {self.artifact} does not round-trip"]
        before, after = self.parse(old), self.parse(new)
        removed = sorted(set(before) - set(after))
        added = set(after) - set(before)
        if removed:
            problems.append(f"{self.id}: keys removed {removed}")
        if added != set(owned):
            problems.append(
                f"{self.id}: keys added {sorted(added)} are not the owned "
                f"keys {sorted(owned)}")
        changed = sorted(key for key in set(before) & set(after)
                         if canonical(before[key]) != canonical(after[key]))
        if changed:
            problems.append(f"{self.id}: old values changed {changed}")
        if not problems:
            stripped = {key: value for key, value in after.items() if key not in added}
            if self.render(stripped) != old:
                problems.append(
                    f"{self.id}: bytes outside the inserted keys moved")
        return problems


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def json_bytes(value: Any) -> bytes:
    """The golden JSON form: `indent=2, sort_keys=True`, trailing newline."""
    return (json.dumps(value, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


@dataclass(frozen=True)
class JsonMapPin(Pin):
    """A JSON object whose top-level keys are the pin's keys."""

    def parse(self, data: bytes) -> Doc:
        return json.loads(data.decode("utf-8"))

    def render(self, doc: Doc) -> bytes:
        return json_bytes(doc)


@dataclass(frozen=True)
class Parameter:
    """A module-level number in a corpus test that is not pinned data: a
    tolerance, a report limit, a fixed request size. Registered with the
    reason it is not a pin, so the registry test can tell the two apart."""

    module: str
    name: str
    reason: str


@dataclass(frozen=True)
class CuratedSet:
    """A per-model collection a test keeps on purpose for a chosen subset of
    the corpus. A new model does not join it and the admitter never writes
    it."""

    module: str
    name: str
    reason: str
    artifact: str = ""
