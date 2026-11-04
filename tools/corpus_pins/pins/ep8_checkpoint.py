"""P5: the checkpoint comparison table under `dp_attention=8, ep_moe=8`.

Per snapshot with a non-empty table: how many rows fall under each
(verdict, evidence) pair, the row total, and a digest over the rows' facts
(`checkpoint_prefix, artifact_key, pattern_names, verdict, evidence,
expected_entries, header_entries, header_count_provenance`). A row is one
(prefix, artifact key) group, so the total moves exactly when artifact keys
appear, disappear or fold; the digest catches a change inside a row the
counts cannot see.

The file keeps three sections keyed by snapshot file name — `corpus`
(`{"match/header_ok": n}`), `row_totals`, `digests` — and every section
names the same snapshots.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pin import Doc, Pin, Row, Scope, json_bytes

PARALLEL = (("dp_attention", 8), ("ep_moe", 8))
SECTIONS = ("corpus", "row_totals", "digests")


def verdict_key(verdict: str, evidence: str) -> str:
    return f"{verdict}/{evidence}"


@functools.lru_cache(maxsize=8)
def step(stem: str, pc: Tuple[Tuple[str, int], ...] = PARALLEL) -> Any:
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step
    resolved = CorpusModel(stem).resolved(shared_loader())
    return evaluate_step(
        resolved, ParallelConfig(**dict(pc)),
        phase="prefill", batch=1, input_tokens=8, output_tokens=0,
    )


def checkpoint_rows(stem: str) -> List[Any]:
    """The comparison rows: every expert-bank artifact against the header.
    An expert-bank tensor whose encoding is not declared raises here."""
    from backend.arch.layout import copy_checkpoint as ckpt
    return list(ckpt.compare_checkpoint(step(stem), CorpusModel(stem).header))


def digest(rows: List[Any]) -> str:
    facts = [
        (
            row.checkpoint_prefix,
            row.artifact_key,
            row.pattern_names,
            row.verdict,
            row.evidence,
            row.expected_entries,
            row.header_entries,
            row.header_count_provenance,
        )
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(facts, separators=(",", ":")).encode()).hexdigest()[:16]


def value_of(rows: List[Any]) -> Dict[str, Any]:
    counts = collections.Counter(
        verdict_key(row.verdict, row.evidence) for row in rows)
    return {"corpus": dict(sorted(counts.items())), "row_totals": len(rows),
            "digests": digest(rows)}


@dataclass(frozen=True)
class Ep8CheckpointPin(Pin):
    def parse(self, data: bytes) -> Doc:
        raw = json.loads(data.decode("utf-8"))
        if sorted(raw) != sorted(SECTIONS):
            raise ValueError(f"{self.artifact}: sections {sorted(raw)}")
        names = {section: set(raw[section]) for section in SECTIONS}
        if len({frozenset(v) for v in names.values()}) != 1:
            raise ValueError(f"{self.artifact}: sections name different snapshots")
        return {name: {section: raw[section][name] for section in SECTIONS}
                for name in raw["corpus"]}

    def render(self, doc: Doc) -> bytes:
        return json_bytes({section: {name: value[section]
                                     for name, value in doc.items()}
                           for section in SECTIONS})

    def applies(self, model: CorpusModel) -> bool:
        return bool(checkpoint_rows(model.stem))

    def key_exists(self, key: str, stems, components) -> bool:
        return key[:-len(".json")] in stems

    def measure(self, model: CorpusModel) -> Row:
        rows = checkpoint_rows(model.stem)
        return {model.filename: value_of(rows)} if rows else {}


EP8_CHECKPOINT = Ep8CheckpointPin(
    id="P5",
    title="EP8 checkpoint rows",
    scope=Scope.CONDITIONAL,
    artifact="tests/golden/pins/ep8_checkpoint.json",
    owner_tests=(
        "tests/backend/arch/layout/test_expert_weight_checkpoint.py::"
        "test_ep8_snapshot_corpus_matches_the_checkpoint_pin",
    ),
    waits_on=("G4",),
)
