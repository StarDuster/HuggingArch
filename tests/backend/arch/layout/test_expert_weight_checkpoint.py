"""How a checkpoint stores an expert bank's expert axis, declared and checked.

An expert bank's weights lead with the expert axis (`shape: [n_expert_copies,
out, in]`), and the checkpoint stores that axis one of two ways. The bank
says which (`expert_weight`): `unrolled` — one entry per expert, its index
just before the weight's own module (`experts.17.gate_proj.weight`), or
`leading_axis` — one entry whose leading axis is the experts. Nothing else
about the bank is declared beside its shapes: where the axis is placed is its
`weight_axes` entry, like any other axis.

What this file pins:

* the declarations are an explicit closed set, recorded in
  `tests/golden/pins/copy_contracts.json` (D2) as `(file, node, encoding)`;
* the checkpoint ledger (`copy_checkpoint`) reads every expert-bank artifact
  off the stored tensors and their encoding, and finds each in a real
  safetensors header — unrolled per expert, packed as one tensor, scales and
  biases included;
* a weight led by the expert axis whose encoding nobody declared is refused;
* over the corpus, at `ep_moe=8 dp_attention=8`, the ledger's rows are the
  ones the P5 pin records (`tests/golden/pins/ep8_checkpoint.json`).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout import copy_checkpoint as ckpt
from backend.arch.layout.persistent import SAVED_METADATA, STORED
from tests.snapshots import SNAPSHOT_DIR, any_snapshot, snapshot_header
from tools.corpus_pins.decisions import copy_contracts
from tools.corpus_pins.pins import ep8_checkpoint

#: P5's data, read from the pin's own file.
EXPECTED_CHECKPOINT_CORPUS = json.loads(
    (PROJECT_ROOT / ep8_checkpoint.EP8_CHECKPOINT.artifact)
    .read_text(encoding="utf-8"))


def _rows(stem: str):
    return ep8_checkpoint.checkpoint_rows(stem)


class TestTheDeclarationsAreAClosedSet:
    def test_the_31_expert_weight_declarations_are_an_explicit_closed_set(
            self):
        declared = [copy_contracts.contract(*d)
                    for d in copy_contracts.declarations()]
        # Counted over the LIST, so a duplicate declaration is red too.
        assert len(declared) == len(set(declared)) == 31
        assert set(declared) == copy_contracts.load()
        assert copy_contracts.encoding_counts(declared) == collections.Counter(
            {"unrolled": 18, "leading_axis": 13})

    def test_every_encoding_is_one_of_the_two(self):
        assert {c[2] for c in copy_contracts.load()} == {
            "unrolled", "leading_axis"}


class TestTheLedgerFindsTheHeader:
    def test_an_unrolled_bank_is_one_entry_per_expert(self):
        rows = _rows("Qwen__Qwen3-30B-A3B")
        experts = [r for r in rows if ".experts." in r.artifact_key]
        assert {r.artifact_key.rsplit(".", 2)[-2] for r in experts} == {
            "gate_proj", "up_proj", "down_proj"}
        for row in experts:
            assert row.checkpoint == "unrolled"
            assert ".experts.X." in row.artifact_key
            # 48 layers x 128 experts, every one a header entry.
            assert row.expected_entries == row.header_entries == 48 * 128
            assert (row.verdict, row.evidence) == (
                ckpt.MATCH, ckpt.EVIDENCE_HEADER_OK)

    def test_a_packed_bank_with_biases_and_scales_is_one_entry(self):
        rows = _rows("openai__gpt-oss-20b")
        keys = {r.artifact_key.rsplit(".", 1)[-1] for r in rows}
        assert keys == {"gate_up_proj_blocks", "gate_up_proj_scales",
                        "gate_up_proj_bias", "down_proj_blocks",
                        "down_proj_scales", "down_proj_bias"}
        for row in rows:
            assert row.checkpoint == "leading_axis"
            assert row.global_parts == 32
            assert (row.verdict, row.evidence) == (
                ckpt.MATCH, ckpt.EVIDENCE_HEADER_OK), row.artifact_key

    def test_an_unrolled_banks_saved_scales_are_artifacts_too(self):
        rows = _rows("MiniMaxAI__MiniMax-M2.7")
        assert {(r.verdict, r.evidence) for r in rows} == {
            (ckpt.MATCH, ckpt.EVIDENCE_HEADER_OK)}
        assert any(r.artifact_key.endswith("weight_scale_inv") for r in rows)
        assert all(".experts.X." in r.artifact_key for r in rows)

    def test_the_ledger_reads_stored_tensors_and_their_saved_metadata(self):
        artifacts = ckpt.expert_artifacts(
            ep8_checkpoint.step("openai__gpt-oss-20b").analysis)
        assert {a.surface for a in artifacts} == {STORED, SAVED_METADATA}
        assert all(a.checkpoint_shape[0] == 32 for a in artifacts)


class TestAnUndeclaredEncodingIsRefused:
    def test_a_weight_led_by_the_expert_axis_needs_expert_weight(self):
        analysis = ep8_checkpoint.step("Qwen__Qwen3-30B-A3B").analysis
        bank = next(node for node in analysis.program.nodes()
                    if node.expert_weight is not None)
        declared = bank.expert_weight
        bank.expert_weight = None
        try:
            with pytest.raises(ValueError, match="expert_weight"):
                ckpt.expert_artifacts(analysis)
        finally:
            bank.expert_weight = declared

    def test_a_mismatched_header_is_unexplained(self):
        """The same header, read as the other encoding, does not match: the
        verdict is a measurement of the declaration, not an echo of it."""
        step = ep8_checkpoint.step("Qwen__Qwen3-30B-A3B")
        bank = next(node for node in step.analysis.program.nodes()
                    if node.expert_weight == "unrolled")
        bank.expert_weight = "leading_axis"
        try:
            rows = ckpt.compare_checkpoint(
                step, snapshot_header("Qwen__Qwen3-30B-A3B"))
        finally:
            bank.expert_weight = "unrolled"
        verdicts = {(r.verdict, r.evidence) for r in rows
                    if ".experts." in r.artifact_key}
        assert (ckpt.MATCH, ckpt.EVIDENCE_HEADER_OK) not in verdicts


@pytest.mark.regression
@pytest.mark.parametrize(
    "snapshot", sorted(EXPECTED_CHECKPOINT_CORPUS["corpus"]))
def test_ep8_snapshot_corpus_matches_the_checkpoint_pin(snapshot):
    any_snapshot()
    if not (SNAPSHOT_DIR / snapshot).exists():
        pytest.skip(f"{snapshot} not checked out")
    stem = snapshot[:-len(".json")]
    got = ep8_checkpoint.value_of(_rows(stem))
    assert got == {section: EXPECTED_CHECKPOINT_CORPUS[section][snapshot]
                   for section in ep8_checkpoint.SECTIONS}
