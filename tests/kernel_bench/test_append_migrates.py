"""Appending to a results file written under an older schema brings the file
forward instead of refusing it.

The v16 deploy (`timing_method`) stopped every model on the boxes with
"cannot append to …/fa2.csv: header does not match schema": `OPTIONAL_COLUMNS`
let an older file be READ, but the harness's append path still demanded the
current header exactly, and every box resumes into its existing results
files. The append path now runs the one migration
(`benchmark_migration.migrate_text`, the fills `tools/migrate_schema.py`
writes), so the next additive bump needs only its `FILL` entry.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tests.corpus_row as _corpus
from backend.inference import benchmarks as bm
from kernel_bench.emit import IncrementalCsvWriter


def _older_file(path: Path, dropped: set, version: str) -> dict:
    """A results file of one row, written by a harness whose schema lacked
    `dropped`."""
    old = _corpus.row(schema_version=version)
    header = [c for c in bm.ALL_COLUMNS if c not in dropped]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerow([old[c] for c in header])
    return old


def _read(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fh.seek(0)
        header = next(csv.reader(fh))
    return header, rows


def test_a_v15_file_takes_a_v16_row_and_ends_up_v16(tmp_path):
    path = tmp_path / "fa2.csv"
    old = _older_file(path, {"timing_method"}, "15")
    new = _corpus.row(model="other/model", timing_method="l2_drain")
    with IncrementalCsvWriter(str(path), append=True) as writer:
        assert writer.write(new) is None
    header, rows = _read(path)
    assert header == list(bm.ALL_COLUMNS)
    assert [r["model"] for r in rows] == [old["model"], "other/model"]
    assert rows[0]["timing_method"] == ""            # the old row: pre-drain clock
    assert rows[0]["schema_version"] == str(bm.KNOWN_SCHEMA_VERSION)
    assert rows[1]["timing_method"] == "l2_drain"
    for r in rows:
        bm.validate_row(r)
    assert not (tmp_path / "fa2.csv.migrating").exists()


def test_the_oldest_header_the_reader_accepts_is_appendable(tmp_path):
    # Every optional column absent at once: whatever the next additive bump
    # adds is covered as soon as it is declared optional with a fill.
    path = tmp_path / "old.csv"
    _older_file(path, set(bm.OPTIONAL_COLUMNS), "7")
    with IncrementalCsvWriter(str(path), append=True) as writer:
        assert writer.write(_corpus.row(model="other/model",
                                        timing_method="l2_drain")) is None
    header, rows = _read(path)
    assert header == list(bm.ALL_COLUMNS) and len(rows) == 2


def test_a_header_with_a_column_the_schema_lacks_is_still_refused(tmp_path):
    # A newer harness's file, or not a results file: migrating would drop
    # the column. Left untouched.
    path = tmp_path / "newer.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join([*bm.ALL_COLUMNS, "from_the_future"]) + "\n")
    before = path.read_text()
    with pytest.raises(ValueError, match="header does not match schema"):
        IncrementalCsvWriter(str(path), append=True)
    assert path.read_text() == before


def test_the_identity_check_sees_the_migrated_rows(tmp_path):
    # The rows already in the file still count: a repeat of one is refused.
    path = tmp_path / "fa2.csv"
    old = _older_file(path, {"timing_method"}, "15")
    with IncrementalCsvWriter(str(path), append=True) as writer:
        assert writer.write({**old, "timing_method": "l2_drain"}) is not None


def test_a_reordered_current_header_is_rewritten_in_canonical_order(tmp_path):
    # Legal to read (the reader takes columns by name), but the writer appends
    # in `ALL_COLUMNS` order: appended beneath a reversed header, every new
    # value would land under another column's name.
    path = tmp_path / "reversed.csv"
    old = _corpus.row(timing_method="l2_drain")
    header = list(reversed(bm.ALL_COLUMNS))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerow([old[c] for c in header])
    with IncrementalCsvWriter(str(path), append=True) as writer:
        assert writer.write(_corpus.row(model="other/model",
                                        timing_method="l2_drain")) is None
    got_header, rows = _read(path)
    assert got_header == list(bm.ALL_COLUMNS)
    assert rows[0] == {c: old[c] for c in bm.ALL_COLUMNS}
    assert rows[1]["model"] == "other/model"
    for r in rows:
        bm.validate_row(r)


@pytest.mark.parametrize("malformed", ["ragged_row", "repeated_column"])
def test_malformed_input_is_refused_and_left_byte_for_byte(tmp_path, malformed):
    # The migration maps rows by name: a repeated column or a row wider than
    # its header would lose a value silently, so neither is rewritten.
    path = tmp_path / "bad.csv"
    old = _corpus.row()
    header = [c for c in bm.ALL_COLUMNS if c != "timing_method"]
    values = [old[c] for c in header]
    if malformed == "ragged_row":
        values.append("extra")
    else:
        header.append("model")
        values.append("shadow/model")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerow(values)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="header does not match schema"):
        IncrementalCsvWriter(str(path), append=True)
    assert path.read_bytes() == before
    assert not (tmp_path / "bad.csv.migrating").exists()
