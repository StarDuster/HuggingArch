"""Remove rows a results CSV holds twice, byte for byte.

A results CSV with two rows the corpus cannot tell apart stops EVERY env of
EVERY model on a measuring box: the queue hands each run every other model's
results as `--measured-in`, so one such row makes the whole sweep exit before
measuring anything. It happened twice in twenty-four hours (16 models with
zero cells, then four models and an idle card), and the repo fix is at the
two layers that owned it: the write gate refuses a repeat
(`emit.IncrementalCsvWriter`), and reading keys no longer applies the publish
gate (`emit.existing_run_keys`). This is the operational other half — what to
do about the rows a box already wrote under the old code.

It removes only a row that is IDENTICAL IN EVERY COLUMN to one above it.
That needs no schema knowledge (so it runs against a box whose repo copy is
older than this one) and it cannot be wrong: both copies say the same thing,
and the file is unreadable while both are there. A collision that is NOT
byte-identical — the same identity with different numbers — is reported and
left alone, because which measurement to keep is a judgement, not a cleanup.

    python -m tools.dedup_results_csv <csv> [<csv> …]
    python -m tools.dedup_results_csv --dir <box work_dir>/results
    python -m tools.dedup_results_csv --dir … --dry-run
    python -m tools.dedup_results_csv <csv> --no-appender

A file the harness touched in the last few minutes is being appended to, and
rewriting it in place would leave the appender writing into an orphaned inode
— its measurements lost. Those are skipped and counted, not raced.

`--no-appender` is for the one caller that KNOWS better: a file named by a
read failure is one the sweep could not get past, so no run of it is writing
— and waiting out the window there costs the card the minutes the file is
blocking every env. The caller states that; the tool does not guess it.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.unserved_rows import RECORDS_DIR, append_record, result_files

#: Long enough to be sure no `run` is mid-append. The queue writes a row the
#: moment it is measured, so a live file moves every few seconds.
_SETTLED_S = 300

#: What this removed, written beside the results it removed it from, in the
#: corpus's purge-record convention: every removed row in full, plus the file
#: and line it came from and the reason ON EVERY ROW. A removal with no
#: record is the one case where nobody can tell a year later whether a gap
#: in the corpus was a defect or a cleanup — and the record travels with the
#: data, because the mirror pulls the results tree.
_REASON = ("a byte-identical repeat of the row above it; the file is "
           "unreadable while both are there and no measurement is lost")

#: The identity columns' width, by position: the schema's first twelve, and
#: the outcome beside them — a cell may hold a measurement AND the record of
#: having once been refused on this card, and those are two rows by design.
#: Only used to TELL APART the two kinds of collision (identical vs
#: disagreeing), never to decide what to drop; the outcome is compared as the
#: raw cell, which is all that takes `oom` apart from a measurement's empty
#: one, so this needs no import from a repo whose schema may be newer than
#: the file's.
_IDENTITY_WIDTH = 12


def _collisions(rows: List[List[str]], outcome: Optional[int]
                ) -> Tuple[List[List[str]], List[Tuple[int, List[str]]],
                           List[int]]:
    """`rows` without byte-identical repeats, the (line, row) pairs that
    went, and the lines where an identity repeats with DIFFERENT contents."""
    seen: dict = {}
    identities: dict = {}
    kept: List[List[str]] = []
    dropped: List[Tuple[int, List[str]]] = []
    disagreeing: List[int] = []
    for line, row in enumerate(rows, start=2):
        whole = tuple(row)
        if whole in seen:
            dropped.append((line, row))
            continue
        ident = whole[:_IDENTITY_WIDTH] + (
            () if outcome is None or outcome >= len(row) else (row[outcome],))
        if ident in identities:
            disagreeing.append(line)
        seen[whole] = line
        identities.setdefault(ident, line)
        kept.append(row)
    return kept, dropped, disagreeing


def _record(path: str, header: List[str],
            dropped: List[Tuple[int, List[str]]], record: Optional[str]) -> str:
    """Append what was removed to the purge record; returns where it went.

    In `RECORDS_DIR`, outside every results tree. It used to go beside the
    tree, so the next reader to glob `<tree>/*.csv` — this tool's own reader
    among them — took the record for results and could delete rows from the
    record of their own removal (it happened on 2026-10-29, from a reader we
    do not control). Written through the one writer, which aligns by column
    NAME: these rows carry whatever schema their file had.
    """
    where = record or os.path.join(
        RECORDS_DIR, f"bench_purge_{time.strftime('%Y-%m-%d')}_duplicate_row.csv")
    return append_record(where, [
        {**dict(zip(header, row, strict=False)),
         "source_csv": path, "row_id": str(line),
         "purge_reason": _REASON} for line, row in dropped])


def _files(args: argparse.Namespace) -> List[str]:
    """The files to read: those named, plus every results CSV of each
    `--dir`, through the one entry point — which skips purge records and
    refuses a tree that still holds one."""
    return list(args.csv) + result_files(args.dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="*", default=[])
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", default=None,
                        help="where to append what was removed (default: "
                             "bench_purge_<date>_duplicate_row.csv beside the "
                             "results tree)")
    parser.add_argument("--no-appender", action="store_true",
                        help="the caller knows no run is writing these files "
                             "(they are named by a read failure), so do not "
                             "wait out the settle window")
    args = parser.parse_args(argv)

    files = _files(args)
    if not files:
        print("no CSVs named", file=sys.stderr)
        return 2
    fixed = removed = skipped = 0
    for path in files:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            continue
        header, body = rows[0], rows[1:]
        outcome = header.index("outcome") if "outcome" in header else None
        kept, dropped, disagreeing = _collisions(body, outcome)
        for line in disagreeing:
            print(f"{path}:{line}: an identity repeats with DIFFERENT "
                  f"contents — left alone, pick one by hand")
        if not dropped:
            continue
        if (not args.no_appender
                and time.time() - os.path.getmtime(path) < _SETTLED_S):
            skipped += 1
            print(f"{path}: {len(dropped)} identical repeats, but the harness "
                  f"wrote this file in the last {_SETTLED_S // 60} minutes "
                  f"— skipped")
            continue
        print(f"{path}: {len(dropped)} identical repeats "
              f"({len(kept)} rows kept)")
        removed += len(dropped)
        if args.dry_run:
            continue
        # Recorded BEFORE the file is rewritten: a record written after a
        # successful rewrite is a record that does not exist when the
        # rewrite is what failed.
        where = _record(path, header, dropped, args.record)
        temporary = f"{path}.dedup"
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(kept)
        os.replace(temporary, path)
        fixed += 1
        print(f"  recorded in {where}")
    print(f"{len(files)} files, {removed} rows removed from {fixed}"
          + (f", {skipped} skipped as live" if skipped else "")
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
