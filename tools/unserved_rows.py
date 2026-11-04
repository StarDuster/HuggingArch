"""Measured rows whose provider no longer says it serves that cell.

A provider declares the cells its kernel is the kernel FOR
(`KernelSignature`: the phases, the enumerated axis values, the divisibility,
the ceilings). When that declaration narrows — because someone read the
kernel and found it serves less than the harness had been handing it — the
rows already measured under the old reading do not become doubtful, they
become WRONG: each one timed a launch that is not the one its row prices.
`causal_conv1d`'s decode build is the case this was written for. It took
`x[:, :, -1]` at any `s_query`, so 733 rows timed one token against a row
billed for S of them; the provider now declares `s_query = 1`, and those
733 are exactly the rows this finds.

Derived rather than listed: nothing here names an op, a kernel or a model.
It asks the registry the same question the plan asks before building a cell,
and a row the registry refuses is a row no run would make today.

    python -m tools.unserved_rows                      # every local mirror
    python -m tools.unserved_rows --apply              # delete them
    python -m tools.unserved_rows --dir <results dir>  # one tree
    python -m tools.unserved_rows --lake [--apply]     # the bench lake

The lake is asked through the same verdict, because a row has to be refused
for one reason wherever it is stored. Order matters when both hold the same
rows: the BOX first, then the mirror, then the lake — the mirror pulls with
`rsync -a`, which restores whatever the box still has.

A row whose provider is no longer registered at all is reported apart and
never deleted: an impl that was removed says nothing about whether its
measurements were true.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import sys
from collections import Counter
from typing import Any, Mapping, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#: Every tree a box's results are kept in: the mirrors of each H100 box, the
#: layout that predates the per-box subdirectory, and the local queue's own
#: tree. All plain CSV written by `kernel_bench.emit`.
#:
#: `.cache/h100_results/<box>/results` is a LIVE mirror — `upload_loop.sh`
#: rsyncs the box into it, so a purge there has to follow the box's, or the
#: next pull restores what it removed. `.cache/h100_results_fresh` is a
#: frozen snapshot (its own jobs/ and queue.log; nothing rsyncs into it) and
#: `.cache/bench4090/results` is the local queue's own tree: purging either
#: is durable on its own. `.cache/prov_rows_pending_upload` holds rows
#: measured and not yet in the lake — left there, a revoked row is uploaded
#: again later.
#:
#: Named one by one rather than globbed. `.cache/h100_results/*/results`
#: reads as "every box's mirror" but the legacy tree's results directory
#: sits a level HIGHER, so the glob skipped it — and a purge that skips a
#: tree deletes the lake's copy of a row while leaving the box's (846
#: indexer rows, found 2026-10-29). A missing name is now visible here
#: instead of hiding in a pattern.
DEFAULT_TREES = (
    ".cache/h100_results/results",          # before the per-box subdirectory
    ".cache/h100_results/h100-a/results",
    ".cache/h100_results/h100-b/results",
    ".cache/h100_results_fresh/results",
    ".cache/bench4090/results",             # the local queue's own
    ".cache/prov_rows_pending_upload",      # measured, not yet uploaded
)


#: Where results trees live, so one that nobody listed can still be FOUND.
CACHE_ROOT = ".cache"
#: The service's own directory. Everything under it — the data files, the
#: staging area, the corpus exports — is the LAKE's copy of a row, which is
#: revoked by id through the service (`store.delete_rows`) and re-exported
#: from what survives. Not a tree anything here rewrites.
LAKE_DIR = os.path.join(CACHE_ROOT, "bench_lake")


#: What a purge record is called (`_record`). Matched anywhere in the file
#: name, because a record gets copied and relabelled — `box_bench_purge_…`
#: off a box, `<tree>__bench_purge_…` when records from several trees are
#: gathered, `….dryrun.csv` beside an applied one — and every one of those
#: is still a record, never a results file.
RECORD_MARK = "bench_purge"
#: Where records belong: OUTSIDE every results tree. A record kept in the
#: tree it documents is read back as results by the next tool that globs
#: `<tree>/*.csv`, and every row in it is a row that tool may select again
#: — only to "remove" it from the record of its own removal. Teaching each
#: reader to skip them is not enough: on 2026-10-29 a purge running from
#: another checkout, at a commit without the skip, moved 21 rows out of a
#: record this way. So the records live apart, and a tree that still holds
#: one stops the run (`result_files`).
RECORDS_DIR = os.path.join(CACHE_ROOT, "purge_records")


def is_record(path: str) -> bool:
    """Whether `path` is a purge record rather than results."""
    return RECORD_MARK in os.path.basename(path)


def results_trees(root: str = CACHE_ROOT) -> Tuple[str, ...]:
    """Every directory under `root` holding result CSVs, found by shape
    rather than by name — so a tree nobody listed is still visible.

    The shape is the one `_files` reads: CSVs one subdirectory down (per
    model) or directly in the directory, whose header carries
    `kernel_impl`, the column that makes a row a measurement or a refusal
    of one. A directory under another tree is that tree's (`_files` globs
    both depths). Purge records (`is_record`) are not results, and the
    lake's own directory (`LAKE_DIR`) is not a tree.
    """
    import glob
    found = set()
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        if (is_record(path)
                or os.path.commonpath([LAKE_DIR, path]) == LAKE_DIR
                or os.path.commonpath([RECORDS_DIR, path]) == RECORDS_DIR):
            continue
        try:
            with open(path, newline="") as handle:
                header = [c.strip().strip('"')
                          for c in (handle.readline() or "").split(",")]
        except OSError:
            continue
        if "kernel_impl" not in header:
            continue
        model = os.path.dirname(path)
        tree = os.path.dirname(model)
        found.add(model if tree == root or not tree else tree)
    # A tree inside a tree is the outer one's: `--dir <outer>` reads it.
    return tuple(sorted(
        t for t in found
        if not any(o != t and t.startswith(o + os.sep) for o in found)))


def default_trees(trees: Iterable[str] = DEFAULT_TREES,
                  root: Optional[str] = CACHE_ROOT) -> Tuple[str, ...]:
    """The trees to read when the caller named none: `DEFAULT_TREES`, and
    a refusal if the disk and the list disagree either way.

    A LISTED tree that is not here, while others are: on this machine that
    means a tree moved or a box was renamed, and a run over the rest would
    report a complete pass over an incomplete set. (No tree at all is
    simply another machine, and the caller's own "no result CSVs found" is
    the right answer.)

    A tree that is HERE and not listed: the failure that bit. A purge or an
    audit reading the list alone passes straight over a new box's mirror and
    says nothing — 120,497 rows were left behind in one unlisted tree on
    2026-10-29, 113,581 of them measured. So the trees are also FOUND
    (`results_trees`), and a found tree nobody listed stops the run.
    """
    here = tuple(t for t in trees if os.path.isdir(t))
    missing = [t for t in trees if t not in here]
    if here and missing:
        raise SystemExit(
            "results trees are missing: " + ", ".join(missing)
            + f"\n{len(here)} of {len(trees)} are here, so this is not a "
            f"machine without results — a tree moved, or a box was renamed "
            f"and `DEFAULT_TREES` was not told. Name the trees with --dir, "
            f"or add the new one there; a run over the rest would report a "
            f"complete pass over an incomplete set.")
    unlisted = [t for t in results_trees(root) if t not in set(trees)
                ] if root is not None else []
    if unlisted:
        raise SystemExit(
            "results trees are here that `DEFAULT_TREES` does not name: "
            + ", ".join(unlisted)
            + "\nA new box's mirror must not be invisible to a purge or an "
            "audit: reading the list alone, this run would pass over those "
            "trees and report a complete pass. Add them to "
            "`DEFAULT_TREES`, or name the trees to read with --dir.")
    return here
#: A results CSV written this recently is one the harness may still be
#: appending to; an affected one stops the run before any file is rewritten.
FRESH_SECONDS = 600


def _rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


#: Where a removal is written down, in the corpus's purge-record
#: convention: every row removed, in full, plus the file it came from and
#: the reason ON EVERY ROW. Without it a gap in the corpus is unreadable a
#: year out — nobody can tell a defect from a cleanup — and the reason has
#: to say which CLAUSE refused and what evidence settled it, because the
#: same refusal covers rows that must go and rows that must stay.
#:
#: An EMPTY `outcome` is a measurement, not a missing value: that is how
#: `benchmarks.outcome_of` and the store's `OUTCOME` macro read it, and the
#: rows of a CSV written before the column existed have no field to carry.
_RECORD_COLUMNS = ("source_csv", "row_id", "purge_reason")


def append_record(where: str, rows: Iterable[Dict[str, str]]) -> str:
    """Append `rows` to the purge record at `where`, by COLUMN NAME.

    The record is the audit trail, so its columns may never shift under a
    row. A `DictWriter` built from one source file's header wrote a LATER
    file's row in that later file's own order while the record's first line
    still declared the first header: 82 rows of a 46-column record came out
    36 or 37 columns wide, their outcome read as a kernel symbol and their
    reason gone (2026-10-29). So the record's own header is read back, and
    a row carrying a column the record does not have WIDENS it: the file is
    rewritten once, by name, with the union — never by position, and never
    dropping a column a row had.
    """
    columns: List[str] = []
    existing: List[Dict[str, str]] = []
    if os.path.isfile(where) and os.path.getsize(where):
        columns, existing = _rows(where)
    incoming = list(rows)
    # The records directory may not exist yet on a machine whose first
    # purge this is; a record that cannot be written is a removal with no
    # audit trail, and the rewrite is about to follow.
    os.makedirs(os.path.dirname(os.path.abspath(where)), exist_ok=True)
    union = list(dict.fromkeys(
        list(columns) + [c for row in incoming for c in row]))
    if columns and union != columns:
        # Widen in place, atomically: same rows, same values, more columns.
        temporary = f"{where}.widening"
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=union)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(temporary, where)
        columns = union
    with open(where, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=union, extrasaction="raise")
        if not columns:
            writer.writeheader()
        writer.writerows(incoming)
    return where


def _record(path: str, header: List[str], dropped: List[Tuple[int, Dict[str, str]]],
            reason: str, record: Optional[str]) -> str:
    """Append the removed rows of one results file to the purge record;
    returns where it went. `header` is that file's own column order, kept
    for readability — `append_record` aligns by name.

    Written BEFORE the file is rewritten: a record written after a
    successful rewrite does not exist in the case where the rewrite is what
    failed.
    """
    where = record or os.path.join(
        RECORDS_DIR, f"bench_purge_{time.strftime('%Y-%m-%d')}_unserved.csv")
    return append_record(where, [
        {**{c: row.get(c, "") for c in header}, **row,
         "source_csv": path, "row_id": str(line), "purge_reason": reason}
        for line, row in dropped])


def _json_column(value: Any, default: str = "{}") -> Dict[str, Any]:
    """One of a row's JSON columns, whichever shape it arrived in.

    A CSV hands these over as TEXT and the bench service hands them over
    already PARSED, and the difference is not cosmetic: `str()` on the
    service's dict gives a Python repr, `json.loads` refuses it, and every
    lake row then reads as `unregistered` — which is exactly how this tool's
    `--lake` mode found nothing to do, and the same mistake
    `tools/restamp_peaks.py` made against the same boundary two days before
    (there it was swallowed and defaulted to bf16, which rewrote 24,920 rows
    with the wrong peak). So the shape is ASKED, never assumed, and never
    re-stringified on the way in.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(value or default)


def _invocation(row: Dict[str, Any]):
    """The cell this row says it measured, as the plan would state it."""
    from kernel_bench.invocation import Invocation
    return Invocation.from_job({
        "op": row["op"],
        "dims": _json_column(row["shape_json"]),
        "dtype_env": _json_column(row["dtype_env_json"]),
        "phase": row.get("phase") or "prefill",
        "kernel_params": _json_column(row.get("kernel_params_json")),
    })


def _verdict(row: Dict[str, str], providers) -> str:
    """`"served"`, `"refused"`, or `"unregistered"` for this row.

    Asked of the CAPABILITY clauses alone (`samples_up_to` dropped): a
    sampling budget declines to measure more of a kernel that runs the cell
    perfectly well, so the rows it already has are honest and a budget that
    arrives later must not delete them."""
    import dataclasses

    from kernel_bench.providers import catalog_kernel
    # A row is keyed by its implementation's `impl_id`: the kernel_impl,
    # then `@` and the deployment parameters it folds in
    # (`Provider.impl_id`, e.g. `flashinfer_paged_decode@_b16_tc`). Matched
    # whole, every parameterised implementation read as "no longer
    # registered", and its refused rows were kept (350k on the mirrors,
    # 2026-10-26).
    impl = catalog_kernel(row.get("kernel_impl") or "")
    library = row.get("framework") or ""
    matches = [p for p in providers
               if p.kernel_impl == impl and (not library or p.library == library)]
    if not matches:
        return "unregistered", "no provider of that name"
    try:
        inv = _invocation(row)
    except Exception as error:
        return "unregistered", f"{type(error).__name__}: {error}"
    why: List[str] = []
    for provider in matches:
        try:
            capability = dataclasses.replace(provider.serves, samples_up_to=())
            said = list(capability.refusals(inv))
            if not said:
                return "served", ""
            why.extend(said)
        except Exception:
            # A declaration that cannot judge the row keeps it: the row is
            # evidence and the gate is the thing in doubt.
            return "served", ""
    # WHICH CLAUSE refused is the whole of what a caller needs to decide
    # whether the row is wrong or the declaration moved under it. A cell
    # whose int32 shape overflowed is a bad row; one refused by
    # `divisible:s_expert_rows%128` after a dim was renamed is a good
    # measurement whose declaration was edited, and deleting it loses a
    # measurement nobody will re-take.
    return "refused", ",".join(dict.fromkeys(why))


def _purge_lake(providers, apply: bool, because: str = "",
                reason: str = "", record: Optional[str] = None
                ) -> Tuple[int, int]:
    """Every row the lake holds, through the same verdict.

    `because` selects which clause's refusals go, exactly as in file mode:
    the lake holds the same two kinds of refusal (a row whose cell is
    impossible, and a good measurement whose declaration moved under it) and
    a delete that cannot tell them apart is not offered here either.
    """
    from tools import lake_rows
    doomed: List[str] = []
    refused: Counter = Counter()
    scanned = foreign = 0
    # Read off a read-only copy of the catalog (`tools.lake_rows`), never by
    # paging the service: over the whole lake that ran it out of memory.
    for row in lake_rows.rows():
        scanned += 1
        flat = {k: ("" if v is None else v) for k, v in row.items()}
        verdict, why = _verdict(flat, providers)
        if verdict != "refused":
            continue
        if because and not any(
                token.startswith(because) for token in why.split(",")):
            continue                # refused, but not for the named reason
        if flat.pop("owner") != lake_rows.OWNER:
            foreign += 1            # the service deletes only its owner's
            continue
        doomed.append((flat["id"], why, flat))
        refused[f"{flat.get('framework', '')}:"
                f"{flat.get('kernel_impl', '')} ({why})"] += 1
    print(f"lake: {scanned} rows scanned, {len(doomed)} refused"
          + (f" ({foreign} more another owner holds, not deletable here)"
             if foreign else ""))
    for label, count in refused.most_common():
        print(f"  {count:>7}  {label}")
    if apply and doomed:
        where = record or os.path.join(
            RECORDS_DIR,
            f"bench_purge_{time.strftime('%Y-%m-%d')}_unserved_lake.csv")
        append_record(where, [
            {**row, "source_csv": "lake", "row_id": rid,
             "purge_reason": reason or f"refused by {why}"}
            for rid, why, row in doomed])
        print(f"recorded what was removed in {where}")
        deleted = lake_rows.delete([r[0] for r in doomed])
        print(f"  deleted {deleted}")
    else:
        deleted = 0
    return len(doomed), deleted


def result_files(dirs: Iterable[str]) -> List[str]:
    """Every results CSV of `dirs` — the one answer to "which files hold
    measured rows", for every tool that reads or rewrites them.

    Each tool used to carry this glob itself, and only one of them skipped
    the purge RECORDS that then lived in the same directories:
    `restamp_peaks --apply` would have rewritten an audit record's peak
    columns, and `launch_blind_spots` counted deleted rows as corpus. One
    entry point, so a reader cannot forget the exclusion.

    Two guards, because the skip alone only protects the readers we own: a
    record here is not read as results, AND a tree that holds one stops the
    run. Records belong in `RECORDS_DIR`, outside every tree, where no
    `<tree>/*.csv` glob — ours or another checkout's — can reach them.
    """
    found = _files(dirs)
    stray = sorted({f for d in dirs for f in _glob_tree(d) if is_record(f)})
    if stray:
        raise SystemExit(
            "purge records are inside a results tree: " + ", ".join(stray)
            + f"\nA record read as results is a row deleted from the record "
            f"of its own removal — it has happened. Move them to "
            f"{RECORDS_DIR}/, which is where records belong, and rerun.")
    return found


def _glob_tree(pattern: str) -> List[str]:
    """Every CSV a results tree holds, records included: both depths
    `_files` reads."""
    import glob
    out: List[str] = []
    for directory in glob.glob(pattern):
        out.extend(sorted(glob.glob(os.path.join(directory, "*", "*.csv"))))
        out.extend(sorted(glob.glob(os.path.join(directory, "*.csv"))))
    return out


def _files(dirs: Iterable[str]) -> List[str]:
    return [f for pattern in dirs for f in _glob_tree(pattern)
            if not is_record(f)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", action="append", default=[],
                        help="a results tree (repeatable); default: every "
                             "tree in `DEFAULT_TREES`")
    parser.add_argument("--reason", default="",
                        help="what goes in the record's `purge_reason` on "
                             "every removed row: the clause AND the evidence "
                             "that settled it (a source anchor, a measured "
                             "number). Defaults to naming the clause alone, "
                             "which is rarely enough a year out")
    parser.add_argument("--record", default=None,
                        help="where to append the purge record (default: "
                             "<RECORDS_DIR>/bench_purge_<date>_unserved.csv, "
                             "outside every results tree)")
    parser.add_argument("--because", default="",
                        help="only rows refused by a clause starting with "
                             "this (`max_dims`, `divisible:s_expert_rows`, "
                             "…). REQUIRED with --apply: a blanket delete "
                             "cannot tell a row whose cell is wrong from one "
                             "whose declaration was edited under it, and the "
                             "second is a measurement nobody will re-take")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite each file without its refused rows")
    parser.add_argument("--lake", action="store_true",
                        help="the bench lake instead of files: rows read "
                             "off a read-only catalog copy, deletes through "
                             "the local service (tools.lake_rows)")
    args = parser.parse_args(argv)
    if args.apply and not args.because:
        parser.error(
            "--apply needs --because <clause>: the refusals are of two "
            "kinds and only one of them should go. A row whose declared "
            "cell is impossible (an int32 shape that overflowed) is a bad "
            "row; a row refused by `divisible:s_expert_rows%128` after that "
            "dim was renamed is a good measurement whose declaration moved "
            "under it, and deleting it loses a measurement nobody will "
            "re-take. Run without --apply first and read the clause column")

    from kernel_bench.providers import all_providers
    providers = list(all_providers())
    if args.lake:
        found, deleted = _purge_lake(providers, args.apply, args.because,
                                     args.reason, args.record)
        if not args.apply:
            print("\n(dry run; --apply to delete)")
            return 0
        if deleted != found:
            print(f"FAILED: the lake deleted {deleted} of the {found} rows "
                  f"listed", file=sys.stderr)
            return 5
        return 0
    files = _files(args.dir or default_trees())
    if not files:
        print("no result CSVs found", file=sys.stderr)
        return 1

    refused: Counter = Counter()
    unregistered: Counter = Counter()
    touched = 0
    records: set = set()
    planned: List[Tuple[str, List[str], List[Dict[str, str]],
                        List[Tuple[int, Dict[str, str]]]]] = []
    for path in files:
        fields, rows = _rows(path)
        if not rows or "kernel_impl" not in (fields or []):
            continue
        keep: List[Dict[str, str]] = []
        removed: List[Tuple[int, Dict[str, str]]] = []
        dropped = 0
        for row in rows:
            verdict, why = _verdict(row, providers)
            label = f"{row.get('framework','')}:{row.get('kernel_impl','')}"
            if verdict == "refused":
                refused[(label, row.get("op", ""), why)] += 1
                if args.because and not any(
                        token.startswith(args.because) for token in why.split(",")):
                    keep.append(row)   # refused, but not for the named reason
                    continue
                dropped += 1
                removed.append((len(keep) + dropped + 1, row))
                continue
            if verdict == "unregistered":
                unregistered[label] += 1
            keep.append(row)
        if dropped:
            planned.append((path, fields, keep, removed))

    # FAIL CLOSED: every file is listed before any is rewritten, and none is
    # if an affected one is still being written — the harness appends to the
    # CSV it is measuring into, and a rewrite moved over it loses those rows.
    # The run then exits non-zero, so a box -> mirror chain stops there.
    busy = [path for path, *_ in planned
            if time.time() - os.path.getmtime(path) < FRESH_SECONDS]
    if busy:
        print(f"ABORT: {len(busy)} affected file(s) written in the last "
              f"{FRESH_SECONDS}s (the harness is appending); nothing "
              f"rewritten — rerun when they are quiet:", file=sys.stderr)
        for path in busy:
            print(f"    {path}", file=sys.stderr)
        return 3
    for path, fields, keep, removed in planned if args.apply else ():
        wrote = _record(path, fields, removed,
                        args.reason or f"refused by {args.because}",
                        args.record)
        records.add(wrote)
        # Written beside and moved into place: a reader (the upload loop)
        # sees the old file or the new one, never half of one.
        temporary = f"{path}.rewriting"
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep)
        os.replace(temporary, path)
        touched += 1

    if records:
        for where in sorted(records):
            print(f"recorded what was removed in {where}")
    print(f"{len(files)} files, {sum(refused.values())} rows the registry "
          f"refuses:")
    for (label, op, why), count in refused.most_common():
        print(f"  {count:>7}  {label}  ({op})  {why}")
    if unregistered:
        print(f"\n{sum(unregistered.values())} rows of impls no longer "
              f"registered (kept — a removed impl says nothing about its "
              f"measurements):")
        for label, count in unregistered.most_common(10):
            print(f"  {count:>7}  {label}")
    if args.apply:
        print(f"\nrewrote {touched} files")
    else:
        print("\n(dry run; --apply to delete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
